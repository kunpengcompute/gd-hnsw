#include "hnsw_urma/core/page_aligned_buffer.hpp"

#include <atomic>
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <utility>

#include <unistd.h>

#include "hnsw_urma/core/checked_math.hpp"

namespace hnsw_urma {
namespace {

std::atomic<std::uint32_t> shared_name_sequence{
    static_cast<std::uint32_t>(
        std::chrono::steady_clock::now().time_since_epoch().count())};

Status SharedMemoryStatus(int error, std::string operation,
                          std::string_view name = {}) {
  auto status = Status(error == ENOMEM || error == ENOSPC
                           ? StatusCode::kResourceExhausted
                           : StatusCode::kInternalError,
                       std::move(operation))
                    .AddContext("errno", std::to_string(error));
  if (!name.empty()) {
    status.AddContext("shared_memory_name", std::string(name));
  }
  return status;
}

bool ValidSharedMemoryName(std::string_view name) {
  return name.size() > 1 && name.front() == '/' &&
         name.find('/', 1) == std::string_view::npos;
}

}  // namespace

PageAlignedBuffer::~PageAlignedBuffer() { Reset(); }

PageAlignedBuffer::PageAlignedBuffer(PageAlignedBuffer&& other) noexcept {
  *this = std::move(other);
}

PageAlignedBuffer& PageAlignedBuffer::operator=(PageAlignedBuffer&& other) noexcept {
  if (this != &other) {
    Reset();
    data_ = std::exchange(other.data_, nullptr);
    logical_bytes_ = std::exchange(other.logical_bytes_, 0);
    registered_bytes_ = std::exchange(other.registered_bytes_, 0);
    page_size_ = std::exchange(other.page_size_, 0);
    storage_ = std::exchange(other.storage_, Storage::kEmpty);
    shared_owner_ = std::exchange(other.shared_owner_, false);
    shared_name_ = std::move(other.shared_name_);
  }
  return *this;
}

Result<std::size_t> PageAlignedBuffer::SystemPageSize() {
  const long page_size = ::sysconf(_SC_PAGESIZE);
  if (page_size <= 0) {
    return Status(StatusCode::kInternalError, "unable to determine the host page size");
  }
  const auto value = static_cast<std::size_t>(page_size);
  if (value < sizeof(void*) || (value & (value - 1U)) != 0) {
    return Status(StatusCode::kInternalError,
                  "host page size is not a valid allocation alignment")
        .AddContext("page_size", std::to_string(value));
  }
  return value;
}

Result<PageAlignedBuffer> PageAlignedBuffer::Create(std::size_t logical_bytes) {
  if (logical_bytes == 0) {
    return Status(StatusCode::kInvalidArgument,
                  "page-aligned buffer length must be non-zero");
  }
  auto page_size = SystemPageSize();
  if (!page_size.ok()) {
    return page_size.status();
  }
  auto registered_bytes = CheckedAlignUp(logical_bytes, page_size.value());
  if (!registered_bytes.ok()) {
    return registered_bytes.status();
  }

  void* allocation = nullptr;
  const int allocate_status =
      ::posix_memalign(&allocation, page_size.value(), registered_bytes.value());
  if (allocate_status != 0 || allocation == nullptr) {
    return Status(allocate_status == ENOMEM ? StatusCode::kResourceExhausted
                                            : StatusCode::kInternalError,
                  "unable to allocate page-aligned memory")
        .AddContext("logical_bytes", std::to_string(logical_bytes))
        .AddContext("registered_bytes", std::to_string(registered_bytes.value()))
        .AddContext("page_size", std::to_string(page_size.value()))
        .AddContext("allocator_status", std::to_string(allocate_status));
  }

  PageAlignedBuffer buffer;
  buffer.data_ = static_cast<std::byte*>(allocation);
  buffer.logical_bytes_ = logical_bytes;
  buffer.registered_bytes_ = registered_bytes.value();
  buffer.page_size_ = page_size.value();
  buffer.storage_ = Storage::kHeap;
  std::memset(buffer.data_, 0, buffer.registered_bytes_);
  return buffer;
}

Result<PageAlignedBuffer> PageAlignedBuffer::CopyFrom(
    std::span<const std::byte> source) {
  auto buffer = Create(source.size());
  if (!buffer.ok()) {
    return buffer.status();
  }
  std::memcpy(buffer.value().data_, source.data(), source.size());
  return std::move(buffer).value();
}

Result<PageAlignedBuffer> PageAlignedBuffer::CreateShared(
    std::size_t logical_bytes) {
  if (logical_bytes == 0) {
    return Status(StatusCode::kInvalidArgument,
                  "shared buffer length must be non-zero");
  }
  auto page_size = SystemPageSize();
  if (!page_size.ok()) {
    return page_size.status();
  }
  auto registered_bytes = CheckedAlignUp(logical_bytes, page_size.value());
  if (!registered_bytes.ok() ||
      registered_bytes.value() >
          static_cast<std::size_t>(std::numeric_limits<off_t>::max())) {
    return registered_bytes.ok()
               ? Status(StatusCode::kOutOfRange,
                        "shared buffer exceeds the shm object size limit")
               : registered_bytes.status();
  }

  int descriptor = -1;
  std::string name;
  int open_error = EEXIST;
  for (unsigned attempt = 0; attempt < 64 && descriptor < 0; ++attempt) {
    char candidate[32]{};
    const auto sequence =
        shared_name_sequence.fetch_add(1, std::memory_order_relaxed);
    std::snprintf(candidate, sizeof(candidate), "/hu%08x%08x",
                  static_cast<unsigned>(::getpid()),
                  static_cast<unsigned>(sequence));
    descriptor = ::shm_open(candidate, O_RDWR | O_CREAT | O_EXCL, 0600);
    if (descriptor >= 0) {
      name = candidate;
      break;
    }
    open_error = errno;
    if (open_error != EEXIST) {
      break;
    }
  }
  if (descriptor < 0) {
    return SharedMemoryStatus(open_error,
                              "unable to create shared memory object");
  }
  if (::ftruncate(descriptor,
                  static_cast<off_t>(registered_bytes.value())) != 0) {
    const int resize_error = errno;
    ::close(descriptor);
    ::shm_unlink(name.c_str());
    return SharedMemoryStatus(resize_error,
                              "unable to size shared memory object", name);
  }
  void* mapping = ::mmap(nullptr, registered_bytes.value(),
                         PROT_READ | PROT_WRITE, MAP_SHARED, descriptor, 0);
  const int map_error = errno;
  ::close(descriptor);
  if (mapping == MAP_FAILED) {
    ::shm_unlink(name.c_str());
    return SharedMemoryStatus(map_error,
                              "unable to map shared memory object", name);
  }

  PageAlignedBuffer buffer;
  buffer.data_ = static_cast<std::byte*>(mapping);
  buffer.logical_bytes_ = logical_bytes;
  buffer.registered_bytes_ = registered_bytes.value();
  buffer.page_size_ = page_size.value();
  buffer.storage_ = Storage::kShared;
  buffer.shared_owner_ = true;
  buffer.shared_name_ = std::move(name);
  std::memset(buffer.data_, 0, buffer.registered_bytes_);
  return buffer;
}

Result<PageAlignedBuffer> PageAlignedBuffer::CopyFromShared(
    std::span<const std::byte> source) {
  auto buffer = CreateShared(source.size());
  if (!buffer.ok()) {
    return buffer.status();
  }
  std::memcpy(buffer.value().data_, source.data(), source.size());
  return std::move(buffer).value();
}

Result<PageAlignedBuffer> PageAlignedBuffer::OpenShared(
    std::string_view name, std::size_t logical_bytes) {
  if (!ValidSharedMemoryName(name) || logical_bytes == 0) {
    return Status(StatusCode::kInvalidArgument,
                  "shared memory name or length is invalid");
  }
  auto page_size = SystemPageSize();
  if (!page_size.ok()) {
    return page_size.status();
  }
  auto registered_bytes = CheckedAlignUp(logical_bytes, page_size.value());
  if (!registered_bytes.ok()) {
    return registered_bytes.status();
  }
  const std::string owned_name(name);
  const int descriptor = ::shm_open(owned_name.c_str(), O_RDWR, 0600);
  if (descriptor < 0) {
    return SharedMemoryStatus(errno, "unable to open peer shared memory",
                              name);
  }
  struct stat metadata {};
  if (::fstat(descriptor, &metadata) != 0) {
    const int stat_error = errno;
    ::close(descriptor);
    return SharedMemoryStatus(stat_error,
                              "peer shared memory size is invalid", name);
  }
  if (metadata.st_size < 0 ||
      static_cast<std::uintmax_t>(metadata.st_size) <
          registered_bytes.value()) {
    ::close(descriptor);
    return SharedMemoryStatus(EINVAL,
                              "peer shared memory size is invalid", name);
  }
  void* mapping = ::mmap(nullptr, registered_bytes.value(),
                         PROT_READ | PROT_WRITE, MAP_SHARED, descriptor, 0);
  const int map_error = errno;
  ::close(descriptor);
  if (mapping == MAP_FAILED) {
    return SharedMemoryStatus(map_error,
                              "unable to map peer shared memory", name);
  }

  PageAlignedBuffer buffer;
  buffer.data_ = static_cast<std::byte*>(mapping);
  buffer.logical_bytes_ = logical_bytes;
  buffer.registered_bytes_ = registered_bytes.value();
  buffer.page_size_ = page_size.value();
  buffer.storage_ = Storage::kShared;
  buffer.shared_name_ = owned_name;
  return buffer;
}

std::span<std::byte> PageAlignedBuffer::logical_memory() noexcept {
  return {data_, logical_bytes_};
}

std::span<const std::byte> PageAlignedBuffer::logical_memory() const noexcept {
  return {data_, logical_bytes_};
}

std::span<std::byte> PageAlignedBuffer::registration_memory() noexcept {
  return {data_, registered_bytes_};
}

std::span<const std::byte> PageAlignedBuffer::registration_memory() const noexcept {
  return {data_, registered_bytes_};
}

Status PageAlignedBuffer::UnlinkSharedMemory() {
  if (storage_ != Storage::kShared || !shared_owner_ ||
      shared_name_.empty()) {
    return Status(StatusCode::kInvalidArgument,
                  "buffer does not own a shared memory name");
  }
  if (::shm_unlink(shared_name_.c_str()) != 0 && errno != ENOENT) {
    return SharedMemoryStatus(errno,
                              "unable to unlink shared memory object",
                              shared_name_);
  }
  shared_owner_ = false;
  shared_name_.clear();
  return Status::Ok();
}

void PageAlignedBuffer::Reset() noexcept {
  if (storage_ == Storage::kHeap) {
    std::free(data_);
  } else if (storage_ == Storage::kShared && data_ != nullptr) {
    ::munmap(data_, registered_bytes_);
  }
  if (shared_owner_ && !shared_name_.empty()) {
    ::shm_unlink(shared_name_.c_str());
  }
  data_ = nullptr;
  logical_bytes_ = 0;
  registered_bytes_ = 0;
  page_size_ = 0;
  storage_ = Storage::kEmpty;
  shared_owner_ = false;
  shared_name_.clear();
}

}  // namespace hnsw_urma

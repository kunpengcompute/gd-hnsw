#include "hnsw_urma/transport/urma/urma_transport.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <deque>
#include <limits>
#include <mutex>
#include <new>
#include <unordered_map>
#include <utility>

#include <urma_api.h>
#if HNSW_URMA_HAS_BONDING_HEADER
#include <urma_ubagg.h>
#endif

#include "hnsw_urma/core/page_aligned_buffer.hpp"

namespace hnsw_urma {
namespace {

constexpr std::size_t kProviderPollLimit = 16;
constexpr std::size_t kNoBounceSlot = static_cast<std::size_t>(-1);
constexpr std::size_t kNoPendingSlot = static_cast<std::size_t>(-1);

std::mutex g_provider_runtime_mutex;
std::size_t g_provider_runtime_users{0};

Status AcquireProviderRuntime() {
  std::lock_guard lock(g_provider_runtime_mutex);
  if (g_provider_runtime_users == 0) {
    urma_init_attr_t attributes{};
    const auto status = urma_init(&attributes);
    if (status != URMA_SUCCESS) {
      return MapUrmaProviderStatus(status, "urma_init");
    }
  }
  ++g_provider_runtime_users;
  return Status::Ok();
}

void ReleaseProviderRuntime() noexcept {
  std::lock_guard lock(g_provider_runtime_mutex);
  if (g_provider_runtime_users == 0) {
    return;
  }
  --g_provider_runtime_users;
  if (g_provider_runtime_users == 0) {
    (void)urma_uninit();
  }
}

Status NullProviderResult(const char* operation) {
  return Status(StatusCode::kTransportError, std::string(operation) + " returned null")
      .AddContext("errno", std::to_string(errno));
}

OperationClass NormalizeOperationClass(OperationClass operation_class) {
  return operation_class;
}

const char* OperationClassName(OperationClass operation_class) noexcept {
  switch (operation_class) {
    case OperationClass::kRead:
      return "read";
    case OperationClass::kRequestWrite:
      return "request_write";
    case OperationClass::kResponseWrite:
      return "response_write";
  }
  return "unknown";
}

const char* SegmentKindName(SegmentKind kind) noexcept {
  switch (kind) {
    case SegmentKind::kLayer0Graph:
      return "layer0_graph";
    case SegmentKind::kVectors:
      return "vectors";
    case SegmentKind::kRequests:
      return "requests";
    case SegmentKind::kResponses:
      return "responses";
    case SegmentKind::kNotify:
      return "notify";
    case SegmentKind::kRpcPool:
      return "rpc_pool";
  }
  return "unknown";
}

const char* CompletionQueueClassName(std::size_t queue_index) noexcept {
  switch (queue_index) {
    case 0:
      return "read";
    case 1:
      return "request_write";
    case 2:
      return "response_write";
    default:
      return "unknown";
  }
}

const char* CompletionStatusName(urma_cr_status_t status) noexcept {
  switch (status) {
    case URMA_CR_SUCCESS:
      return "URMA_CR_SUCCESS";
    case URMA_CR_LOC_ACCESS_ERR:
      return "URMA_CR_LOC_ACCESS_ERR";
    default:
      // Keep provider extensions numeric instead of guessing their meaning.
      return "unknown";
  }
}

std::size_t AcceptedPrefix(const urma_jfs_wr_t* begin, std::size_t count,
                           const urma_jfs_wr_t* bad_work_request) noexcept {
  if (begin == nullptr || bad_work_request == nullptr) {
    return 0;
  }
  const auto* current = begin;
  for (std::size_t index = 0; index < count && current != nullptr;
       ++index, current = current->next) {
    if (current == bad_work_request) {
      return index;
    }
  }
  return 0;
}

#if HNSW_URMA_HAS_BONDING_HEADER
urma_context_aggr_mode_t NativeAggregationMode(UmdkBondingMode mode) noexcept {
  switch (mode) {
    case UmdkBondingMode::kStandalone:
      return URMA_AGGR_MODE_STANDALONE;
    case UmdkBondingMode::kActiveBackup:
      return URMA_AGGR_MODE_ACTIVE_BACKUP;
    case UmdkBondingMode::kBalance:
      return URMA_AGGR_MODE_BALANCE;
  }
  return URMA_AGGR_MODE_STANDALONE;
}
#endif

}  // namespace

struct UmdkResourceBackend::Impl {
  struct Entry {
    UrmaResourceKind kind{UrmaResourceKind::kContext};
    void* primary{nullptr};
    void* secondary{nullptr};
    bool owned{true};
  };

  UmdkBackendOptions options;
  UrmaCapabilities capabilities;
  urma_device_t* device{nullptr};
  urma_device_attr_t device_attributes{};
  urma_context_t* context{nullptr};
  std::uint32_t selected_eid{0};
  std::uint64_t next_handle{1};
  bool provider_runtime_acquired{false};
  std::unordered_map<std::uint64_t, Entry> entries;
  std::vector<std::uint64_t> creation_order;

  Entry* Find(UrmaResourceHandle handle, UrmaResourceKind expected) noexcept {
    auto iterator = entries.find(handle.value);
    if (iterator == entries.end() || iterator->second.kind != expected) {
      return nullptr;
    }
    return &iterator->second;
  }

  const Entry* Find(UrmaResourceHandle handle, UrmaResourceKind expected) const noexcept {
    auto iterator = entries.find(handle.value);
    if (iterator == entries.end() || iterator->second.kind != expected) {
      return nullptr;
    }
    return &iterator->second;
  }

  void DestroyNative(const Entry& entry) noexcept {
    if (!entry.owned) return;
    switch (entry.kind) {
      case UrmaResourceKind::kContext:
        if (entry.primary != nullptr) {
          (void)urma_delete_context(static_cast<urma_context_t*>(entry.primary));
          context = nullptr;
        }
        break;
      case UrmaResourceKind::kRegisteredSegment:
        (void)urma_unregister_seg(static_cast<urma_target_seg_t*>(entry.primary));
        break;
      case UrmaResourceKind::kSendJfc:
      case UrmaResourceKind::kTargetJfc:
        (void)urma_delete_jfc(static_cast<urma_jfc_t*>(entry.primary));
        break;
      case UrmaResourceKind::kSendJfs:
        (void)urma_delete_jfs(static_cast<urma_jfs_t*>(entry.primary));
        break;
      case UrmaResourceKind::kSendJetty:
        (void)urma_delete_jetty(static_cast<urma_jetty_t*>(entry.primary));
        break;
      case UrmaResourceKind::kTargetJfr:
        (void)urma_delete_jfr(static_cast<urma_jfr_t*>(entry.primary));
        break;
      case UrmaResourceKind::kImportedSegment:
        (void)urma_unimport_seg(static_cast<urma_target_seg_t*>(entry.primary));
        break;
      case UrmaResourceKind::kImportedEndpoint:
        if (options.configure_bonding) {
          (void)urma_unimport_jetty(static_cast<urma_target_jetty_t*>(entry.primary));
        } else {
          (void)urma_unimport_jfr(static_cast<urma_target_jetty_t*>(entry.primary));
        }
        break;
      case UrmaResourceKind::kAdvice:
        if (options.configure_bonding) {
          (void)urma_unadvise_jetty(static_cast<urma_jetty_t*>(entry.primary),
                                  static_cast<urma_target_jetty_t*>(entry.secondary));
        } else {
          (void)urma_unadvise_jfr(static_cast<urma_jfs_t*>(entry.primary),
                                static_cast<urma_target_jetty_t*>(entry.secondary));
        }
        break;
    }
  }

  Result<UrmaResourceHandle> Record(UrmaResourceKind kind, void* primary,
                                    void* secondary = nullptr, bool owned = true) {
    const UrmaResourceHandle handle{next_handle++};
    Entry entry{kind, primary, secondary, owned};
    try {
      entries.emplace(handle.value, entry);
      creation_order.push_back(handle.value);
    } catch (const std::bad_alloc&) {
      entries.erase(handle.value);
      DestroyNative(entry);
      return Status(StatusCode::kResourceExhausted, "unable to track UMDK resource handle");
    }
    return handle;
  }

  ~Impl() {
    for (auto iterator = creation_order.rbegin(); iterator != creation_order.rend(); ++iterator) {
      auto entry = entries.find(*iterator);
      if (entry != entries.end()) {
        DestroyNative(entry->second);
        entries.erase(entry);
      }
    }
    if (provider_runtime_acquired) {
      ReleaseProviderRuntime();
    }
  }
};

UmdkResourceBackend::UmdkResourceBackend(std::unique_ptr<Impl> implementation)
    : impl_(std::move(implementation)) {}

UmdkResourceBackend::~UmdkResourceBackend() = default;

Result<std::unique_ptr<UmdkResourceBackend>> UmdkResourceBackend::Create(
    const UmdkBackendOptions& options) {
  if (options.device_name.empty()) {
    return Status(StatusCode::kInvalidArgument, "--urma-device must not be empty");
  }
  auto implementation = std::unique_ptr<Impl>(new (std::nothrow) Impl());
  if (!implementation) {
    return Status(StatusCode::kResourceExhausted, "unable to allocate UMDK backend");
  }
  implementation->options = options;
  if (options.external_context == nullptr) {
    auto runtime_status = AcquireProviderRuntime();
    if (!runtime_status.ok()) return runtime_status;
    implementation->provider_runtime_acquired = true;
  }
  implementation->device = urma_get_device_by_name(implementation->options.device_name.data());
  if (implementation->device == nullptr) {
    return NullProviderResult("urma_get_device_by_name");
  }
  const auto query_status =
      urma_query_device(implementation->device, &implementation->device_attributes);
  if (query_status != URMA_SUCCESS) {
    return MapUrmaProviderStatus(query_status, "urma_query_device");
  }
  if (options.eid_index.has_value()) {
    implementation->selected_eid = *options.eid_index;
  } else {
    std::uint32_t count = 0;
    urma_eid_info_t* eids = urma_get_eid_list(implementation->device, &count);
    if (eids == nullptr || count == 0) {
      if (eids != nullptr) {
        urma_free_eid_list(eids);
      }
      return Status(StatusCode::kTransportError, "URMA device has no configured EID");
    }
    implementation->selected_eid = static_cast<std::uint32_t>(eids[0].eid_index);
    urma_free_eid_list(eids);
  }
  const auto& capability = implementation->device_attributes.dev_cap;
  implementation->capabilities = {
      .max_jfs = capability.max_jfs,
      .max_jfr = capability.max_jfr,
      .max_jfc = capability.max_jfc,
      .max_jfs_depth = capability.max_jfs_depth,
      .max_jfr_depth = capability.max_jfr_depth,
      .max_jfc_depth = capability.max_jfc_depth,
      .max_inline_bytes = capability.max_jfs_inline_len,
      .max_read_bytes = capability.max_read_size,
      .max_write_bytes = capability.max_write_size,
      .max_jetty = capability.max_jetty,
  };
  auto backend = std::unique_ptr<UmdkResourceBackend>(
      new (std::nothrow) UmdkResourceBackend(std::move(implementation)));
  if (!backend) {
    return Status(StatusCode::kResourceExhausted, "unable to allocate UMDK backend wrapper");
  }
  return backend;
}

const UrmaCapabilities& UmdkResourceBackend::capabilities() const noexcept {
  return impl_->capabilities;
}

bool UmdkResourceBackend::requires_endpoint_advice() const noexcept {
  // B129 perftest connects RM Jettys on UB by import alone; RC binding is not
  // part of this transport. Non-UB and legacy simplex retain explicit advice.
  return !impl_->options.configure_bonding || impl_->device->type != URMA_TRANSPORT_UB;
}

std::uint32_t UmdkResourceBackend::selected_eid_index() const noexcept {
  return impl_->selected_eid;
}

Result<std::vector<std::byte>> UmdkResourceBackend::LocalEid() const {
  if (impl_->context == nullptr) {
    return Status(StatusCode::kInvalidArgument, "context must exist before exporting the local EID");
  }
  try {
    std::vector<std::byte> bytes(sizeof(impl_->context->eid));
    std::memcpy(bytes.data(), &impl_->context->eid, bytes.size());
    return bytes;
  } catch (const std::bad_alloc&) {
    return Status(StatusCode::kResourceExhausted, "unable to copy local EID descriptor");
  }
}

Result<std::vector<std::byte>> UmdkResourceBackend::ExportDescriptor(
    UrmaResourceHandle handle) const {
  auto iterator = impl_->entries.find(handle.value);
  if (iterator == impl_->entries.end()) {
    return Status(StatusCode::kInvalidArgument,
                  "UMDK resource handle does not exist")
        .AddContext("handle", std::to_string(handle.value))
        .AddContext("tracked_resource_count",
                    std::to_string(impl_->entries.size()));
  }
  try {
    if (iterator->second.kind == UrmaResourceKind::kRegisteredSegment) {
      auto* target_segment =
          static_cast<urma_target_seg_t*>(iterator->second.primary);
      // Preserve the provider's full descriptor, including bonding extension
      // data. UMDK 25.12 B129 uses it to reconstruct remote physical segments
      // without importing a virtual segment through the kernel.
      urma_seg_t* descriptor = nullptr;
      std::uint32_t bytes = 0;
      const auto status =
          urma_get_seg_ctx(target_segment, &descriptor, &bytes);
      const std::unique_ptr<urma_seg_t, decltype(&urma_put_seg_ctx)> owner(
          descriptor, &urma_put_seg_ctx);
      if (status != URMA_SUCCESS) {
        return MapUrmaProviderStatus(status, "urma_get_seg_ctx");
      }
      if (descriptor == nullptr || bytes < sizeof(urma_seg_t)) {
        return Status(StatusCode::kTransportError, "provider returned an invalid segment descriptor");
      }
      std::vector<std::byte> output(bytes);
      std::memcpy(output.data(), descriptor, bytes);
      return output;
    }
    if (iterator->second.kind == UrmaResourceKind::kSendJetty) {
      urma_rjetty_t* descriptor = nullptr;
      std::uint32_t bytes = 0;
      const auto status = urma_get_rjetty(
          static_cast<urma_jetty_t*>(iterator->second.primary), &descriptor, &bytes);
      const std::unique_ptr<urma_rjetty_t, decltype(&urma_put_rjetty)> owner(
          descriptor, &urma_put_rjetty);
      if (status != URMA_SUCCESS) {
        return MapUrmaProviderStatus(status, "urma_get_rjetty");
      }
      if (descriptor == nullptr || bytes < sizeof(urma_rjetty_t)) {
        return Status(StatusCode::kTransportError,
                      "provider returned an invalid Jetty descriptor");
      }
      // The bonding extension contains the physical Jetty mappings. Copy the
      // full provider length, as perftest does, rather than just the Jetty ID.
      std::vector<std::byte> output(bytes);
      std::memcpy(output.data(), descriptor, bytes);
      return output;
    }
    if (iterator->second.kind == UrmaResourceKind::kTargetJfr) {
      const auto* jfr = static_cast<const urma_jfr_t*>(iterator->second.primary);
      std::vector<std::byte> output(sizeof(jfr->jfr_id));
      std::memcpy(output.data(), &jfr->jfr_id, output.size());
      return output;
    }
  } catch (const std::bad_alloc&) {
    return Status(StatusCode::kResourceExhausted, "unable to copy provider descriptor");
  }
  return Status(StatusCode::kInvalidArgument, "resource kind has no exportable descriptor");
}

Result<UrmaResourceHandle> UmdkResourceBackend::Create(const UrmaResourceRequest& request) {
  if (request.kind != UrmaResourceKind::kContext && impl_->context == nullptr) {
    return Status(StatusCode::kInvalidArgument, "UMDK context must be created first");
  }
  try {
    switch (request.kind) {
    case UrmaResourceKind::kContext: {
      if (impl_->context != nullptr) {
        return Status(StatusCode::kInvalidArgument, "only one UMDK context is allowed per rank");
      }
      if (impl_->options.external_context != nullptr) {
        impl_->context = static_cast<urma_context_t*>(impl_->options.external_context);
        return impl_->Record(request.kind, impl_->context, nullptr, false);
      }
      impl_->context = urma_create_context(impl_->device, impl_->selected_eid);
      if (impl_->context == nullptr) {
        return NullProviderResult("urma_create_context");
      }
#if HNSW_URMA_HAS_BONDING_HEADER
      if (impl_->options.configure_bonding) {
        const auto mode = NativeAggregationMode(impl_->options.bonding_mode);
        const auto status = urma_set_context_opt(
            impl_->context, URMA_OPT_AGGR_MODE, &mode, sizeof(mode));
        if (status != URMA_SUCCESS) {
          (void)urma_delete_context(impl_->context);
          impl_->context = nullptr;
          return MapUrmaProviderStatus(status,
                                       "urma_set_context_opt(URMA_OPT_AGGR_MODE)");
        }
      }
#else
      if (impl_->options.configure_bonding) {
        (void)urma_delete_context(impl_->context);
        impl_->context = nullptr;
        return Status(StatusCode::kInvalidArgument,
                      "bonding was requested but urma_ubagg.h was not "
                      "available at build time");
      }
#endif
      return impl_->Record(request.kind, impl_->context);
    }
    case UrmaResourceKind::kRegisteredSegment: {
      if (request.address == nullptr || request.bytes == 0) {
        return Status(StatusCode::kInvalidArgument, "registered segment memory is empty");
      }
      auto page_size = PageAlignedBuffer::SystemPageSize();
      if (!page_size.ok()) {
        return page_size.status();
      }
      const auto address =
          static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(request.address));
      const auto page_bytes = static_cast<std::uint64_t>(page_size.value());
      if (address % page_bytes != 0 || request.bytes % page_bytes != 0) {
        return Status(StatusCode::kInvalidArgument,
                      "registered segment address and length must be page aligned")
            .AddContext("page_size", std::to_string(page_size.value()))
            .AddContext("address_page_remainder",
                        std::to_string(address % page_bytes))
            .AddContext("length_page_remainder",
                        std::to_string(request.bytes % page_bytes))
            .AddContext("device", impl_->options.device_name)
            .AddContext("eid_index", std::to_string(impl_->selected_eid));
      }
      urma_seg_cfg_t config{};
      config.va = address;
      config.len = request.bytes;
      config.token_value.token = impl_->options.token;
      config.flag.bs.token_policy = URMA_TOKEN_NONE;
      config.flag.bs.cacheable = URMA_NON_CACHEABLE;
      config.flag.bs.access = URMA_ACCESS_READ | URMA_ACCESS_WRITE;
      errno = 0;
      auto* segment = urma_register_seg(impl_->context, &config);
      if (segment == nullptr) {
        auto status = NullProviderResult("urma_register_seg");
        status.AddContext("page_size", std::to_string(page_size.value()))
            .AddContext("address_page_remainder",
                        std::to_string(address % page_bytes))
            .AddContext("length_page_remainder",
                        std::to_string(request.bytes % page_bytes))
            .AddContext("device", impl_->options.device_name)
            .AddContext("eid_index", std::to_string(impl_->selected_eid));
        return status;
      }
      return impl_->Record(request.kind, segment);
    }
    case UrmaResourceKind::kSendJfc:
    case UrmaResourceKind::kTargetJfc: {
      urma_jfc_cfg_t config{};
      config.depth = request.depth;
      auto* jfc = urma_create_jfc(impl_->context, &config);
      return jfc == nullptr ? Result<UrmaResourceHandle>(NullProviderResult("urma_create_jfc"))
                            : impl_->Record(request.kind, jfc);
    }
    case UrmaResourceKind::kSendJfs:
    case UrmaResourceKind::kSendJetty: {
      const bool duplex = request.kind == UrmaResourceKind::kSendJetty;
      if (duplex != impl_->options.configure_bonding) {
        return Status(StatusCode::kInvalidArgument,
                      "send queue mode does not match the UMDK backend");
      }
      auto* jfc_entry = impl_->Find(request.dependency, UrmaResourceKind::kSendJfc);
      if (jfc_entry == nullptr) {
        return Status(StatusCode::kInvalidArgument, "send JFS requires its send JFC");
      }
      const bool write_path = request.operation_class != OperationClass::kRead;
      if (write_path && impl_->device_attributes.dev_cap.max_jfs_rsge < 2) {
        return Status(StatusCode::kResourceExhausted,
                      "WRITE_NOTIFY requires two remote SGEs per WR");
      }
      urma_jfs_cfg_t config{};
      config.depth = request.depth;
      config.trans_mode = URMA_TM_RM;
      config.priority = request.operation_class == OperationClass::kRead ? 15 :
                        request.operation_class == OperationClass::kResponseWrite ? 14 : 12;
      config.max_sge = 1;
      config.max_rsge = write_path ? 2 : 1;
      config.max_inline_data = impl_->capabilities.max_inline_bytes;
      config.rnr_retry = URMA_TYPICAL_RNR_RETRY;
      config.err_timeout = URMA_TYPICAL_ERR_TIMEOUT;
      config.jfc = static_cast<urma_jfc_t*>(jfc_entry->primary);
      if (duplex) {
        auto* jfr_entry = impl_->Find(request.secondary_dependency,
                                     UrmaResourceKind::kTargetJfr);
        if (jfr_entry == nullptr) {
          return Status(StatusCode::kInvalidArgument,
                        "duplex Jetty requires its shared target JFR");
        }
        auto* jfr = static_cast<urma_jfr_t*>(jfr_entry->primary);
        urma_jetty_cfg_t jetty_config{};
        jetty_config.flag.bs.share_jfr = URMA_SHARE_JFR;
        jetty_config.jfs_cfg = config;
        jetty_config.shared.jfr = jfr;
        jetty_config.shared.jfc = jfr->jfr_cfg.jfc;
        auto* jetty = urma_create_jetty(impl_->context, &jetty_config);
        return jetty == nullptr
                   ? Result<UrmaResourceHandle>(NullProviderResult("urma_create_jetty"))
                   : impl_->Record(request.kind, jetty);
      }
      auto* jfs = urma_create_jfs(impl_->context, &config);
      return jfs == nullptr ? Result<UrmaResourceHandle>(NullProviderResult("urma_create_jfs"))
                            : impl_->Record(request.kind, jfs);
    }
    case UrmaResourceKind::kTargetJfr: {
      auto* jfc_entry = impl_->Find(request.dependency, UrmaResourceKind::kTargetJfc);
      if (jfc_entry == nullptr || request.depth == 0) {
        return Status(StatusCode::kInvalidArgument, "target JFR requires a non-zero depth and target JFC");
      }
      urma_jfr_cfg_t config{};
      config.depth = request.depth;
      config.trans_mode = URMA_TM_RM;
      config.max_sge = 1;
      config.min_rnr_timer = URMA_TYPICAL_MIN_RNR_TIMER;
      config.jfc = static_cast<urma_jfc_t*>(jfc_entry->primary);
      config.token_value.token = impl_->options.token;
      auto* jfr = urma_create_jfr(impl_->context, &config);
      return jfr == nullptr ? Result<UrmaResourceHandle>(NullProviderResult("urma_create_jfr"))
                            : impl_->Record(request.kind, jfr);
    }
    case UrmaResourceKind::kImportedSegment: {
      if (request.descriptor.size() < sizeof(urma_seg_t)) {
        return Status(StatusCode::kInvalidArgument, "remote segment descriptor is truncated");
      }
      std::vector<std::byte> descriptor(request.descriptor.begin(), request.descriptor.end());
      urma_token_t token{};
      token.token = static_cast<std::uint32_t>(request.token);
      urma_import_seg_flag_t flags{};
      flags.bs.cacheable = URMA_NON_CACHEABLE;
      flags.bs.access = URMA_ACCESS_READ | URMA_ACCESS_WRITE;
      flags.bs.mapping = URMA_SEG_NOMAP;
      auto* segment = urma_import_seg(impl_->context,
                                      reinterpret_cast<urma_seg_t*>(descriptor.data()),
                                      &token, 0, flags);
      return segment == nullptr ? Result<UrmaResourceHandle>(NullProviderResult("urma_import_seg"))
                                : impl_->Record(request.kind, segment);
    }
    case UrmaResourceKind::kImportedEndpoint: {
      urma_token_t token{};
      token.token = static_cast<std::uint32_t>(request.token);
      if (impl_->options.configure_bonding) {
        if (request.descriptor.size() < sizeof(urma_rjetty_t)) {
          return Status(StatusCode::kInvalidArgument,
                        "remote Jetty descriptor is truncated; all ranks must use duplex metadata");
        }
        std::vector<std::byte> descriptor(request.descriptor.begin(), request.descriptor.end());
        auto* remote = reinterpret_cast<urma_rjetty_t*>(descriptor.data());
        if (remote->trans_mode != URMA_TM_RM || remote->type != URMA_JETTY) {
          return Status(StatusCode::kInvalidArgument,
                        "remote endpoint must be an RM duplex Jetty");
        }
        remote->tp_type = impl_->options.tp_type == UmdkTpType::kCtp ? URMA_CTP : URMA_RTP;
        auto* endpoint = urma_import_jetty(impl_->context, remote, &token);
        return endpoint == nullptr
                   ? Result<UrmaResourceHandle>(NullProviderResult("urma_import_jetty"))
                   : impl_->Record(request.kind, endpoint);
      }
      if (request.descriptor.size() != sizeof(urma_jfr_id_t)) {
        return Status(StatusCode::kInvalidArgument, "remote JFR descriptor has the wrong size");
      }
      urma_rjfr_t remote{};
      std::memcpy(&remote.jfr_id, request.descriptor.data(), sizeof(remote.jfr_id));
      remote.trans_mode = URMA_TM_RM;
      remote.tp_type = impl_->options.tp_type == UmdkTpType::kCtp ? URMA_CTP : URMA_RTP;
      auto* endpoint = urma_import_jfr(impl_->context, &remote, &token);
      return endpoint == nullptr
                 ? Result<UrmaResourceHandle>(NullProviderResult("urma_import_jfr"))
                 : impl_->Record(request.kind, endpoint);
    }
    case UrmaResourceKind::kAdvice: {
      auto* endpoint = impl_->Find(request.dependency, UrmaResourceKind::kImportedEndpoint);
      const bool duplex = impl_->options.configure_bonding;
      auto* jfs = impl_->Find(request.secondary_dependency, duplex
                                 ? UrmaResourceKind::kSendJetty : UrmaResourceKind::kSendJfs);
      if (endpoint == nullptr || jfs == nullptr) {
        return Status(StatusCode::kInvalidArgument, "advice requires a local send queue and imported endpoint");
      }
      if (!requires_endpoint_advice()) {
        return Status(StatusCode::kInvalidArgument, "UB RM duplex endpoints do not require advice");
      }
      auto* target = static_cast<urma_target_jetty_t*>(endpoint->primary);
      const auto status = duplex
                              ? urma_advise_jetty(static_cast<urma_jetty_t*>(jfs->primary), target)
                              : urma_advise_jfr(static_cast<urma_jfs_t*>(jfs->primary), target);
      if (status != URMA_SUCCESS && status != URMA_EEXIST) {
        return MapUrmaProviderStatus(status, duplex ? "urma_advise_jetty" : "urma_advise_jfr");
      }
      return impl_->Record(request.kind, jfs->primary, endpoint->primary);
    }
    }
  } catch (const std::bad_alloc&) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate temporary UMDK provider metadata");
  }
  return Status(StatusCode::kInternalError, "unknown UMDK resource kind");
}

void UmdkResourceBackend::Destroy(UrmaResourceKind kind, UrmaResourceHandle handle) noexcept {
  auto iterator = impl_->entries.find(handle.value);
  if (iterator == impl_->entries.end() || iterator->second.kind != kind) {
    return;
  }
  impl_->DestroyNative(iterator->second);
  impl_->entries.erase(iterator);
}

Status UmdkResourceBackend::AcquireDeviceWrites() {
#if defined(__aarch64__)
  asm volatile("dmb oshld" ::: "memory");
#elif defined(__x86_64__)
  asm volatile("lfence" ::: "memory");
#else
  return Status(StatusCode::kTransportError,
                "no device-read barrier is implemented for this architecture");
#endif
  return Status::Ok();
}

Result<UrmaResourceHandle> UmdkResourceBackend::ImportExternal(
    UrmaResourceKind kind, void* native_handle) noexcept {
  if (!native_handle || kind != UrmaResourceKind::kImportedSegment) {
    return Status(StatusCode::kInvalidArgument, "expected a borrowed imported segment");
  }
  return impl_->Record(kind, native_handle, nullptr, false);
}

void* UmdkResourceBackend::Native(UrmaResourceHandle handle,
                                  UrmaResourceKind expected_kind) const noexcept {
  const auto* entry = impl_->Find(handle, expected_kind);
  return entry == nullptr ? nullptr : entry->primary;
}

struct UmdkLanePostEngine::Impl {
  struct SendQueueHandle {
    urma_jfs_t* jfs{nullptr};
    urma_jetty_t* jetty{nullptr};

    bool valid() const noexcept { return jfs != nullptr || jetty != nullptr; }
    urma_status_t Post(urma_jfs_wr_t* work, urma_jfs_wr_t** bad_work) const {
      return jetty != nullptr ? urma_post_jetty_send_wr(jetty, work, bad_work)
                              : urma_post_jfs_wr(jfs, work, bad_work);
    }
    const char* PostOperation() const noexcept {
      return jetty != nullptr ? "urma_post_jetty_send_wr" : "urma_post_jfs_wr";
    }
  };
  struct LocalSegment {
    urma_target_seg_t* native{nullptr};
    std::byte* begin{nullptr};
    std::size_t bytes{0};
  };
  struct RemoteSegment {
    SegmentKind kind{SegmentKind::kLayer0Graph};
    urma_target_seg_t* native{nullptr};
    std::uint64_t bytes{0};
  };
  struct RemoteEndpoint {
    OperationClass operation_class{OperationClass::kRead};
    urma_target_jetty_t* native{nullptr};
    std::uint32_t send_queue_index{0};
  };
  struct Peer {
    std::uint32_t rank{0};
    std::vector<RemoteSegment> segments;
    std::vector<RemoteEndpoint> endpoints;
  };
  struct Pending {
    OperationTicket ticket;
    OperationClass operation_class{OperationClass::kRead};
    std::uint64_t bytes{0};
    std::size_t bounce_slot{kNoBounceSlot};
    std::uint32_t peer_rank{0};
    SegmentKind payload_segment{SegmentKind::kLayer0Graph};
    SegmentKind notify_segment{SegmentKind::kNotify};
    bool inline_source{false};
  };
  struct PendingSlot {
    Pending value;
    // ubagg may retain every node and SGE in a native WR list until all of
    // that list's CRs have been polled. Keep provider-visible storage here
    // instead of reusing one per-call scratch array.
    urma_jfs_wr_t work{};
    urma_sge_t source{};
    urma_sge_t read_destination{};
    std::array<urma_sge_t, 2> write_destinations{};
    std::size_t batch_leader{kNoPendingSlot};
    std::size_t batch_next{kNoPendingSlot};
    std::size_t batch_remaining{0};
    std::size_t batch_size{0};
    std::uint32_t generation{0};
    bool active{false};
    bool native_busy{false};
  };

  UmdkResourceBackend* backend{nullptr};
  UmdkLaneOptions options;
  SendQueueHandle read_jfs;
  std::vector<SendQueueHandle> request_jfs;
  std::vector<SendQueueHandle> response_jfs;
  urma_jfc_t* read_jfc{nullptr};
  urma_jfc_t* request_jfc{nullptr};
  urma_jfc_t* response_jfc{nullptr};
  std::vector<LocalSegment> local_segments;
  std::vector<Peer> peers;
  urma_target_seg_t* bounce_segment{nullptr};
  std::byte* bounce_begin{nullptr};
  std::size_t bounce_bytes{0};
  std::size_t bounce_stride{0};
  std::vector<bool> bounce_busy;
  std::vector<PendingSlot> pending;
  std::size_t pending_count{0};
  std::size_t native_busy_count{0};
  std::size_t native_busy_high_watermark{0};
  std::size_t pending_cursor{0};
  std::vector<std::uint64_t> post_contexts;
  std::vector<std::size_t> post_bounce_slots;
  std::size_t next_poll_class{0};

  Status ValidateTicket(const OperationTicket& ticket) const {
    if (ticket.value == 0 || ticket.generation == 0 ||
        ticket.owner_lane != options.owner_lane) {
      return Status(StatusCode::kProtocolError, "ticket does not belong to this UMDK lane");
    }
    return Status::Ok();
  }

  const Peer* FindPeer(std::uint32_t rank) const noexcept {
    for (const auto& peer : peers) {
      if (peer.rank == rank) {
        return &peer;
      }
    }
    return nullptr;
  }

  const RemoteSegment* FindRemoteSegment(std::uint32_t rank, SegmentKind kind) const noexcept {
    const auto* peer = FindPeer(rank);
    if (peer != nullptr) {
      for (const auto& segment : peer->segments) {
        if (segment.kind == kind) {
          return &segment;
        }
      }
    }
    return nullptr;
  }

  const RemoteEndpoint* FindEndpoint(
      std::uint32_t rank,
      OperationClass operation_class) const noexcept {
    const auto* peer = FindPeer(rank);
    if (peer != nullptr) {
      for (const auto& endpoint : peer->endpoints) {
        if (endpoint.operation_class == operation_class) {
          return &endpoint;
        }
      }
    }
    return nullptr;
  }

  urma_target_seg_t* FindLocalSegment(const std::byte* address, std::size_t bytes) const noexcept {
    const auto requested = reinterpret_cast<std::uintptr_t>(address);
    for (const auto& segment : local_segments) {
      const auto begin = reinterpret_cast<std::uintptr_t>(segment.begin);
      if (requested >= begin && requested - begin <= segment.bytes &&
          bytes <= segment.bytes - (requested - begin)) {
        return segment.native;
      }
    }
    return nullptr;
  }

  const SendQueueHandle* SendQueue(OperationClass operation_class,
                                   std::uint32_t queue_index = 0) const noexcept {
    if (operation_class == OperationClass::kRead) {
      return queue_index == 0 && read_jfs.valid() ? &read_jfs : nullptr;
    }
    const auto& queues = operation_class == OperationClass::kRequestWrite
                             ? request_jfs
                             : response_jfs;
    return queue_index < queues.size() && queues[queue_index].valid()
               ? &queues[queue_index] : nullptr;
  }

  std::optional<std::size_t> AcquireBounce(std::size_t bytes) noexcept {
    if (bounce_segment == nullptr || bytes > bounce_stride) {
      return std::nullopt;
    }
    for (std::size_t slot = 0; slot < bounce_busy.size(); ++slot) {
      if (!bounce_busy[slot]) {
        bounce_busy[slot] = true;
        return slot;
      }
    }
    return std::nullopt;
  }

  void ReleaseBounce(std::size_t slot) noexcept {
    if (slot != kNoBounceSlot && slot < bounce_busy.size()) {
      bounce_busy[slot] = false;
    }
  }

  Result<std::uint64_t> ReservePending(Pending value) {
    for (std::size_t checked = 0; checked < pending.size(); ++checked) {
      const auto index = (pending_cursor + checked) % pending.size();
      auto& slot = pending[index];
      if (slot.active || slot.native_busy) {
        continue;
      }
      ++slot.generation;
      if (slot.generation == 0) {
        ++slot.generation;
      }
      slot.value = value;
      slot.work = {};
      slot.source = {};
      slot.read_destination = {};
      slot.write_destinations = {};
      slot.batch_leader = kNoPendingSlot;
      slot.batch_next = kNoPendingSlot;
      slot.batch_remaining = 0;
      slot.batch_size = 0;
      slot.active = true;
      slot.native_busy = true;
      ++native_busy_count;
      native_busy_high_watermark = std::max(native_busy_high_watermark, native_busy_count);
      ++pending_count;
      pending_cursor = (index + 1U) % pending.size();
      return (static_cast<std::uint64_t>(slot.generation) << 32U) |
             (static_cast<std::uint64_t>(index) + 1U);
    }
    return Status(StatusCode::kResourceExhausted,
                  "UMDK lane operation table is full");
  }

  PendingSlot* FindPendingSlot(std::uint64_t context) noexcept {
    const std::uint64_t encoded_slot = context & 0xffffffffULL;
    const auto generation = static_cast<std::uint32_t>(context >> 32U);
    if (encoded_slot == 0 || encoded_slot > pending.size() || generation == 0) {
      return nullptr;
    }
    auto& slot = pending[static_cast<std::size_t>(encoded_slot - 1U)];
    return slot.active && slot.native_busy && slot.generation == generation
               ? &slot
               : nullptr;
  }

  void ReleasePrepared(std::uint64_t context) noexcept {
    auto* slot = FindPendingSlot(context);
    if (slot == nullptr || slot->batch_leader != kNoPendingSlot) {
      return;
    }
    slot->active = false;
    slot->native_busy = false;
    --native_busy_count;
    slot->value = {};
    slot->work = {};
    slot->source = {};
    slot->read_destination = {};
    slot->write_destinations = {};
    slot->batch_leader = kNoPendingSlot;
    slot->batch_next = kNoPendingSlot;
    slot->batch_remaining = 0;
    slot->batch_size = 0;
    --pending_count;
  }

  Status CommitPostedBatch(std::span<const std::uint64_t> contexts) noexcept {
    if (contexts.empty()) {
      return Status::Ok();
    }
    auto* leader = FindPendingSlot(contexts.front());
    if (leader == nullptr || leader->batch_leader != kNoPendingSlot) {
      return Status(StatusCode::kProtocolError,
                    "UMDK native WR batch leader is invalid");
    }
    const auto leader_index = static_cast<std::size_t>(leader - pending.data());
    for (std::size_t index = 0; index < contexts.size(); ++index) {
      auto* slot = FindPendingSlot(contexts[index]);
      if (slot == nullptr || slot->batch_leader != kNoPendingSlot) {
        return Status(StatusCode::kProtocolError,
                      "UMDK native WR batch member is invalid");
      }
      for (std::size_t previous = 0; previous < index; ++previous) {
        if (contexts[previous] == contexts[index]) {
          return Status(StatusCode::kProtocolError,
                        "UMDK native WR batch contains a duplicate member");
        }
      }
    }
    leader->batch_remaining = contexts.size();
    leader->batch_size = contexts.size();
    PendingSlot* previous = nullptr;
    for (std::size_t index = 0; index < contexts.size(); ++index) {
      auto* member = FindPendingSlot(contexts[index]);
      member->batch_leader = leader_index;
      member->batch_next = kNoPendingSlot;
      if (previous != nullptr) {
        previous->batch_next =
            static_cast<std::size_t>(member - pending.data());
      }
      previous = member;
    }
    return Status::Ok();
  }

  Result<Pending> CompletePending(std::uint64_t context) noexcept {
    auto* slot = FindPendingSlot(context);
    if (slot == nullptr || slot->batch_leader >= pending.size()) {
      return Status(StatusCode::kProtocolError,
                    "UMDK completion has an unknown native WR batch");
    }
    const auto leader_index = slot->batch_leader;
    auto& leader = pending[leader_index];
    if (!leader.native_busy || leader.batch_leader != leader_index ||
        leader.batch_remaining == 0 || leader.batch_size == 0) {
      return Status(StatusCode::kProtocolError,
                    "UMDK native WR batch completion accounting is invalid");
    }
    Pending completed = slot->value;
    slot->active = false;
    --pending_count;
    --leader.batch_remaining;
    if (leader.batch_remaining == 0) {
      // A member CR may arrive before the provider has finished releasing
      // other nodes in the same list, so release native storage as a group.
      // Follow the list captured at post time instead of scanning the full
      // max-outstanding table.
      const auto batch_size = leader.batch_size;
      auto member_index = leader_index;
      for (std::size_t released = 0; released < batch_size; ++released) {
        if (member_index >= pending.size()) {
          return Status(StatusCode::kProtocolError,
                        "UMDK native WR batch member chain is truncated");
        }
        auto& member = pending[member_index];
        if (!member.native_busy || member.batch_leader != leader_index) {
          return Status(StatusCode::kProtocolError,
                        "UMDK native WR batch member chain is invalid");
        }
        const auto next_member = member.batch_next;
        ReleaseBounce(member.value.bounce_slot);
        member.value = {};
        member.work = {};
        member.source = {};
        member.read_destination = {};
        member.write_destinations = {};
        member.batch_leader = kNoPendingSlot;
        member.batch_next = kNoPendingSlot;
        member.batch_remaining = 0;
        member.batch_size = 0;
        member.native_busy = false;
        --native_busy_count;
        member_index = next_member;
      }
      if (member_index != kNoPendingSlot) {
        return Status(StatusCode::kProtocolError,
                      "UMDK native WR batch member chain exceeds its size");
      }
    }
    return completed;
  }
};

UmdkLanePostEngine::UmdkLanePostEngine(std::unique_ptr<Impl> implementation)
    : impl_(std::move(implementation)) {}

UmdkLanePostEngine::~UmdkLanePostEngine() = default;

Result<std::unique_ptr<UmdkLanePostEngine>> UmdkLanePostEngine::Create(
    UmdkResourceBackend& backend, UmdkLaneOptions options) {
  if (options.max_outstanding == 0 ||
      options.max_outstanding > std::numeric_limits<std::uint32_t>::max() ||
      options.wr_list_max == 0 || options.poll_batch == 0 ||
      options.poll_batch > kProviderPollLimit || options.resources.lane != options.owner_lane) {
    return Status(StatusCode::kInvalidArgument, "UMDK lane bounds or owner are invalid");
  }
  auto implementation = std::unique_ptr<Impl>(new (std::nothrow) Impl());
  if (!implementation) {
    return Status(StatusCode::kResourceExhausted, "unable to allocate UMDK lane engine");
  }
  implementation->backend = &backend;
  implementation->options = std::move(options);
  const auto& resources = implementation->options.resources;
  const auto native_queue = [&](UrmaResourceHandle handle) {
    Impl::SendQueueHandle queue;
    if (resources.duplex_jetty) {
      queue.jetty = static_cast<urma_jetty_t*>(backend.Native(handle, UrmaResourceKind::kSendJetty));
    } else {
      queue.jfs = static_cast<urma_jfs_t*>(backend.Native(handle, UrmaResourceKind::kSendJfs));
    }
    return queue;
  };
  implementation->read_jfs = native_queue(resources.read_jfs);
  try {
    implementation->request_jfs.reserve(resources.request_jfs.size());
    implementation->response_jfs.reserve(resources.response_jfs.size());
    for (const auto resource : resources.request_jfs) {
      implementation->request_jfs.push_back(native_queue(resource));
    }
    for (const auto resource : resources.response_jfs) {
      implementation->response_jfs.push_back(native_queue(resource));
    }
  } catch (const std::bad_alloc&) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate UMDK send queue table");
  }
  implementation->read_jfc = static_cast<urma_jfc_t*>(
      backend.Native(resources.read_jfc, UrmaResourceKind::kSendJfc));
  implementation->request_jfc = static_cast<urma_jfc_t*>(
      backend.Native(resources.request_jfc, UrmaResourceKind::kSendJfc));
  implementation->response_jfc = static_cast<urma_jfc_t*>(
      backend.Native(resources.response_jfc, UrmaResourceKind::kSendJfc));
  const bool has_read =
      implementation->read_jfs.valid() &&
      implementation->read_jfc != nullptr;
  const bool has_request =
      !implementation->request_jfs.empty() &&
      std::all_of(implementation->request_jfs.begin(),
                  implementation->request_jfs.end(),
                  [](const auto& queue) { return queue.valid(); }) &&
      implementation->request_jfc != nullptr;
  const bool has_response =
      !implementation->response_jfs.empty() &&
      std::all_of(implementation->response_jfs.begin(),
                  implementation->response_jfs.end(),
                  [](const auto& queue) { return queue.valid(); }) &&
      implementation->response_jfc != nullptr;
  const bool read_configured = resources.read_jfs.value != 0 ||
                               implementation->read_jfc != nullptr;
  const bool request_configured = !implementation->request_jfs.empty() ||
                                  implementation->request_jfc != nullptr;
  const bool response_configured = !implementation->response_jfs.empty() ||
                                   implementation->response_jfc != nullptr;
  const bool complete_role =
      read_configured == has_read &&
      request_configured == has_request &&
      response_configured == has_response &&
      ((has_read && has_request && has_response) ||
       (has_read && !has_request && !has_response) ||
       (!has_read && has_request && has_response));
  if (!complete_role) {
    return Status(StatusCode::kInvalidArgument, "UMDK lane resource handle is invalid");
  }
  try {
    implementation->local_segments.reserve(implementation->options.local_segments.size());
    for (const auto& binding : implementation->options.local_segments) {
      auto* native = static_cast<urma_target_seg_t*>(
          backend.Native(binding.resource, UrmaResourceKind::kRegisteredSegment));
      if (native == nullptr || binding.memory.empty()) {
        return Status(StatusCode::kInvalidArgument, "local UMDK segment binding is invalid");
      }
      implementation->local_segments.push_back(
          {native, binding.memory.data(), binding.memory.size()});
    }
    if (implementation->options.write_bounce.has_value()) {
      const auto& bounce = *implementation->options.write_bounce;
      auto* native = static_cast<urma_target_seg_t*>(
          backend.Native(bounce.resource, UrmaResourceKind::kRegisteredSegment));
      if (native == nullptr || bounce.memory.empty() || bounce.slot_bytes == 0 ||
          bounce.memory.size() < bounce.slot_bytes) {
        return Status(StatusCode::kInvalidArgument,
                      "UMDK write bounce binding is invalid");
      }
      implementation->bounce_segment = native;
      implementation->bounce_begin = bounce.memory.data();
      implementation->bounce_bytes = bounce.memory.size();
      implementation->bounce_stride = bounce.slot_bytes;
      implementation->bounce_busy.assign(bounce.memory.size() / bounce.slot_bytes, false);
    }
    implementation->peers.reserve(implementation->options.peers.size());
    for (const auto& binding : implementation->options.peers) {
      Impl::Peer peer;
      peer.rank = binding.rank;
      for (const auto& segment : binding.segments) {
        auto* native = static_cast<urma_target_seg_t*>(
            backend.Native(segment.resource, UrmaResourceKind::kImportedSegment));
        if (native == nullptr || segment.bytes == 0) {
          return Status(StatusCode::kInvalidArgument, "remote UMDK segment binding is invalid");
        }
        peer.segments.push_back({segment.kind, native, segment.bytes});
      }
      for (const auto& endpoint : binding.endpoints) {
        auto* native = static_cast<urma_target_jetty_t*>(
            backend.Native(endpoint.resource, UrmaResourceKind::kImportedEndpoint));
        if (native == nullptr) {
          return Status(StatusCode::kInvalidArgument, "remote UMDK endpoint binding is invalid");
        }
        if (implementation->SendQueue(endpoint.operation_class,
                                      endpoint.send_queue_index) == nullptr) {
          return Status(StatusCode::kInvalidArgument,
                        "remote UMDK endpoint send queue is invalid");
        }
        peer.endpoints.push_back({endpoint.operation_class, native,
                                  endpoint.send_queue_index});
      }
      implementation->peers.push_back(std::move(peer));
    }
    implementation->pending.resize(implementation->options.max_outstanding);
    implementation->post_contexts.resize(implementation->options.wr_list_max);
    implementation->post_bounce_slots.resize(implementation->options.wr_list_max,
                                             kNoBounceSlot);
  } catch (const std::bad_alloc&) {
    return Status(StatusCode::kResourceExhausted, "unable to allocate UMDK lane tables");
  }
  auto engine = std::unique_ptr<UmdkLanePostEngine>(
      new (std::nothrow) UmdkLanePostEngine(std::move(implementation)));
  if (!engine) {
    return Status(StatusCode::kResourceExhausted, "unable to allocate UMDK lane wrapper");
  }
  return engine;
}

std::uint64_t UmdkLanePostEngine::native_descriptors_high_watermark() const noexcept {
  return impl_->native_busy_high_watermark;
}

Status UmdkLanePostEngine::PostRead(const ReadOperation& operation) {
  auto result = PostReads(std::span(&operation, 1));
  return result.status;
}

Status UmdkLanePostEngine::PostWriteNotify(const WriteNotifyOperation& operation) {
  auto result = PostWriteNotifies(std::span(&operation, 1));
  return result.status;
}

LanePostBatchResult UmdkLanePostEngine::PostReads(
    std::span<const ReadOperation> operations) {
  LanePostBatchResult result;
  if (!impl_->read_jfs.valid()) {
    result.status = Status(
        StatusCode::kInvalidArgument,
        "RPC-only UMDK lane cannot post READ operations");
    return result;
  }
  std::size_t count = std::min(operations.size(), impl_->options.wr_list_max);
  if (count == 0) {
    return result;
  }
  const auto peer_rank = operations.front().peer_rank;
  const auto remote_segment = operations.front().remote_segment;
  for (std::size_t index = 1; index < count; ++index) {
    if (operations[index].peer_rank != peer_rank ||
        operations[index].remote_segment != remote_segment) {
      count = index;
      break;
    }
  }
  count = std::min({count, impl_->options.max_outstanding - impl_->pending_count,
                    impl_->pending.size() - impl_->native_busy_count});
  if (count == 0) {
    result.status = Status(StatusCode::kResourceExhausted, "UMDK lane has no READ WR descriptors or credits");
    return result;
  }
  for (std::size_t index = 0; index < count; ++index) {
    const auto& operation = operations[index];
    auto ticket_status = impl_->ValidateTicket(operation.ticket);
    const auto* remote = impl_->FindRemoteSegment(operation.peer_rank,
                                                   operation.remote_segment);
    auto* endpoint = impl_->FindEndpoint(operation.peer_rank, OperationClass::kRead);
    auto* local = impl_->FindLocalSegment(operation.destination.data(),
                                          operation.destination.size());
    if (!ticket_status.ok() || operation.destination.empty() || remote == nullptr ||
        endpoint == nullptr || local == nullptr ||
        operation.destination.size() > std::numeric_limits<std::uint32_t>::max() ||
        operation.remote_offset > remote->bytes ||
        operation.destination.size() > remote->bytes - operation.remote_offset) {
      result.status = ticket_status.ok()
                          ? Status(StatusCode::kInvalidArgument,
                                   "READ binding, size, or remote offset is invalid")
                          : ticket_status;
      return result;
    }
  }
  for (std::size_t index = 0; index < count; ++index) {
    const auto& operation = operations[index];
    const auto* remote = impl_->FindRemoteSegment(operation.peer_rank,
                                                   operation.remote_segment);
    auto* endpoint = impl_->FindEndpoint(operation.peer_rank, OperationClass::kRead);
    auto* local = impl_->FindLocalSegment(operation.destination.data(),
                                          operation.destination.size());
    auto context = impl_->ReservePending({
        .ticket = operation.ticket,
        .operation_class = OperationClass::kRead,
        .bytes = operation.destination.size(),
        .peer_rank = operation.peer_rank,
        .payload_segment = operation.remote_segment,
    });
    if (!context.ok()) {
      for (std::size_t rollback = 0; rollback < index; ++rollback) {
        impl_->ReleasePrepared(impl_->post_contexts[rollback]);
      }
      result.status = context.status();
      return result;
    }
    impl_->post_contexts[index] = context.value();
    auto* pending = impl_->FindPendingSlot(context.value());
    if (pending == nullptr) {
      for (std::size_t rollback = 0; rollback <= index; ++rollback) {
        impl_->ReleasePrepared(impl_->post_contexts[rollback]);
      }
      result.status = Status(StatusCode::kProtocolError,
                             "reserved READ native WR slot is unavailable");
      return result;
    }
    auto& source = pending->source;
    source = {};
    source.addr = remote->native->seg.ubva.va + operation.remote_offset;
    source.len = static_cast<std::uint32_t>(operation.destination.size());
    source.tseg = remote->native;
    auto& destination = pending->read_destination;
    destination = {};
    destination.addr = reinterpret_cast<std::uint64_t>(operation.destination.data());
    destination.len = static_cast<std::uint32_t>(operation.destination.size());
    destination.tseg = local;
    auto& work = pending->work;
    work = {};
    work.opcode = URMA_OPC_READ;
    work.flag.bs.complete_enable = 1;
    work.tjetty = endpoint->native;
    work.user_ctx = context.value();
    work.rw.src = {&source, 1};
    work.rw.dst = {&destination, 1};
    work.next = nullptr;
    if (index != 0) {
      impl_->FindPendingSlot(impl_->post_contexts[index - 1U])->work.next = &work;
    }
  }
  urma_jfs_wr_t* bad_work_request = nullptr;
  auto* first_work = &impl_->FindPendingSlot(impl_->post_contexts.front())->work;
  result.post_calls = 1;
  const auto provider_status = impl_->read_jfs.Post(first_work, &bad_work_request);
  std::size_t posted = count;
  if (provider_status != URMA_SUCCESS) {
    posted = AcceptedPrefix(first_work, count, bad_work_request);
    result.status = MapUrmaProviderStatus(
                        provider_status, std::string(impl_->read_jfs.PostOperation()) + "(READ)")
                        .AddContext("bad_wr_index", std::to_string(posted));
  }
  if (posted != 0 && posted < count) {
    impl_->FindPendingSlot(impl_->post_contexts[posted - 1U])->work.next = nullptr;
  }
  for (std::size_t index = posted; index < count; ++index) {
    impl_->ReleasePrepared(impl_->post_contexts[index]);
  }
  if (posted != 0) {
    auto committed = impl_->CommitPostedBatch(
        std::span(impl_->post_contexts).first(posted));
    if (!committed.ok()) {
      result.status = committed;
    }
  }
  result.posted = posted;
  return result;
}

LanePostBatchResult UmdkLanePostEngine::PostWriteNotifies(
    std::span<const WriteNotifyOperation> operations) {
  LanePostBatchResult result;
  std::size_t count = std::min(operations.size(), impl_->options.wr_list_max);
  if (count == 0) {
    return result;
  }
  const auto operation_class = NormalizeOperationClass(operations.front().operation_class);
  if (operation_class == OperationClass::kRead) {
    result.status = Status(
        StatusCode::kInvalidArgument,
        "READ-only UMDK lane cannot post WRITE_NOTIFY operations");
    return result;
  }
  // bonding_dev_0 retains one target-jetty reference for a linked WR list.
  // Mixing peers changes work.tjetty inside that list and crashes B015 while
  // bondp_poll_jfc releases the provider-side WR reference. Keep each native
  // list on one imported endpoint; the caller rolls back and retries the
  // unaccepted suffix on a later progress iteration.
  const auto peer_rank = operations.front().peer_rank;
  const auto* first_endpoint =
      impl_->FindEndpoint(peer_rank, operation_class);
  auto* send_queue = first_endpoint == nullptr
                         ? nullptr
                         : impl_->SendQueue(
                               operation_class,
                               first_endpoint->send_queue_index);
  if (send_queue == nullptr) {
    result.status = Status(
        StatusCode::kInvalidArgument,
        "WRITE_NOTIFY peer endpoint or send queue is invalid");
    return result;
  }
  const auto payload_segment = operations.front().payload_segment;
  const auto notify_segment = operations.front().notify_segment;
  for (std::size_t index = 1; index < count; ++index) {
    const auto& operation = operations[index];
    if (operation.operation_class != operation_class ||
        operation.peer_rank != peer_rank ||
        operation.payload_segment != payload_segment ||
        operation.notify_segment != notify_segment) {
      count = index;
      break;
    }
  }
  if (operation_class == OperationClass::kRead ||
      count > impl_->options.max_outstanding - impl_->pending_count) {
    result.status = Status(StatusCode::kInvalidArgument,
                           "WRITE_NOTIFY class or operation capacity is invalid");
    return result;
  }
  for (std::size_t index = 0; index < count; ++index) {
    const auto& operation = operations[index];
    auto ticket_status = impl_->ValidateTicket(operation.ticket);
    const auto provider_write_limit =
        impl_->backend->capabilities().max_write_bytes;
    if (provider_write_limit == 0 ||
        operation.payload.size() > provider_write_limit) {
      result.status =
          Status(StatusCode::kResourceExhausted,
                 "WRITE_NOTIFY payload exceeds the provider write limit")
              .AddContext("payload_bytes",
                          std::to_string(operation.payload.size()))
              .AddContext("max_write_bytes",
                          std::to_string(provider_write_limit));
      return result;
    }
    const auto* payload = impl_->FindRemoteSegment(operation.peer_rank,
                                                    operation.payload_segment);
    const auto* notify = impl_->FindRemoteSegment(operation.peer_rank,
                                                   operation.notify_segment);
    auto* endpoint = impl_->FindEndpoint(operation.peer_rank, operation_class);
    if (!ticket_status.ok() || operation.operation_class != operation_class ||
        operation.payload.empty() || operation.notify_value == 0 || payload == nullptr ||
        notify == nullptr || endpoint == nullptr ||
        (operation.inline_source &&
         operation.payload.size() > impl_->backend->capabilities().max_inline_bytes) ||
        operation.payload.size() > std::numeric_limits<std::uint32_t>::max() ||
        operation.remote_payload_offset > payload->bytes ||
        operation.payload.size() > payload->bytes - operation.remote_payload_offset ||
        operation.remote_notify_offset % alignof(std::uint64_t) != 0 ||
        operation.remote_notify_offset > notify->bytes ||
        sizeof(std::uint64_t) > notify->bytes - operation.remote_notify_offset) {
      result.status = ticket_status.ok()
                          ? Status(StatusCode::kInvalidArgument,
                                   "WRITE_NOTIFY binding, size, offset, or inline mode is invalid")
                          : ticket_status;
      return result;
    }
  }
  std::fill_n(impl_->post_bounce_slots.begin(), count, kNoBounceSlot);
  for (std::size_t index = 0; index < count; ++index) {
    const auto& operation = operations[index];
    const auto* payload = impl_->FindRemoteSegment(operation.peer_rank,
                                                    operation.payload_segment);
    const auto* notify = impl_->FindRemoteSegment(operation.peer_rank,
                                                   operation.notify_segment);
    auto* endpoint = impl_->FindEndpoint(operation.peer_rank, operation_class);
    auto* local = impl_->FindLocalSegment(operation.payload.data(),
                                          operation.payload.size());
    const std::byte* source_address = operation.payload.data();
    // Some bonding providers retain every SGE tseg even for inline WRs.
    // Keep the source segment valid by staging unregistered payloads in the
    // registered bounce pool instead of relying on a null inline source tseg.
    if (local == nullptr) {
      auto bounce = impl_->AcquireBounce(operation.payload.size());
      if (!bounce.has_value()) {
        for (std::size_t rollback = 0; rollback < index; ++rollback) {
          impl_->ReleasePrepared(impl_->post_contexts[rollback]);
          impl_->ReleaseBounce(impl_->post_bounce_slots[rollback]);
        }
        result.status = Status(StatusCode::kResourceExhausted,
                               "UMDK registered write bounce pool is exhausted");
        return result;
      }
      impl_->post_bounce_slots[index] = *bounce;
      auto* destination = impl_->bounce_begin + *bounce * impl_->bounce_stride;
      std::memcpy(destination, operation.payload.data(), operation.payload.size());
      source_address = destination;
      local = impl_->bounce_segment;
    }
    auto context = impl_->ReservePending({
        .ticket = operation.ticket,
        .operation_class = operation_class,
        .bytes = operation.payload.size(),
        .bounce_slot = impl_->post_bounce_slots[index],
        .peer_rank = operation.peer_rank,
        .payload_segment = operation.payload_segment,
        .notify_segment = operation.notify_segment,
        .inline_source = operation.inline_source,
    });
    if (!context.ok()) {
      for (std::size_t rollback = 0; rollback < index; ++rollback) {
        impl_->ReleasePrepared(impl_->post_contexts[rollback]);
      }
      for (std::size_t rollback = 0; rollback <= index; ++rollback) {
        impl_->ReleaseBounce(impl_->post_bounce_slots[rollback]);
      }
      result.status = context.status();
      return result;
    }
    impl_->post_contexts[index] = context.value();
    auto* pending = impl_->FindPendingSlot(context.value());
    if (pending == nullptr) {
      for (std::size_t rollback = 0; rollback <= index; ++rollback) {
        impl_->ReleasePrepared(impl_->post_contexts[rollback]);
        impl_->ReleaseBounce(impl_->post_bounce_slots[rollback]);
      }
      result.status = Status(StatusCode::kProtocolError,
                             "reserved WRITE_NOTIFY native WR slot is unavailable");
      return result;
    }
    auto& source = pending->source;
    source = {};
    source.addr = reinterpret_cast<std::uint64_t>(source_address);
    source.len = static_cast<std::uint32_t>(operation.payload.size());
    source.tseg = local;
    auto& destinations = pending->write_destinations;
    destinations = {};
    destinations[0].addr = payload->native->seg.ubva.va + operation.remote_payload_offset;
    destinations[0].len = static_cast<std::uint32_t>(operation.payload.size());
    destinations[0].tseg = payload->native;
    destinations[1].addr = notify->native->seg.ubva.va + operation.remote_notify_offset;
    destinations[1].len = sizeof(std::uint64_t);
    destinations[1].tseg = notify->native;
    auto& work = pending->work;
    work = {};
    work.opcode = URMA_OPC_WRITE_NOTIFY;
    // Each operation requests a CR in the correctness path. Grouped completion
    // is enabled only after S2 establishes the provider's partial-list rule.
    work.flag.bs.complete_enable = 1;
    work.flag.bs.inline_flag = operation.inline_source ? 1 : 0;
    work.tjetty = endpoint->native;
    work.user_ctx = context.value();
    work.rw.src = {&source, 1};
    work.rw.dst = {destinations.data(), 2};
    work.rw.notify_data = operation.notify_value;
    work.next = nullptr;
    if (index != 0) {
      impl_->FindPendingSlot(impl_->post_contexts[index - 1U])->work.next = &work;
    }
  }
  urma_jfs_wr_t* bad_work_request = nullptr;
  auto* first_work = &impl_->FindPendingSlot(impl_->post_contexts.front())->work;
  const auto provider_post_started =
      impl_->options.profiling_enabled
          ? std::chrono::steady_clock::now()
          : std::chrono::steady_clock::time_point{};
  const auto provider_status = send_queue->Post(first_work, &bad_work_request);
  if (impl_->options.profiling_enabled) {
    result.provider_post_nanoseconds =
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() -
                provider_post_started)
                .count());
  }
  std::size_t posted = count;
  if (provider_status != URMA_SUCCESS) {
    posted = AcceptedPrefix(first_work, count, bad_work_request);
    result.status = MapUrmaProviderStatus(
                        provider_status, std::string(send_queue->PostOperation()) + "(WRITE_NOTIFY)")
                        .AddContext("bad_wr_index", std::to_string(posted));
  }
  if (posted != 0 && posted < count) {
    impl_->FindPendingSlot(impl_->post_contexts[posted - 1U])->work.next = nullptr;
  }
  for (std::size_t index = posted; index < count; ++index) {
    impl_->ReleasePrepared(impl_->post_contexts[index]);
    impl_->ReleaseBounce(impl_->post_bounce_slots[index]);
  }
  if (posted != 0) {
    auto committed = impl_->CommitPostedBatch(
        std::span(impl_->post_contexts).first(posted));
    if (!committed.ok()) {
      result.status = committed;
    }
  }
  result.posted = posted;
  return result;
}

Result<std::size_t> UmdkLanePostEngine::Poll(
    std::span<TransportCompletion> completions) {
  std::size_t produced = 0;
  const std::array<urma_jfc_t*, 3> queues{impl_->read_jfc, impl_->request_jfc,
                                         impl_->response_jfc};
  for (std::size_t offset = 0; offset < queues.size() && produced < completions.size(); ++offset) {
    const std::size_t queue_index = (impl_->next_poll_class + offset) % queues.size();
    if (queues[queue_index] == nullptr) {
      continue;
    }
    const auto capacity = std::min({impl_->options.poll_batch,
                                    completions.size() - produced, kProviderPollLimit});
    std::array<urma_cr_t, kProviderPollLimit> records{};
    const int count = urma_poll_jfc(queues[queue_index], static_cast<int>(capacity), records.data());
    if (count < 0) {
      return Status(StatusCode::kTransportError, "urma_poll_jfc failed")
          .AddContext("errno", std::to_string(errno))
          .AddContext("queue_class", std::to_string(queue_index));
    }
    if (queue_index == 0 && count != 0) {
      // Order completed DMA writes before the release publication to Search.
      auto visible = impl_->backend->AcquireDeviceWrites();
      if (!visible.ok()) return visible;
    }
    for (int index = 0; index < count; ++index) {
      const auto context = records[static_cast<std::size_t>(index)].user_ctx;
      auto completed = impl_->CompletePending(context);
      if (!completed.ok()) {
        return Status(StatusCode::kProtocolError, "UMDK completion has an unknown user_ctx")
            .AddContext("user_ctx", std::to_string(context))
            .AddContext("cause", completed.status().ToString());
      }
      const auto& pending = completed.value();
      auto& output = completions[produced++];
      output.ticket = pending.ticket;
      output.operation_class = pending.operation_class;
      output.bytes = pending.bytes;
      output.status = Status::Ok();
      if (records[static_cast<std::size_t>(index)].status != URMA_CR_SUCCESS) {
        const auto native_status =
            records[static_cast<std::size_t>(index)].status;
        const auto cr_status = static_cast<int>(native_status);
        output.status =
            Status(StatusCode::kTransportError,
                   "UMDK completion reported an error")
                .AddContext("cr_status", std::to_string(cr_status))
                .AddContext("cr_status_name",
                            CompletionStatusName(native_status))
                .AddContext("queue_class",
                            CompletionQueueClassName(queue_index))
                .AddContext("operation_class",
                            OperationClassName(pending.operation_class))
                .AddContext("peer_rank",
                            std::to_string(pending.peer_rank))
                .AddContext("ticket",
                            std::to_string(pending.ticket.value))
                .AddContext("operation_slot",
                            std::to_string(
                                pending.ticket.operation_slot))
                .AddContext("owner_lane",
                            std::to_string(pending.ticket.owner_lane))
                .AddContext("generation",
                            std::to_string(pending.ticket.generation))
                .AddContext("bytes",
                            std::to_string(pending.bytes))
                .AddContext("remote_payload_segment",
                            SegmentKindName(pending.payload_segment))
                .AddContext(
                    "remote_notify_segment",
                    pending.operation_class == OperationClass::kRead
                        ? "none"
                        : SegmentKindName(pending.notify_segment))
                .AddContext("inline_source",
                            pending.inline_source ? "true" : "false")
                .AddContext(
                    "bounce_source",
                    pending.bounce_slot == kNoBounceSlot ? "false"
                                                         : "true");
      }
    }
  }
  impl_->next_poll_class = (impl_->next_poll_class + 1U) % 3U;
  return produced;
}

}  // namespace hnsw_urma

#include "hnsw_urma/core/status.hpp"

#include <sstream>

namespace hnsw_urma {

Status &Status::AddContext(std::string key, std::string value) {
  context_.emplace_back(std::move(key), std::move(value));
  return *this;
}

std::optional<std::string_view>
Status::FindContext(std::string_view key) const noexcept {
  for (auto it = context_.rbegin(); it != context_.rend(); ++it) {
    if (it->first == key) {
      return it->second;
    }
  }
  return std::nullopt;
}

std::string Status::ToString() const {
  std::ostringstream out;
  out << StatusCodeName(code_);
  if (!message_.empty()) {
    out << ": " << message_;
  }
  for (const auto &[key, value] : context_) {
    out << " [" << key << '=' << value << ']';
  }
  return out.str();
}

std::string_view StatusCodeName(StatusCode code) noexcept {
  switch (code) {
  case StatusCode::kOk:
    return "OK";
  case StatusCode::kInvalidArgument:
    return "INVALID_ARGUMENT";
  case StatusCode::kInvalidFormat:
    return "INVALID_FORMAT";
  case StatusCode::kOutOfRange:
    return "OUT_OF_RANGE";
  case StatusCode::kResourceExhausted:
    return "RESOURCE_EXHAUSTED";
  case StatusCode::kProtocolError:
    return "PROTOCOL_ERROR";
  case StatusCode::kTimeout:
    return "TIMEOUT";
  case StatusCode::kCancelled:
    return "CANCELLED";
  case StatusCode::kTransportError:
    return "TRANSPORT_ERROR";
  case StatusCode::kInternalError:
    return "INTERNAL_ERROR";
  }
  return "UNKNOWN";
}

} // namespace hnsw_urma

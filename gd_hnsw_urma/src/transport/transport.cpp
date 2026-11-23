#include "hnsw_urma/transport/transport.hpp"

namespace hnsw_urma {
void TransportModuleAnchor() {}

LanePostBatchResult LanePostEngine::PostReads(std::span<const ReadOperation> operations) {
  LanePostBatchResult result;
  for (const auto& operation : operations) {
    ++result.post_calls;
    result.status = PostRead(operation);
    if (!result.status.ok()) {
      return result;
    }
    ++result.posted;
  }
  return result;
}

LanePostBatchResult LanePostEngine::PostWriteNotifies(
    std::span<const WriteNotifyOperation> operations) {
  LanePostBatchResult result;
  for (const auto& operation : operations) {
    result.status = PostWriteNotify(operation);
    if (!result.status.ok()) {
      return result;
    }
    ++result.posted;
  }
  return result;
}
}  // namespace hnsw_urma

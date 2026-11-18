#include "hnsw_urma/search/mock_io.hpp"

#include <algorithm>

namespace hnsw_urma {

Result<GraphTicket> MockGraphReader::Submit(QueryKey query,
                                           std::span<const NodeToExpand> nodes,
                                           NeighborBatch* output) {
  if (!next_rejection_.ok()) {
    Status rejection = std::move(next_rejection_);
    next_rejection_ = Status::Ok();
    return rejection;
  }
  if (nodes.empty() || output == nullptr) {
    return Status(StatusCode::kInvalidArgument, "invalid mock graph request");
  }
  GraphTicket ticket{next_ticket_++, query.generation};
  Operation operation{ticket, std::vector<NodeToExpand>(nodes.begin(), nodes.end()), output, false,
                      std::move(next_failure_)};
  next_failure_ = Status::Ok();
  operations_.emplace(ticket.value, std::move(operation));
  return ticket;
}

Result<bool> MockGraphReader::Poll(GraphTicket ticket) {
  const auto found = operations_.find(ticket.value);
  if (found == operations_.end() || found->second.ticket.generation != ticket.generation) {
    return Status(StatusCode::kProtocolError, "unknown mock graph ticket");
  }
  return found->second.ready;
}

Status MockGraphReader::Finish(GraphTicket ticket) {
  const auto found = operations_.find(ticket.value);
  if (found == operations_.end() || !found->second.ready) {
    return Status(StatusCode::kProtocolError, "mock graph ticket is not complete");
  }
  Status status = found->second.status;
  operations_.erase(found);
  return status;
}

Result<std::span<const node_id_t>> MockGraphReader::ReadUpperNeighbors(
    node_id_t id, std::int32_t level) const noexcept {
  auto result = lookup_(id, level);
  if (!result.ok()) {
    return result.status();
  }
  upper_neighbors_ = std::move(result).value();
  return std::span<const node_id_t>(upper_neighbors_);
}

void MockGraphReader::Complete(GraphTicket ticket) {
  auto found = operations_.find(ticket.value);
  if (found == operations_.end() || found->second.ready) {
    return;
  }
  auto& operation = found->second;
  if (operation.status.ok()) {
    operation.output->Clear();
    for (const auto& node : operation.nodes) {
      auto result = lookup_(node.global_id, node.level);
      if (!result.ok()) {
        operation.status = result.status();
        break;
      }
      operation.output->neighbors.insert(operation.output->neighbors.end(), result.value().begin(),
                                         result.value().end());
      operation.output->offsets.push_back(
          static_cast<std::uint32_t>(operation.output->neighbors.size()));
    }
  }
  operation.ready = true;
}

void MockGraphReader::CompleteAll() {
  const auto tickets = pending_tickets();
  for (const auto& ticket : tickets) {
    Complete(ticket);
  }
}

std::vector<GraphTicket> MockGraphReader::pending_tickets() const {
  std::vector<GraphTicket> tickets;
  for (const auto& [id, operation] : operations_) {
    static_cast<void>(id);
    if (!operation.ready) {
      tickets.push_back(operation.ticket);
    }
  }
  std::sort(tickets.begin(), tickets.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.value < rhs.value; });
  return tickets;
}

Result<DistanceTicket> MockDistanceOracle::Submit(QueryKey query,
                                                  std::span<const float> query_vector,
                                                  shard_id_t owner_shard,
                                                  std::span<const node_id_t> ids,
                                                  std::span<float> output,
                                                  DistanceRequestOptions options) {
  static_cast<void>(owner_shard);
  saw_upper_distance_ |= options.upper_distance.has_value();
  if (!next_rejection_.ok()) {
    Status rejection = std::move(next_rejection_);
    next_rejection_ = Status::Ok();
    return rejection;
  }
  if (ids.empty() || ids.size() != output.size()) {
    return Status(StatusCode::kInvalidArgument, "invalid mock distance request");
  }
  DistanceTicket ticket{next_ticket_++, query.generation};
  Operation operation{ticket, std::vector<float>(query_vector.begin(), query_vector.end()),
                      std::vector<node_id_t>(ids.begin(), ids.end()), output, false,
                      std::move(next_failure_)};
  next_failure_ = Status::Ok();
  operations_.emplace(ticket.value, std::move(operation));
  return ticket;
}

Result<bool> MockDistanceOracle::Poll(DistanceTicket ticket) {
  const auto found = operations_.find(ticket.value);
  if (found == operations_.end() || found->second.ticket.generation != ticket.generation) {
    return Status(StatusCode::kProtocolError, "unknown mock distance ticket");
  }
  return found->second.ready;
}

Status MockDistanceOracle::Finish(DistanceTicket ticket) {
  const auto found = operations_.find(ticket.value);
  if (found == operations_.end() || !found->second.ready) {
    return Status(StatusCode::kProtocolError, "mock distance ticket is not complete");
  }
  Status status = found->second.status;
  operations_.erase(found);
  return status;
}

void MockDistanceOracle::Complete(DistanceTicket ticket) {
  auto found = operations_.find(ticket.value);
  if (found == operations_.end() || found->second.ready) {
    return;
  }
  auto& operation = found->second;
  if (operation.status.ok()) {
    for (std::size_t index = 0; index < operation.ids.size(); ++index) {
      auto result = lookup_(operation.query, operation.ids[index]);
      if (!result.ok()) {
        operation.status = result.status();
        break;
      }
      operation.output[index] = result.value();
    }
  }
  operation.ready = true;
}

void MockDistanceOracle::CompleteAll(bool reverse) {
  auto tickets = pending_tickets();
  if (reverse) {
    std::reverse(tickets.begin(), tickets.end());
  }
  for (const auto& ticket : tickets) {
    Complete(ticket);
  }
}

std::vector<DistanceTicket> MockDistanceOracle::pending_tickets() const {
  std::vector<DistanceTicket> tickets;
  for (const auto& [id, operation] : operations_) {
    static_cast<void>(id);
    if (!operation.ready) {
      tickets.push_back(operation.ticket);
    }
  }
  std::sort(tickets.begin(), tickets.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.value < rhs.value; });
  return tickets;
}

}  // namespace hnsw_urma

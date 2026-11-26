/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * You can use this software according to the terms and conditions of the Mulan
 * PSL v2. You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE. See the
 * Mulan PSL v2 for more details.
 */

#pragma once

#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace hnsw_urma {

enum class StatusCode {
  kOk = 0,
  kInvalidArgument,
  kInvalidFormat,
  kOutOfRange,
  kResourceExhausted,
  kProtocolError,
  kTimeout,
  kCancelled,
  kTransportError,
  kInternalError,
};

class Status {
public:
  Status() = default;
  Status(StatusCode code, std::string message)
      : code_(code), message_(std::move(message)) {}

  [[nodiscard]] static Status Ok() { return {}; }
  [[nodiscard]] bool ok() const noexcept { return code_ == StatusCode::kOk; }
  [[nodiscard]] StatusCode code() const noexcept { return code_; }
  [[nodiscard]] const std::string &message() const noexcept { return message_; }
  [[nodiscard]] const std::vector<std::pair<std::string, std::string>> &
  context() const noexcept {
    return context_;
  }

  Status &AddContext(std::string key, std::string value);
  [[nodiscard]] std::optional<std::string_view>
  FindContext(std::string_view key) const noexcept;
  [[nodiscard]] std::string ToString() const;

private:
  StatusCode code_{StatusCode::kOk};
  std::string message_;
  std::vector<std::pair<std::string, std::string>> context_;
};

[[nodiscard]] std::string_view StatusCodeName(StatusCode code) noexcept;

template <typename T> class [[nodiscard]] Result {
public:
  Result(const T &value) : value_(value) {}
  Result(T &&value) : value_(std::move(value)) {}
  Result(Status status) : status_(std::move(status)) {
    if (status_.ok()) {
      status_ = Status(StatusCode::kInternalError,
                       "Result constructed without a value");
    }
  }

  [[nodiscard]] bool ok() const noexcept { return value_.has_value(); }
  [[nodiscard]] const Status &status() const noexcept { return status_; }

  [[nodiscard]] const T &value() const & {
    if (!value_) {
      std::abort();
    }
    return *value_;
  }

  [[nodiscard]] T &value() & {
    if (!value_) {
      std::abort();
    }
    return *value_;
  }

  [[nodiscard]] T &&value() && {
    if (!value_) {
      std::abort();
    }
    return std::move(*value_);
  }

private:
  std::optional<T> value_;
  Status status_;
};

} // namespace hnsw_urma

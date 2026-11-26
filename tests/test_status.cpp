/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

// Migrated from Milvus ub_alignment; source identity in
// manifests/reference-tests.json.
#include "test_framework.hpp"

#include <string>

#include "hnsw_urma/core/status.hpp"

using hnsw_urma::Status;
using hnsw_urma::StatusCode;

TEST(ReferenceUnitStatus, StatusPreservesStructuredContext) {
  Status status(StatusCode::kTransportError, "post failed");
  status.AddContext("rank", "3")
      .AddContext("peer", "7")
      .AddContext("phase", "graph-read");

  UB_REQUIRE_EQ(status.code(), StatusCode::kTransportError);
  UB_REQUIRE_EQ(status.FindContext("rank").value(), "3");
  UB_REQUIRE_EQ(status.FindContext("peer").value(), "7");
  UB_REQUIRE_EQ(status.FindContext("phase").value(), "graph-read");
  UB_REQUIRE_TRUE(status.ToString().find("TRANSPORT_ERROR") !=
                  std::string::npos);
}

TEST(ReferenceUnitStatus, ResultCarriesValueOrError) {
  hnsw_urma::Result<int> value(42);
  UB_REQUIRE_TRUE(value.ok());
  UB_REQUIRE_EQ(value.value(), 42);

  hnsw_urma::Result<int> error(
      Status(StatusCode::kInvalidArgument, "bad value"));
  UB_REQUIRE_FALSE(error.ok());
  UB_REQUIRE_EQ(error.status().code(), StatusCode::kInvalidArgument);
}

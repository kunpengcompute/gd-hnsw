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

#include <cstdlib>
#include <filesystem>
#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
  const auto pattern =
      (std::filesystem::temp_directory_path() / "gd-hnsw-ut-XXXXXX").string();
  std::vector<char> buffer(pattern.begin(), pattern.end());
  buffer.push_back('\0');
  const char *temporary = ::mkdtemp(buffer.data());
  if (!temporary) {
    std::cerr << "Unable to create isolated test directory\n";
    return 2;
  }
  const std::filesystem::path isolated(temporary);
  if (::setenv("TMPDIR", temporary, 1) != 0)
    return 2;
  const int result = RUN_ALL_TESTS();
  std::error_code error;
  std::filesystem::remove_all(isolated, error);
  if (error) {
    std::cerr << "Temporary directory cleanup: " << error.message() << '\n';
    return 2;
  }
  return result;
}

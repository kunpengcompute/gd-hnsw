#pragma once
#include "hnsw_urma/index/builder.hpp"
#include "hnsw_urma/index/format.hpp"
#include <filesystem>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace ub_complete {
inline const std::vector<float> kVectors{0, 0, 1, 0, 2, 0, 3, 0,
                                         0, 2, 1, 2, 2, 2, 3, 2};
class IndexFixture : public ::testing::Test {
protected:
  void SetUp() override {
    root_ = std::filesystem::temp_directory_path() / "fixture";
    std::error_code error;
    std::filesystem::remove_all(root_, error);
    Build();
  }
  void TearDown() override {
    std::error_code error;
    std::filesystem::remove_all(root_, error);
    EXPECT_FALSE(error) << error.message();
  }
  void Build(std::uint32_t metric = 0) {
    hnsw_urma::ReferenceBuildOptions options;
    options.num_shards = 2;
    options.M = 2;
    options.M0 = 7;
    options.ef_search = 64;
    options.metric = metric;
    auto status =
        hnsw_urma::BuildReferenceIndex(root_, kVectors, 8, 2, options, 100);
    ASSERT_TRUE(status.ok()) << status.ToString();
  }
  std::filesystem::path root_;
};
} // namespace ub_complete

// Migrated from Milvus ub_alignment; source identity in
// manifests/reference-tests.json.
#include "test_framework.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "hnsw_urma/compute/distance.hpp"

using hnsw_urma::ComputeDistance;
using hnsw_urma::ComputeDistanceBatch;
using hnsw_urma::ComputeDistanceBatch4;
using hnsw_urma::ComputeL2DotNormBatch4;
using hnsw_urma::DistanceImplementation;
using hnsw_urma::Metric;
using hnsw_urma::ScalarDot;
using hnsw_urma::ScalarL2;

TEST(ReferenceUnitDistance, DistanceScalarMatchesKnownValues) {
  const std::vector<float> lhs{1.0F, 2.0F, 3.0F};
  const std::vector<float> rhs{4.0F, 0.0F, -1.0F};
  UB_REQUIRE_NEAR(ScalarL2(lhs, rhs), 29.0F, 1e-6F);
  UB_REQUIRE_NEAR(ScalarDot(lhs, rhs), 1.0F, 1e-6F);
  UB_REQUIRE_NEAR(ComputeDistance(lhs, rhs, Metric::kInnerProduct,
                                  DistanceImplementation::kScalar),
                  -1.0F, 1e-6F);
}

TEST(ReferenceUnitDistance, DistanceSimdMatchesScalarWithinTolerance) {
  std::mt19937 random(123456U);
  std::uniform_real_distribution<float> distribution(-2.0F, 2.0F);
  for (std::size_t dim : {1U, 3U, 4U, 7U, 16U, 129U}) {
    std::vector<float> lhs(dim);
    std::vector<float> rhs(dim);
    for (std::size_t index = 0; index < dim; ++index) {
      lhs[index] = distribution(random);
      rhs[index] = distribution(random);
    }
    const float scalar =
        ComputeDistance(lhs, rhs, Metric::kL2, DistanceImplementation::kScalar);
    const float optimized =
        ComputeDistance(lhs, rhs, Metric::kL2, DistanceImplementation::kAuto);
    UB_REQUIRE_NEAR(scalar, optimized, 1e-4F * static_cast<float>(dim));
  }
}

TEST(ReferenceUnitDistance, DistanceBatchValidatesShapeAndPreservesRows) {
  const std::vector<float> query{1.0F, 2.0F};
  const std::vector<float> vectors{1.0F, 2.0F, 2.0F, 4.0F};
  std::vector<float> output(2);
  UB_REQUIRE_TRUE(
      ComputeDistanceBatch(query, vectors, 2, Metric::kL2, output).ok());
  UB_REQUIRE_NEAR(output[0], 0.0F, 1e-6F);
  UB_REQUIRE_NEAR(output[1], 5.0F, 1e-6F);
  UB_REQUIRE_FALSE(
      ComputeDistanceBatch(query, vectors, 3, Metric::kL2, output).ok());
}

TEST(ReferenceUnitDistance, DistanceBatch4MatchesIndependentL2Rows) {
  const std::vector<float> query{1.0F, -2.0F, 3.0F, 4.0F, -5.0F,
                                 6.0F, 7.0F,  8.0F, 9.0F};
  std::array<std::vector<float>, 4> vectors;
  std::array<const float *, 4> rows{};
  std::array<float, 4> output{};
  for (std::size_t row = 0; row < vectors.size(); ++row) {
    vectors[row].resize(query.size());
    for (std::size_t index = 0; index < query.size(); ++index) {
      vectors[row][index] =
          static_cast<float>(row * query.size() + index) * 0.25F;
    }
    rows[row] = vectors[row].data();
  }
  UB_REQUIRE_TRUE(ComputeDistanceBatch4(query, rows, Metric::kL2, output,
                                        DistanceImplementation::kAuto)
                      .ok());
  for (std::size_t row = 0; row < rows.size(); ++row) {
    UB_REQUIRE_NEAR(output[row],
                    ComputeDistance(query, vectors[row], Metric::kL2,
                                    DistanceImplementation::kScalar),
                    1e-4F * static_cast<float>(query.size()));
  }
}

TEST(ReferenceUnitDistance, DotNormBatch4MatchesDirectL2WithinTolerance) {
  const std::vector<float> query{1.0F,  -2.0F, 3.0F, 4.0F,
                                 -5.0F, 6.0F,  7.0F, 8.0F};
  std::array<std::vector<float>, 4> vectors;
  std::array<const float *, 4> rows{};
  std::array<float, 4> norms{};
  std::array<float, 4> output{};
  for (std::size_t row = 0; row < vectors.size(); ++row) {
    vectors[row].resize(query.size());
    for (std::size_t column = 0; column < query.size(); ++column) {
      vectors[row][column] = static_cast<float>(row * 3 + column) * 0.125F;
    }
    rows[row] = vectors[row].data();
    norms[row] = ScalarDot(vectors[row], vectors[row]);
  }
  UB_REQUIRE_TRUE(ComputeL2DotNormBatch4(query, ScalarDot(query, query), rows,
                                         norms, output)
                      .ok());
  for (std::size_t row = 0; row < rows.size(); ++row) {
    UB_REQUIRE_NEAR(output[row], ScalarL2(query, vectors[row]),
                    1e-4F * static_cast<float>(query.size()));
    UB_REQUIRE_TRUE(output[row] >= 0.0F);
  }
}

TEST(ReferenceUnitDistance, DistancePropagatesSpecialFloatingPointValues) {
  const std::vector<float> lhs{std::numeric_limits<float>::infinity(), 1.0F};
  const std::vector<float> rhs{0.0F, std::numeric_limits<float>::quiet_NaN()};
  UB_REQUIRE_TRUE(std::isnan(ScalarL2(lhs, rhs)));
}

#pragma once
#include <gtest/gtest.h>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string_view>

// The reference's requirements were fatal, including in non-void helpers.
// Keep that contract instead of continuing into Result::value() after failure.
// Native GoogleTest registers/runs every TEST and catches/report exceptions.
namespace hnsw_urma::test {
[[noreturn]] inline void Fail(std::string_view expression, std::string_view file,
                              int line, std::string_view detail = {}) {
  std::ostringstream message;
  message << file << ':' << line << ": " << expression;
  if (!detail.empty()) message << " (" << detail << ')';
  throw std::runtime_error(message.str());
}
}
#define UB_REQUIRE_TRUE(expression) do { if (!(expression)) \
  ::hnsw_urma::test::Fail(#expression, __FILE__, __LINE__); } while (false)
#define UB_REQUIRE_FALSE(expression) UB_REQUIRE_TRUE(!(expression))
#define UB_REQUIRE_EQ(lhs, rhs) do { const auto& ub_left = (lhs); const auto& ub_right = (rhs); \
  if (!(ub_left == ub_right)) ::hnsw_urma::test::Fail(#lhs " == " #rhs, __FILE__, __LINE__, \
    ::testing::PrintToString(ub_left) + " vs " + ::testing::PrintToString(ub_right)); } while (false)
#define UB_REQUIRE_NEAR(lhs, rhs, tolerance) do { const double ub_left=(lhs); \
  const double ub_right=(rhs); const double ub_tolerance=(tolerance); \
  if (!(ub_tolerance >= 0) || std::isnan(ub_left) || std::isnan(ub_right) || \
      !(ub_left == ub_right || std::fabs(ub_left-ub_right) <= ub_tolerance)) \
    ::hnsw_urma::test::Fail(#lhs " ~= " #rhs, __FILE__, __LINE__); } while(false)

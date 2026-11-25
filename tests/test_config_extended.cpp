#include "hnsw_urma/runtime/config.hpp"
#include "hnsw_urma/runtime/topology.hpp"
#include <array>
#include <gtest/gtest.h>
#include <string>
#include <string_view>
#include <vector>
using namespace hnsw_urma;
namespace {
Result<BenchmarkCommand> Parse(std::initializer_list<std::string_view> more) {
  std::vector<std::string_view> args{"--index", "fixture", "--dry-run"};
  args.insert(args.end(), more.begin(), more.end());
  return ParseBenchmarkCommand(args);
}
GlobalIndexMeta Meta() {
  GlobalIndexMeta m;
  m.ntotal = 8;
  m.dim = 2;
  m.M = 2;
  m.M0 = 7;
  m.ef_search_default = 8;
  m.entry_point = 1;
  m.max_level = 1;
  m.num_shards = 2;
  return m;
}
} // namespace
TEST(ConfigExtended, HelpVersionAndRequiredPaths) {
  const std::array<std::string_view, 1> help{"--help"}, version{"--version"};
  ASSERT_TRUE(ParseBenchmarkCommand(help).ok());
  EXPECT_EQ(ParseBenchmarkCommand(help).value().action, BenchmarkAction::kHelp);
  EXPECT_EQ(ParseBenchmarkCommand(version).value().action,
            BenchmarkAction::kVersion);
  EXPECT_FALSE(ParseBenchmarkCommand({}).ok());
  EXPECT_FALSE(BenchmarkHelp().empty());
  EXPECT_FALSE(BenchmarkVersion().empty());
  EXPECT_FALSE(GitCommit().empty());
}
TEST(ConfigExtended, NumericOptionsRejectMissingInvalidAndOverflowValues) {
  const std::vector<std::string_view> options{
      "--topk",
      "--ef-search",
      "--expand-batch",
      "--pull-threshold",
      "--search-workers",
      "--distance-workers",
      "--read-progress-threads",
      "--rpc-progress-threads",
      "--max-active-queries-per-rank",
      "--visited-capacity",
      "--read-submit-depth",
      "--rpc-submit-depth",
      "--rpc-operation-slots-per-search",
      "--read-result-slots-per-search",
      "--rpc-request-batch-max-tasks",
      "--rpc-request-batch-max-bytes",
      "--rpc-response-batch-max-bytes",
      "--rpc-response-progress-max-bytes",
      "--read-jfs-depth",
      "--request-jfs-depth",
      "--response-jfs-depth",
      "--read-jfc-depth",
      "--request-jfc-depth",
      "--response-jfc-depth",
      "--request-jfs-per-rpc-lane",
      "--response-jfs-per-rpc-lane",
      "--wr-list-max",
      "--request-ring-depth",
      "--operation-timeout-ms",
      "--query-timeout-ms",
      "--drain-timeout-ms",
      "--rounds"};
  for (const auto option : options) {
    SCOPED_TRACE(option);
    EXPECT_FALSE(Parse({option}).ok());
    for (const auto value : {"invalid", "-1", "18446744073709551616"}) {
      SCOPED_TRACE(value);
      EXPECT_FALSE(Parse({option, value}).ok());
    }
  }
}
TEST(ConfigExtended, AcceptedOptionBranchesPreserveValues) {
  for (const auto option : {"--topk",
                            "--ef-search",
                            "--expand-batch",
                            "--pull-threshold",
                            "--search-workers",
                            "--distance-workers",
                            "--read-progress-threads",
                            "--rpc-progress-threads",
                            "--max-active-queries-per-rank",
                            "--read-submit-depth",
                            "--rpc-submit-depth",
                            "--rpc-operation-slots-per-search",
                            "--read-result-slots-per-search",
                            "--rpc-request-batch-max-tasks",
                            "--rpc-request-batch-max-bytes",
                            "--rpc-response-batch-max-bytes",
                            "--rpc-response-progress-max-bytes",
                            "--read-jfs-depth",
                            "--request-jfs-depth",
                            "--response-jfs-depth",
                            "--read-jfc-depth",
                            "--request-jfc-depth",
                            "--response-jfc-depth",
                            "--request-jfs-per-rpc-lane",
                            "--response-jfs-per-rpc-lane",
                            "--wr-list-max",
                            "--request-ring-depth",
                            "--visited-capacity",
                            "--warmup",
                            "--rounds",
                            "--read-progress-spin-us",
                            "--rpc-progress-spin-us",
                            "--queue-memory-budget-bytes",
                            "--registered-rpc-memory-budget-bytes",
                            "--query-offset",
                            "--query-count"}) {
    SCOPED_TRACE(option);
    auto parsed = Parse({option, "8"});
    ASSERT_TRUE(parsed.ok()) << parsed.status().ToString();
    EXPECT_EQ(parsed.value().config.index, std::filesystem::path("fixture"));
    EXPECT_TRUE(parsed.value().config.dry_run);
  }
  auto parsed = Parse({"--topk",
                       "3",
                       "--queries",
                       "queries.bin",
                       "--ground-truth",
                       "truth.bin",
                       "--query-offset",
                       "2",
                       "--query-count",
                       "5",
                       "--output",
                       "out",
                       "--profile-output",
                       "profile",
                       "--profile",
                       "--use-dot-norm",
                       "--operation-timeout-ms",
                       "10",
                       "--query-timeout-ms",
                       "20",
                       "--drain-timeout-ms",
                       "30"});
  ASSERT_TRUE(parsed.ok());
  const auto &c = parsed.value().config;
  EXPECT_EQ(c.topk, 3U);
  EXPECT_EQ(c.query_offset, 2U);
  EXPECT_EQ(c.query_count, 5U);
  EXPECT_EQ(c.queries, std::filesystem::path("queries.bin"));
  EXPECT_TRUE(c.profile);
  EXPECT_TRUE(c.use_dot_norm);
  EXPECT_EQ(c.operation_timeout_ms, 10U);
}
TEST(ConfigExtended, EnumsAndAutoValues) {
  const std::vector<std::pair<std::string_view, std::vector<std::string_view>>>
      enums{{"--distance-policy", {"auto", "pull", "pushdown"}},
            {"--layer0-sort", {"all", "local-only", "upper-only"}},
            {"--cpu-bind", {"auto", "none"}},
            {"--cpu-layout",
             {"auto", "contiguous", "llc-isolated", "llc-spread"}},
            {"--read-progress-mode", {"busy", "adaptive"}},
            {"--rpc-progress-mode", {"busy", "adaptive"}},
            {"--tp-type", {"ctp", "rtp"}},
            {"--intra-node-transport", {"urma", "shm"}},
            {"--output-format", {"text", "json", "csv"}}};
  for (const auto &[option, values] : enums) {
    SCOPED_TRACE(option);
    for (auto value : values)
      EXPECT_TRUE(Parse({option, value}).ok());
    EXPECT_FALSE(Parse({option, "invalid"}).ok());
  }
  for (const auto option :
       {"--ef-search", "--search-workers", "--distance-workers",
        "--read-progress-threads", "--rpc-progress-threads",
        "--visited-capacity", "--read-submit-depth", "--rpc-submit-depth",
        "--eid-index", "--rpc-operation-slots-per-search"})
    EXPECT_TRUE(Parse({option, "auto"}).ok());
  EXPECT_EQ(ToString(DistancePolicy::kPull), "pull");
  EXPECT_EQ(ToString(OutputFormat::kJson), "json");
  EXPECT_EQ(ToString(CpuLayout::kContiguous), "contiguous");
  EXPECT_EQ(ToString(Layer0SortMode::kUpperOnly), "upper-only");
  EXPECT_EQ(ToString(IntraNodeTransport::kSharedMemory), "shm");
}
TEST(ConfigExtended, CrossFieldValidationAndDeviceMaps) {
  const std::vector<std::vector<std::string_view>> invalid{
      {"--expand-batch", "65"},
      {"--request-ring-depth", "3"},
      {"--visited-capacity", "7"},
      {"--operation-timeout-ms", "50", "--query-timeout-ms", "10"},
      {"--urma-device", "a", "--urma-device-map", "0=b"},
      {"--unknown", "8"}};
  for (const auto &args : invalid) {
    std::vector<std::string_view> full{"--index", "fixture", "--dry-run"};
    full.insert(full.end(), args.begin(), args.end());
    EXPECT_FALSE(ParseBenchmarkCommand(full).ok());
  }
  auto map = Parse({"--urma-device-map", "0=dev0,1=dev1"});
  ASSERT_TRUE(map.ok()) << map.status().ToString();
  EXPECT_EQ(map.value().config.urma_device_map.at(0), "dev0");
  for (const auto value : {"bad", "0=", "x=dev", "0=a,0=b"})
    EXPECT_FALSE(Parse({"--urma-device-map", value}).ok());
}
TEST(ConfigExtended, MemoryPlansRejectZeroAndOverflow) {
  std::array<std::uint32_t, 6> dims{2, 1, 8, 8, 4096, 4096};
  for (std::size_t i = 0; i < dims.size(); ++i) {
    auto bad = dims;
    bad[i] = 0;
    EXPECT_FALSE(
        BuildRpcMemoryPlan(bad[0], bad[1], bad[2], bad[3], bad[4], bad[5])
            .ok());
  }
  EXPECT_FALSE(BuildRpcMemoryPlan(UINT32_MAX, UINT32_MAX, UINT32_MAX,
                                  UINT32_MAX, UINT32_MAX, UINT32_MAX)
                   .ok());
  auto plan = BuildRpcMemoryPlan(2, 1, 8, 8, 4096, 4096);
  ASSERT_TRUE(plan.ok());
  EXPECT_EQ(plan.value().registered_rpc_bytes,
            plan.value().request_segment_bytes +
                plan.value().response_segment_bytes +
                plan.value().notify_segment_bytes);
}
TEST(ConfigExtended, CommunicationOwnersAndCpuLayouts) {
  EXPECT_FALSE(BuildCommunicationOwnerMap(0, 0, 1, 1).ok());
  EXPECT_FALSE(BuildCommunicationOwnerMap(2, 2, 1, 1).ok());
  EXPECT_FALSE(BuildCommunicationOwnerMap(2, 0, 0, 1).ok());
  for (std::uint32_t world : {1U, 2U, 8U}) {
    auto map = BuildCommunicationOwnerMap(world, 0, 2, 3);
    ASSERT_TRUE(map.ok());
    EXPECT_EQ(map.value().read_owner_by_rank[0], kNoCommunicationOwner);
    for (std::uint32_t rank = 1; rank < world; ++rank)
      EXPECT_LT(map.value().rpc_owner_by_rank[rank], 3U);
    if (world > 1U)
      EXPECT_FALSE(FormatCommunicationOwnerMap(map.value()).empty());
  }
  const std::array<std::uint32_t, 5> cpus{4, 1, 3, 0, 2};
  const CpuRoleCounts counts{1, 1, 1, 1, 1};
  auto layout = BuildContiguousCpuLayout(cpus, counts);
  ASSERT_TRUE(layout.ok());
  EXPECT_EQ(layout.value().search.size(), 1U);
  EXPECT_EQ(layout.value().reserved.size(), 1U);
  const std::array<std::uint32_t, 5> duplicates{0, 0, 1, 2, 3};
  EXPECT_FALSE(BuildContiguousCpuLayout(duplicates, counts).ok());
  EXPECT_FALSE(BuildContiguousCpuLayout({}, counts).ok());
  const std::array<CpuTopologyEntry, 5> topology{
      {{0, 0, 0}, {1, 1, 0}, {2, 2, 1}, {3, 3, 2}, {4, 4, 3}}};
  auto spread = BuildLlcSpreadCpuLayout(cpus, topology, counts);
  ASSERT_TRUE(spread.ok());
  auto isolated = BuildLlcIsolatedCpuLayout(cpus, topology, counts);
  ASSERT_TRUE(isolated.ok());
  EXPECT_FALSE(BuildLlcSpreadCpuLayout(cpus, {}, counts).ok());
}

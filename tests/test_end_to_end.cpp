#include "fixture.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include "hnsw_urma/search/local_io.hpp"
#include "hnsw_urma/search/search.hpp"
using namespace hnsw_urma;
namespace {
class EndToEnd : public ub_complete::IndexFixture {
 protected:
  void Check(std::uint32_t metric,std::uint32_t topk,const std::array<float,2>& query) {
    auto global=LoadGlobalIndexFile(root_);ASSERT_TRUE(global.ok());
    auto map=ShardMap::Create(global.value().ranges,global.value().meta.ntotal);ASSERT_TRUE(map.ok());
    auto upper=LoadUpperGraphFile(root_,global.value().meta);ASSERT_TRUE(upper.ok());
    std::vector<LocalShard> shards;shards.reserve(2);
    for(const auto& range:global.value().ranges) {
      auto shard=LoadShardFile(root_,global.value().meta,range);ASSERT_TRUE(shard.ok());
      shards.push_back(std::move(shard).value());
    }
    std::vector<const LocalShard*> views;for(const auto& shard:shards) views.push_back(&shard);
    auto graph=LocalGraphReader::Create(map.value(),upper.value(),views);ASSERT_TRUE(graph.ok());
    auto distance=LocalDistanceOracle::Create(map.value(),views,metric==0?Metric::kL2:Metric::kInnerProduct);
    ASSERT_TRUE(distance.ok());
    SearchParams params;params.topk=topk;params.ef_search=8;params.expand_batch=3;
    auto task=HnswSearchTask::Create({0,0,1},global.value().meta,map.value(),graph.value(),distance.value(),query,params);
    ASSERT_TRUE(task.ok())<<task.status().ToString();
    auto result=RunToCompletion(std::move(task).value());ASSERT_TRUE(result.ok())<<result.status().ToString();
    std::vector<std::pair<double,node_id_t>> expected;
    for(node_id_t id=0;id<8;++id) {
      double score=0;
      for(std::size_t dim=0;dim<2;++dim) {
        const double value=ub_complete::kVectors[id*2+dim];
        const double delta=double(query[dim])-value;
        score+=metric==0?delta*delta:-double(query[dim])*value;
      }
      expected.push_back({score,id});
    }
    std::sort(expected.begin(),expected.end());
    ASSERT_EQ(result.value().neighbors.size(),topk);
    for(std::size_t i=0;i<topk;++i) {
      EXPECT_EQ(result.value().neighbors[i].id,expected[i].second);
      ASSERT_TRUE(std::isfinite(result.value().neighbors[i].distance));
      EXPECT_NEAR(result.value().neighbors[i].distance,expected[i].first,1e-5);
    }
  }
};
}
TEST_F(EndToEnd, L2BuildReloadMatchesIndependentBruteForce) {Check(0,3,{1.1F,0.2F});}
TEST_F(EndToEnd, InnerProductBuildReloadMatchesIndependentBruteForce) {
  Build(1);Check(1,3,{1.0F,0.25F});
}
TEST_F(EndToEnd, RepeatedQueriesAndTopKBoundaries) {
  for(auto k:{1U,2U,3U,8U}) {SCOPED_TRACE(k);Check(0,k,{1.1F,0.2F});}
  Check(0,3,{2.8F,1.8F});Check(0,3,{1.1F,0.2F});
}
TEST_F(EndToEnd, CorruptShardRejectedThenRebuildRestoresSearch) {
  const auto file=ShardFilePath(root_,0);
  std::fstream output(file,std::ios::binary|std::ios::in|std::ios::out);
  const std::uint64_t invalid=0;output.write(reinterpret_cast<const char*>(&invalid),sizeof(invalid));output.close();
  EXPECT_FALSE(LoadIndexForShard(root_,0).ok());
  Build();Check(0,3,{1.1F,0.2F});
}
TEST_F(EndToEnd, LoadedIndexOwnsDataAfterFilesAreRemoved) {
  auto loaded=LoadIndexForShard(root_,0);ASSERT_TRUE(loaded.ok());
  std::filesystem::remove_all(root_);
  auto vector=loaded.value().local_shard.Vector(0);ASSERT_TRUE(vector.ok());
  EXPECT_EQ(vector.value().size(),2U);EXPECT_FLOAT_EQ(vector.value()[0],0.0F);
  EXPECT_TRUE(loaded.value().local_shard.Neighbors(0).ok());
}

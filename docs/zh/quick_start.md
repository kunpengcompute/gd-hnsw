# 快速入门

本文档提供gd_hnsw核心功能快速入门的简易指导。完整的依赖安装与编译方法请参见《[安装指南](./installation_guide.md)》，接口详细语义请参见《[API参考](./api_reference.md)》。

## gd_hnsw核心概念介绍

gd_hnsw的核心是`Context`对象，即封装分布式检索全流程的有状态句柄。使用它只需理解以下核心概念。

- **分片（shard）**：全局HNSW图被切成N份，每份是一个分片索引文件。
- **离线建库**：单进程调用`build`，包括读数据集，建HNSW，抽N分片，落盘。
- **在线检索**：N个容器各自`load`自己的分片，然后与`node_init_and_sync`同步并启动service，最后`search`全局检索。
- **pushdown**：本节点搜索时，远端分片的距离由对端节点的service线程就近计算，结果回传。任一节点可独立给出完整top-K。
- **核心流程**：`initialize`，`build`/`load`，`node_init_and_sync`，`search`，`finalize`。

## 接口介绍

全部接口在 `namespace gd_hnsw`，状态封装在`Context`对象（无全局变量）。

| 接口 | 作用 |
|------|------|
| Context::initialize | 初始化 |
| Context::build | 构建索引并落盘 | 
| Context::load | 加载本地分片文件 |
| Context::node_init_and_sync | 节点初始化及同步 |
| Context::search | 对本节点query做全局检索，返回完整top-K |
| Context::finalize | 释放资源 |
| Context::route_queries（辅助） | 按质心/均匀把query分到各节点 |

## 快速入门示例

### 离线构建索引

- 可参考当前算法benchmarks目录下[build_index.cpp](../../benchmarks/build_index.cpp)的实现。

   ```cpp
   #include "gd_hnsw_api.h"

   int main() {
       using namespace gd_hnsw;
       Context ctx;
       ctx.initialize();
       auto br = ctx.build({
           .dataset_path = "/data/gist_1M_960/base.bin",
           .output_path  = "/data/idx",
           .num_shards   = 4,
           .M            = 16,
           .ef_construction = 200,
           .ef_search    = 64,
           .build_threads = 0,
           .cluster_partition = false,
           .dataset_format = "auto",
       });
       if (br.status != Status::Ok) { /* br.error */ }
       ctx.finalize();
       return 0;
   }
   ```

- 也可继续用现成的命令行工具`./build_index`。

   ```bash
   ./build_index -d /path/to/data/gist_1M_960/base.bin -o /path/to/data/idx -M 64 -c 40 -e 355 -s 8 -t 32
   ```

- 构建产物。

- `/data/idx.shard_0 ... /data/idx.shard_<N-1>` —— 分片索引（含图 + 向量）。
- `/data/idx.idmap` —— （仅`cluster_partition`）新gid到原始gid的映射。
- `/data/idx.centroids` —— （仅`cluster_partition`）每分片质心，用于query路由。

   >![表示说明的图片](./public_sys-resources/icon-note.gif) **说明：**
   >部署前把上述文件放到**每个容器都能读到**的位置（共享存储 / 各节点本地副本）。

### 在线检索（N个容器）

1. 可参考当前算法benchmarks目录下[main.cpp](../../benchmarks/main.cpp)的实现。

   ```cpp
   #include "gd_hnsw_api.h"
   #include <vector>
   #include <cstdio>

   int main(int argc, char** argv) {
       using namespace gd_hnsw;

       Context ctx;
       if (ctx.initialize(&argc, &argv) != Status::Ok) {
           fprintf(stderr, "init failed\n"); return 1;
       }

       const uint32_t rank = ctx.rank();

       auto lr = ctx.load(rank, "/data/idx");
       if (lr.status != Status::Ok) { fprintf(stderr, "load failed: %s\n", lr.error.c_str()); return 1; }

       DeployOptions deploy{};
       deploy.search_threads  = 0;
       deploy.service_threads = 0;
       deploy.expand_batch    = 1;
       deploy.use_dot_norm    = false;
       if (ctx.node_init_and_sync(rank, deploy) != Status::Ok) { fprintf(stderr, "sync failed\n"); return 1; }

       const uint32_t dim = ctx.dim();
       std::vector<float> my_queries = /* ... my_nq * dim ... */;
       const uint64_t my_nq = my_queries.size() / dim;

       const int32_t K = 10;
       std::vector<int32_t> ids(my_nq * K, -1);
       std::vector<float>   dists(my_nq * K, 0.0f);
       SearchParams sp{ K };
       sp.ef_search = 64;
       auto sr = ctx.search(my_queries.data(), my_nq, sp, ids.data(), dists.data());
       if (sr.status != Status::Ok) { fprintf(stderr, "search failed: %s\n", sr.error.c_str()); }

       ctx.finalize();
       return 0;
   }
   ```

2. 参考当前算法benchmarks目录下[CMakeLists.txt](../../benchmarks/CMakeLists.txt), 编译得到可执行文件例如：./build/benchmarks/gd_hnsw_bench。

3. 进入当前算法conifgs目录下，配置对应数据集的相关配置参数。

4. 启动算法（例如8节点）。

   ```bash
   mpirun --allow-run-as-root -np 8 --hostfile hostfile --map-by ppr:1:numa --bind-to numa ./build/benchmarks/gd_hnsw_bench --config configs/gist-960-euclidean.config
   ```

   >![表示说明的图片](./public_sys-resources/icon-note.gif) **说明：**
   >`hostfile`中为服务器IP。形如`xxx.xxx.xxx.xxx`。

### 分配query

本章节介绍分配query的两种常见方法。

- **均匀切片**：`rank r`处理`[r*ceil(nq/N), (r+1)*ceil(nq/N))`的query。简单，负载可能不均。
- **质心路由**：`Context::route_queries(ctx, all_queries, nq)`返回本rank应处理的query下标列表（需建库时开启`cluster_partition`，会落盘centroids）。负载更均衡。

## 修订记录

| 文档版本 | 发布日期 | 修改说明 |
|------|----------|------------|
| 01 | 2026-09-30 | 第一次正式发布。 |

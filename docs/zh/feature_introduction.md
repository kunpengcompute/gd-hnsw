# 特性介绍

本文档介绍gd_hnsw的系统架构、核心算法原理，帮助三方开发者理解其工作机理并正确使用。

## 架构介绍

以下从分层结构、功能模块和部署模型三方面介绍当前架构。

### 分层结构

gd_hnsw系统总体分为**对外接口层、核心算法层、底层依赖层**三部分：核心以C++17实现，对外只暴露精简的`Context`句柄对象，内部依赖MPI、ubs-mem、Faiss、HDF5等。

- **对外接口层**：`gd_hnsw_api.h`中的`Context`类及其配套结构（`Status`、`InitOptions`、`BuildOptions`、`LoadResult`、`DeployOptions`、`SearchParams`、`SearchResult`）。用户只需在容器进程的入口函数`main`中按序调用`initialize`,`build`/`load`,`node_init_and_sync`,`search`,`finalize`即可完成分布式部署与检索。
- **核心算法层**
  - `gd_hnsw_core`（纯算法静态库）：包含布局（`gd_layout`）、搜索器（`GdHnswSearcher`）、索引I/O（`index_io`），不依赖MPI/ubs-mem/faiss/HDF5，便于复用与单测。
  - `gd_hnsw_faiss`（Faiss 抽取层）：负责基于Faiss构建HNSW并流式抽取分片。
- **底层依赖层**：ub、MPI（节点同步与集合通信）、OpenMP（线程级并行）、HDF5/.bin（数据集加载）、NEON距离核。

### 功能模块

**表 1** gd_hnsw 功能模块<a id="gd_hnsw功能模块"></a>

| 模块名称 | 功能描述 |
|----------|----------|
| Context（对外API） | 封装分布式检索全流程：初始化、建库、加载、同步、搜索、释放；状态全部在对象内，无全局变量 |
| GdHnswSearcher | 有状态搜索器，支持单分片本地搜索与多分片pushdown搜索、批量搜索 |
| SuperBlockV1（gd_layout） | 数据布局：固定4KB超级块头+向量/层级/偏移/邻居/累计邻居数5个数据块，含CRC64校验与状态机 |
| FaissExtractor | 调用 Faiss 构建HNSW图，流式抽N 片逐个落盘，可选K-Means聚类分片（生成idmap/centroids） |
| DualDistService（dist_service） | pushdown双通道service：task_inbox/result_inbox跨节点通道 |
| VisitedTable | 访问表：小图用密集数组（O(1) 直接下标），大图用epoch戳flat开放寻址哈希表（512KB固定，fit L2） |
| MinimaxHeap | 双堆结构：结果max-heap（bounded ef_search+候选min-heap（lazy失效），候选出堆O(log n)摊还 |
| distance_kernel / distance_kernel_dot_norm | NEON向量化距离核：L2平方距离（batch 4/16）与dot+norm内积距离 |
| index_io | 分片文件、idmap、centroids的读写与跨分片一致性校验 |
| hdf5_loader/bin_loader/ann_dataset | 数据集加载：支持HDF5（ann-benchmarks 格式）与.bin（fbin/ibin）格式 |

### 部署模型

gd_hnsw采用 **1:1:1**部署模型。

```text
N个容器=N个进程=N个分片
```

- 每个节点持有一个分片的HNSW图。
- 多分片部署时，本节点分片的向量加载到本进程堆，远端分片只保留图结构。
- 检索时本节点需要远端分片距离时通过pushdown通道下发到对应节点就近计算，最后将结果回传。

## 算法原理

### HNSW算法概述

HNSW（Hierarchical Navigable Small World，分层可导航小世界图）是一种基于多层图的近似最近邻搜索算法。

- **多层图结构**：上层稀疏、下层稠密。查询时从顶层入口点（entry point）出发，逐层贪心下探到第0层，在第0层做beam search（`ef_search`宽度的候选集维护）。
- **关键参数**
  - `M`：每个节点在第0层之上每层的最大邻居数。
  - `ef_construction`：建图时候选集宽度，越大图质量越高、构建越慢。
  - `ef_search`：查询时候选集宽度，越大召回越高、时延越高。
- **距离度量**：当前实现以L2平方距离为主（`METRIC_L2`），并提供可选的dot+norm内积距离核。

### pushdown机制

任一节点可独立发起`search`。当本节点在图遍历中需要计算某个**远端分片**向量的距离时。

1. 本节点搜索线程通过`DualDistProxy`把query向量的批量距离请求写入对端节点的计算任务通道。
2. 对端节点常驻的`DualDistService`工作线程poll到任务后，在本地就近计算距离，把结果写入本节点的计算结果通道。
3. 本节点搜索线程从计算结果通道读取结果，继续图遍历。

这一机制把"大块向量跨节点搬运"转化为"少量距离结果跨节点回传"，并让计算发生在数据所在节点，显著降低访存与通信开销。`expand_batch`（multi-pop扩展）允许每轮弹出多个候选、合并其邻居后批量下发，进一步摊薄通道往返成本。

## 修订记录

| 文档版本 | 发布日期 | 修改说明 |
|------|----------|------------|
| 01 | 2026-09-30 | 第一次正式发布。 |

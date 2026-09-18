# 版本说明书

## 版本配套说明

### 产品版本信息

| 项目 | 说明 |
|------|------|
| 产品名称 | Kunpeng BoostKit |
| 产品版本 | 26.2.RC1 |
| 软件名称 | gd_hnsw |
| 软件版本 | v1.0.0 |
| 底层依赖 | Faiss（HNSW构建）、UB相关组件与驱动、MPI、OpenMP、HDF5 |
| 目标架构 | 鲲鹏ARM |
| C++ 标准 | C++17 |
| 构建系统 | CMake≥3.18 |

### 与操作系统、编译器和处理器配套说明

**表 1**gd_hnsw已验证环境<a id="gd_hnsw"></a>

| 操作系统 | 处理器类型 | 内存 | 编译器 | CMake版本号 |
|----------|------------|------|--------|--------------|
| openEuler 24.03 LTS SP3 | 鲲鹏950处理器 | 24*64GB | GCC 12.3.1 | CMake>=3.18 |

>![表示说明的图片](./public_sys-resources/icon-note.gif) **说明：**
>gd_hnsw算法依赖NUMA，当前仅支持Linux+NUMA。非Linux或非NUMA环境编译会报错。

## 版本使用注意事项

### 使用注意事项

请参见《[最佳实践](./best_practices.md)》中的"[部署Checklist](./best_practices.md#部署checklist)"。

## v1.0.0

### 更新说明

**新增特性**

- 提供基于灵衢通信能力的分布式HNSW全局检索算法。
- 提供`initialize/build/load/node_init_and_sync/search/finalize/route_queries`七个对外接口，覆盖"离线建库、在线部署、全局检索、释放"全流程。
- 提供单分片快速路径与多分片pushdown 路径（远端距离就近下推）。
- 提供K-Means聚类分片（`cluster_partition`），生成`idmap`（新gid映射原始gid）与`centroids`，支持质心路由query。
- 提供NEON向量化L2距离核（单向量/batch-4）与可选dot+norm内积距离核。
- 提供命令行建库工`build_index`与最小示例`gd_hnsw_bench`。

## 版本配套文档

### v1.0.0版本配套文档

| 文档名称 | 内容简介 | 交付形式 |
|----------|----------|----------|
| 《版本说明书》 | 本文档提供gd_hnsw的版本发布信息。 | 开源仓 |
| 《快速入门》 | 本文档提供gd_hnsw的快速入门指导。 | 开源仓 |
| 《安装指南》 | 本文档提供gd_hnsw编译安装指导。 | 开源仓 |
| 《API参考》 | 本文档提供gd_hnsw对外接口定义、接口说明。 | 开源仓 |
| 《最佳实践》 | 本文档提供gd_hnsw使用的实践案例。 | 开源仓 |
| 《特性介绍》 | 本文档提供gd_hnsw架构介绍、核心算法原理及特性优化说明。 | 开源仓 |

### 获取文档的方法

您可以通过访问[gd_hnsw仓库](https://gitcode.com/boostkit/gd-hnsw)浏览和获取相关文档。

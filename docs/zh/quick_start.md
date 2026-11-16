# 快速入门

本文带你在 5 分钟内完成 gd-hnsw 的第一次索引构建与分布式检索。前置条件：已完成 [安装指南](installation_guide.md) 中的构建。

## 1. 准备数据集

gd-hnsw 支持两类数据集格式（自动识别）：

- **HDF5**：包含 base / query / groundtruth 的标准 ANN 数据集（如 bigann、gist、bge）；
- **BIN**：BIN / IVEC / FBIN 原始向量文件。

`configs/` 目录提供 4 个预置数据集配置：

| 配置文件 | 数据集 | 维度 |
| --- | --- | --- |
| `bigann-128.config` | BigANN | 128 |
| `gist-960-euclidean.config` | GIST | 960 |
| `bge-1024.config` | BGE | 1024 |
| `openai-1536.config` | OpenAI | 1536 |

以 bigann 为例，打开 `configs/bigann-128.config`，修改两处路径指向你的实际数据：

```ini
dataset = /dataset/bigann-100m.hdf5   # 数据集路径
index   = /data/idx-bigann            # 索引输出前缀
```

## 2. 离线构建索引

```bash
./build_index --config configs/bigann-128.config
```

构建为**单进程离线流程**（无需 mpirun）。配置中的构建参数：

```ini
build.M = 64                # HNSW 每节点邻居数
build.ef-construction = 40  # 构建期搜索宽度
build.shards = 8            # 分片数，必须与后续 mpirun -np 一致
build.threads = 0           # 0 = 使用全部核
build.cluster = false       # true = k-means 聚类分片（提升数据局部性）
```

构建完成后生成 `index.shard_0 .. index.shard_<N-1>` 共 N 个分片文件；启用聚类分片时还会生成 idmap / centroids 伴生文件。

## 3. 运行分布式检索基准

```bash
mpirun -np 8 ./gd_hnsw_bench --config configs/bigann-128.config
```

> **注意：`-np` 的进程数必须等于 `build.shards`**，每个进程加载一个分片（rank 号与分片号一一对应）。

基准输出：recall@k、聚合 QPS、每查询时延分位数。常用检索参数（命令行覆盖配置文件）：

```bash
mpirun -np 8 ./gd_hnsw_bench --config configs/bigann-128.config \
    --k 10 --ef-search 265 \
    --search-threads 64 --service-threads 30 \
    --expand-batch 16 --use-dot-norm \
    --rounds 3 --warmup 1
```

先跑 warmup 再计时，`rounds` 轮取统计值。调参建议见 [最佳实践](best_practices.md)。

## 4. 集成到自己的程序

检索库以进程为单位嵌入宿主程序，每个进程（容器）完成一次完整生命周期：

1. **初始化**：建立 MPI / URMA 运行环境（集合操作，所有进程共同参与）；
2. **加载分片**：每个进程加载自己负责的一个索引分片，进程号（rank）与分片号一一对应（本地操作）；
3. **节点同步**：发布本节点的 URMA 注册内存并发现对端内存，导入对端句柄、建立通信通道，启动运行时线程（集合操作）；
4. **检索**：提交查询向量与检索参数（k、ef_search），沿全局图完成跨分片搜索，取得 top-k 的 ID 与距离；
5. **释放**：回收通信资源与查询池，退出运行环境（集合操作）。

> URMA 版具体接口签名以 `include/` 头文件与 [API 参考](api_reference.md) 为准。多节点查询分发支持按最近聚类中心路由（启用聚类分片时）或均匀切分。

## 5. 下一步

- 理解全局图与距离下推的设计：[特性介绍](feature_introduction.md)
- 性能调优（ef_search / 线程配比 / 路由策略）：[最佳实践](best_practices.md)
- 接口细节与返回码：[API 参考](api_reference.md)

# API 参考

> **本文档待补全。** URMA 版接口代码尚未合入本仓库，本文暂不提供真实接口签名。代码合入后将按 `include/` 头文件补全。以下列出计划覆盖的接口范围与概念模型，便于提前了解集成方式。

## 1. 生命周期（概念模型）

检索库以进程为单位嵌入宿主程序，每个进程（容器）执行一次完整生命周期：

| 阶段 | 操作类型 | 说明 |
| --- | --- | --- |
| 初始化 | 集合操作 | 建立 MPI / URMA 运行环境，所有进程共同参与 |
| 加载分片 | 本地操作 | 每个进程加载一个索引分片，rank 与分片号一一对应 |
| 节点同步 | 集合操作 | 发布本节点 URMA 注册内存、发现并导入对端句柄，建立通信通道，启动运行时线程 |
| 检索 | 本地操作 | 提交查询向量与参数（k、ef_search），取得 top-k 的 ID 与距离 |
| 释放 | 集合操作 | 回收通信资源与查询池，退出运行环境 |

## 2. 部署与检索参数（计划覆盖）

- 运行时线程配置：Search / Distance / READ Progress / RPC Progress 数量（auto 或显式），Control 固定为 1；
- 查询参数：`k`、`ef_search`、`expand_batch`、距离策略（`pull` / `pushdown` / `auto`）；
- 资源控制：`max_active_queries`（查询池大小）；
- 查询路由：按最近聚类中心路由（启用聚类分片时）或均匀切分。

参数含义与调优方法见 [最佳实践](best_practices.md)。

## 3. Milvus 接入面（概念）

接入 Milvus QueryNode 时，算法经独立 CGO 桥直达运行时，不经过原生 segcore / knowhere 检索链：

- 每个分片以 5 个角色注册 URMA 内存：`topology` / `vectors` / `distance-tasks` / `distance-results` / `distance-notify`；
- 注册记录经 etcd（绑定 session 租约）发布与发现，对端导入 peer 句柄后启用分布式检索；
- `shardDelegator.Search` 内的 `ub.TrySearch` 为分流闸门，由 `queryNode.ub.enabled` / `queryNode.urma.globalIndex.enabled` 配置决定；
- 计划约束：仅裸向量检索（无标量过滤 / group-by / 迭代器），`0 < topk ≤ ef`；URMA 数据面不可用时安全回退原生检索路径。

## 4. 工具程序

- `build_index`：离线索引构建（单进程，无需 mpirun），用法见 [快速入门](quick_start.md)；
- `gd_hnsw_bench`：分布式检索基准，输出 recall@k、聚合 QPS 与时延分位数。

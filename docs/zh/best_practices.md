# 最佳实践

本文档提供使用gd_hnsw的部署Checklist、典型场景实践与调优建议。

## 部署Checklist

部署分布式检索前，请逐项确认。

1. **每个进程能读本进程分片文件**：索引三元组（`global.meta`/`upper_graph.index`/`shard_N.index`）在共享存储或各节点本地副本上可读。
2. **URMA（Unified Remote Memory Access，统一远端内存访问）环境**：所有节点已安装UB（Unified Bus，统一总线）相关组件与驱动（参见《[安装指南](./installation_guide.md)》），设备可枚举，传输模式与设备能力匹配（当前驱动要求`rm`）。
3. **进程数=分片数**：采用"一个进程一个分片"的对等部署，进程数须与索引分片数一致。
4. **资源目录可用**：各节点分片的五块注册内存（topology/vectors/distance-tasks/distance-results/distance-notify）已在资源目录发布，且对端完成链路激活与句柄导入（Milvus场景经etcd，参见《[快速入门](./quick_start.md)》"Milvus接入"）。
5. **基础一致性校验**：导入对端资源前用`UbValidateShardLayout`校验分片数量、编号范围与入口点归属。注意该校验不含构建版本/内容指纹比对，不能单独用于排除过期发布。

## 调优建议

推荐调优顺序：**先建索引达标召回 → 再调ef_search → 再调运行时参数提QPS（Queries Per Second，每秒查询数） → 最后验收时延**。

### HNSW参数

**表 1** HNSW参数调优说明

| 参数名称 | 参数说明 |
|------|------|
| M / M0 | 上层 / 第0层最大邻居数。越大图越稠密、召回越高、内存与建图时间增加 |
| ef_construction | 建图候选宽度，越大图质量越高、构建越慢 |
| ef_search | 查询候选宽度，越大召回越高、时延越高；须≥topk，可在线调整 |

**ef_search标定方法**：固定其他参数，在一组`ef_search`取值（如32/64/128/256/512）下分别测量召回率与QPS，画出召回-QPS曲线，选择召回达标（建议留有余量）且QPS最高的取值。

### 距离策略选择

- `pushdown`：远端距离就近计算，只回传ID与距离。高维向量、批量较大或跨节点带宽敏感时优先。
- `pull`：拉回向量本地计算。低维向量、组内ID零散时可避免RPC（Remote Procedure Call，远程过程调用）任务组织开销。
- `auto`（默认）：组内ID数量小于`pull_threshold`（默认4）时拉取、否则下推，大多数场景的稳妥选择。

辅助手段：开启组内ID排序（`layer0_sort_mode`）让同分片访问更集中；结果集合满时下推请求自动携带最差距离上界，远端只回传不劣于上界的结果，减少响应量。

### expand_batch与内存预算联动

`expand_batch`（第0层每轮展开节点数，默认1）调大可摊薄每轮通信往返，但启动计划容量按`maximum_batch = max(1, M, M0 × expand_batch)`放大：

- 调大`expand_batch`后，检查容量规划是否仍在队列与注册RPC内存预算（`UbSetMemoryBudget`，默认1 GiB / 8 GiB）内；仅当规划用量超过现有预算时才需提高相应预算或降低容量相关参数，超限配置会在启动校验时被拒绝。
- `read_jfs_depth`缺省解析为`max(64, maximum_batch, max_active_queries × expand_batch × 2)`；显式调大`max_active_queries`或`expand_batch`后注意其headroom。

### 并发与查询池

- `max_active_queries`（默认16）控制在途查询占用。调大可提升并发，但会放大槽位池、队列与URMA lane缓冲，调大前确认设备与进程内存。
- 大批量nq场景（maxActiveQueries × 单查询时延）受查询级超时（默认30 s）约束，不要仅凭超时判定运行时卡死，先按批量调整`query_timeout_ms`。

### 线程与CPU

- 五类线程默认`auto`：Distance=max(1,⌊C/6⌋)、READ=max(1,min(2,⌊C/24⌋))、RPC=max(1,min(4,⌊C/12⌋))，Search取扣除后剩余。
- 跨节点pushdown压力大、Distance队列积压时显式调大Distance线程；追求单节点QPS时调大Search线程，但须为Distance与通信线程留足核数。
- 生产环境建议保持`cpu_bind_mode=auto`（按NUMA（Non-Uniform Memory Access，非一致内存访问）拓扑绑核），并按机器拓扑选择`cpu_layout`；时延敏感用`progress_mode=busy`，功耗/共享环境可试`adaptive`。

### 基准测量方法

- 使用检索基准应用：先按`warmup`轮预热（建立通道、触碰内存），再执行`rounds`轮取统计值。
- 关注三类指标：召回率（recall@k）、吞吐（QPS）、时延分位（P50/P95/P99）。
- 开启`--profile`导出分阶段耗时（上层搜索/图等待/距离等待/堆提交等）与队列水线，用于定位瓶颈在搜索、通信还是距离服务。
- 变更单一变量对比（如只改`ef_search`或只改距离策略），避免多参数同时变化无法归因。

## Milvus接入实践

1. 按《[快速入门](./quick_start.md)》"Milvus接入"配置`queryNode.urma`与`queryNode.ub`。注意`seedDir`/`shardIds`的启动装载仅在`globalIndex.enabled=false`时生效；开启全局索引时该路径被跳过，数据随segment装载。
2. 伪分布式/单机验证时`shardIds`留空（装载全部分片）；真分布式按部署为每个QN（QueryNode，查询节点）分配其持有的分片（每QN一个分片）。
3. UB链路只支持裸向量检索（单FloatVector placeholder、无过滤/group-by/迭代器、`round_decimal=-1`、`search_params`不含未声明字段）且要求`0 < topk <= ef`；不满足的计划由UB解析返回错误，回退行为取决于全局索引开关与错误发生位置，见下一项。
4. 开启全局索引（`globalIndex.enabled=true`）时UB查询失败直接返回错误；关闭全局索引时，本地`RunSearchAsWorker`失败（包括计划或placeholder解析失败）可回退原生链路，转发等前置步骤报错仍直接返回。资源就绪后可经`UbRetryEnableForPushdown`升级为pushdown路径。

## 验收Checklist

1. 功能测试用例全部通过（单元/运行时/集成测试）。
2. 目标数据集上召回率大于0.95。
3. 查询时延小于50ms。
4. 同等算力下，全局索引相比32分片索引QPS提升100%（设计目标，按相同召回率口径对比）。

## 缩略语

| 缩略语 | 英文全称 | 中文全称 |
|--------|----------|----------|
| CPU | Central Processing Unit | 中央处理器 |
| HNSW | Hierarchical Navigable Small World | 分层可导航小世界 |
| NUMA | Non-Uniform Memory Access | 非一致内存访问 |
| QN | QueryNode | 查询节点 |
| QPS | Queries Per Second | 每秒查询数 |
| RPC | Remote Procedure Call | 远程过程调用 |
| UB | Unified Bus | 统一总线 |
| URMA | Unified Remote Memory Access | 统一远端内存访问 |

## 修订记录

| 文档版本 | 发布日期 | 修改说明 |
|------|----------|------------|
| 01 | 2026-12-30 | 第一次正式发布。 |

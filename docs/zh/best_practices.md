# 最佳实践

本文档提供使用gd_hnsw的部署Checklist、典型场景实践与调优建议。

## 部署Checklist

部署分布式检索前，请逐项确认。

**环境与网络**

1. **跨节点连通性实测**：用TCP（Transmission Control Protocol，传输控制协议）/HTTP（HyperText Transfer Protocol，超文本传输协议）实测跨节点互通，不要只看ping通；各节点时间同步（偏差过大会引起TLS（Transport Layer Security，传输层安全）握手类失败）、防火墙放通所需端口。
2. **磁盘与内存余量充足**：存放索引的数据盘建议在写入前至少有本节点索引三元组副本大小2倍的可用空间，并单独为系统盘、日志和临时文件预留空间。装载前完成`UbConfigure`，再调用`UbEstimateRuntimeMemory`取得运行时内存估算值；该值包含队列、RPC与READ缓冲、操作池及访问表，不包含索引图、向量和宿主全部开销。默认的队列1 GiB与RPC注册内存8 GiB是配置预算上限，不能相加后当作固定占用或整个进程的内存需求。建议按“索引驻留内存 + 运行时估算值 + 宿主及其他开销”规划，并额外预留至少25%的余量。例如三项分别为20 GiB、6 GiB和2 GiB时，本进程规划容量至少为35 GiB；同机多个进程须分别计入。该示例不是所有数据集的统一最低要求，最终应以部署配置下的峰值实测验收。
3. **URMA（Unified Remote Memory Access，统一远端内存访问）环境**：所有节点已安装UB（Unified Bus，统一总线）相关组件与驱动（参见《[安装指南](./installation_guide.md)》），设备可枚举，传输模式与设备能力匹配（当前驱动要求`rm`）。
4. **URMA数据面已编译启用**：算法库以`ENABLE_URMA=ON`构建；`OFF`构建的库不含跨节点数据面，启用与检索接口将返回`kInvalidArgument`。

**索引与进程**

5. **每个进程能读本进程分片文件**：索引三元组（`global.meta`/`upper_graph.index`/`shard_N.index`）在共享存储或各节点本地副本上可读。
6. **进程数=分片数**：采用"一个进程一个分片"的对等部署，进程数须与索引分片数一致。

**资源发布与启用**

7. **资源目录可用**：各节点分片的五块注册内存（topology/vectors/distance-tasks/distance-results/distance-notify）已在资源目录发布，且对端完成链路激活与句柄导入。
8. **基础一致性校验**：导入对端资源前用`UbValidateShardLayout`校验分片数量、编号范围与入口点归属。注意该校验不含构建版本/内容指纹比对，不能单独用于排除过期发布。
9. **配置变更须重启生效**：接入侧配置下发后重启相关组件，不要假设配置热加载。

## 调优建议

推荐调优顺序：先把索引构建到达标召回，然后调`ef_search`，再调运行时参数提升QPS（Queries Per Second，每秒查询数），最后验收时延。

### HNSW参数

**表 1** HNSW参数调优说明

| 参数名称 | 类型 | 取值范围 | 默认值 | 参数说明 |
|------|------|----------|--------|----------|
| M | uint32_t | 参考构建须大于0；Milvus构建取1至2048 | 16 | 上层最大邻居数。越大图越稠密、召回越高，内存与建图时间增加 |
| M0 | uint32_t | 参考构建须大于0；Milvus构建取2×M | 32 | 第0层最大邻居数；参考构建可单独设置，Milvus构建由M推导 |
| ef_construction | uint32_t | Milvus构建取1至2147483647 | 200（Milvus构建） | 建图候选宽度，越大图质量越高、构建越慢；Milvus参数键为`efConstruction`，本库参考构建接口未暴露该参数 |
| ef_search | uint32_t | 大于或等于topk | 64（参考构建写入的索引默认值） | 查询候选宽度，越大召回越高、时延越高；可在每次查询时传入 |

上表区分参考构建接口与Milvus构建的约束。Milvus查询请求中的`ef`对应本库`ef_search`；未传入`ef`时，当前接入链路采用`topk`，不会直接采用参考索引中的默认值64。

**ef_search标定方法**：固定其他参数，在一组`ef_search`取值（如32/64/128/256/512）下分别测量召回率与QPS，画出召回-QPS曲线，选择召回达标（建议留有余量）且QPS最高的取值。

### 并发与查询池

- `max_active_queries`控制在途查询占用，建议初始设为16。通过C接口配置时必须显式填写大于0的值，填0不会采用默认16。调大可提升并发，但会放大槽位池、队列与URMA lane缓冲，调大前确认设备与进程内存。
- 大批量场景（nq较大）下，整批查询共用同一个截止时间，排队时间也计入预算；端到端耗时更接近`ceil(nq / 有效并发数) × 单查询时延`（忽略其他开销）。不要仅凭超时判定运行时卡死，先按批量调整`query_timeout_ms`。

### 线程与CPU

- 五类线程默认`auto`（C为进程可用CPU核数）：Control固定1个；Distance=max(1,⌊C/6⌋)；READ=max(1,min(2,⌊C/24⌋))；RPC（Remote Procedure Call，远程过程调用）=max(1,min(4,⌊C/12⌋))；Search取扣除Distance、READ、RPC与系统预留（max(1,⌊C/12⌋)）后的剩余核数。
- 跨节点pushdown压力大、Distance队列积压时显式调大Distance线程；追求单节点QPS时调大Search线程，但须为Distance与通信线程留足核数。
- 生产环境建议保持`cpu_bind_mode=auto`（按NUMA（Non-Uniform Memory Access，非一致内存访问）拓扑绑核），并按机器拓扑选择`cpu_layout`；时延敏感用`progress_mode=busy`，功耗/共享环境可试`adaptive`。

### 基准测量方法

- 使用VectorDBBench或宿主侧压测工具：先预热（建立通道、触碰内存）再取统计值。
- 关注三类指标：召回率（recall@k）、吞吐（QPS）、时延分位（P50/P95/P99）。
- 变更单一变量对比（如只改`ef_search`或某一运行时选项），避免多参数同时变化无法归因。

## 接入实践（以Milvus为例）

以下以Milvus为例给出接入侧的实践建议；其他宿主可参照执行。

1. 按Milvus侧资料配置接入参数（配置键与默认值以Milvus侧为准）；接入侧配置下发后须重启相关组件生效。
2. 伪分布式/单机验证时可在单进程装载全部分片；真分布式按部署为每个查询节点分配其持有的分片（每个进程一个分片）。
3. UB链路只支持裸向量检索（单FloatVector查询、无过滤/group-by/迭代器、`search_params`不含未声明字段）且要求`0 < topk <= ef_search`；不满足的检索请求由接入链路返回错误。
4. 启用接口幂等可重试：对端资源未就绪导致启用失败时，待资源就绪后重新发起启用即可。

## 验收Checklist

功能测试用例全部通过（单元/运行时/集成测试）。回显示例（以实际执行为准）：

```output
[==========] 120 tests from 18 test suites ran. (5321 ms total)
[  PASSED  ] 120 tests.
```

## 缩略语

| 缩略语 | 英文全称 | 中文全称 |
|--------|----------|----------|
| CPU | Central Processing Unit | 中央处理器 |
| HNSW | Hierarchical Navigable Small World | 分层可导航小世界 |
| HTTP | HyperText Transfer Protocol | 超文本传输协议 |
| NUMA | Non-Uniform Memory Access | 非一致内存访问 |
| QPS | Queries Per Second | 每秒查询数 |
| RPC | Remote Procedure Call | 远程过程调用 |
| TCP | Transmission Control Protocol | 传输控制协议 |
| TLS | Transport Layer Security | 传输层安全 |
| UB | Unified Bus | 统一总线 |
| URMA | Unified Remote Memory Access | 统一远端内存访问 |

## 修订记录

| 文档版本 | 发布日期 | 修改说明 |
|------|----------|------------|
| 01 | 2026-12-30 | 第一次正式发布。 |

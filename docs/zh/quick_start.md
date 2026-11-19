# 快速入门

本文档提供gd_hnsw核心功能快速入门的简易指导。完整的依赖安装与编译方法请参见《[安装指南](./installation_guide.md)》，接口详细语义请参见《[API参考](./api_reference.md)》。

## gd_hnsw核心概念介绍

- **分片（shard）**：覆盖全部数据的一张HNSW（Hierarchical Navigable Small World，分层可导航小世界）全局图被切成N份存储，每份是一个分片；**一个进程负责一个分片**。
- **索引三元组**：离线构建产物为一个目录下的三类文件——`global.meta`（全局元信息与分片范围）、`upper_graph.index`（上层图完整邻接关系，每进程一份副本）、`shard_N.index`（分片N的第0层邻接表与向量）。
- **离线建库**：基于Faiss在全部数据上构图，再按分片落盘三元组（构建组件由宿主侧提供）。
- **在线检索**：各进程装载自己的分片，发布本地内存（图/向量/邮箱）、导入对端句柄后启动五类线程运行时；任一进程可接纳查询，沿全局图跨分片搜索，直接产出全局top-K。
- **距离下推（pushdown）**：远端分片向量的距离可下推到数据所在进程就近计算，只回传ID与距离。

## 跑通路径总览

以下以集成gd_hnsw的Milvus集群为示例，给出端到端跑通路径；使用其他宿主时，Milvus部署环节替换为对应宿主的集成部署即可。

```text
环境准备（URMA 组网、依赖安装）
  ↓ 验证：各节点 URMA 设备可枚举，节点间网络互通
编译 / 获取算法库（libub_hnsw.so）
  ↓ 验证：编译产物完整（见《安装指南》）
部署集成 gd_hnsw 的 Milvus 集群（进程数 = 分片数）
  ↓ 验证：各组件 Running
离线构建全局索引三元组
  ↓ 验证：三元组文件齐全（global.meta / upper_graph.index / shard_N.index）
装载并启用全局检索
  ↓ 验证：装载状态就绪、查询正常返回
查询验证
  ↓ 验证：返回全局 top-K
```

>![表示说明的图片](./public_sys-resources/icon-note.gif) **说明：**
>任一环节的验证点不通过时，请先排查再继续，不要带障进入下一步。

## 环境准备

1. 按《[安装指南](./installation_guide.md)》完成依赖安装与算法库编译（跨节点数据面需`ENABLE_URMA=ON`）。
2. 确认各节点URMA（Unified Remote Memory Access，统一远端内存访问）设备可用（`ubctl`等工具枚举）、传输模式与设备能力匹配（当前驱动要求`rm`可靠消息模式）。
3. 确认节点间网络互通、时间同步、防火墙放通所需端口；跨节点连通性建议以TCP（Transmission Control Protocol，传输控制协议）/HTTP（HyperText Transfer Protocol，超文本传输协议）实测为准，不要只看ping通。

## 离线构建索引三元组

索引三元组是检索的输入：`global.meta`（全局元信息与各分片ID范围）、`upper_graph.index`（全局上层图完整邻接关系）、`shard_N.index`（分片N的第0层邻接表与原始向量，每分片一个）。

生产环境大规模索引的Faiss适配构建由宿主侧构建组件提供，不属于本库接口面。本库提供确定性参考构建接口`BuildReferenceIndex`（`index/builder.hpp`），适用于小规模数据（默认上限100,000行，可调）的验证与测试：

```cpp
#include "hnsw_urma/index/builder.hpp"

int main() {
    hnsw_urma::ReferenceBuildOptions opts{};
    opts.num_shards = 8;    // 分片数（须与后续运行进程数一致）
    opts.M            = 16; // 上层每层最大邻居数
    opts.M0           = 32; // 第0层最大邻居数
    opts.ef_search    = 64; // 写入索引的默认ef_search
    opts.metric       = 0;  // 0=L2, 1=负内积

    // vectors：rows × dim 连续排布的float向量
    auto st = hnsw_urma::BuildReferenceIndex("/data/idx", vectors, rows, dim, opts);
    if (!st.ok()) { /* st.ToString() */ }
    return 0;
}
```

构建产物（写入`/data/idx`）：

- `global.meta` —— 全局元信息与各分片ID范围。
- `upper_graph.index` —— 全局上层图（第1层及以上）完整邻接关系。
- `shard_0.index ... shard_<N-1>.index` —— 各分片的第0层邻接表与原始向量。

>![表示说明的图片](./public_sys-resources/icon-note.gif) **说明：**
>部署前把上述文件放到**每个进程都能读到**的位置（共享存储 / 各节点本地副本）。

## 经C ABI集成（生命周期顺序）

宿主进程按以下顺序调用`ub_c_api.h`（接口详细语义见《[API参考](./api_reference.md)》）：

1. **注入上下文**：`UbSetUrmaContext`注入宿主持有的URMA上下文（服务模式必需，本库不自行初始化URMA；启用后不可再设置）。
2. **配置**：`UbConfigure`一次性暂存线程/内存预算/超时等配置。须提供有效的非空设备名`urma_device_name`、非空CPU列表`allowed_cpus`及其长度，并将`max_active_queries`设为≥1（如16）。其他字段取0/NULL一般保留内置默认；`max_active_queries=0`会被暂存，但后续解析运行时配置时返回`kOutOfRange`，CPU列表为空则无法创建邮箱。
3. **装载**：`UbSeedFromDir`（全量或分片子集）或`UbLoadShardFromDir`（单个分片，内置一致性校验），随后`UbSetLocalShard`钉住本分片；可用`UbGetShardStatus`读取装载状态快照。
4. **绑定图身份**：`UbBindGlobalGraphIdentity`把全局图绑定到其产物身份（graph_id + epoch，相同身份重复调用幂等）。
5. **邮箱**：`UbEnsureRpcMailbox`构建RPC（Remote Procedure Call，远程过程调用）距离邮箱，并取回三段注册内存地址用于发布（requests/responses对应角色2/3，notify面向对端角色4）。
6. **发布**：`UbGetGraphPublishSnapshot`取回本分片图/向量注册内存与全局入口点信息，经资源目录发布。
7. **导入**：对每个对端分片与角色（0~4）调用`UbImportPeerHandles`；导入前可用`UbValidateShardLayout`校验分片数量、编号范围与入口点归属。该接口不比对构建版本或内容指纹，不能单独保证目录记录与本地索引来自同一次构建。
8. **启用**：`UbEnableDistributedSearch`启动五类线程运行时。启动前提（任一不满足即拒绝）：恰好一个本地分片、本地RPC邮箱已构建、每个远端分片全部五种角色的对端句柄均已导入、已注入URMA上下文并配置有效的非空设备名。启用失败的原因可写入`msg`，资源就绪后可重试。`UbRpcDistanceReady`仅表示运行时对象是否存在，不能代替健康检查。
9. **检索**：`UbSearch`/`UbSearchWithTimeout`同步返回全局top-K（不足填充-1/+inf）。每次调用要求`1 ≤ nq ≤ 4096`，更多向量须拆分批次；查询向量不得含NaN或无穷值。
10. **释放**：`UbStopDistributedSearch`停止运行时，`UbShutdown`释放全部状态。

## 接入示例：Milvus

本节以Milvus为例介绍宿主接入形态；其他宿主可参照上文"经C ABI集成"的生命周期自行接入。

gd_hnsw运行时随Milvus集成分发，经C ABI（Application Binary Interface，应用二进制接口）接入Milvus QueryNode。Milvus侧提供`queryNode.urma`/`queryNode.ub`配置项控制启用与设备、线程等参数，配置键与默认值以Milvus侧资料为准。

接入链路当前仅支持**裸向量检索**：单个FloatVector查询、无标量过滤/group-by/迭代器、`search_params`仅接受声明的字段（未知字段拒绝），并要求`0 < topk <= ef`。

## 部署验证清单

部署完成后逐项确认：

1. 集群各组件Running，查询节点进程数与索引分片数一致（一个进程一个分片）。
2. 各分片的五块注册内存（topology/vectors/distance-tasks/distance-results/distance-notify）已在资源目录发布，对端完成链路激活与句柄导入。
3. 集合装载完成，检索请求返回全局top-K结果。
4. 时延、吞吐等验收口径见《[最佳实践](./best_practices.md)》"验收Checklist"。

## 缩略语

| 缩略语 | 英文全称 | 中文全称 |
|--------|----------|----------|
| ABI | Application Binary Interface | 应用二进制接口 |
| HNSW | Hierarchical Navigable Small World | 分层可导航小世界 |
| HTTP | HyperText Transfer Protocol | 超文本传输协议 |
| IP | Inner Product | 内积 |
| L2 | L2 Norm | 欧氏距离 |
| RPC | Remote Procedure Call | 远程过程调用 |
| TCP | Transmission Control Protocol | 传输控制协议 |
| URMA | Unified Remote Memory Access | 统一远端内存访问 |

## 修订记录

| 文档版本 | 发布日期 | 修改说明 |
|------|----------|------------|
| 01 | 2026-12-30 | 第一次正式发布。 |

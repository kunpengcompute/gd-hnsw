# 快速入门

本文档提供gd_hnsw核心功能快速入门的简易指导。完整的依赖安装与编译方法请参见《[安装指南](./installation_guide.md)》，接口详细语义请参见《[API参考](./api_reference.md)》。

## gd_hnsw核心概念介绍

- **分片（shard）**：覆盖全部数据的一张HNSW全局图被切成N份存储，每份是一个分片；**一个进程负责一个分片**。
- **索引三元组**：离线构建产物为一个目录下的三类文件——`global.meta`（全局元信息与分片范围）、`upper_graph.index`（上层图完整邻接关系，每进程一份副本）、`shard_N.index`（分片N的第0层邻接表与向量）。
- **离线建库**：基于Faiss在全部数据上构图，再按分片落盘三元组。
- **在线检索**：各进程装载自己的分片，发布本地内存（图/向量/邮箱）、导入对端句柄后启动五类线程运行时；任一进程可接纳查询，沿全局图跨分片搜索，直接产出全局top-K。
- **距离下推（pushdown）**：远端分片向量的距离可下推到数据所在进程就近计算，只回传ID与距离；也可选择拉取（pull）向量本地计算，或由auto策略按组自动选择。

## 数据集准备

数据集采用bin格式：文件头为两个uint32字段（行数、维度），后接行主序的float32数据（`LoadFloatBin`）。基准运行还需要查询集与（可选的）ground truth，用于统计召回率。

## 快速入门示例

### 离线构建索引

使用Faiss适配构建（C++接口示例）。注意：Faiss构建接口位于**全局命名空间**（构建插件头文件`faiss_builder.hpp`），不要加`hnsw_urma::`前缀。

```cpp
#include "faiss_builder.hpp"  // Faiss构建插件接口（全局命名空间）

int main() {
    hnsw_urma::ReferenceBuildOptions opts{};
    opts.num_shards = 8;    // 分片数（须与后续运行进程数一致）
    opts.M            = 16; // 上层每层最大邻居数
    opts.M0           = 32; // 第0层最大邻居数
    opts.ef_search    = 64; // 写入索引的默认ef_search
    opts.metric       = 0;  // 0=L2, 1=负内积

    // vectors：rows × dim 连续排布的float向量
    auto st = BuildFaissIndex("/data/idx", vectors, rows, dim, opts,
                              /*ef_construction=*/200);
    if (!st.ok()) { /* st.ToString() */ }
    return 0;
}
```

数据量较大时建议使用流式构建器`FaissStreamBegin`/`FaissStreamAdd`/`FaissStreamFinish`分块喂入，避免一次性物化全部向量。

构建产物（写入`/data/idx`）：

- `global.meta` —— 全局元信息与各分片ID范围。
- `upper_graph.index` —— 全局上层图（第1层及以上）完整邻接关系。
- `shard_0.index ... shard_<N-1>.index` —— 各分片的第0层邻接表与原始向量。

>![表示说明的图片](./public_sys-resources/icon-note.gif) **说明：**
>部署前把上述文件放到**每个进程都能读到**的位置（共享存储 / 各节点本地副本）。

### 运行检索基准

以8个分片为例，使用MPI启动8个进程（进程数=分片数）。可执行文件名以实际交付为准。

```bash
mpirun -np 8 ./benchmark \
    --index /data/idx \
    --queries /data/queries.bin \
    --ground-truth /data/truth.bin \
    --topk 10 --ef-search 64 \
    --distance-policy auto --expand-batch 1 \
    --rounds 5 --warmup 2
```

基准按`warmup`轮预热后执行`rounds`轮，输出召回率（recall@k）、QPS与时延分位（P50/P95/P99），并可用`--profile`导出分阶段耗时与队列水线画像（`--profile-output`指定路径，`--output-format`可选text/json/csv）。

**表 1** 基准常用参数（完整参数以`--help`输出为准）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --index | 必填 | 索引三元组目录 |
| --queries | 必填（--dry-run时除外） | 查询集 |
| --ground-truth | 可选 | ground truth（提供时统计召回率） |
| --query-offset / --query-count | 0 / 全部 | 只取查询集的子区间 |
| --topk | 10 | top-K结果数 |
| --ef-search | auto | 查询候选集宽度，缺省用索引内默认值 |
| --distance-policy | auto | 距离策略：auto / pull / pushdown |
| --pull-threshold | 4 | auto策略的组内ID数量阈值 |
| --expand-batch | 1 | 第0层每轮展开节点数，取值[1,64] |
| --use-dot-norm | false | 启用dot+norm平方L2距离优化，仅对L2度量生效 |
| --layer0-sort | all | 第0层组内ID排序范围：all / local-only / upper-only |
| --max-active-queries-per-rank | 16 | 查询池大小（每进程在途查询上限） |
| --search-workers / --distance-workers | 0=auto | Search / Distance线程数 |
| --read-progress-threads / --rpc-progress-threads | 0=auto | READ / RPC Progress线程数 |
| --cpu-bind | auto | CPU绑定：auto / none |
| --cpu-layout | auto | CPU布局：auto / contiguous / llc-isolated / llc-spread |
| --read-progress-mode / --rpc-progress-mode | busy | 进度线程模式：busy / adaptive |
| --query-timeout-ms / --operation-timeout-ms | 30000 / 5000 | 查询级 / 操作级超时（毫秒），前者不得小于后者 |
| --rounds / --warmup | 5 / 2 | 正式轮数 / 预热轮数 |
| --dry-run | 关闭 | 只做配置解析，不实际运行基准（此时--queries可省略） |

### 经C ABI集成（生命周期顺序）

宿主进程（如Milvus QueryNode）按以下顺序调用`ub_c_api.h`：

1. **配置**：`UbSetConfig`（线程/CPU）、`UbSetUrmaEndpoint`（URMA设备）、可选`UbSetRuntimeOptions` / `UbSetMemoryBudget` / `UbSetTimeouts` / `UbSetExpandBatch`。
2. **装载**：`UbSeedFromDir`（装载全部分片，单机验证）或`UbSeedFromDirWithShards`（只装载本进程分片，真分布式），随后`UbSetLocalShard`钉住本分片。
3. **邮箱**：`UbEnsureRpcMailbox`构建RPC邮箱，经`UbGetRpcMailboxRegions`取得三段地址用于发布。
4. **发布**：`UbGetShardRegions`枚举本分片图/向量注册内存，连同邮箱一起注册进UMDK并经资源目录发布；`UbGetEntryPoint`/`UbGetEntryShard`提供发布身份。
5. **导入**：对每个对端分片与角色（0~4）调用`UbImportPeerHandles`；导入前可用`UbValidateShardLayout`校验分片数量、编号范围与入口点归属。该接口不比对构建版本或内容指纹，不能单独保证目录记录与本地索引来自同一次构建。
6. **启用**：`UbEnableDistributedSearch`启动五类线程运行时；`UbRpcDistanceReady`报告距离数据面就绪，资源不全时可用`UbRetryEnableForPushdown`重试。
7. **检索**：`UbSearch`同步返回全局top-K（不足填充-1/+inf）。
8. **释放**：`UbStopDistributedSearch`停止运行时，`UbShutdown`释放全部状态。

### Milvus接入

gd_hnsw运行时随Milvus QueryNode集成（vendored），经CGO桥（`ubcgo`）调用C ABI。相关配置（`configs/milvus.yaml`）：

**表 2** queryNode.urma配置项

| 配置 | 默认值 | 说明 |
|------|--------|------|
| enabled | true | QueryNode启动时初始化UMDK URMA；初始化失败时，默认因failFast=true而启动失败；设为failFast=false时继续使用既有存储路径 |
| device | "" | URMA设备名，空则取枚举到的第一个 |
| eidIdx / uasid | 0 / 0 | urma_create_context的EID索引 / 地址空间ID（0=自动分配） |
| failFast | true | URMA引导或端点解析失败时使QueryNode启动失败 |
| transportMode | rm | 本地传输模式；当前驱动要求rm（rc会被urma_create_jfs拒绝） |
| jfcDepth / jfsDepth / jfrDepth | 1024 | 完成/发送/接收队列深度，不得超过设备能力 |
| globalIndex.enabled | true | 发布全局索引数据面能力 |
| globalIndex.builder | faiss | 全局索引构建器 |

**表 3** queryNode.ub配置项

| 配置 | 默认值 | 说明 |
|------|--------|------|
| enabled | true | 启用UB（hnsw_urma）分布式检索运行时 |
| seedDir | 路径 | 本地索引三元组目录，QN启动时装载；仅`globalIndex.enabled=false`时生效，开启全局索引时该路径被跳过、数据随segment装载 |
| shardIds | "" | 本QN持有的分片号（逗号分隔，如"0,1,2,3"）；空=装载全部分片（伪分布式）；同样仅在`globalIndex.enabled=false`的启动装载路径生效 |
| threads.search / distance / readProgress / rpcProgress | 0 | 各类线程数，0=auto（内置公式） |
| cpuBindMode | auto | CPU绑定：auto / none |
| cpuLayout | contiguous | CPU布局：contiguous / llc-isolated / llc-spread |
| progressMode | busy | 进度线程模式：busy / adaptive |
| allowedCpusSource | cgroup | CPU来源：cgroup / yaml / numa |
| allowedCpus | "[]" | 显式CPU列表（仅allowedCpusSource=yaml时生效） |
| numaNode | 0 | NUMA节点（仅allowedCpusSource=numa时生效） |
| runtimeOptions | `--distance-policy pushdown --use-dot-norm --layer0-sort local-only --request-ring-depth 32` | 空格分隔的运行时选项串，覆盖单项配置；默认值意味着Milvus接入默认启用pushdown、dot+norm、local-only排序、环深度32，与底层RuntimeConfig默认值（auto/false/all/8）不同 |

**检索分流**：`shardDelegator.Search`中的`ub.TrySearch`按开关分流（`queryNode.ub.enabled`或全局索引开关任一开启即进入UB链路），不预判计划形态。UB链路当前只支持**裸向量检索**：单个FloatVector placeholder、无标量过滤/group-by/迭代器、`round_decimal=-1`、`search_params`仅接受声明的字段（未知字段拒绝），并要求`0 < topk <= ef`。不支持的计划由UB解析返回错误；开启全局索引（`queryNode.urma.globalIndex.enabled=true`）时，UB查询失败直接返回错误；关闭全局索引时，本地`RunSearchAsWorker`失败（包括计划或placeholder解析失败）可回退原生segcore链路，转发等前置步骤报错仍直接返回。另外，接入层会把COSINE查询向量归一化后按IP处理（与构建侧对基向量的归一化一致）。

## 修订记录

| 文档版本 | 发布日期 | 修改说明 |
|------|----------|------------|
| 01 | 2026-11-17 | 第一次正式发布。 |

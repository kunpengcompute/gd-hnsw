# API参考

本文档提供gd_hnsw对外C ABI与C++接口定义、参数说明与返回值。

## 概述

gd_hnsw以C++20实现，对外提供两层接口。

- **C ABI（Application Binary Interface，应用二进制接口）**：头文件`ub_c_api.h`，纯C接口（无C++构造、签名不含STL（Standard Template Library，标准模板库）），位于全局命名空间（extern "C"），面向Milvus等宿主的CGO集成，覆盖"配置→装载→资源发布/导入→启用→检索→释放"全生命周期。
- **C++ API（Application Programming Interface，应用程序编程接口）**：位于`namespace hnsw_urma`，含索引离线构建（`index/builder.hpp`）与索引文件读写（`index/format.hpp`），用于构建工具与自定义集成。

错误处理约定：C ABI返回`0`表示成功，非`0`为`hnsw_urma::StatusCode`（个别接口除外，以各接口返回值说明为准）；C++ API通过`Status`/`Result<T>`报告常规错误，但不保证所有异常均被转换为状态码，例如内存分配异常仍可能抛出。

## 状态码

```cpp
enum class StatusCode {
  kOk = 0,
  kInvalidArgument,
  kInvalidFormat,
  kOutOfRange,
  kResourceExhausted,
  kProtocolError,
  kTimeout,
  kCancelled,
  kTransportError,
  kInternalError,
};
```

`Status`携带错误消息与键值对上下文（`AddContext`），`ToString()`输出完整描述。

## C ABI（ub_c_api.h）

### 配置接口

以下接口在首次装载分片前调用（"staged"语义），启用分布式检索（`UbEnableDistributedSearch`）前生效。

#### UbSetConfig

**接口定义**

```cpp
void UbSetConfig(uint32_t search_workers,
                 uint32_t distance_workers,
                 uint32_t read_progress_workers,
                 uint32_t rpc_progress_workers,
                 const char* cpu_bind_mode,
                 const char* cpu_layout,
                 const char* progress_mode,
                 const uint32_t* allowed_cpus,
                 uint32_t allowed_cpu_count,
                 uint32_t max_active_queries,
                 uint32_t read_jfs_depth);
```

**接口用途**

设置运行时线程与CPU（Central Processing Unit，中央处理器）配置。线程数传`0`表示`auto`，按内置公式解析（见《[特性介绍](./feature_introduction.md)》"线程模型"）。

**参数说明**

| 参数名称 | 参数类型 | 参数说明 | 取值范围 |
|----------|----------|------|----------|
| search_workers | uint32_t | Search线程数 | 0=auto |
| distance_workers | uint32_t | Distance线程数 | 0=auto |
| read_progress_workers | uint32_t | READ Progress线程数 | 0=auto |
| rpc_progress_workers | uint32_t | RPC（Remote Procedure Call，远程过程调用） Progress线程数 | 0=auto |
| cpu_bind_mode | const char* | CPU绑定模式 | "auto"（按NUMA（Non-Uniform Memory Access，非一致内存访问）拓扑绑核）/ "none" |
| cpu_layout | const char* | CPU布局 | "contiguous" / "llc-isolated" / "llc-spread" |
| progress_mode | const char* | 进度线程模式 | "busy"（自旋）/ "adaptive"（休眠） |
| allowed_cpus | const uint32_t* | 显式CPU列表 | allowed_cpu_count==0表示不设置，由C++侧自行解析 |
| allowed_cpu_count | uint32_t | CPU列表长度 | ≥0 |
| max_active_queries | uint32_t | 查询池大小（在途查询上限） | 默认16；调大前确认设备内存 |
| read_jfs_depth | uint32_t | URMA（Unified Remote Memory Access，统一远端内存访问） READ JFS（Jetty Frame Sender，Jetty帧发送端）（WR（Work Request，工作请求）槽位池）深度 | 0=auto（解析为max(64, maximum_batch, max_active_queries × expand_batch × 2)） |

#### UbSetMemoryBudget

**接口定义**

```cpp
void UbSetMemoryBudget(uint64_t queue_memory_budget_bytes,
                       uint64_t registered_rpc_memory_budget_bytes);
```

**接口用途**

设置启动期内存上限：队列计划与注册RPC计划。传`0`保留内置默认值（队列1 GiB、注册RPC 8 GiB）。两项计划随`maximum_batch = max(1, M, M0 × expand_batch)`放大，调大`expand_batch`超出默认值可容纳的范围时需同步调大本预算。须在`UbEnableDistributedSearch`前调用，与`UbSetConfig`相互独立。

#### UbSetTimeouts

**接口定义**

```cpp
void UbSetTimeouts(uint32_t query_timeout_ms, uint32_t operation_timeout_ms);
```

**接口用途**

设置查询级墙钟超时与单次尝试的阶段超时（毫秒）。传`0`保留内置默认值（查询30 s、操作5 s）。调度器截止时间与此预算共用，大批量nq（maxActiveQueries × 单查询时延）由查询超时约束。

#### UbSetExpandBatch

**接口定义**

```cpp
void UbSetExpandBatch(uint32_t expand_batch);
```

**接口用途**

设置HNSW（Hierarchical Navigable Small World，分层可导航小世界）第0层束宽（每轮展开的节点数）。传`0`保留编译期默认值（1）。该值同时作用于搜索任务与启动计划容量（`maximum_batch = max(1, M, M0 × expand_batch)`），避免运行期触发"batch exceeds prepared slot space"限制。

#### UbSetUrmaEndpoint / UbSetRuntimeOptions

**接口定义**

```cpp
void UbSetUrmaEndpoint(const char* device_name, uint32_t eid_index);
void UbSetRuntimeOptions(const char* options);
```

**接口用途**

- `UbSetUrmaEndpoint`：把URMA设备名传给运行时单例，`UbEnableDistributedSearch`据此构建UMDK（Unified Memory Development Kit，统一内存开发工具包）后端。`device_name`为NUL结尾字符串，传`nullptr`清除。
- `UbSetRuntimeOptions`：以空格分隔的选项串覆盖单项运行时配置（与基准CLI（Command Line Interface，命令行界面）同名的选项，例如`--distance-policy pushdown --use-dot-norm`）。

### 装载接口

#### UbSeedFromDir / UbSeedFromDirWithShards

**接口定义**

```cpp
int UbSeedFromDir(const char* triple_dir);
int UbSeedFromDirWithShards(const char* triple_dir,
                            const uint32_t* shard_ids,
                            uint32_t count);
```

**接口用途**

从本地"三元组"目录（`global.meta` + `upper_graph.index` + `shard_N.index`）装载索引到进程单例。

- `UbSeedFromDir`：按`global.meta`声明的全部分片范围逐个装载（单进程持有全部分片，用于伪分布式/单机验证）。幂等，已装载的分片跳过。
- `UbSeedFromDirWithShards`：只装载`shard_ids`列出的分片（真分布式，每个进程持有自己的分片，远端分片经URMA访问）。`count==0`不装载并返回成功；分片号超出`[0, num_shards)`报错。按分片幂等。

**返回值**：`0`成功；非`0`为`StatusCode`。

#### UbSetLocalShard

**接口定义**

```cpp
int UbSetLocalShard(uint32_t shard_id);
```

**接口用途**

钉住本进程服务的分片。一个进程只拥有一个分片；调用会校验该分片已在本地装载，并释放其他已装载分片的向量数据（全局元信息与分片映射保留，被释放分片的数据由对端经URMA提供）。变更分片归属须先`UbShutdown`。

**返回值**：`0`成功；非`0`为`StatusCode`。

#### UbLoadedShardCount / UbExpectedShardCount

**接口定义**

```cpp
uint32_t UbLoadedShardCount(void);
uint32_t UbExpectedShardCount(void);
```

**接口用途**

诊断：已装载分片数 / 期望分片数（`shard_map`声明的分片总数）。

### 资源发布与导入接口

#### UbGetShardRegions

**接口定义**

```cpp
int UbGetShardRegions(uint32_t capacity,
                      uint32_t* out_shard_ids,
                      void** out_graph_addrs,
                      uint64_t* out_graph_bytes,
                      void** out_vector_addrs,
                      uint64_t* out_vector_bytes);
```

**接口用途**

发布侧：枚举每个已装载分片的注册内存（图地址/长度、向量地址/长度），供调用方注册进UMDK并经资源目录发布。数组容量须不小于已装载分片数（可用`UbLoadedShardCount()`取得）。内存归运行时所有，`UbShutdown`前有效，调用方不得释放。

**返回值**：成功返回填充的条目数；失败返回正值状态码（如参数非法或容量不足时返回`kInvalidArgument`）。注意：成功返回1个条目时与`kInvalidArgument`(1)同值，建议结合`UbLoadedShardCount()`核对结果。

#### UbGetEntryPoint / UbGetEntryShard

**接口定义**

```cpp
int UbGetEntryPoint(uint64_t* out_entry_vertex_id, int32_t* out_entry_level);
int UbGetEntryShard(uint32_t* out_entry_shard_id);
```

**接口用途**

- `UbGetEntryPoint`：返回全局图入口点（`meta.entry_point`，最高层节点的全局ID）及其层数。
- `UbGetEntryShard`：返回入口点所在分片。该值才是资源目录发布用的正确`EntryShardID`（入口点的层数不是分片号）。

**返回值**：`0`成功；非`0`为`StatusCode`。

#### UbValidateShardLayout

**接口定义**

```cpp
int UbValidateShardLayout(uint32_t shard_id,
                          uint32_t shard_count,
                          uint32_t entry_shard_id,
                          uint64_t entry_vertex_id);
```

**接口用途**

基础一致性校验：把外部发布的分片身份（etcd ResourceRecord：shard_count、shard_id、入口点）与本地`global.meta`拓扑比对。激活某分片链路前使用。拒绝情形包括：shard_count与`meta.num_shards`不一致、shard_id/entry_shard_id越界、入口点无法在本地分片映射中解析或不属于entry_shard_id。

>![表示说明的图片](./public_sys-resources/icon-note.gif) **说明：**
>该校验只比对分片数量、编号范围与入口点的分片归属，不包含构建版本或内容指纹比对，也不要求发布的入口点等于本地`meta.entry_point`，因此不能单独用于排除过期发布或确认"同一次构建"。

**返回值**：`0`匹配；非`0`为`StatusCode`。

#### UbImportPeerHandles / UbRemovePeerHandles

**接口定义**

```cpp
int UbImportPeerHandles(uint32_t shard_id,
                        uint32_t role,
                        void* target_jetty,
                        void* target_segment);
void UbRemovePeerHandles(uint32_t shard_id, uint32_t role);
```

**接口用途**

对端句柄注入：采纳宿主（如UrmaManager）为拥有`shard_id`的对端激活链路后得到的不透明句柄对（`UrmaPeerEndpointInfo*`/`urma_target_seg_t*`）。运行时不持有句柄所有权，调用方须保持链路存活直至`UbShutdown`或`UbRemovePeerHandles`。同键重复调用为幂等重激活。

**role取值**

| 取值 | 角色 |
|------|------|
| 0 | topology（第0层图结构） |
| 1 | vectors（基向量） |
| 2 | distance-tasks（距离任务队列） |
| 3 | distance-results（距离结果队列） |
| 4 | distance-notify（WRITE_NOTIFY通知） |

**返回值**（`UbImportPeerHandles`）：`0`成功；非`0`为`StatusCode`。

### RPC邮箱接口

#### UbEnsureRpcMailbox

**接口定义**

```cpp
int UbEnsureRpcMailbox(void);
```

**接口用途**

构建进程本地RPC距离邮箱（幂等）。须在`UbSeedFromDir`/装载分片后调用（需已知world_size）。邮箱承载角色2（distance-tasks）与角色3（distance-results）的发布，并在`UbEnableDistributedSearch`时复用为requests/responses/notify三段。

**返回值**：`0`成功；非`0`为`StatusCode`。

#### UbGetRpcMailboxRegions / UbRpcLayout

**接口定义**

```cpp
int UbGetRpcMailboxRegions(void** out_requests_address,
                          uint64_t* out_requests_bytes,
                          void** out_responses_address,
                          uint64_t* out_responses_bytes,
                          void** out_notify_address,
                          uint64_t* out_notify_bytes,
                          uint32_t* out_ring_depth);
const char* UbRpcLayout(void);
```

**接口用途**

- `UbGetRpcMailboxRegions`：返回邮箱三段（requests/responses/notify）的地址与长度及环深度，供CGO层注册并经资源目录发布（角色2=TaskQueue、角色3=ResultQueue、notify为对端角色4）。`UbEnsureRpcMailbox`未成功时返回非`0`。
- `UbRpcLayout`：返回邮箱布局描述串（线程局部，下次调用前有效）。

### 启用与检索接口

#### UbEnableDistributedSearch

**接口定义**

```cpp
int UbEnableDistributedSearch(void);
```

**接口用途**

构建并启动分布式线程组运行时（调度器 + Control/Search/Distance/READ Progress/RPC Progress）。启动前提（任一不满足即拒绝）：恰好一个本地分片（多个已装载分片可先经`UbSetLocalShard`钉住本分片、释放其余）、本地RPC邮箱已构建（`UbEnsureRpcMailbox`）、每个远端分片全部五种角色（0~4）的对端句柄均已导入（`UbImportPeerHandles`）。幂等：运行时已启动时返回`0`。

**返回值**：`0`成功；非`0`为`StatusCode`。

#### UbSearch

**接口定义**

```cpp
int UbSearch(const float* query, uint32_t nq, uint32_t dim, uint32_t topk,
             uint32_t ef_search, const char* metric_name,
             int64_t* out_ids, float* out_dists);
```

**接口用途**

同步检索服务入口：对`nq`个查询向量执行全局图检索。结果共`nq × topk`项，不足处以`-1`（id）/`+inf`（距离）填充。`metric_name`须与已装载索引的度量一致（L2（L2 Norm，欧氏距离）或IP（Inner Product，内积）），未知度量报错。

**参数说明**

| 参数名称 | 参数类型 | 参数说明 |
|----------|----------|------|
| query | const float* | 查询向量，nq × dim连续排布 |
| nq | uint32_t | 查询向量数 |
| dim | uint32_t | 向量维度，须与索引一致 |
| topk | uint32_t | top-K结果数 |
| ef_search | uint32_t | 查询候选集宽度，应≥topk（Milvus接入层强制校验`0 < topk <= ef`） |
| metric_name | const char* | 度量名："L2"或"IP" |
| out_ids | int64_t* | 输出id（调用方分配nq × topk） |
| out_dists | float* | 输出距离（调用方分配nq × topk） |

**返回值**：`0`成功；非`0`为`StatusCode`。

#### UbRpcDistanceReady / UbRetryEnableForPushdown

**接口定义**

```cpp
int UbRpcDistanceReady(void);
int UbRetryEnableForPushdown(char* msg, int msg_capacity);
```

**接口用途**

- `UbRpcDistanceReady`：报告分布式运行时是否已启动且距离数据面（RPC邮箱）可用。启用前与停止后为未就绪。
- `UbRetryEnableForPushdown`：重新执行分布式启用。成功时构建完整RPC距离树；对端资源不全时拒绝并把原因写入`msg`（容量`msg_capacity`，NUL结尾）。

**返回值**：`UbRpcDistanceReady`返回`1`=就绪、`0`=未就绪（布尔语义，非状态码）；`UbRetryEnableForPushdown`返回`0`成功，非`0`为`StatusCode`。

### 释放接口

#### UbStopDistributedSearch / UbShutdown

**接口定义**

```cpp
int UbStopDistributedSearch(void);
int UbShutdown(void);
```

**接口用途**

- `UbStopDistributedSearch`：停止分布式检索运行时。可安全重复调用。
- `UbShutdown`：释放全部单例状态，可安全重复调用；之后可由下一次装载重新seed。

**返回值**：`0`成功；非`0`为`StatusCode`。

### 诊断接口

#### UbPing

**接口定义**

```cpp
int UbPing(int x);
```

**接口用途**

连通性冒烟测试：返回`x + 1`。

## 运行时配置（RuntimeConfig）

以下配置经`UbSetConfig`/`UbSetMemoryBudget`/`UbSetTimeouts`/`UbSetExpandBatch`或基准CLI设置，默认值见`config.hpp`。

**表 1** 主要运行时配置

| 配置 | 默认值 | 说明 |
|------|--------|------|
| topk | 10 | top-K结果数 |
| ef_search | auto（索引内默认值） | 查询候选集宽度 |
| expand_batch | 1 | 第0层每轮展开节点数；maximum_batch = max(1, M, M0 × expand_batch) |
| distance_policy | auto | 距离策略：auto / pull / pushdown |
| pull_threshold | 4 | auto策略下组内ID数量阈值，小于则拉取否则下推 |
| use_dot_norm | false | 是否启用dot+norm L2距离优化路径（以‖q‖²+‖x‖²−2q·x计算平方L2距离，仅限L2度量） |
| layer0_sort_mode | all | 第0层组内ID排序范围 |
| max_active_queries | 16 | 查询池大小（在途查询上限） |
| operation_timeout_ms | 5000 | 单次尝试阶段超时（毫秒） |
| query_timeout_ms | 30000 | 查询级墙钟超时（毫秒） |
| drain_timeout_ms | 30000 | 优雅退出超时（毫秒） |
| queue_memory_budget_bytes | 1 GiB | 队列计划内存上限 |
| registered_rpc_memory_budget_bytes | 8 GiB | 注册RPC计划内存上限 |
| read_jfs_depth | auto | URMA READ JFS深度：max(64, maximum_batch, max_active_queries × expand_batch × 2) |
| request_ring_depth | 8 | RPC请求环深度 |
| warmup / rounds | 2 / 5 | 基准预热轮数 / 正式轮数 |

## 索引构建（C++，index/builder.hpp）

### ReferenceBuildOptions

```cpp
struct ReferenceBuildOptions {
  std::uint32_t num_shards{1};
  std::uint32_t M{16};
  std::uint32_t M0{32};
  std::uint32_t ef_search{64};
  std::uint32_t metric{0};  // 0=L2, 1=负内积
};
```

### BuildReferenceIndex

**接口定义**

```cpp
Status BuildReferenceIndex(const std::filesystem::path& root,
                           std::span<const float> vectors,
                           std::uint64_t rows, std::uint32_t dim,
                           const ReferenceBuildOptions& options,
                           std::uint64_t max_rows = 100000);
```

**接口用途**

确定性O(N²)构建器，用于测试夹具与本地正确性校验。输入超过`max_rows`（默认100,000）即拒绝，除非调用方显式调高上限；生产大索引请使用Faiss适配构建。

### Faiss适配构建

**接口定义**

```cpp
bool FaissBuilderAvailable() noexcept;
Status BuildFaissIndex(const std::filesystem::path& root,
                       std::span<const float> vectors,
                       std::uint64_t rows, std::uint32_t dim,
                       const ReferenceBuildOptions& options,
                       std::uint32_t ef_construction);
Status FaissStreamBegin(const std::filesystem::path& root,
                        std::uint64_t rows, std::uint32_t dim,
                        const ReferenceBuildOptions& options,
                        std::uint32_t ef_construction,
                        FaissStreamBuilder** out);
Status FaissStreamAdd(FaissStreamBuilder* builder,
                      std::span<const float> vectors, std::uint64_t rows);
Status FaissStreamFinish(FaissStreamBuilder* builder);
void FaissStreamDestroy(FaissStreamBuilder* builder) noexcept;
```

**接口用途**

基于Faiss构建生产级HNSW图并产出索引三元组。`FaissStreamBuilder`为增量构建器：按有界数据块喂入向量，调用方无需一次性物化全部数据；句柄不透明，用毕以`FaissStreamDestroy`释放。

>![表示说明的图片](./public_sys-resources/icon-note.gif) **说明：**
>Faiss构建接口声明于构建插件头文件`faiss_builder.hpp`，位于**全局命名空间**（不在`namespace hnsw_urma`内；签名中的`Status`/`ReferenceBuildOptions`来自`hnsw_urma`）。`BuildReferenceIndex`位于`namespace hnsw_urma`（`index/builder.hpp`），两者命名空间不同，调用时注意区分。

**构建产物**（写入`root`目录）

| 文件 | 内容 |
|------|------|
| `global.meta` | 全局元信息与各分片ID范围 |
| `upper_graph.index` | 全局上层图（第1层及以上）完整邻接关系 |
| `shard_N.index` | 分片N的第0层邻接表与原始向量 |

## 索引文件读写（C++，index/format.hpp）

format V2：魔数`0x3258444E494D5248`，版本2，4096字节定长头（`IndexHeaderV2`），数据块4KB对齐并带校验。文件类型三种：`kGlobalMeta`/`kUpperGraph`/`kShard`。

| 接口 | 用途 |
|------|------|
| `WriteGlobalIndexFile` / `WriteUpperGraphFile` / `WriteShardFile` | 分别写出三元组文件 |
| `LoadGlobalIndexFile` | 读取`global.meta`（元信息+分片范围） |
| `LoadUpperGraphFile` | 读取并物化上层图副本（`UpperGraphReplica`） |
| `LoadShardFile` | 读取分片（`LocalShard`：第0层邻接表+向量） |
| `LoadIndexForShard` | 一次性装载某分片运行所需全部数据（`LoadedIndex`） |
| `ShardFilePath` | 计算分片文件路径（`shard_N.index`） |

`GlobalIndexMeta`关键字段：`ntotal`、`dim`、`metric`、`M`、`M0`、`ef_search_default`、`entry_point`、`max_level`、`num_shards`。装载时校验元信息一致性（`ValidateGlobalMeta`），不一致即拒绝。

## 缩略语

| 缩略语 | 英文全称 | 中文全称 |
|--------|----------|----------|
| ABI | Application Binary Interface | 应用二进制接口 |
| API | Application Programming Interface | 应用程序编程接口 |
| CLI | Command Line Interface | 命令行界面 |
| CPU | Central Processing Unit | 中央处理器 |
| HNSW | Hierarchical Navigable Small World | 分层可导航小世界 |
| IP | Inner Product | 内积 |
| JFS | Jetty Frame Sender | Jetty帧发送端 |
| L2 | L2 Norm | 欧氏距离 |
| NUMA | Non-Uniform Memory Access | 非一致内存访问 |
| RPC | Remote Procedure Call | 远程过程调用 |
| STL | Standard Template Library | 标准模板库 |
| UMDK | Unified Memory Development Kit | 统一内存开发工具包 |
| URMA | Unified Remote Memory Access | 统一远端内存访问 |
| WR | Work Request | 工作请求 |

## 修订记录

| 文档版本 | 发布日期 | 修改说明 |
|------|----------|------------|
| 01 | 2026-12-30 | 第一次正式发布。 |

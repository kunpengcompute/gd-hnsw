# API参考

## 概述

本文介绍gd_hnsw的API（Application Programming Interface，应用程序编程接口）。gd_hnsw采用HNSW（Hierarchical Navigable Small World，分层可导航小世界）图索引，跨节点通信基于URMA（Unified Remote Memory Access，统一远端内存访问）。

本库提供两类接口：

- **C接口**：头文件`ub_c_api.h`，供Milvus等宿主接入检索运行时。
- **C++接口**：位于`namespace hnsw_urma`，用于参考建库和索引文件读写。

调用顺序：**注入URMA上下文 → 配置 → 装载分片 → 创建邮箱 → 发布、导入资源 → 启用 → 检索 → 释放**。完整接入流程见《[快速入门](./quick_start.md#经c-abi集成生命周期顺序)》。

除另有说明外，C接口返回`0`表示成功，非`0`表示失败。配置和输出结构体应先清零，`reserved`字段保持为0。宿主应顺序执行装载、资源发布、启用和释放操作，避免并发变更运行时状态。

## 状态码

| 状态码 | 说明 |
|--------|------|
| kOk（0） | 成功 |
| kInvalidArgument | 参数不合法 |
| kInvalidFormat | 索引格式不合法 |
| kOutOfRange | 数值超出范围 |
| kResourceExhausted | 资源不足 |
| kProtocolError | 协议错误 |
| kTimeout | 操作超时 |
| kCancelled | 操作被取消 |
| kTransportError | 通信错误 |
| kInternalError | 内部错误 |

C++接口通过`Status`或`Result<T>`返回常规错误，使用`ok()`判断成功，错误说明由`Status::ToString()`提供；内存分配等异常仍可能抛出。

## 配置接口

### UbSetUrmaContext

**接口定义**

```cpp
int UbSetUrmaContext(void* urma_context);
```

**接口用途**

设置宿主已创建的URMA上下文（`urma_context_t*`）。本库借用上下文，由宿主负责创建和释放。

**参数说明**

| 参数 | 类型 | 说明 |
|------|------|------|
| urma_context | void* | 服务模式下须为有效上下文；在启用检索前设置 |

运行时启用后不能更换上下文。宿主须等运行时成功停止后再释放上下文。

### UbConfigure

**接口定义**

```cpp
int UbConfigure(const UbConfig* cfg);
```

**接口用途**

保存线程、CPU（Central Processing Unit，中央处理器）和内存预算等运行时配置，在首次装载分片前调用。`cfg`须非空；启用检索后不能修改配置。

**UbConfig参数说明**

| 字段 | 类型 | 说明 |
|------|------|------|
| search_workers | uint32_t | Search线程数；0=自动 |
| distance_workers | uint32_t | Distance线程数；0=自动 |
| read_progress_workers | uint32_t | READ Progress线程数；0=自动 |
| rpc_progress_workers | uint32_t | RPC（Remote Procedure Call，远程过程调用）Progress线程数；0=自动 |
| cpu_bind_mode | const char* | `auto`或`none`；NULL=`auto` |
| cpu_layout | const char* | `auto`、`contiguous`、`llc-isolated`或`llc-spread`；NULL=`contiguous` |
| progress_mode | const char* | `busy`或`adaptive`；NULL=`busy` |
| allowed_cpus | const uint32_t* | 必填，进程可用的非空CPU列表 |
| allowed_cpu_count | uint32_t | CPU列表长度，须大于0 |
| max_active_queries | uint32_t | 最大在途查询数，须≥1；建议初始设为16 |
| read_jfs_depth | uint32_t | URMA READ发送队列深度；0=自动 |
| queue_memory_budget_bytes | uint64_t | 队列内存预算；0=默认1 GiB（Gibibyte，吉比字节） |
| registered_rpc_memory_budget_bytes | uint64_t | RPC注册内存预算；0=默认8 GiB |
| query_timeout_ms | uint32_t | 查询超时，单位毫秒；0=默认30000 |
| operation_timeout_ms | uint32_t | 操作超时，单位毫秒；0=默认5000，不得大于查询超时 |
| expand_batch | uint32_t | 每轮展开节点数；0=默认1，显式值范围[1,64] |
| runtime_options | const char* | 可选配置串，覆盖对应字段；NULL=空串 |
| urma_device_name | const char* | 启用分布式检索前须提供有效的非空URMA设备名 |
| urma_eid_index | uint32_t | URMA设备上的EID（Endpoint Identifier，端点标识符）索引，可为0 |

>![表示说明的图片](./public_sys-resources/icon-note.gif) **说明：**
>不要直接用全零配置启动。须填写设备名、CPU列表及长度、`max_active_queries`。该接口只保存配置；`max_active_queries=0`会在后续配置解析时返回`kOutOfRange`，CPU列表为空会导致邮箱创建失败。

`runtime_options`使用空格分隔，例如：

```text
--distance-policy pushdown --use-dot-norm --layer0-sort local-only --request-ring-depth 32
```

下表列出常用选项；不支持的选项会在配置解析时被拒绝。

| 选项 | 默认值 | 说明 |
|------|--------|------|
| --distance-policy | auto | 距离策略：`auto`、`pull`或`pushdown` |
| --pull-threshold | 4 | auto策略下，远端分组ID（Identifier，标识符）数量小于此值时拉取，否则下推 |
| --use-dot-norm | 关闭 | 启用L2（L2 Norm，欧氏距离）的平方距离优化，仅对L2生效 |
| --layer0-sort | all | 第0层排序范围：`all`、`local-only`或`upper-only` |
| --request-ring-depth | 8 | RPC请求环深度，须为正的2的幂 |
| --drain-timeout-ms | 30000 | 停止时等待在途任务排空的超时，单位毫秒 |
| --profile | 关闭 | 开启运行时画像统计 |

### UbEstimateRuntimeMemory

```cpp
int UbEstimateRuntimeMemory(const char* root, uint32_t shard, uint64_t* bytes);
```

装载分片前估算运行时内存。`root`为索引三元组目录，`shard`为本地分片号，`bytes`为非空的输出指针。

## 装载接口

### UbSeedFromDir

**接口定义**

```cpp
int UbSeedFromDir(const char* triple_dir,
                  const uint32_t* shard_ids, uint32_t count);
```

**接口用途**

从索引三元组目录装载全部分片或指定分片。

同一索引按分片幂等装载；从空运行时开始时，若分片装载返回错误，将回滚本次已装载数据。

**参数说明**

| 参数 | 类型 | 说明 |
|------|------|------|
| triple_dir | const char* | 包含`global.meta`、`upper_graph.index`和分片文件的目录 |
| shard_ids | const uint32_t* | 要装载的分片号数组，各分片号须在[0, num_shards)内 |
| count | uint32_t | 数组长度；`shard_ids==NULL`或`count==0`时装载全部分片 |

分布式服务每个进程只保留一个本地分片，可在装载后调用`UbSetLocalShard`确定。

### UbLoadShardFromDir / UbReadIndexMeta

**接口定义**

```cpp
int UbLoadShardFromDir(const char* triple_dir, uint32_t shard_id,
                       UbIndexMeta* out_meta);
int UbReadIndexMeta(const char* triple_dir, uint32_t shard_id,
                    UbIndexMeta* out_meta);
```

**接口用途**

- `UbLoadShardFromDir`：装载一个分片，校验分片存在且各分片声明的行数之和等于`ntotal`；`out_meta`可为NULL。
- `UbReadIndexMeta`：只读元信息，不装载分片；`out_meta`须非空。

`triple_dir`为三元组目录，`shard_id`为分片号。只读元信息时，分片号超出范围会使`local_count`为0；装载时须提供有效分片号。

**UbIndexMeta输出字段**

| 字段 | 类型 | 说明 |
|------|------|------|
| dim | uint32_t | 向量维度 |
| ntotal | uint64_t | 全局向量总数 |
| shard_count | uint32_t | 全局分片数 |
| local_count | uint64_t | 指定分片的向量数 |

### UbSetLocalShard

```cpp
int UbSetLocalShard(uint32_t shard_id);
```

指定本进程服务的分片，须在启用检索前调用。该分片必须已装载；若已装载多个分片，调用后仅保留指定分片。启用后不能通过此接口更换分片，应先成功关闭运行时。

### 装载状态与图复用

```cpp
int UbGetShardStatus(UbShardStatus* out);
int UbBindGlobalGraphIdentity(const char* graph_id, uint64_t epoch);
int UbReuseGlobalGraphShard(const char* graph_id, uint64_t epoch,
                            uint32_t shard, int* out_reused,
                            UbSeededShardInfo* out_info);
```

| 接口 | 用途及参数要求 |
|------|----------------|
| UbGetShardStatus | 读取装载状态；`out`须非空，读取时避免并发装载或关闭 |
| UbBindGlobalGraphIdentity | 装载成功后绑定图身份；`graph_id`非空，`epoch>0`；更换身份须先成功关闭运行时 |
| UbReuseGlobalGraphShard | 查询图和分片是否可复用；`graph_id`非空，`out_reused`、`out_info`须非空 |

复用命中时，`out_reused=1`并填写`out_info`；尚未绑定图身份时返回成功且`out_reused=0`，继续正常装载。身份不匹配或目标分片未在本地装载时返回错误。

| 输出结构体 | 字段 | 类型 | 说明 |
|------------|------|------|------|
| UbShardStatus | local_shard_id | uint32_t | 唯一本地分片号；未确定唯一分片时为`UB_NO_LOCAL_SHARD` |
| UbShardStatus | loaded_count | uint32_t | 已装载分片数 |
| UbShardStatus | expected_count | uint32_t | 全局分片数 |
| UbSeededShardInfo | dim | uint32_t | 已装载分片的向量维度 |
| UbSeededShardInfo | count | uint64_t | 已装载分片的向量数 |

## 资源发布与导入接口

### UbEnsureRpcMailbox

```cpp
int UbEnsureRpcMailbox(UbRpcMailboxRegions* out);
```

装载分片后创建本地RPC邮箱，并返回供宿主注册、发布的内存信息。`out`须非空，可重复调用。

**UbRpcMailboxRegions输出字段**

| 字段 | 类型 | 说明 |
|------|------|------|
| requests_address / requests_bytes | void* / uint64_t | 请求内存地址/字节数，对应角色2 |
| responses_address / responses_bytes | void* / uint64_t | 响应内存地址/字节数，对应角色3 |
| notify_address / notify_bytes | void* / uint64_t | 通知内存地址/字节数，对应角色4 |
| ring_depth | uint32_t | 邮箱环深度 |

### UbGetGraphPublishSnapshot

**接口定义**

```cpp
int UbGetGraphPublishSnapshot(uint32_t capacity,
                              UbShardRegion* out_regions,
                              UbGraphPublishSnapshot* out_snapshot);
```

**接口用途**

获取本地分片的图、向量内存地址和全局入口信息，供宿主发布资源。

**参数说明**

| 参数 | 类型 | 说明 |
|------|------|------|
| capacity | uint32_t | `out_regions`数组容量 |
| out_regions | UbShardRegion* | 调用方分配的分片内存信息数组 |
| out_snapshot | UbGraphPublishSnapshot* | 非空的输出结构体 |

先传`capacity=0`、`out_regions=NULL`查询`region_count`，再分配足够容量的数组并再次调用。结果从`out_regions`读取，`out_snapshot->regions`始终为NULL。

| 输出结构体 | 字段 | 说明 |
|------------|------|------|
| UbShardRegion | shard_id | 分片号 |
| UbShardRegion | graph_address / graph_bytes | 第0层图内存地址/字节数 |
| UbShardRegion | vector_address / vector_bytes | 向量内存地址/字节数 |
| UbGraphPublishSnapshot | region_count | 返回的分片内存信息条目数 |
| UbGraphPublishSnapshot | shard_count | 全局分片数 |
| UbGraphPublishSnapshot | entry_shard_id | 全局入口点所在分片 |
| UbGraphPublishSnapshot | entry_vertex_id / entry_level | 全局入口点ID/层数 |

先用`UbSetLocalShard`确定本地分片，再读取、发布资源。内存由运行时管理，仅在对应分片保持装载期间有效；分片装载集合变化后须重新获取。

### UbValidateShardLayout

```cpp
int UbValidateShardLayout(uint32_t shard_id, uint32_t shard_count,
                          uint32_t entry_shard_id, uint64_t entry_vertex_id);
```

导入对端资源前，校验其分片数量、分片编号和入口点归属是否符合本地索引。成功返回0；该校验不能单独确认资源来自同一次索引构建。

### UbImportPeerHandles / UbRemovePeerHandles

**接口定义**

```cpp
int UbImportPeerHandles(uint32_t shard_id, uint32_t role,
                        const void* rjetty_descriptor,
                        uint32_t descriptor_bytes, uint32_t jetty_token,
                        void* target_segment);
void UbRemovePeerHandles(uint32_t shard_id, uint32_t role);
```

**接口用途**

导入或移除宿主已激活的对端资源句柄。导入成功返回0；移除接口无返回值。

**参数说明**

| 参数 | 类型 | 说明 |
|------|------|------|
| shard_id | uint32_t | 对端分片号 |
| role | uint32_t | 资源角色，见下表 |
| rjetty_descriptor | const void* | 非空的对端端点描述数据 |
| descriptor_bytes | uint32_t | 描述数据字节数，须大于0 |
| jetty_token | uint32_t | 对端端点令牌 |
| target_segment | void* | 宿主提供的有效目标段指针，不能为NULL |

| role | 资源 |
|------|------|
| 0 | topology：第0层图结构 |
| 1 | vectors：基向量 |
| 2 | distance-tasks：距离请求 |
| 3 | distance-results：距离响应 |
| 4 | distance-notify：完成通知 |

>![表示说明的图片](./public_sys-resources/icon-note.gif) **说明：**
>句柄由宿主管理。只有运行时成功停止并排空后，宿主才可释放链路资源；停止失败时须继续保留。仅调用`UbRemovePeerHandles`不会停止正在使用该资源的运行时。

## 启用与检索接口

### UbEnableDistributedSearch / UbEnableDistributedSearchWithTimeout

**接口定义**

```cpp
int UbEnableDistributedSearch(char* msg, int msg_capacity);
int UbEnableDistributedSearchWithTimeout(int64_t timeout_ns);
```

**接口用途**

启动分布式检索。调用前须已提供有效的非空URMA设备名、注入上下文、保留一个本地分片、创建邮箱，并导入所有远端分片的五类资源句柄。

| 参数 | 类型 | 说明 |
|------|------|------|
| msg | char* | 可选的失败原因缓冲区，可为NULL |
| msg_capacity | int | 缓冲区字节数；大于0时最多写入该容量（含字符串结束符） |
| timeout_ns | int64_t | 启动预算，单位纳秒；-1=不限时，0=立即超时，小于-1=参数错误 |

成功返回0。资源未就绪时，补齐资源后可重试；运行时已存在时，重复调用返回其当前状态，健康时为0。

### UbSearch / UbSearchWithTimeout

**接口定义**

```cpp
int UbSearch(const float* query, uint32_t nq, uint32_t dim, uint32_t topk,
             uint32_t ef_search, const char* metric_name,
             int64_t* out_ids, float* out_dists);
int UbSearchWithTimeout(const float* query, uint32_t nq, uint32_t dim,
                        uint32_t topk, uint32_t ef_search,
                        const char* metric_name, int64_t timeout_ns,
                        int64_t* out_ids, float* out_dists);
```

**接口用途**

对一批查询向量执行全局检索，同步返回top-K结果。须先成功启用分布式检索。

**参数说明**

| 参数 | 类型 | 说明及取值范围 |
|------|------|----------------|
| query | const float* | 非空，`nq × dim`个连续float值，不含NaN（Not a Number，非数值）或无穷值 |
| nq | uint32_t | 本批查询向量数，范围[1,4096]；更多向量须分批调用 |
| dim | uint32_t | 向量维度，须与索引一致 |
| topk | uint32_t | 每条查询返回的结果数，须大于0 |
| ef_search | uint32_t | 搜索候选集宽度，须≥topk |
| metric_name | const char* | L2或IP（Inner Product，内积），须与索引一致 |
| timeout_ns | int64_t | 剩余请求预算，单位纳秒；-1=使用配置超时，0=立即超时，小于-1=参数错误 |
| out_ids | int64_t* | 非空，由调用方分配`nq × topk`项 |
| out_dists | float* | 非空，由调用方分配`nq × topk`项 |

**输出约定**：结果按查询顺序连续排列；不足top-K时，以`id=-1`、`distance=+inf`填充。

`UbSearch`等价于`timeout_ns=-1`的`UbSearchWithTimeout`。同一批查询共用截止时间，其排队和执行受`query_timeout_ms`及请求剩余预算共同限制。

## 释放接口

```cpp
int UbStopDistributedSearch(void);
int UbShutdown(void);
```

| 接口 | 用途 |
|------|------|
| UbStopDistributedSearch | 停止接纳查询，等待在途任务排空并停止运行时 |
| UbShutdown | 停止运行时并释放已装载索引、邮箱和暂存句柄，可在成功后重新装载 |

两者均可重复调用。须检查返回码；停止或关闭失败时，不要释放宿主持有的URMA上下文或链路资源。

## 诊断接口

```cpp
int UbPing(int x);
int UbRpcDistanceReady(void);
const char* UbLastError(void);
const char* UbRpcLayout(void);
```

| 接口 | 返回值及说明 |
|------|--------------|
| UbPing | 返回`x + 1`，用于验证接口调用是否连通 |
| UbRpcDistanceReady | 运行时对象存在时返回1，否则返回0；不能代替健康检查 |
| UbLastError | 返回线程保存的错误详情；为空时返回NULL。部分失败分支不更新详情，可能保留旧值，应以本次返回码为准 |
| UbRpcLayout | 返回本地RPC邮箱的布局描述字符串 |

返回的字符串由库管理，调用方不得释放，应及时复制。

## 索引构建（C++）

头文件：`hnsw_urma/index/builder.hpp`。

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

用小规模数据构建索引三元组，供测试和正确性验证。生产环境的大规模Faiss建库由宿主侧构建组件提供。

**参数说明**

| 参数 | 说明 |
|------|------|
| root | 输出目录 |
| vectors | `rows × dim`个连续float值 |
| rows / dim | 向量数/维度，均须大于0 |
| options | 构建参数，见下表 |
| max_rows | 允许的最大向量数，默认100000；超出时返回错误 |

**ReferenceBuildOptions参数说明**

| 字段 | 类型 | 说明 |
|------|------|------|
| num_shards | uint32_t | 分片数，默认1，范围[1, rows] |
| M | uint32_t | 上层最大邻居数，默认16，须大于0 |
| M0 | uint32_t | 第0层最大邻居数，默认32，须大于0 |
| ef_search | uint32_t | 写入索引的默认搜索宽度，默认64，须大于0 |
| metric | uint32_t | 0=L2，1=负内积；默认0 |

**返回值**：`Status`，使用`ok()`判断是否成功。

## 索引文件读写（C++）

头文件：`hnsw_urma/index/format.hpp`。索引使用format V2，由`global.meta`、`upper_graph.index`和`shard_N.index`组成。

| 接口 | 用途 |
|------|------|
| WriteGlobalIndexFile / LoadGlobalIndexFile | 写入/读取全局元信息和分片范围 |
| WriteUpperGraphFile / LoadUpperGraphFile | 写入/读取上层图 |
| WriteShardFile / LoadShardFile | 写入/读取一个分片的第0层图和向量 |
| LoadIndexForShard | 装载指定分片检索所需的数据 |
| ShardFilePath | 获取`shard_N.index`文件路径 |

## 缩略语

| 缩略语 | 英文全称 | 中文全称 |
|--------|----------|----------|
| API | Application Programming Interface | 应用程序编程接口 |
| CPU | Central Processing Unit | 中央处理器 |
| EID | Endpoint Identifier | 端点标识符 |
| GiB | Gibibyte | 吉比字节 |
| HNSW | Hierarchical Navigable Small World | 分层可导航小世界 |
| ID | Identifier | 标识符 |
| IP | Inner Product | 内积 |
| L2 | L2 Norm | 欧氏距离 |
| NaN | Not a Number | 非数值 |
| RPC | Remote Procedure Call | 远程过程调用 |
| URMA | Unified Remote Memory Access | 统一远端内存访问 |

## 修订记录

| 文档版本 | 发布日期 | 修改说明 |
|------|----------|------------|
| 01 | 2026-12-30 | 第一次正式发布。 |

# API参考

本文介绍gd_hnsw的对外接口定义、接口说明与调用约定。

## 概述

本文介绍gd_hnsw的API。gd_hnsw采用HNSW图索引，跨节点通信基于URMA（Unified Remote Memory Access，统一远端内存访问）。

本库提供两类接口：

- **C接口**：头文件`ub_c_api.h`，供Milvus等宿主接入检索运行时。
- **C++接口**：位于`namespace hnsw_urma`，用于参考建库和索引文件读写。

调用顺序：先注入URMA上下文并配置，然后装载分片、创建邮箱，再发布并导入资源，之后启用并执行检索，最后释放。完整接入流程请参见《[快速入门](./quick_start.md#集成c-abi)》。

调用约定如下：

- 除另有说明外，C接口返回`0`表示成功，非`0`表示失败。
- 配置和输出结构体应先清零，`reserved`字段保持为0。
- 宿主应顺序执行装载、资源发布、启用和释放操作，避免并发变更运行时状态。

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

**参数说明**

| 参数 | 类型 | 说明 |
|------|------|------|
| cfg | const UbConfig* | 非空的配置结构体指针；各字段见下表 |

**UbConfig字段说明**

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
>不要直接用全零配置启动。须填写设备名、CPU列表及长度、`max_active_queries`。`UbConfigure`只暂存配置，装载接口不负责解析运行时配置。建议在装载前调用`UbEstimateRuntimeMemory`提前校验：`max_active_queries=0`会在该步返回`kOutOfRange`。若未调用内存估算接口，该错误会在创建邮箱（`UbEnsureRpcMailbox`）时返回；启用接口也会再次解析配置。CPU列表为空会导致内存估算或创建邮箱失败。

`runtime_options`使用空格分隔，例如：

```text
--use-dot-norm --layer0-sort local-only --request-ring-depth 32
```

下表列出常用选项；不支持的选项会在配置解析时被拒绝。

| 选项 | 默认值 | 说明 |
|------|--------|------|
| --use-dot-norm | 关闭 | 启用L2（L2 Norm，欧氏距离）的平方距离优化，仅对L2生效 |
| --layer0-sort | all | 第0层排序范围：`all`、`local-only`或`upper-only` |
| --request-ring-depth | 8 | RPC请求环深度，须为正的2的幂 |
| --drain-timeout-ms | 30000 | 停止时等待在途任务排空的超时，单位毫秒 |
| --profile | 关闭 | 开启运行时画像统计 |

### UbEstimateRuntimeMemory

**接口定义**

```cpp
int UbEstimateRuntimeMemory(const char* root, uint32_t shard, uint64_t* bytes);
```

**接口用途**

在`UbConfigure`之后、装载分片之前校验运行时配置并估算内存。返回值包括队列、RPC与READ缓冲、操作池及访问表，不包括索引图、向量和宿主全部开销。

**参数说明**

| 参数 | 类型 | 说明 |
|------|------|------|
| root | const char* | 索引三元组目录 |
| shard | uint32_t | 本地分片号 |
| bytes | uint64_t* | 非空的输出指针，返回估算字节数 |

## 装载接口

### UbSeedFromDir

**接口定义**

```cpp
int UbSeedFromDir(const char* triple_dir,
                  const uint32_t* shard_ids, uint32_t count);
```

**接口用途**

从索引三元组目录装载全部分片或指定分片。与`UbLoadShardFromDir`的差异：`UbSeedFromDir`面向一次性装载全量或分片子集，适用于本地调试与验证场景；`UbLoadShardFromDir`装载单个分片，内置各分片行数总和与`ntotal`的一致性校验并可回写元数据，推荐用于“一个进程一个分片”的服务部署。

同一索引按分片幂等装载；从空运行时开始时，若分片装载返回错误，将回滚本次已装载数据。

**参数说明**

| 参数 | 类型 | 说明 |
|------|------|------|
| triple_dir | const char* | 包含`global.meta`、`upper_graph.index`和分片文件的目录 |
| shard_ids | const uint32_t* | 要装载的分片号数组，各分片号须大于或等于0且小于num_shards |
| count | uint32_t | 数组长度；`shard_ids==NULL`或`count==0`时装载全部分片 |

分布式服务每个进程只保留一个本地分片，可在装载后调用`UbSetLocalShard`确定。

### UbLoadShardFromDir

**接口定义**

```cpp
int UbLoadShardFromDir(const char* triple_dir, uint32_t shard_id,
                       UbIndexMeta* out_meta);
```

**接口用途**

装载一个分片，校验分片存在且各分片声明的行数之和等于`ntotal`。适用于“一个进程一个分片”的服务部署；需要同时读取装载元信息时，可传入`out_meta`。

**参数说明**

| 参数 | 类型 | 说明 |
|------|------|------|
| triple_dir | const char* | 包含索引三元组的非空目录路径 |
| shard_id | uint32_t | 本地分片号，须为索引中已声明的有效分片 |
| out_meta | UbIndexMeta* | 可为NULL；非空时返回装载元信息 |

**UbIndexMeta输出字段**

| 字段 | 类型 | 说明 |
|------|------|------|
| dim | uint32_t | 向量维度 |
| ntotal | uint64_t | 全局向量总数 |
| shard_count | uint32_t | 全局分片数 |
| local_count | uint64_t | 指定分片的向量数 |
| reserved | uint64_t | 保留字段，返回0 |

### UbReadIndexMeta

**接口定义**

```cpp
int UbReadIndexMeta(const char* triple_dir,
                    uint32_t shard_id,
                    UbIndexMeta* out_meta);
```

**接口用途**

只读`global.meta`中的元信息，不装载分片；适用于读取索引规模、维度或分片数量。

**参数说明**

| 参数 | 类型 | 说明 |
|------|------|------|
| triple_dir | const char* | 包含索引三元组的非空目录路径 |
| shard_id | uint32_t | 选择要统计的分片；超出范围时`local_count`返回0 |
| out_meta | UbIndexMeta* | 非空输出指针；字段请参见[UbLoadShardFromDir](#ubloadshardfromdir)中的输出字段表 |

### UbSetLocalShard

**接口定义**

```cpp
int UbSetLocalShard(uint32_t shard_id);
```

**接口用途**

指定本进程服务的分片，须在启用检索前调用。该分片必须已装载；若已装载多个分片，调用后仅保留指定分片。启用后不能通过此接口更换分片，应先成功关闭运行时。

**参数说明**

| 参数 | 类型 | 说明 |
|------|------|------|
| shard_id | uint32_t | 本进程服务的分片号，须已装载 |

### UbGetShardStatus

**接口定义**

```cpp
int UbGetShardStatus(UbShardStatus* out);
```

**接口用途**

读取本进程装载状态快照；读取时应避免并发装载或关闭。

**参数说明**

| 参数 | 类型 | 说明 |
|------|------|------|
| out | UbShardStatus* | 非空输出结构体指针 |

**UbShardStatus输出字段**

| 输出结构体 | 字段 | 类型 | 说明 |
|------------|------|------|------|
| UbShardStatus | local_shard_id | uint32_t | 唯一本地分片号；未确定唯一分片时为`UB_NO_LOCAL_SHARD` |
| UbShardStatus | loaded_count | uint32_t | 已装载分片数 |
| UbShardStatus | expected_count | uint32_t | 全局分片数 |
| UbShardStatus | reserved | uint32_t | 保留字段，返回0 |

### UbBindGlobalGraphIdentity

**接口定义**

```cpp
int UbBindGlobalGraphIdentity(const char* graph_id, uint64_t epoch);
```

**接口用途**

装载成功后绑定图身份。同一图身份重复调用幂等；更换身份须先成功关闭运行时。

**参数说明**

| 参数 | 类型 | 说明 |
|------|------|------|
| graph_id | const char* | 非空的图身份字符串 |
| epoch | uint64_t | 图版本，须大于0 |

### UbReuseGlobalGraphShard

**接口定义**

```cpp
int UbReuseGlobalGraphShard(const char* graph_id, uint64_t epoch,
                            uint32_t shard, int* out_reused,
                            UbSeededShardInfo* out_info);
```

**接口用途**

查询已绑定图身份与目标分片是否可复用。复用命中时，`out_reused=1`并填写`out_info`。尚未绑定图身份时返回成功且`out_reused=0`，宿主继续正常装载；身份不匹配或目标分片未在本地装载时返回错误。

**参数说明**

| 参数 | 类型 | 说明 |
|------|------|------|
| graph_id | const char* | 非空的图身份字符串 |
| epoch | uint64_t | 待匹配的图版本，应与已绑定版本一致 |
| shard | uint32_t | 待复用的分片号 |
| out_reused | int* | 非空输出指针；1表示命中，0表示未命中 |
| out_info | UbSeededShardInfo* | 非空输出结构体指针；仅在命中时读取字段 |

**UbSeededShardInfo输出字段**

| 输出结构体 | 字段 | 类型 | 说明 |
|------------|------|------|------|
| UbSeededShardInfo | dim | uint32_t | 已装载分片的向量维度 |
| UbSeededShardInfo | count | uint64_t | 已装载分片的向量数 |
| UbSeededShardInfo | reserved | uint32_t | 保留字段，命中时返回0 |

## 资源发布与导入接口

### UbEnsureRpcMailbox

**接口定义**

```cpp
int UbEnsureRpcMailbox(UbRpcMailboxRegions* out);
```

**接口用途**

装载分片后创建本地RPC邮箱，并返回供宿主注册、发布的内存信息。可重复调用。

**参数说明**

| 参数 | 类型 | 说明 |
|------|------|------|
| out | UbRpcMailboxRegions* | 非空的输出结构体，返回三段注册内存地址/字节数与环深度 |

**UbRpcMailboxRegions输出字段**

| 字段 | 类型 | 说明 |
|------|------|------|
| requests_address / requests_bytes | void* / uint64_t | 请求内存地址/字节数，对应角色2 |
| responses_address / responses_bytes | void* / uint64_t | 响应内存地址/字节数，对应角色3 |
| notify_address / notify_bytes | void* / uint64_t | 通知内存地址/字节数，对应角色4 |
| ring_depth | uint32_t | 邮箱环深度 |
| reserved | uint32_t | 保留字段，返回0 |

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

| 输出结构体 | 字段 | 类型 | 说明 |
|------------|------|------|------|
| UbShardRegion | shard_id | uint32_t | 分片号 |
| UbShardRegion | graph_address / graph_bytes | void* / uint64_t | 第0层图内存地址/字节数 |
| UbShardRegion | vector_address / vector_bytes | void* / uint64_t | 向量内存地址/字节数 |
| UbGraphPublishSnapshot | regions | const UbShardRegion* | 始终为NULL；结果从调用方传入的`out_regions`读取 |
| UbGraphPublishSnapshot | region_count | uint32_t | 返回的分片内存信息条目数 |
| UbGraphPublishSnapshot | shard_count | uint32_t | 全局分片数 |
| UbGraphPublishSnapshot | entry_shard_id | uint32_t | 全局入口点所在分片 |
| UbGraphPublishSnapshot | entry_vertex_id / entry_level | uint64_t / int32_t | 全局入口点ID/层数 |
| UbGraphPublishSnapshot | reserved | uint32_t | 保留字段，返回0 |

先用`UbSetLocalShard`确定本地分片，再读取、发布资源。内存由运行时管理，仅在对应分片保持装载期间有效；分片装载集合变化后须重新获取。

### UbValidateShardLayout

**接口定义**

```cpp
int UbValidateShardLayout(uint32_t shard_id,
                          uint32_t shard_count,
                          uint32_t entry_shard_id,
                          uint64_t entry_vertex_id);
```

**接口用途**

导入对端资源前，校验其分片数量、分片编号和入口点归属是否符合本地索引。成功返回0；该校验不能单独确认资源来自同一次索引构建。

**参数说明**

| 参数 | 类型 | 说明 |
|------|------|------|
| shard_id | uint32_t | 对端资源所属的分片号 |
| shard_count | uint32_t | 对端发布的全局分片数 |
| entry_shard_id | uint32_t | 对端发布的全局入口点所属分片 |
| entry_vertex_id | uint64_t | 对端发布的全局入口点ID |

### UbImportPeerHandles

**接口定义**

```cpp
int UbImportPeerHandles(uint32_t shard_id,
                        uint32_t role,
                        const void* rjetty_descriptor,
                        uint32_t descriptor_bytes,
                        uint32_t jetty_token,
                        void* target_segment);
```

**接口用途**

导入宿主已激活的对端资源句柄；成功返回0。运行时借用目标段，宿主须在运行时成功停止并排空前保留链路资源。

**参数说明**

| 参数 | 类型 | 说明 |
|------|------|------|
| shard_id | uint32_t | 对端分片号 |
| role | uint32_t | 资源角色，取值0至4，见下表 |
| rjetty_descriptor | const void* | 非空的对端端点描述数据 |
| descriptor_bytes | uint32_t | 描述数据字节数，须大于0 |
| jetty_token | uint32_t | 对端端点令牌 |
| target_segment | void* | 宿主提供的有效目标段指针，不能为NULL |

**资源角色说明**

| role | 资源 |
|------|------|
| 0 | topology：第0层图结构 |
| 1 | vectors：基向量 |
| 2 | distance-tasks：距离请求，对应邮箱requests段 |
| 3 | distance-results：距离响应，对应邮箱responses段 |
| 4 | distance-notify：完成通知，对应邮箱notify段 |

>![表示说明的图片](./public_sys-resources/icon-note.gif) **说明：**
>句柄由宿主管理。只有运行时成功停止并排空后，宿主才可释放链路资源；停止失败时须继续保留。

### UbRemovePeerHandles

**接口定义**

```cpp
void UbRemovePeerHandles(uint32_t shard_id, uint32_t role);
```

**接口用途**

移除暂存的指定对端资源句柄，不释放宿主持有的资源。该接口无返回值，也不会停止正在使用该资源的运行时；释放资源前仍须成功停止并排空运行时。

**参数说明**

| 参数 | 类型 | 说明 |
|------|------|------|
| shard_id | uint32_t | 对端分片号 |
| role | uint32_t | 要移除的资源角色，含义请参见[UbImportPeerHandles](#ubimportpeerhandles)中的资源角色表 |

## 启用与检索接口

### UbEnableDistributedSearch

**接口定义**

```cpp
int UbEnableDistributedSearch(char* msg, int msg_capacity);
```

**接口用途**

启动分布式检索，适用于宿主没有启动截止时间、需要读取失败原因的常规启用场景。失败时将原因写入有效的`msg`缓冲区，并更新`UbLastError`。宿主有启动期限时应使用[UbEnableDistributedSearchWithTimeout](#ubenabledistributedsearchwithtimeout)。

**参数说明**

| 参数 | 类型 | 说明 |
|------|------|------|
| msg | char* | 可选的失败原因缓冲区，可为NULL |
| msg_capacity | int | 缓冲区字节数；大于0时最多写入该容量，含字符串结束符 |

调用前须提供有效的非空URMA设备名、注入上下文、仅保留一个本地分片、创建邮箱，并导入所有远端分片的五类资源句柄。资源未就绪时，补齐资源后可重试；运行时已存在时，重复调用返回其当前状态，健康时为0。

### UbEnableDistributedSearchWithTimeout

**接口定义**

```cpp
int UbEnableDistributedSearchWithTimeout(int64_t timeout_ns);
```

**接口用途**

在调用方给定的剩余启动预算内启动分布式检索。宿主有启动截止时间（如探活窗口）时，推荐使用本接口，避免等待超过宿主预算。该变体不提供`msg`缓冲区，也不保证更新`UbLastError`，排查时应以本次返回码和宿主日志为准。

**参数说明**

| 参数 | 类型 | 说明 |
|------|------|------|
| timeout_ns | int64_t | 启动预算，单位纳秒；-1=不限时，0=立即超时，小于-1=参数错误 |

调用前须提供有效的非空URMA设备名、注入上下文、仅保留一个本地分片、创建邮箱，并导入所有远端分片的五类资源句柄。资源未就绪时，补齐资源后可重试；运行时已存在时，重复调用返回其当前状态，健康时为0。

### UbSearch

**接口定义**

```cpp
int UbSearch(const float* query, uint32_t nq, uint32_t dim, uint32_t topk,
             uint32_t ef_search, const char* metric_name,
             int64_t* out_ids, float* out_dists);
```

**接口用途**

对一批查询向量执行全局检索，同步返回top-K结果。须先成功启用分布式检索。本接口使用配置超时`query_timeout_ms`，等价于`timeout_ns=-1`的[UbSearchWithTimeout](#ubsearchwithtimeout)；宿主有请求截止时间时应使用带超时版本。

**参数说明**

| 参数 | 类型 | 说明 |
|------|------|------|
| query | const float* | 非空，`nq × dim`个连续float值，不含NaN（Not a Number，非数值）或无穷值 |
| nq | uint32_t | 本批查询向量数，范围为1至4096；更多向量须分批调用 |
| dim | uint32_t | 向量维度，须与索引一致 |
| topk | uint32_t | 每条查询返回的结果数，须大于0 |
| ef_search | uint32_t | 搜索候选集宽度，须大于或等于topk |
| metric_name | const char* | L2或IP（Inner Product，内积），须与索引一致 |
| out_ids | int64_t* | 非空，由调用方分配`nq × topk`项 |
| out_dists | float* | 非空，由调用方分配`nq × topk`项 |

**输出约定**：结果按查询顺序连续排列；不足top-K时，以`id=-1`、`distance=+inf`填充。

### UbSearchWithTimeout

**接口定义**

```cpp
int UbSearchWithTimeout(const float* query, uint32_t nq, uint32_t dim, uint32_t topk,
                        uint32_t ef_search, const char* metric_name,
                        int64_t timeout_ns, int64_t* out_ids, float* out_dists);
```

**接口用途**

按本次请求的剩余时间预算执行一批查询，同步返回全局top-K。须先成功启用分布式检索；同一批查询共用截止时间，排队和执行受配置超时与请求剩余预算共同限制。

**参数说明**

| 参数 | 类型 | 说明 |
|------|------|------|
| query | const float* | 非空，`nq × dim`个连续float值，不含NaN（Not a Number，非数值）或无穷值 |
| nq | uint32_t | 本批查询向量数，范围为1至4096；更多向量须分批调用 |
| dim | uint32_t | 向量维度，须与索引一致 |
| topk | uint32_t | 每条查询返回的结果数，须大于0 |
| ef_search | uint32_t | 搜索候选集宽度，须大于或等于topk |
| metric_name | const char* | L2或IP（Inner Product，内积），须与索引一致 |
| timeout_ns | int64_t | 剩余请求预算，单位纳秒；-1=使用配置超时，0=立即超时，小于-1=参数错误 |
| out_ids | int64_t* | 非空，由调用方分配`nq × topk`项 |
| out_dists | float* | 非空，由调用方分配`nq × topk`项 |

**输出约定**：结果按查询顺序连续排列；不足top-K时，以`id=-1`、`distance=+inf`填充。

## 释放接口

### UbStopDistributedSearch

**接口定义**

```cpp
int UbStopDistributedSearch(void);
```

**接口用途**

停止接纳查询，等待在途任务排空并停止运行时。

**参数说明**

无。

### UbShutdown

**接口定义**

```cpp
int UbShutdown(void);
```

**接口用途**

停止运行时并释放已装载索引、邮箱和暂存句柄，可在成功后重新装载。

**参数说明**

无。

两者均可重复调用。须检查返回码；停止或关闭失败时，不要释放宿主持有的URMA上下文或链路资源。

## 诊断接口

### UbPing

**接口定义**

```cpp
int UbPing(int x);
```

**接口用途**

返回`x + 1`，用于验证接口调用是否连通。

**参数说明**

| 参数 | 类型 | 说明 |
|------|------|------|
| x | int | 用于连通性验证的整数；应保证`x + 1`不超出int范围 |

### UbRpcDistanceReady

**接口定义**

```cpp
int UbRpcDistanceReady(void);
```

**接口用途**

运行时对象存在时返回1，否则返回0；不能代替健康检查。

**参数说明**

无。

### UbLastError

**接口定义**

```cpp
const char* UbLastError(void);
```

**接口用途**

返回调用线程保存的错误详情；为空时返回NULL。部分失败分支不更新详情，可能保留旧值，应以本次返回码为准。字符串由库管理，调用方不得释放，应在同线程下一次库调用前及时复制。

**参数说明**

无。

### UbRpcLayout

**接口定义**

```cpp
const char* UbRpcLayout(void);
```

**接口用途**

返回本地RPC邮箱的布局描述字符串；邮箱未创建时为空字符串。字符串由库管理，调用方不得释放，并应在同线程下次调用本接口前复制。

**参数说明**

无。

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

| 参数 | 类型 | 说明 |
|------|------|------|
| root | const std::filesystem::path& | 输出目录 |
| vectors | std::span&lt;const float&gt; | `rows × dim`个连续float值 |
| rows | std::uint64_t | 向量数，须大于0且不超过max_rows |
| dim | std::uint32_t | 向量维度，须大于0 |
| options | const ReferenceBuildOptions& | 构建参数，见下表 |
| max_rows | std::uint64_t | 允许的最大向量数，默认100000；超出时返回错误 |

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

### WriteGlobalIndexFile

**接口定义**

```cpp
Status WriteGlobalIndexFile(const std::filesystem::path& root,
                                          const GlobalIndexMeta& meta,
                                          std::span<const ShardRange> ranges);
```

**接口用途**

写入`global.meta`，保存全局元信息与分片范围。返回`Status`。

**参数说明**

| 参数 | 类型 | 说明 |
|------|------|------|
| root | const std::filesystem::path& | 索引三元组目录 |
| meta | const GlobalIndexMeta& | 待写入的全局元信息 |
| ranges | std::span&lt;const ShardRange&gt; | 覆盖全部分片的范围表 |

### LoadGlobalIndexFile

**接口定义**

```cpp
Result<GlobalIndexFile> LoadGlobalIndexFile(const std::filesystem::path& root);
```

**接口用途**

读取并校验`global.meta`。成功返回包含全局元信息和分片范围的`Result<GlobalIndexFile>`。

**参数说明**

| 参数 | 类型 | 说明 |
|------|------|------|
| root | const std::filesystem::path& | 索引三元组目录 |

### WriteUpperGraphFile

**接口定义**

```cpp
Status WriteUpperGraphFile(const std::filesystem::path& root,
                                         const GlobalIndexMeta& meta,
                                         std::span<const UpperNodeMeta> nodes,
                                         std::span<const node_id_t> neighbors);
```

**接口用途**

写入`upper_graph.index`，保存上层节点元信息与邻接关系。返回`Status`。

**参数说明**

| 参数 | 类型 | 说明 |
|------|------|------|
| root | const std::filesystem::path& | 索引三元组目录 |
| meta | const GlobalIndexMeta& | 与该图一致的全局元信息 |
| nodes | std::span&lt;const UpperNodeMeta&gt; | 上层节点元信息 |
| neighbors | std::span&lt;const node_id_t&gt; | 上层邻接表中的节点ID |

### LoadUpperGraphFile

**接口定义**

```cpp
Result<UpperGraphReplica> LoadUpperGraphFile(const std::filesystem::path& root,
                                                           const GlobalIndexMeta& expected_meta);
```

**接口用途**

读取并校验上层图。成功返回`Result<UpperGraphReplica>`。

**参数说明**

| 参数 | 类型 | 说明 |
|------|------|------|
| root | const std::filesystem::path& | 索引三元组目录 |
| expected_meta | const GlobalIndexMeta& | 用于校验上层图的预期全局元信息 |

### WriteShardFile

**接口定义**

```cpp
Status WriteShardFile(const std::filesystem::path& root,
                                    const GlobalIndexMeta& meta, const ShardRange& range,
                                    std::span<const node_id_t> layer0,
                                    std::span<const float> vectors);
```

**接口用途**

写入一个`shard_N.index`，保存该分片的第0层邻接表与向量。返回`Status`。

**参数说明**

| 参数 | 类型 | 说明 |
|------|------|------|
| root | const std::filesystem::path& | 索引三元组目录 |
| meta | const GlobalIndexMeta& | 与该分片一致的全局元信息 |
| range | const ShardRange& | 分片编号、起始ID与向量数 |
| layer0 | std::span&lt;const node_id_t&gt; | 第0层邻接表，含各节点的M0个邻居槽位 |
| vectors | std::span&lt;const float&gt; | 按行连续存储的本分片向量 |

### LoadShardFile

**接口定义**

```cpp
Result<LocalShard> LoadShardFile(const std::filesystem::path& root,
                                               const GlobalIndexMeta& expected_meta,
                                               const ShardRange& expected_range,
                                               LocalShardStorage storage =
                                                   LocalShardStorage::kPrivate);
```

**接口用途**

读取并校验一个分片的第0层图与向量。成功返回`Result<LocalShard>`。

**参数说明**

| 参数 | 类型 | 说明 |
|------|------|------|
| root | const std::filesystem::path& | 索引三元组目录 |
| expected_meta | const GlobalIndexMeta& | 预期全局元信息 |
| expected_range | const ShardRange& | 预期分片范围 |
| storage | LocalShardStorage | 存储方式，默认`kPrivate`；`kShared`使用共享内存 |

### LoadIndexForShard

**接口定义**

```cpp
Result<LoadedIndex> LoadIndexForShard(const std::filesystem::path& root,
                                                   shard_id_t shard_id,
                                                   LocalShardStorage storage =
                                                       LocalShardStorage::kPrivate);
```

**接口用途**

装载指定分片检索所需的全局元信息、分片映射、上层图及本分片数据。成功返回`Result<LoadedIndex>`。

**参数说明**

| 参数 | 类型 | 说明 |
|------|------|------|
| root | const std::filesystem::path& | 索引三元组目录 |
| shard_id | shard_id_t | 要装载的有效分片号 |
| storage | LocalShardStorage | 本分片数据的存储方式，默认`kPrivate` |

### ShardFilePath

**接口定义**

```cpp
std::filesystem::path ShardFilePath(const std::filesystem::path& root,
                                                  shard_id_t shard_id);
```

**接口用途**

返回`root`下指定`shard_N.index`的文件路径，不检查文件是否存在。

**参数说明**

| 参数 | 类型 | 说明 |
|------|------|------|
| root | const std::filesystem::path& | 索引三元组目录 |
| shard_id | shard_id_t | 用于生成文件名的分片号 |

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

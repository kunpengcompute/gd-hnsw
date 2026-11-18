# 特性介绍

本文档介绍gd_hnsw（代码命名空间`hnsw_urma`）的系统架构、核心算法原理与关键特性，帮助三方开发者理解其工作机理并正确使用。

gd_hnsw是一套基于URMA（Unified Remote Memory Access，统一远端内存访问）的全局分布式HNSW（Hierarchical Navigable Small World，分层可导航小世界）检索运行时：多个进程共同持有一张覆盖全部数据的HNSW全局图（按分片存储），任一进程可发起查询，沿图遍历时通过URMA READ直接读取远端进程的图结构与向量，并可将距离计算下推（pushdown）到数据所在进程就近执行。核心以C++20实现，跨节点通信基于UMDK（Unified Memory Development Kit，统一内存开发工具包）/URMA；独立基准的集群协同基于MPI（Message Passing Interface，消息传递接口）（Milvus集成不依赖MPI，资源发现走Milvus目录机制）。

## 架构介绍

以下从分层结构、功能模块和部署模型三方面介绍当前架构。

### 分层结构

```text
┌──────────────────────────────────────────────────────────┐
│ 接入层    ub_c_api.h（C ABI，面向Milvus CGO）/ 基准与构建应用 │
├──────────────────────────────────────────────────────────┤
│ 运行时层  runtime（config/thread_group/scheduler/topology/ │
│           hybrid_distance/load_balancing/drain/rpc）       │
│ 搜索层    search（query_context/query_slot_pool/io/         │
│           local_io/remote_io/read_resources）              │
│ 传输层    transport（urma_transport/read_batch_lane/        │
│           shared_memory_transport/urma_resources）         │
├──────────────────────────────────────────────────────────┤
│ 核心层    core（status/types/shard_map/visited_table/       │
│           minimax_heap/query_key/page_aligned_buffer/ring） │
│ 索引层    index（format/builder）  计算层 compute（dataset/ │
│           distance）  协议层 protocol（distance_batch_wire） │
├──────────────────────────────────────────────────────────┤
│ 底层依赖  UMDK/URMA / MPI / Faiss（离线图构建）/ OpenMP      │
└──────────────────────────────────────────────────────────┘
```

- **接入层**：`ub_c_api.h`以纯C ABI（Application Binary Interface，应用二进制接口）暴露运行时能力（配置、装载、资源发布/导入、启用、检索、释放），供Milvus等宿主经CGO调用；索引构建与检索基准以独立应用形式提供。
- **运行时层**：五类线程的编排与调度（详见"线程模型"），以及距离策略（`HybridDistanceOracle`）、负载均衡、优雅退出（drain）。
- **搜索层**：查询任务（`HnswSearchTask`）的搜索状态机、查询池（`QuerySlotPool`）、本地/远端I/O路径。
- **传输层**：URMA READ/WRITE_NOTIFY的封装与批处理（lane），以及节点内共享内存传输。
- **核心层**：`ShardMap`分片映射、`VisitedTable`访问表、`MinimaxHeap`双堆、无锁环（SPSC（Single Producer Single Consumer，单生产者单消费者）/MPMC（Multi Producer Multi Consumer，多生产者多消费者））、页对齐缓冲区、统一`Status`/`Result`返回。
- **索引层**：索引文件格式（format V2）与离线构建接口（builder，含Faiss适配）。

### 功能模块

**表 1** gd_hnsw功能模块

| 模块名称 | 功能描述 |
|----------|----------|
| ShardMap（core） | 分片映射：把全局节点ID转换为"所属分片+分片内编号"，支撑数据直接寻址 |
| UpperGraphReplica（index） | 上层图副本：每个进程物化一份完整上层图邻接关系，本地读取任意上层节点邻居 |
| LocalShard（index） | 本分片数据：第0层邻接表与原始向量，含向量范数缓存与预取 |
| HnswSearchTask（search） | 有状态搜索任务：维护结果集合R、候选集C与已展开集合，逐轮"取邻接表→算距离→更新集合" |
| QuerySlotPool（search） | 查询池：固定数量的查询位置（`QuerySlot`），由`max_active_queries`控制并发查询占用 |
| HybridDistanceOracle（runtime） | 距离策略适配器：按目标分片与策略（auto/pull/pushdown）为每组邻居选择执行方式 |
| Scheduler（runtime） | 查询接纳与分配：准入队列、就绪通知、重试与超时管理（`query_timeout_ms`/`operation_timeout_ms`） |
| Distance线程（runtime） | 距离计算服务：处理其他进程下推的距离请求，使用本地向量计算指定ID的距离 |
| READ/RPC Progress（transport/runtime） | 通信线程：推进远端邻接表/向量的URMA READ，收发RPC（Remote Procedure Call，远程过程调用）距离请求与结果（WRITE_NOTIFY） |
| RpcMailbox（runtime/rpc） | RPC邮箱：requests/responses/notify三块注册内存段，承载距离任务与结果队列 |
| index format V2（index） | 索引文件格式：4KB定长头+数据块，含校验；三类文件组成"三元组"（见"索引文件组织"） |
| builder（index） | 离线构建：`BuildReferenceIndex`（小规模校验用）与Faiss适配构建（生产，支持流式） |
| benchmark（runtime） | 检索基准：召回率、QPS（Queries Per Second，每秒查询数）、时延分位（P50/P95/P99）与详细运行时画像计数 |

### 部署模型

gd_hnsw采用对等部署：**一个进程负责一个数据分片**（one worker owns one shard）。

- 每个进程运行相同的程序，加载自己的分片（`LocalShard`）并持有完整的上层图副本与全局元信息。
- 任一进程都可以接纳查询并维护该查询的全部搜索状态（结果集合R、候选集C、已展开集合E）；遇到跨分片边时，通过URMA访问邻居所在进程的数据，继续同一次搜索。
- 分片边界只决定数据在哪里，不限制图搜索的范围；一次查询可以访问多个乃至全部分片，但各分片无需分别完成一次本地top-k检索。
- 进程内线程按五类划分（见"线程模型"），既执行分配给自己的查询，也为其他进程提供数据读取和距离计算服务。

## 算法原理

### HNSW算法概述

HNSW是一种基于多层图的近似最近邻搜索算法。

- **多层图结构**：上层稀疏、下层稠密（第0层包含全部节点）。查询从入口点出发，逐层贪心下探到第0层，在第0层做beam search（`ef_search`宽度的候选集维护）。
- **关键参数**（`ReferenceBuildOptions`）
  - `M`：上层每层最大邻居数（默认16）。
  - `M0`：第0层最大邻居数（默认32）。
  - `ef_construction`：建图候选宽度，越大图质量越高、构建越慢（Faiss构建时传入）。
  - `ef_search`：查询候选宽度（索引默认64），越大召回越高、时延越高。
- **距离度量**：支持L2（L2 Norm，欧氏距离）与IP（Inner Product，内积）两类；构建选项`metric`取值0=L2、1=负内积。运行时可选dot+norm优化路径（`use_dot_norm`）：以‖q‖²+‖x‖²−2q·x计算平方L2距离，仅限L2度量，不是IP开关。

### 全局图索引与分片基线对比

传统分片式检索将数据划分为多个分片，各分片独立建图、独立检索，再由协调节点汇聚重排序。gd_hnsw则先在全部数据上构建一张完整的HNSW图，再将图分片存储，**保留跨分片的图边**。

**表 2** 分片基线方案与gd_hnsw全局图方案对比

| 对比项 | 分片基线方案 | gd_hnsw全局图方案 |
|--------|--------------|-------------------|
| 索引组织 | 先划分数据，各分片独立建立局部图 | 在全部数据上构图，再将图分片存储 |
| 图的连接范围 | 每张局部图只连接本分片内的向量 | 保留不同分片之间的连接，逻辑上仍是一张完整图 |
| 一次查询 | 在各局部图分别搜索，再汇聚结果 | 沿同一张全局图跨分片搜索，生成一份结果 |
| 集群总距离计算量（规模模型） | P·log(N/P) | log N |

以距离计算次数衡量搜索工作量。设数据集包含N个向量，均匀分布在P个分片上，图搜索工作量随数据规模按对数增长，则独立分片索引的总计算量按`P·log(N/P)`增长，高于全局图的`log N`。图检索工作量随数据规模次线性增长，数据缩小为1/P并不能让单次局部搜索同比缩小，而查询仍要执行P次，因而放大了总计算量。全局图保留跨分片导航路径，一次搜索可持续向更近区域推进，避免各局部图分别导航的额外开销。

>![表示说明的图片](./public_sys-resources/icon-note.gif) **说明：**
>上述对数关系用于说明计算量放大的原因，不能直接换算为并行查询耗时，实际比较应采用相同的目标召回率。

### 距离下推与混合距离策略

远端邻居的距离有两种取得方式：**向量拉取（pull）**把远端向量经URMA READ取回本进程计算；**距离下推（pushdown）**把距离计算任务发送到向量所在进程，由其Distance线程就地计算后回传"ID+距离"。

`HybridDistanceOracle`按组选择执行方式，策略由`distance_policy`配置：

| 策略 | 行为 |
|------|------|
| `pull` | 远端分组一律拉回向量，本进程Search线程计算 |
| `pushdown` | 远端分组一律下推到数据所在进程计算 |
| `auto`（默认） | 该分片本组ID数量小于`pull_threshold`（默认4）时拉取，否则下推 |

同一轮可以同时包含本地计算、远端拉取和远端下推；Search线程先发布远端任务再计算本地分组，使本地计算与远端通信重叠。结果集合已满时，下推请求还会携带当前最差结果距离作为上界，远端只回传距离不大于该上界的ID与距离，减少响应数据量。

**带宽估算**

按10亿条、1024维float32向量、2节点均分数据估算。假设节点A每秒处理1,000次查询（QPS），每次计算1万条向量，其中A、B各有5,000条。

**表 3** 向量拉取与距离下推的开销对比（1B×1024维、2节点、1,000 QPS）

| 指标 | 拉回向量本地计算 | 距离下推就近计算 |
|------|------------------|------------------|
| 每次查询的距离计算 | A：10,000次；B：0次 | A：5,000次；B：5,000次 |
| 每次查询的跨节点传输 | 5,000×1024×4B ≈ 0.02048GB | 请求+响应 ≈ 0.000064GB |
| 1,000 QPS所需有效带宽 | 20.48GB/s | 约0.064GB/s |

距离下推不减少需要评价的图节点数量，主要收益是大幅减少原始向量的跨节点传输，并分担查询进程的搜索线程负担。

### 线程模型

每个进程运行五类线程，数量按进程配置（`RuntimeConfig`）：

**表 4** 五类线程职责

| 线程 | 主要职责 |
|------|----------|
| Control | 接纳查询、分配Search线程，管理活动查询数量（`max_active_queries`）与超时，回收查询资源；固定1个 |
| Search | 推进查询、维护搜索状态，计算本地及拉回向量的距离，生成最终结果 |
| Distance | 处理其他进程的距离计算请求，使用本地向量计算指定ID的距离 |
| READ Progress | 推进远端邻接表与向量的URMA READ，并参与查询阶段的就绪通知 |
| RPC Progress | 发送本进程的距离请求、接收返回结果，并把本进程Distance的计算结果发回请求方 |

设当前进程可用逻辑CPU（Central Processing Unit，中央处理器）数为C，除Control固定1个外，其余四类默认`auto`（0），按以下规则解析（`ResolveRuntimeConfig`）：

| 线程 | auto解析规则 |
|------|--------------|
| Distance | max(1, ⌊C/6⌋) |
| READ Progress | max(1, min(2, ⌊C/24⌋)) |
| RPC Progress | max(1, min(4, ⌊C/12⌋)) |
| Search | 扣除Distance、READ/RPC Progress及预留（max(1, ⌊C/12⌋)，含Control）后的剩余数量 |

线程支持CPU绑定（`cpu_bind_mode`：auto/none）、拓扑布局（`cpu_layout`：auto/contiguous/llc-isolated/llc-spread）与进度模式（`progress_mode`：busy自旋/adaptive休眠）。查询等待远端操作时，Search线程可以继续处理其他就绪查询，单条查询的搜索状态始终由同一个Search线程维护。

### 索引文件组织

索引以"三元组"形式组织在一个目录下（format V2，4KB定长头+数据块，含校验）：

| 文件 | 内容 | 查询中的用途 |
|------|------|--------------|
| `global.meta` | 全局元信息（`GlobalIndexMeta`：ntotal/dim/metric/M/M0/ef_search默认值/入口点/最大层数/分片数）与各分片的ID范围 | 确定图参数与全局ID的分片归属 |
| `upper_graph.index` | 全局上层图（第1层及以上）的完整邻接关系 | 每个进程各物化一份副本，本地读取任意上层节点邻居 |
| `shard_N.index` | 分片N的第0层邻接表与原始向量 | 展开节点时读邻居；计算距离时读向量 |

每个分片覆盖一段连续的全局ID。查询通过`ShardMap`把全局ID转换为"所属分片+分片内编号"，即可直接计算本地或远端内存中的数据位置，无需先询问远端。

### URMA注册内存

每个分片以五种角色发布注册内存，构成可被远端导入的内存描述符集合（角色编号见`MilvusUbRuntime::ImportPeerHandles`与`ub_c_api.h`）：

**表 5** URMA五块注册内存

| 角色 | 内容 | 用途 |
|------|------|------|
| 0 topology | 图结构（第0层邻接表） | 远端URMA READ读取邻居 |
| 1 vectors | 基向量数据 | pull策略下远端READ取向量 |
| 2 distance-tasks | 距离任务队列（requests段） | 接收远端下推的距离计算请求 |
| 3 distance-results | 距离结果队列（responses段） | 回传距离计算结果 |
| 4 distance-notify | 完成通知（notify段） | WRITE_NOTIFY信用与就绪通知 |

宿主（如Milvus QueryNode）将本地分片的图/向量/邮箱内存注册进UMDK，经etcd等资源目录发布；对端解析目录记录、激活链路并把句柄导入运行时后，即可在搜索时直接READ/WRITE对端内存。

## 关键特性

- **全局图索引**：在全部数据上构建一张HNSW图再分片存储，保留跨分片边；一次查询沿全局图搜索，直接产出全局top-K，无需各分片独立召回再归并。
- **混合距离策略**：pull/pushdown/auto三档，auto按组内ID数量阈值（默认4）自动选择；支持组内排序（`layer0_sort_mode`）与结果上界过滤，降低传输量。
- **URMA数据面**：远端邻接表/向量经URMA READ单边读取，距离任务经WRITE_NOTIFY投递，配合五块注册内存与资源目录实现零拷贝跨节点访问。
- **多线程并行与查询间重叠**：五类线程分工，通信与计算重叠；查询等待远端操作时Search线程切换处理其他查询。
- **并发资源管理**：查询池（`max_active_queries`，默认16）限制在途查询；查询级超时（默认30s）与操作级超时（默认5s）保障长尾可控；支持优雅退出（drain）。
- **可观测性**：基准输出召回率、QPS与时延分位（P50/P95/P99），并可导出分阶段耗时与队列水线等详细画像（JSON（JavaScript Object Notation，JavaScript对象表示法）/CSV（Comma-Separated Values，逗号分隔值））。
- **Milvus一体化接入**：经C ABI（`ub_c_api.h`）接入Milvus QueryNode，由`ub.TrySearch`分流闸门按开关接管检索请求；开启全局索引时UB（Unified Bus，统一总线）查询失败直接返回错误；关闭全局索引时，本地UB执行失败（包括计划解析失败）可回退原生链路，转发等前置步骤报错仍直接返回。

## 性能规格

以下为设计目标（需求规格），非承诺实测值。

**表 6** gd_hnsw性能规格（设计目标）

| 规格项 | 目标 |
|--------|------|
| 吞吐（QPS） | 同等算力下，全局索引相比32分片索引提升100% |
| 查询时延 | 小于50ms |
| 召回率 | 大于0.95 |
| 验证环境 | 鲲鹏超节点电组网4节点（KP950+），openEuler 24.03 LTS（Long Term Support，长期支持） SP3 |

## 缩略语

| 缩略语 | 英文全称 | 中文全称 |
|--------|----------|----------|
| ABI | Application Binary Interface | 应用二进制接口 |
| CPU | Central Processing Unit | 中央处理器 |
| CSV | Comma-Separated Values | 逗号分隔值 |
| HNSW | Hierarchical Navigable Small World | 分层可导航小世界 |
| IP | Inner Product | 内积 |
| JSON | JavaScript Object Notation | JavaScript对象表示法 |
| L2 | L2 Norm | 欧氏距离 |
| LTS | Long Term Support | 长期支持 |
| MPI | Message Passing Interface | 消息传递接口 |
| MPMC | Multi Producer Multi Consumer | 多生产者多消费者 |
| OpenMP | Open Multi-Processing | 开放多处理 |
| QPS | Queries Per Second | 每秒查询数 |
| RPC | Remote Procedure Call | 远程过程调用 |
| SPSC | Single Producer Single Consumer | 单生产者单消费者 |
| UB | Unified Bus | 统一总线 |
| UMDK | Unified Memory Development Kit | 统一内存开发工具包 |
| URMA | Unified Remote Memory Access | 统一远端内存访问 |

## 修订记录

| 文档版本 | 发布日期 | 修改说明 |
|------|----------|------------|
| 01 | 2026-12-30 | 第一次正式发布。 |

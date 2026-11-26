# 版本说明书

本文档提供gd_hnsw每个发布版本的基础信息、配套环境与特性更新信息。

## 版本配套说明

### 产品版本信息

| 项目 | 说明 |
| --- | --- |
| 产品名称 | Kunpeng BoostKit |
| 产品版本 | 26.3.0 |
| 软件名称 | gd_hnsw |
| 软件版本 | V1.0.0 |

### 与操作系统、编译器和处理器配套说明

**表 1** gd_hnsw发布验证环境

| 操作系统 | 处理器类型 | 编译器 | CMake版本号 |
| --- | --- | --- | --- |
| openEuler 24.03 LTS SP3 | 鲲鹏950处理器 | GCC 12.3 | CMake 3.22及以上版本 |

>![表示说明的图片](./public_sys-resources/icon-note.gif) **说明。**
>
> - gd_hnsw跨节点数据面依赖UMDK/URMA，需在配备URMA设备的鲲鹏环境部署，并以`ENABLE_URMA=ON`构建算法库；未启用URMA数据面的构建无法提供分布式检索服务。
> - 上表列出本版本发布时的验证环境及构建工具要求；UB相关组件与驱动版本请参见《[安装指南](./installation_guide.md#安装ub相关组件与驱动)》。

## 版本使用注意事项

### 使用注意事项

请参见《[最佳实践](./best_practices.md)》中的"[部署Checklist](./best_practices.md#部署checklist)"。

## V1.0.0

### 更新说明

**新增特性**

- 提供基于灵衢统一总线URMA通信能力的分布式HNSW全局检索算法：在全部数据上构建一张全局图再分片存储，保留跨分片图边，一次查询直接产出全局Top-K。
- 提供URMA数据面：远端邻接表/向量经URMA READ单边读取，距离任务经WRITE_NOTIFY投递；每个分片以五种角色发布注册内存（topology/vectors/distance-tasks/distance-results/distance-notify）。
- 提供距离下推（pushdown）：远端分片向量的距离计算下发到数据所在进程就近执行，配合组内ID排序与结果上界过滤，减少跨节点传输。
- 提供五类线程运行时：Control/Search/Distance/READ Progress/RPC（Remote Procedure Call，远程过程调用）Progress，数量支持auto解析（按可用CPU（Central Processing Unit，中央处理器）的内置公式），支持CPU绑定、拓扑布局与进度模式配置。
- 提供并发资源管理：查询池（`max_active_queries`）、查询级与操作级超时、优雅退出（drain）。
- 提供索引文件format V2与离线构建：三元组组织（global.meta/upper_graph.index/shard_N.index）；生产环境大规模构建由宿主侧构建组件承担，本库提供确定性参考构建。
- 提供运行时画像能力：分阶段耗时与队列水线等画像计数，用于定位瓶颈在搜索、通信还是距离服务。
- 提供C ABI（Application Binary Interface，应用二进制接口）（`ub_c_api.h`）与Milvus一体化接入：由接入层按开关分流接管裸向量检索请求；接入行为细节请参见Milvus侧资料。

### 已知限制

- 跨节点数据面依赖UMDK/URMA；目标平台为鲲鹏AArch64，编译环境需支持C++20。
- 采用“一个进程一个分片”的对等部署：参与同一张全局图的算法工作进程数须等于全局图分片数。Milvus接入时对应持有该图分片的工作QueryNode，不包含Proxy、协调组件和其他数据处理组件；快速入门使用8个工作QueryNode承载8个全局图分片。
- 索引须在检索前完整构建，暂不支持对已加载的全局图在线增删向量。Milvus接入流程为先导入数据，再通过`create_index`构建全局索引，完成加载后检索；更换数据或图时需按《[最佳实践](./best_practices.md#配置变化和复测状态处理)》处理原集合和运行时状态。
- Milvus接入链路当前仅支持针对单个FloatVector字段的裸向量检索，一次请求可包含多条查询向量；不支持标量过滤/group-by/迭代器，`search_params`仅接受声明的字段，且要求`0 < topk <= ef`。请求字段`ef`对应本库接口参数`ef_search`。
- 底层索引与C ABI的距离度量支持L2（L2 Norm，欧氏距离）与IP（Inner Product，内积）两类。

#### 约束说明

| 场景 | 约束说明 |
| --- | --- |
| 重复导入数据、将不同业务场景的数据混合存储 | 算法运行时采用进程级单例设计，不支持多 Collection 并存，且不支持动态在线增量索引插入 |
| URMA 运行环境发生故障（设备或链路异常） | 算法运行时无故障恢复与断点续跑能力，环境故障后无法接续运行，须重新初始化并全部重新运行 |
| 查询执行中算法内部进入异常分支（如传输层 fatal 错误、协议错误） | 运行时无自愈能力：fatal 状态置位后不可清除、出错的工作线程直接退出，查询无法自动恢复，须重启运行时恢复 |
| QueryNode 空载（无查询任务）运行 | 算法运行时线程默认采用忙轮询（busy 模式）以降低查询时延，空载仍占用 CPU |

## 缩略语

| 缩略语 | 英文全称 | 中文全称 |
| --- | --- | --- |
| ABI | Application Binary Interface | 应用二进制接口 |
| ARM | Advanced RISC Machine | 高级精简指令集机器 |
| CMake | Cross-platform Make | 跨平台构建工具 |
| CPU | Central Processing Unit | 中央处理器 |
| GCC | GNU Compiler Collection | GNU编译器套件 |
| HNSW | Hierarchical Navigable Small World | 分层可导航小世界 |
| IP | Inner Product | 内积 |
| L2 | L2 Norm | 欧氏距离 |
| LTS | Long Term Support | 长期支持 |
| RPC | Remote Procedure Call | 远程过程调用 |
| UMDK | Unified Memory Development Kit | 统一内存开发工具包 |
| URMA | Unified Remote Memory Access | 统一远端内存访问 |

## 版本配套文档

### V1.0.0版本配套文档

| 文档名称 | 内容简介 | 交付形式 |
| --- | --- | --- |
| 《版本说明书》 | 本文档提供gd_hnsw的版本发布信息。 | 开源仓 |
| 《快速入门》 | 本文档提供gd_hnsw的快速入门指导。 | 开源仓 |
| 《安装指南》 | 本文档提供gd_hnsw编译安装指导。 | 开源仓 |
| 《API参考》 | 本文档提供gd_hnsw对外接口定义、接口说明。 | 开源仓 |
| 《最佳实践》 | 本文档提供gd_hnsw使用的实践案例。 | 开源仓 |
| 《特性介绍》 | 本文档提供gd_hnsw架构介绍、核心算法原理及特性优化说明。 | 开源仓 |
| 《常见问题和故障案例》 | 本文档汇总gd_hnsw使用过程中的常见问题与解答。 | 开源仓 |

### 获取文档的方法

您可以通过访问[gd_hnsw仓库](https://gitcode.com/boostkit/gd-hnsw)浏览和获取相关文档。

## 修订记录

| 文档版本 | 发布日期 | 修改说明 |
| --- | --- | --- |
| 01 | 2026-12-30 | 第一次正式发布。 |

# gd_hnsw全局分布式检索介绍

## 最新消息

\[2026-12-30\]：第一次正式发布。提供基于URMA（Unified Remote Memory Access，统一远端内存访问）的全局分布式检索能力。

## 项目介绍

**gd_hnsw**是一个面向鲲鹏ARM（Advanced RISC Machine，高级精简指令集机器）架构的**全局分布式检索算法**。相较于传统分片检索的方式（各分片独立检索再汇聚，总计算量随之放大），该算法在全部数据上构建一张HNSW（Hierarchical Navigable Small World，分层可导航小世界）全局图再按分片存储，保留跨分片图边；利用灵衢统一总线的高带宽、低时延远端内存访问能力实现分布式全局检索，减少计算量提升吞吐。

核心设计思想如下。

- **一个进程一个分片**：各进程对等部署，任一进程可发起查询并维护该查询的全部搜索状态。
- **全局图跨分片搜索**：沿同一张全局图推进，无需各分片分别完成一次本地top-K再归并。
- **URMA数据面**：远端邻接表/向量经URMA READ单边读取；每个分片以五种角色发布注册内存（topology/vectors/distance-tasks/distance-results/distance-notify）。
- **距离下推（pushdown）**：远端分片向量的距离计算可下发给**拥有该分片的远端进程**就近执行，只回传ID与距离；支持pull/pushdown/auto三种策略。
- **五类线程运行时**：Control/Search/Distance/READ Progress/RPC（Remote Procedure Call，远程过程调用） Progress，通信与计算重叠，查询等待远端操作时Search线程切换处理其他查询。
- **Milvus一体化接入**：经C ABI（Application Binary Interface，应用二进制接口）（`ub_c_api.h`）接入Milvus QueryNode，由接入层按开关分流接管检索请求；当前仅支持裸向量检索；接入行为细节请参见Milvus侧资料。

底层HNSW图基于[Faiss](https://github.com/facebookresearch/faiss)构建。

## 目录结构

代码仓目录结构如下（算法库代码随版本交付合入）。

```text
gd-hnsw/
├─ gd_hnsw_urma/                            // 算法库（独立CMake交付，产出libub_hnsw.so）
│  ├─ include/hnsw_urma/                    // 公共头文件（C++20）
│  │  ├─ compute/                           // 数据集加载（bin）与距离计算（L2/IP；dot+norm为L2优化路径）
│  │  ├─ control/                           // 集群控制面
│  │  ├─ core/                              // 基础组件：status/types/shard_map/visited_table/
│  │  │                                     //   minimax_heap/query_key/page_aligned_buffer/spsc_ring/mpmc_ring
│  │  ├─ index/                             // 索引文件format V2（三元组）与参考构建接口
│  │  ├─ protocol/                          // 距离批量请求/响应线格式（distance_batch_wire）
│  │  ├─ runtime/                           // 运行时：config/thread_group/scheduler/topology/
│  │  │                                     //   hybrid_distance/load_balancing/drain/rpc
│  │  ├─ search/                            // 搜索：query_context/query_slot_pool/io/
│  │  │                                     //   local_io/remote_io/read_resources/search
│  │  └─ transport/                         // 传输：URMA transport/read_batch_lane/
│  │                                        //   shared_memory_transport/urma_resources
│  ├─ src/                                  // 上述各层实现，及Milvus接入层（milvus_*）、C ABI（ub_c_api.cpp）
│  ├─ CMakeLists.txt                        // 独立构建入口
│  ├─ ub_hnsw-config.cmake.in               // find_package(ub_hnsw)包配置模板
│  └─ ub_hnsw.pc.in                         // pkg-config模板
├─ docs/                                    // 文档
│  ├─ zh/                                   // 中文文档
│  │  ├─ api_reference.md                   // API参考
│  │  ├─ best_practices.md                  // 最佳实践
│  │  ├─ FAQ.md                             // 常见问题（FAQ）
│  │  ├─ feature_introduction.md            // 特性介绍
│  │  ├─ installation_guide.md              // 安装指南
│  │  ├─ quick_start.md                     // 快速入门
│  │  └─ release_notes.md                   // 版本说明书
├─ LICENSE                                  // 许可证
└─ README.md                                // 项目介绍
```

## 版本说明

关于gd_hnsw的版本更新情况请参见《[版本说明书](./docs/zh/release_notes.md)》。

## 学习文档

<table><thead align="left"><tr><th valign="top" width="20%">学习资源名称</th>
<th valign="top" width="80%">学习资源简介</th>
</tr>
</thead>
<tbody>
<tr><td valign="top"><a href="./docs/zh/feature_introduction.md">特性介绍</a></td>
<td valign="top">提供gd_hnsw架构介绍、核心算法原理及特性说明。</td>
</tr>
<tr><td valign="top"><a href="./docs/zh/release_notes.md">版本说明书</a></td>
<td valign="top">提供gd_hnsw每个发布版本的基础信息、配套环境与特性更新信息。</td>
</tr>
<tr><td valign="top"><a href="./docs/zh/quick_start.md">快速入门</a></td>
<td valign="top">提供gd_hnsw快速入门指导，含环境准备、索引构建与Milvus接入示例。</td>
</tr>
<tr><td valign="top"><a href="./docs/zh/installation_guide.md">安装指南</a></td>
<td valign="top">提供gd_hnsw依赖安装与编译方法指导。</td>
</tr>
<tr><td valign="top"><a href="./docs/zh/api_reference.md">API参考</a></td>
<td valign="top">提供gd_hnsw对外C ABI与C++接口定义、参数说明与返回值。</td>
</tr>
<tr><td valign="top"><a href="./docs/zh/best_practices.md">最佳实践</a></td>
<td valign="top">提供使用gd_hnsw的部署Checklist与调优建议。</td>
</tr>
<tr><td valign="top"><a href="./docs/zh/FAQ.md">常见问题（FAQ）</a></td>
<td valign="top">汇总使用gd_hnsw过程中的常见问题与解答。</td>
</tr>
</tbody>
</table>

## 免责声明

本项目面向鲲鹏ARM架构进行优化，底层HNSW图构建基于开源Faiss实现（调用faiss::IndexHNSWFlat建图后提取图结构）。软件的任何漏洞与安全问题，凡涉及Faiss上游部分均由相应的上游社区根据其漏洞和安全响应机制解决，请密切关注上游社区发布的通知和版本更新。本项目对软件的漏洞及安全问题不承担任何责任，使用者需自行评估并承担在自身生产环境中使用本软件的风险。

## License

gd_hnsw采用Mulan PSL（Mulan Permissive Software License，木兰宽松许可证） v2许可证授权，支持修改代码和再开源，具体请参见[LICENSE](./LICENSE)文件。

本项目的文档适用CC-BY（Creative Commons Attribution，知识共享署名） 4.0许可证，具体请参见 [LICENSE](./docs/LICENSE)文件。

## 贡献声明

欢迎大家为社区做贡献，如果使用过程中有任何问题/建议，或者需要反馈特性需求和bug报告，可以提交Issue联系我们，具体贡献方法可参考[这里](https://gitcode.com/boostkit/community/blob/master/docs/contributor/contributing.md)。同时也欢迎在[讨论专区](https://gitcode.com/boostkit/gd-hnsw/discussions)展开讨论交流。感谢您的支持。

## 致谢

gd_hnsw在以下方面受益于开源社区。

- [Faiss](https://github.com/facebookresearch/faiss) —— 提供HNSW索引构建能力。

感谢来自社区的每一个PR（Pull Request，拉取请求），欢迎贡献gd_hnsw！

## 缩略语

| 缩略语 | 英文全称 | 中文全称 |
|--------|----------|----------|
| ABI | Application Binary Interface | 应用二进制接口 |
| ARM | Advanced RISC Machine | 高级精简指令集机器 |
| CC-BY | Creative Commons Attribution | 知识共享署名 |
| HNSW | Hierarchical Navigable Small World | 分层可导航小世界 |
| IP | Inner Product | 内积 |
| L2 | L2 Norm | 欧氏距离 |
| PR | Pull Request | 拉取请求 |
| PSL | Mulan Permissive Software License | 木兰宽松许可证 |
| RPC | Remote Procedure Call | 远程过程调用 |
| URMA | Unified Remote Memory Access | 统一远端内存访问 |

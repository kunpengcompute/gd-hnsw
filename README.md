# gd_hnsw全局分布式检索介绍

本文介绍gd_hnsw的核心设计、代码目录、学习文档及开源协作信息。

## 最新消息

\[2026-12-30\]：gd_hnsw提供基于URMA（Unified Remote Memory Access，统一远端内存访问）的全局分布式检索能力。

## 项目介绍

**gd_hnsw**是一个面向鲲鹏ARM架构的**全局分布式检索算法**。相较于传统分片检索的方式（各分片独立检索再汇聚，总计算量随之放大），该算法在全部数据上构建一张HNSW全局图再按分片存储，保留跨分片图边；利用灵衢统一总线的高带宽、低时延远端内存访问能力实现分布式全局检索，减少计算量提升吞吐。

核心设计思想如下。

- **一个进程一个分片**：各进程对等部署，任一进程可发起查询并维护该查询的全部搜索状态。
- **全局图跨分片搜索**：沿同一张全局图推进，无需各分片分别完成一次本地top-K再归并。
- **URMA数据面**：远端邻接表/向量经URMA READ单边读取；每个分片以五种角色发布注册内存（topology/vectors/distance-tasks/distance-results/distance-notify）。
- **距离下推（pushdown）**：远端分片向量的距离计算可下发给**拥有该分片的远端进程**就近执行，只回传ID与距离。
- **五类线程运行时**：Control/Search/Distance/READ Progress/RPC Progress，通信与计算重叠，查询等待远端操作时Search线程切换处理其他查询。
- **Milvus一体化接入**：经C ABI（Application Binary Interface，应用二进制接口）（`ub_c_api.h`）接入Milvus QueryNode，由接入层按开关分流接管检索请求；当前仅支持裸向量检索；接入行为细节请参见Milvus侧资料。

底层HNSW图基于[Faiss](https://github.com/facebookresearch/faiss)构建。

## 目录结构

### 仓库工程目录

以下为`dev_for_26930`分支的工程结构参考，包含配套中文文档。《[快速入门](./docs/zh/quick_start.md)》使用`master`分支的`gd_hnsw_urma/`算法源码，目录说明见下文“[独立算法库交付目录](#独立算法库交付目录)”。获取源码和编译时，应按所选分支及实际Commit核对目录，不要混用不同分支的构建入口。

```text
gd-hnsw/
├─ .clang-format                                         // 代码格式化配置
├─ .gitcode/                                             // 持续集成配置
│  ├─ dockerfiles/                                       // 持续集成构建镜像
│  │  └─ Dockerfile.openeuler_24.03_LTS_SP3
│  └─ workflows/                                         // 构建与测试流水线
│     ├─ PR-pipeline_gd-hnsw.yml
│     ├─ pr_build.yml
│     └─ pr_ut.yml
├─ .gitignore                                            // Git忽略规则
├─ .gitmodules                                           // Git子模块配置
├─ .pre-commit-config.yaml                               // 提交前检查配置
├─ 3rdparty/                                             // 第三方依赖
│  ├─ CMakeLists.txt                                     // 第三方依赖构建入口
│  ├─ faiss/                                             // Faiss构建配置与子模块
│  │  ├─ CMakeLists.txt
│  │  └─ faiss/                                          // Faiss子模块
│  └─ googletest/                                        // GoogleTest构建配置与子模块
│     ├─ CMakeLists.txt
│     └─ googletest/                                     // GoogleTest子模块
├─ CMakeLists.txt                                        // 顶层构建入口
├─ LICENSE                                               // 代码许可证
├─ README.md                                             // 项目介绍
├─ benchmarks/                                           // 索引构建工具与性能基准
│  ├─ CMakeLists.txt
│  ├─ ann_dataset.h
│  ├─ bench_report.cpp                                   // 召回率、吞吐与时延报告
│  ├─ bench_report.h
│  ├─ bench_runner.cpp                                   // 基准执行与预热
│  ├─ bench_runner.h
│  ├─ bin_loader.h
│  ├─ build_index.cpp                                    // 离线索引构建工具
│  ├─ config_parser.h                                    // 数据集与参数配置解析
│  ├─ hdf5_loader.h
│  └─ main.cpp                                           // 基准测试入口
├─ configs/                                              // 数据集与构建、检索参数配置
│  ├─ bge-1024.config
│  ├─ bigann-128.config
│  └─ gist-960-euclidean.config
├─ docs/                                                 // 文档
│  ├─ LICENSE                                            // 文档许可证
│  └─ zh/                                                // 中文文档
│     ├─ FAQ.md                                          // 常见问题
│     ├─ api_reference.md                                // API参考
│     ├─ best_practices.md                               // 最佳实践
│     ├─ feature_introduction.md                         // 特性介绍
│     ├─ installation_guide.md                           // 安装指南
│     ├─ public_sys-resources/                           // 文档提示图标
│     │  ├─ icon-caution.gif
│     │  ├─ icon-danger.gif
│     │  ├─ icon-note.gif
│     │  ├─ icon-notice.gif
│     │  ├─ icon-tip.gif
│     │  └─ icon-warning.gif
│     ├─ quick_start.md                                  // 快速入门
│     └─ release_notes.md                                // 版本说明书
├─ include/                                              // 公共头文件
│  ├─ ann_dataset.h
│  ├─ bin_loader.h
│  ├─ build.h                                            // 离线构建接口
│  ├─ dist_service.h                                     // 距离下推服务
│  ├─ distance_kernel.h                                  // L2距离计算
│  ├─ distance_kernel_dot_norm.h                         // 内积与范数距离计算
│  ├─ faiss_extractor.h                                  // Faiss图结构提取
│  ├─ gd_hnsw_api.h                                      // 宿主接入接口
│  ├─ gd_hnsw_search.h                                   // 图搜索接口
│  ├─ gd_layout.h                                        // 共享内存布局
│  ├─ hdf5_loader.h
│  ├─ index_io.h                                         // 索引读写与校验
│  ├─ ls64_copy.h
│  └─ status.h                                           // 状态码
├─ scripts/                                              // 构建与辅助脚本
│  └─ build.sh                                           // 依赖初始化、编译与单元测试
├─ src/                                                  // 算法与接入实现
│  ├─ core/                                              // 图搜索、内存布局与索引读写
│  │  ├─ gd_hnsw_search.cpp
│  │  ├─ gd_layout.cpp
│  │  └─ index_io.cpp
│  ├─ deploy/                                            // 宿主接入实现
│  │  └─ gd_hnsw_api.cpp
│  └─ index/                                             // 索引构建与Faiss图提取
│     ├─ build.cpp
│     └─ faiss_extractor.cpp
└─ tests/                                                // 单元测试与集成测试
   ├─ CMakeLists.txt
   ├─ run_deploy_test.sh                                 // 部署接入测试脚本
   ├─ test_build.cpp
   ├─ test_concurrency.cpp
   ├─ test_config_parser.cpp
   ├─ test_deploy_e2e.cpp                                // 部署接入端到端测试
   ├─ test_distance_common.hpp
   ├─ test_doorbell.cpp
   ├─ test_dot_norm.cpp
   ├─ test_faiss_extract.cpp
   ├─ test_index_io.cpp
   ├─ test_l2_kernel.cpp
   ├─ test_layout_compute.cpp
   ├─ test_main.cpp
   ├─ test_minimax_heap.cpp
   ├─ test_pushdown_fork.cpp
   ├─ test_pushdown_protocol.cpp
   ├─ test_pushdown_search.cpp
   ├─ test_search_algorithm.cpp
   ├─ test_search_e2e.cpp
   ├─ test_superblock.cpp
   ├─ test_superblock_full.cpp
   ├─ test_visited_table.cpp
   └─ ubs_mem_stub.cpp
```

### 独立算法库交付目录

本版本的URMA运行时以`gd_hnsw_urma/`独立CMake工程交付，其公共头文件、源码与包配置文件如下。算法库的编译和接口说明请参见《[安装指南](./docs/zh/installation_guide.md)》和《[API参考](./docs/zh/api_reference.md)》。

```text
gd_hnsw_urma/
├─ CMakeLists.txt                                        // 算法库构建入口
├─ include/
│  └─ hnsw_urma/                                         // 算法库公共头文件
│     ├─ compute/                                        // 数据集加载与距离计算
│     │  ├─ dataset.hpp
│     │  └─ distance.hpp
│     ├─ control/                                        // 集群控制面
│     │  └─ mpi_control.hpp
│     ├─ core/                                           // 状态、分片映射、队列与内存等基础组件
│     │  ├─ checked_math.hpp
│     │  ├─ log.hpp
│     │  ├─ minimax_heap.hpp
│     │  ├─ mpmc_ring.hpp
│     │  ├─ page_aligned_buffer.hpp
│     │  ├─ query_key.hpp
│     │  ├─ shard_map.hpp
│     │  ├─ spsc_ring.hpp
│     │  ├─ status.hpp
│     │  ├─ types.hpp
│     │  └─ visited_table.hpp
│     ├─ index/                                          // 索引三元组格式与参考构建接口
│     │  ├─ builder.hpp
│     │  └─ format.hpp
│     ├─ milvus_query_executor.hpp
│     ├─ milvus_ub_runtime.hpp
│     ├─ milvus_ub_runtime_impl.hpp
│     ├─ protocol/                                       // 距离批量请求与响应格式
│     │  └─ distance_batch_wire.hpp
│     ├─ runtime/                                        // 线程编排、调度、距离下推与运行时管理
│     │  ├─ benchmark.hpp
│     │  ├─ config.hpp
│     │  ├─ counting_notification.hpp
│     │  ├─ drain.hpp
│     │  ├─ git_commit.hpp
│     │  ├─ hybrid_distance.hpp
│     │  ├─ load_balancing.hpp
│     │  ├─ rpc/                                         // RPC邮箱、传输与距离服务
│     │  │  ├─ distance_oracle.hpp
│     │  │  ├─ mailbox.hpp
│     │  │  └─ transport.hpp
│     │  ├─ rpc_runtime.hpp
│     │  ├─ scheduler.hpp
│     │  ├─ search_event.hpp
│     │  ├─ thread_group.hpp
│     │  └─ topology.hpp
│     ├─ search/                                         // 搜索状态机、查询池与本地、远端I/O
│     │  ├─ io.hpp
│     │  ├─ local_io.hpp
│     │  ├─ mock_io.hpp
│     │  ├─ query_context.hpp
│     │  ├─ query_slot_pool.hpp
│     │  ├─ read_resources.hpp
│     │  ├─ remote_io.hpp
│     │  └─ search.hpp
│     ├─ transport/                                      // 传输抽象、批量读取与共享内存传输
│     │  ├─ mock_lane_post_engine.hpp
│     │  ├─ read_batch_lane.hpp
│     │  ├─ shared_memory_transport.hpp
│     │  ├─ transport.hpp
│     │  ├─ urma/                                        // URMA传输
│     │  │  └─ urma_transport.hpp
│     │  └─ urma_resources.hpp
│     └─ ub_c_api.h                                      // 对外C ABI
├─ src/                                                  // 算法库各模块实现与Milvus接入实现
│  ├─ compute/
│  │  ├─ dataset.cpp
│  │  └─ distance.cpp
│  ├─ control/
│  │  └─ mpi_control.cpp
│  ├─ core/
│  │  ├─ minimax_heap.cpp
│  │  ├─ page_aligned_buffer.cpp
│  │  ├─ query_key.cpp
│  │  ├─ shard_map.cpp
│  │  ├─ status.cpp
│  │  └─ visited_table.cpp
│  ├─ index/
│  │  ├─ builder.cpp
│  │  ├─ format.cpp
│  │  └─ index.cpp
│  ├─ milvus_distributed_runtime.cpp
│  ├─ milvus_query_executor.cpp
│  ├─ milvus_runtime_config.cpp
│  ├─ milvus_ub_runtime.cpp
│  ├─ protocol/
│  │  └─ distance_batch_wire.cpp
│  ├─ runtime/
│  │  ├─ benchmark.cpp
│  │  ├─ config.cpp
│  │  ├─ drain.cpp
│  │  ├─ hybrid_distance.cpp
│  │  ├─ load_balancing.cpp
│  │  ├─ rpc/
│  │  │  ├─ distance_oracle.cpp
│  │  │  ├─ mailbox.cpp
│  │  │  └─ transport.cpp
│  │  ├─ rpc_runtime.cpp
│  │  ├─ scheduler.cpp
│  │  ├─ thread_group.cpp
│  │  └─ topology.cpp
│  ├─ search/
│  │  ├─ local_io.cpp
│  │  ├─ mock_io.cpp
│  │  ├─ query_context.cpp
│  │  ├─ query_slot_pool.cpp
│  │  ├─ read_resources.cpp
│  │  ├─ remote_io.cpp
│  │  └─ search.cpp
│  ├─ transport/
│  │  ├─ mock_lane_post_engine.cpp
│  │  ├─ read_batch_lane.cpp
│  │  ├─ shared_memory_transport.cpp
│  │  ├─ transport.cpp
│  │  ├─ urma/
│  │  │  └─ urma_transport.cpp
│  │  └─ urma_resources.cpp
│  └─ ub_c_api.cpp                                       // 对外C ABI实现
├─ ub_hnsw-config.cmake.in                               // CMake包配置模板
├─ ub_hnsw.pc.in                                         // pkg-config模板
└─ upstream-source.json                                  // 算法库源码来源记录
```

## 版本说明

关于gd_hnsw的版本更新情况请参见《[版本说明书](./docs/zh/release_notes.md)》。

## 学习文档

| 学习文档名称 | 学习文档简介 |
| --- | --- |
| [特性介绍](./docs/zh/feature_introduction.md) | 提供gd_hnsw架构介绍、核心算法原理及特性说明。 |
| [版本说明书](./docs/zh/release_notes.md) | 提供gd_hnsw每个发布版本的基础信息、配套环境与特性更新信息。 |
| [快速入门](./docs/zh/quick_start.md) | 提供gd_hnsw快速入门指导，含环境准备、索引构建与Milvus接入示例。 |
| [安装指南](./docs/zh/installation_guide.md) | 提供gd_hnsw依赖安装与编译方法指导。 |
| [API参考](./docs/zh/api_reference.md) | 提供gd_hnsw对外C ABI与C++接口定义、参数说明与返回值。 |
| [最佳实践](./docs/zh/best_practices.md) | 提供使用gd_hnsw的部署Checklist与调优建议。 |
| [常见问题（FAQ）](./docs/zh/FAQ.md) | 汇总使用gd_hnsw过程中的常见问题与解答。 |

## 缩略语

| 缩略语 | 英文全称 | 中文全称 |
| --- | --- | --- |
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

## 免责声明

本项目面向鲲鹏ARM架构进行优化，底层HNSW图构建基于开源Faiss实现（调用faiss::IndexHNSWFlat建图后提取图结构）。软件的任何漏洞与安全问题，凡涉及Faiss上游部分均由相应的上游社区根据其漏洞和安全响应机制解决，请密切关注上游社区发布的通知和版本更新。本项目对软件的漏洞及安全问题不承担任何责任，使用者需自行评估并承担在自身生产环境中使用本软件的风险。

## License

gd_hnsw采用Mulan PSL（Mulan Permissive Software License，木兰宽松许可证）v2许可证授权，支持修改代码和再开源，具体请参见[LICENSE](./LICENSE)文件。

本项目的文档适用CC-BY（Creative Commons Attribution，知识共享署名）4.0许可证，具体请参见 [LICENSE](./docs/LICENSE)文件。

## 贡献声明

欢迎大家为社区做贡献，如果使用过程中有任何问题/建议，或者需要反馈特性需求和bug报告，可以提交Issue联系我们，具体贡献方法可参考[贡献指南](https://gitcode.com/boostkit/community/blob/master/docs/contributor/contributing.md)。同时也欢迎在[讨论专区](https://gitcode.com/boostkit/gd-hnsw/discussions)展开讨论交流。

## 致谢

gd_hnsw在以下方面受益于开源社区。

- [Faiss](https://github.com/facebookresearch/faiss) —— 提供HNSW索引构建能力。

感谢来自社区的每一个PR（Pull Request，拉取请求），欢迎贡献gd_hnsw！

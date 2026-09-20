# gd_hnsw全局分布式检索介绍

## 最新消息

\[2026.09.30\]：第一次正式发布。提供全局分布式检索能力。

## 项目介绍

**gd_hnsw**是一个面向鲲鹏ARM架构的**全局分布式检索算法**。相较于传统分片检索的方式，该算法充分利用灵衢高带宽、低时延的优势实现了全局分布式检索，减少计算量提升吞吐。

核心设计思想如下。

- **N个容器=N个进程=N个分片**。
- 每个节点持有一个分片的HNSW图。
- 检索时采用**pushdown**（距离下推）：本节点搜索到在远端分片的向量距离时，通过灵渠的通信能力将计算任务下发给**拥有该分片的远端节点**就近计算，最后将结果回传。

底层HNSW图基于[Faiss](https://github.com/facebookresearch/faiss)构建。

## 目录结构

代码仓目录结构如下。

```text
gd-hnsw-master/
├─ 3rdparty/                                  // 第三方依赖（faiss v1.13.2、googletest v1.14.0子模块）
│  ├─ CMakeLists.txt                          // 子模块调度：faiss/gtest检出、构建并安装到dist/3rdparty/
│  ├─ faiss/
│  │  └─ CMakeLists.txt                       // faiss构建脚本（引用3rdparty/faiss/faiss子模块）
│  └─ googletest/
│     └─ CMakeLists.txt                       // googletest构建脚本（引用3rdparty/googletest/googletest子模块）
├─ benchmarks/                                // 性能基准与端到端服务测试
│  ├─ CMakeLists.txt                          // build_index/gd_hnsw_bench目标构建
│  ├─ main.cpp                                // bench 入口：命令行解析与分发
│  ├─ build_index.cpp                         // 离线索引构建工具
│  ├─ bench_runner.cpp                        // 基准运行器（warmup/rounds、召回率与延迟统计）
│  ├─ bench_runner.h                          // 基准运行器接口
│  ├─ bench_report.cpp                        // 基准报告生成（召回率、QPS、时延分位）
│  ├─ bench_report.h                          // 报告接口
│  ├─ config_parser.h                         // configs/*.config解析
│  ├─ ann_dataset.h                           // 数据集抽象（HDF5/BIN自动识别）
│  ├─ hdf5_loader.h                           // HDF5数据集加载
│  └─ bin_loader.h                            // BIN/IVEC/FBIN原始向量加载
├─ configs/                                   // 数据集与检索/构建参数配置
│  ├─ bge-1024.config                         // BGE 1024维
│  ├─ openai-1536.config                      // OpenAI 1536维
│  ├─ gist-960-euclidean.config               // GIST 960维
│  └─ bigann-128.config                       // BigANN 128维
├─ docs/                                      // 文档
│  ├─ zh/                                     // 中文文档
│     ├─ api_reference.md                     // API参考
│     ├─ best_practices.md                    // 最佳实践
│     ├─ feature_introduction.md              // 特性介绍
│     ├─ installation_guide.md                // 安装指南
│     ├─ quick_start.md                       // 快速入门
│     └─ release_notes.md                     // 版本说明书
├─ include/                                   // 公共头文件
│  ├─ gd_hnsw_api.h                          // 公共Context API
│  ├─ gd_layout.h                            // SuperBlockV1共享内存布局、BlockDesc、常量与枚举
│  ├─ gd_hnsw_search.h                       // GdHnswSearcher与VisitedTable
│  ├─ index_io.h                              // 索引读写与校验（CRC64、ValidationResult）
│  ├─ dist_service.h                          // 距离服务（DualDistProxy、pushdown通道）
│  ├─ distance_kernel.h                       // L2距离核
│  ├─ distance_kernel_dot_norm.h              // dot+norm内积距离核（FP16友好）
│  ├─ build.h                                 // 离线构建接口（BuildOptions/BuildResult）
│  ├─ faiss_extractor.h                       // Faiss HNSW图结构提取与格式转换
│  ├─ ann_dataset.h                           // 数据集抽象接口
│  ├─ bin_loader.h                            // BIN原始向量加载接口
│  ├─ hdf5_loader.h                           // HDF5数据集加载接口
│  ├─ ls64_copy.h                             // 64/128位对齐拷贝工具
│  ├─ security_check.h                        // 安全检查（输入校验、边界检查）
│  └─ status.h                                // 统一Status返回码
├─ scripts/
│  └─ build.sh                                // 构建脚本（glibc检查、子模块初始化、cmake构建、UT）
├─ src/
│  ├─ core/                                   // Layer1：gd_hnsw_core（+OpenMP）
│  │  ├─ gd_layout.cpp                       // SuperBlockV1布局组装、block描述符、CRC计算
│  │  ├─ gd_hnsw_search.cpp                  // HNSW图搜索、距离计算、minimax heap、pushdown
│  │  └─ index_io.cpp                         // 索引文件读写、共享内存映射、完整性校验
│  ├─ index/                                  // Layer2/3：提取与构建
│  │  ├─ faiss_extractor.cpp                  // Layer2：从faiss HNSW提取向量/邻居/层级
│  │  └─ build.cpp                            // Layer3：离线多分片构建
│  └─ deploy/                                 // Layer4：gd_hnsw_api
│     └─ gd_hnsw_api.cpp                     // Context 实现：initialize/load/node_init_and_sync/search/finalize
├─ tests/                                     // 单元测试（gtest）
│  ├─ CMakeLists.txt                          // 测试目标构建
│  ├─ test_main.cpp                           // gtest main入口
│  ├─ test_superblock.cpp                     // SuperBlockV1布局单测
│  ├─ test_superblock_full.cpp                // SuperBlock全量字段/CRC校验
│  ├─ test_layout_compute.cpp                 // 布局偏移/对齐计算
│  ├─ test_visited_table.cpp                  // VisitedTable dense/sparse双模式
│  ├─ test_search_algorithm.cpp               // HNSW搜索算法正确性
│  ├─ test_search_e2e.cpp                     // 端到端搜索
│  ├─ test_l2_kernel.cpp                      // L2距离核
│  ├─ test_dot_norm.cpp                       // dot+norm内积距离核
│  ├─ test_distance_common.hpp                // 距离核公共测试夹具
│  ├─ test_minimax_heap.cpp                   // minimax heap优先队列
│  ├─ test_index_io.cpp                       // 索引读写与校验
│  ├─ test_faiss_extract.cpp                  // Faiss图结构提取
│  ├─ test_config_parser.cpp                  // 配置解析
│  ├─ test_concurrency.cpp                    // 并发安全
│  ├─ test_doorbell.cpp                       // doorbell同步原语
│  ├─ test_pushdown_protocol.cpp              // pushdown协议
│  ├─ test_pushdown_search.cpp                // pushdown搜索流程
│  └─ test_pushdown_fork.cpp                  // pushdown fork行为
├─ .clang-format                              // 代码格式化配置
├─ .gitignore                                 // Git忽略规则
├─ .gitmodules                                // Git子模块（faiss v1.13.2、googletest v1.14.0）
├─ .pre-commit-config.yaml                    // pre-commit钩子配置
├─ CMakeLists.txt                             // 顶层构建脚本（分层条件编译）
├─ LICENSE                                    // 许可证
└─ README.md                                  // 项目介绍
```

## 版本说明

关于gd_hnsw的版本更新情况请参见《[版本说明书](./docs/zh/release_notes.md)》。

## 学习文档

<a name="table1191773710200"></a>
<table><thead align="left"><tr><th valign="top" width="20%">学习资源名称</th>
<th valign="top" width="80%">学习资源简介</th>
</tr>
</thead>
<tbody>
<tr><td valign="top"><a href="./docs/zh/feature_introduction.md">特性介绍</a></td>
<td valign="top">提供gd_hnsw架构介绍、核心算法原理及特性优化说明。</td>
</tr>
<tr><td valign="top"><a href="./docs/zh/release_notes.md">版本说明书</a></td>
<td valign="top">提供gd_hnsw每个发布版本的基础信息、配套环境与特性更新信息。</td>
</tr>
<tr><td valign="top"><a href="./docs/zh/quick_start.md">快速入门</a></td>
<td valign="top">提供gd_hnsw快速入门指导，含离线构建索引与在线检索最小示例。</td>
</tr>
<tr><td valign="top"><a href="./docs/zh/installation_guide.md">安装指南</a></td>
<td valign="top">提供gd_hnsw依赖安装与编译方法指导。</td>
</tr>
<tr><td valign="top"><a href="./docs/zh/api_reference.md">API参考</a></td>
<td valign="top">提供gd_hnsw对外Context接口定义、参数说明与返回值。</td>
</tr>
<tr><td valign="top"><a href="./docs/zh/best_practices.md">最佳实践</a></td>
<td valign="top">提供使用gd_hnsw的部署Checklist与调优建议。</td>
</tr>
</tbody>
</table>

## 免责声明

本项目面向鲲鹏ARM架构进行优化，底层HNSW图构建基于开源Faiss实现。代码在保持Fais开源算法语义的前提下进行侵入式改造，不对Faiss上游接口做破坏性修改。软件的任何漏洞与安全问题，凡涉及Faiss上游部分均由相应的上游社区根据其漏洞和安全响应机制解决，请密切关注上游社区发布的通知和版本更新。本项目对软件的漏洞及安全问题不承担任何责任，使用者需自行评估并承担在自身生产环境中使用本软件的风险。

## License

gd_hnsw采用Apache 2.0 License许可证授权，支持修改代码和再开源，具体请参见[LICENSE](./LICENSE)文件。

本项目的文档适用CC-BY 4.0许可证，具体请参见 [LICENSE](./docs/LICENSE)文件。

## 贡献声明

欢迎大家为社区做贡献，如果使用过程中有任何问题/建议，或者需要反馈特性需求和bug报告，可以提交Issue联系我们，具体贡献方法可参考[这里](https://gitcode.com/boostkit/community/blob/master/docs/contributor/contributing.md)。同时也欢迎在[讨论专区](https://gitcode.com/boostkit/gd-hnsw/discussions)展开讨论交流。感谢您的支持。

## 致谢

gd_hnsw在以下方面受益于开源社区。

- [Faiss](https://github.com/facebookresearch/faiss) —— 提供HNSW索引构建能力。
- MPI/OpenMP/HDF5/OpenBLAS等开源基础设施。

感谢来自社区的每一个PR，欢迎贡献gd_hnsw！

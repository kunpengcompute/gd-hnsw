# 常见问题和故障案例（FAQ）

本文提供gd_hnsw常见问题和故障案例，涵盖安装编译、部署环境、原理与使用等类别内容。（条目持续补充）

## 安装编译相关

### 编译报找不到UMDK/URMA（"ENABLE_URMA requires the URMA library, headers and bonding header"）

**问题**

编译算法库时CMake（Cross-platform Make，跨平台构建工具）报错："ENABLE_URMA requires the URMA library, headers and bonding header"。

**回答**

该报错表示开启`ENABLE_URMA`后未找到UMDK（Unified Memory Development Kit，统一内存开发工具包）/URMA（Unified Remote Memory Access，统一远端内存访问）的头文件与库。确认UB（Unified Bus，统一总线）相关组件与驱动已安装（见《[安装指南](./installation_guide.md)》"安装UB相关组件与驱动"）；如UMDK安装在非系统默认路径，请通过`UMDK_ROOT`环境变量或`CMAKE_PREFIX_PATH`指定。不需要跨节点数据面时，可改以`ENABLE_URMA=OFF`构建（构建产物不含跨节点能力）。

### 编译报找不到fmt（"ub_hnsw requires fmt"）

**问题**

编译算法库时CMake报错："ub_hnsw requires fmt"。

**回答**

确认fmt开发包已安装（如`fmt-devel`）；如经conan或源码安装fmt，请将其CMake包路径加入`CMAKE_PREFIX_PATH`。

### 编译报C++20特性缺失

**问题**

编译时报C++20特性缺失（如`std::jthread`）。

**回答**

编译器或标准库过旧。请使用GCC（GNU Compiler Collection，GNU编译器套件） 12.3及以上版本，并确认以C++20或更高标准编译。

## 部署环境相关

### 运行时无跨节点流量

**问题**

部署完成后检索没有跨节点流量，跨节点能力未生效。

**回答**

请逐项确认：

1. 算法库构建时`ENABLE_URMA=ON`；
2. 各节点URMA设备可用（`ubctl`等工具枚举）；
3. 传输模式与设备能力匹配（当前驱动要求`rm`可靠消息模式）；
4. 对端资源已在资源目录中发布并完成导入。

### 跨节点 ping 通但 TCP/HTTP 不通，如何排查？

**问题**

集群节点间 ping 测试正常，但跨节点业务通信（TCP/HTTP）失败。

**回答**

ping 只验证 ICMP 可达，不代表业务连通性，请以 TCP（Transmission Control Protocol，传输控制协议）/HTTP（HyperText Transfer Protocol，超文本传输协议）实测为准，并按以下顺序排查：

1. 确认两节点是否二层直连：`ip route get <对端IP>`，路由应走本机网卡直达，不经过额外网关；
2. 确认邻居表能解析对端地址：`ip neigh show <对端IP>`；
3. 用 `nc`、`wget` 等工具实测跨节点 TCP/HTTP 连通性；
4. 若链路经过隧道封装，检查中间网络设备对隧道报文与大包的限制，必要时改用主机路由直连方式组网。

URMA 跨节点数据面对网络质量敏感，组网验收建议全部以实测通过为准。

### URMA 环境不满足要求时会有什么表现？

**问题**

部署环境缺少 URMA 设备或配置不当时，算法库会出现什么现象？

**回答**

按场景有三种典型表现：

1. 算法库以`ENABLE_URMA=OFF`编译：调用启用或检索接口返回`kInvalidArgument`（"built without URMA support"），须以`ENABLE_URMA=ON`重新编译；
2. URMA 设备不可枚举或传输模式与设备能力不匹配：启用分布式检索失败，当前驱动要求`rm`可靠消息模式；
3. 未注入 URMA 上下文（未调用`UbSetUrmaContext`）：启用分布式检索返回`kInvalidArgument`。

排查建议：用`ubctl`等工具确认设备可枚举，确认库以`ENABLE_URMA=ON`编译，并检查接入流程已注入上下文。

## 原理与使用相关

### 全局索引与分片索引的区别是什么？

**问题**

gd_hnsw 的全局索引与传统分片式索引有何不同？

**回答**

传统分片式检索将数据划分为多个分片，各分片独立建图、独立检索，再由协调节点汇聚重排序；gd_hnsw 先在全部数据上构建一张完整图，再按分片存储并保留跨分片图边，一次查询沿同一张全局图搜索，直接产出全局 top-K。以距离计算次数衡量，设数据 N 个向量均匀分布在 P 个分片上，分片方案的总计算量按 `P·log(N/P)` 增长，全局图方案为 `log N`。详细对比见《[特性介绍](./feature_introduction.md)》"全局图索引与分片基线对比"。

### 查询进程空载时为什么仍占用 CPU？

**问题**

查询进程在没有查询任务运行时，为什么仍占用 CPU（Central Processing Unit，中央处理器）？

**回答**

这是为降低查询时延的设计取舍：Search 线程与 READ/RPC（Remote Procedure Call，远程过程调用） 进度线程默认采用忙轮询（busy 模式），取不到任务时持续轮询而非休眠等待，空载时仍会占用 CPU。其中 READ/RPC 进度线程可配置`progress_mode=adaptive`，空载时转为休眠等待以缓解占用；Search 线程的空转行为无法通过配置关闭。

## 缩略语

| 缩略语 | 英文全称 | 中文全称 |
|--------|----------|----------|
| CMake | Cross-platform Make | 跨平台构建工具 |
| CPU | Central Processing Unit | 中央处理器 |
| FAQ | Frequently Asked Questions | 常见问题 |
| GCC | GNU Compiler Collection | GNU编译器套件 |
| HTTP | HyperText Transfer Protocol | 超文本传输协议 |
| RPC | Remote Procedure Call | 远程过程调用 |
| TCP | Transmission Control Protocol | 传输控制协议 |
| UB | Unified Bus | 统一总线 |
| UMDK | Unified Memory Development Kit | 统一内存开发工具包 |
| URMA | Unified Remote Memory Access | 统一远端内存访问 |

## 修订记录

| 文档版本 | 发布日期 | 修改说明 |
|------|----------|------------|
| 01 | 2026-12-30 | 第一次正式发布。 |

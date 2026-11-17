# 安装指南

本文档介绍gd_hnsw的环境要求、依赖安装与编译方法。

## 环境要求

以下为本版本的设计典配环境。各组件的验证报告与精确驱动配套版本以发布版本的配套表为准。

**表 1** gd_hnsw设计典配环境

| 操作系统 | 处理器类型 | 编译器 | 其他 |
|----------|------------|--------|------|
| openEuler 24.03 LTS SP3 | 鲲鹏950处理器（AArch64） | GCC 12.3 | CMake 3.26及以上版本，C++20编译环境，需UMDK/URMA |

>![表示说明的图片](./public_sys-resources/icon-note.gif) **说明：**
>
> - gd_hnsw核心代码使用C++20（`std::jthread`、`std::span`、`std::stop_token`等），编译器与标准库须支持C++20。
> - 跨节点数据面依赖UMDK/URMA；无URMA设备的环境只能以共享内存传输做本机功能验证，无法获得跨节点能力。

## 依赖说明

gd_hnsw编译依赖如下组件。

**表 2** gd_hnsw依赖组件

| 依赖 | 说明 | 获取方式 |
|------|------|----------|
| CMake 3.26及以上版本 | 构建系统 | 系统包管理器或pip |
| GCC 12.3 | C++20编译器 | 系统包管理器 |
| MPI | 独立基准的集群控制面（Milvus集成不依赖MPI） | yum install openmpi-devel等 |
| UMDK/URMA及UB相关组件 | 跨节点数据面（READ/WRITE_NOTIFY） | 请参见下文"[安装UB相关组件与驱动](./installation_guide.md#安装ub相关组件与驱动)" |
| Faiss 1.13.2及以上版本 | 离线HNSW图构建 | 请参见下文"[获取Faiss](./installation_guide.md#获取faiss)" |
| OpenBLAS | Faiss数学库依赖 | 请参见下文"[获取OpenBLAS](./installation_guide.md#获取openblas)" |
| googletest | 单元测试框架（仅跑测试时需要） | 系统包管理器或源码 |

## 编译安装

### 安装系统依赖

以openEuler 24.03 LTS SP3为例。

```bash
yum install make cmake gcc g++ openmpi-devel blas-devel lapack-devel
```

### 获取OpenBLAS

1. Faiss依赖数学库，从GitHub仓下载开源OpenBLAS源代码（版本以Faiss要求为准，如v0.3.29或更新版本）。保存在编译机器可访问的路径中，假设位于"/path/to/OpenBLAS"。

   ```bash
   git clone https://github.com/OpenMathLib/OpenBLAS.git
   ```

2. 编译源代码获取libopenblas.so。

   ```bash
   cd /path/to/OpenBLAS
   make
   make install
   ```

3. 构建前确保动态库可被找到（以实际安装路径为准，示例如下）。

   ```bash
   export LD_LIBRARY_PATH=/opt/OpenBLAS/lib:$LD_LIBRARY_PATH
   ```

### 获取Faiss

从GitHub仓下载Faiss源代码（1.13.2及以上版本），按其官方文档以CMake编译安装。

```bash
git clone --branch v1.13.2 --single-branch https://github.com/facebookresearch/faiss.git
```

### 安装UB相关组件与驱动

相关版本依赖如下。

**表 3** UB相关组件与驱动版本

| 软件名 | 版本 | 获取路径 |
|------|------|----------|
| kernel | 6.6.0-145.3.24.155.oe2403sp3 | 跟随openEuler 24.03 LTS SP3最新版本发布，通过yum install安装 |
| umdk | 25.12.0-B122.oe2403sp3 | 跟随openEuler 24.03 LTS SP3最新版本发布，通过yum install安装 |
| obmm | 1.0-3oe2403sp3 | 跟随openEuler 24.03 LTS SP3最新版本发布，通过yum install安装 |
| libummu | 1.0.3-3oe2403sp3 | 跟随openEuler 24.03 LTS SP3最新版本发布，通过yum install安装 |
| libcdma | 1.0.4-4oe2403sp3 | 跟随openEuler 24.03 LTS SP3最新版本发布，通过yum install安装 |
| ubutils | 1.0.2-6oe2403sp3 | 跟随openEuler 24.03 LTS SP3最新版本发布，通过yum install安装 |
| ubctl | 1.0.3-4oe2403sp3 | 跟随openEuler 24.03 LTS SP3最新版本发布，通过yum install安装 |
| UBM | 2.0.0.B023-1 | [support网站获取](https://support.huawei.com) |
| UBE | 1.1.10.0.b142-openeuler24.03 | [support网站获取](https://support.huawei.com) |
| BIOS | 2.16.0.B013 | [support网站获取](https://support.huawei.com) |

>![表示说明的图片](./public_sys-resources/icon-note.gif) **说明：**
>UB相关组件的具体配套版本以发布版本的版本配套表为准。

### 获取源码

```bash
git clone https://gitcode.com/boostkit/gd-hnsw.git
```

### 编译gd_hnsw

当前可复现的构建路径是**随Milvus一体构建**：运行时源码位于Milvus仓`internal/core/src/index/ub_hnsw/urma_runtime/`，随`milvus_index`对象库一同编译（Milvus集成定义`HNSW_URMA_HAS_MPI=0`，资源发现走Milvus目录机制，不依赖MPI）；Faiss构建插件见`internal/core/urma_builder/`。按Milvus官方构建流程编译即可包含本运行时。

独立交付包的形态（含CMake包配置，下游经`find_package(hnsw_urma CONFIG)`引用，如`hnsw_urma::builder`目标）**以正式发布包为准**；正式发布后按包内说明执行CMake构建：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel $(nproc)
cmake --install build --prefix /path/to/install
```

>![表示说明的图片](./public_sys-resources/icon-note.gif) **说明：**
>编译需能找到UMDK/URMA头文件与库；如安装在非系统默认路径，请通过`CMAKE_PREFIX_PATH`或对应变量指定。

### 编译产物

**表 4** gd_hnsw编译产物（独立交付包形态，以正式发布包为准）

| 产物 | 类型 | 说明 |
|------|------|------|
| hnsw_urma运行时库 | 库 | 核心检索运行时（core/index/compute/search/transport/runtime/protocol/control） |
| hnsw_urma::builder | 库 | Faiss适配的离线索引构建模块 |
| 索引构建应用 | 可执行 | 基于Faiss的离线建库工具（产出索引三元组） |
| 检索基准应用 | 可执行 | 多进程基准：加载索引、启动五类线程运行时、统计召回率/QPS/时延分位 |

## 单元测试验证

测试基于googletest，覆盖以下范围：

- **单元**：索引格式与校验、ShardMap分片映射、VisitedTable、MinimaxHeap、距离计算、数据集加载、状态码、页对齐缓冲区、无锁环、运行时配置解析；
- **运行时**：调度器、RPC传输/运行时/邮箱、距离下推、负载均衡、优雅退出；
- **集成**：本地搜索、远端拉取、端到端服务；
- **格式**：索引文件format V2的写出/读取与全局元信息一致性校验。

## 常见问题

1. **CMake报找不到UMDK/URMA**

   确认UB相关组件与驱动已安装（见"安装UB相关组件与驱动"），并把头文件/库路径加入`CMAKE_PREFIX_PATH`。

2. **编译报C++20特性缺失（如`std::jthread`）**

   编译器或标准库过旧。请使用GCC 12.3及以上版本，并确认以C++20或更高标准编译。

3. **Faiss链接报缺BLAS符号**

   确认OpenBLAS已编译安装，且`LD_LIBRARY_PATH`包含其lib目录（如`/opt/OpenBLAS/lib`）。

4. **运行时无跨节点流量**

   确认各节点URMA设备可用（`ubctl`等工具枚举）、传输模式与设备能力匹配（当前驱动要求`rm`可靠消息模式），且对端资源已在资源目录中发布并完成导入。

## 修订记录

| 文档版本 | 发布日期 | 修改说明 |
|------|----------|------------|
| 01 | 2026-11-17 | 第一次正式发布。 |

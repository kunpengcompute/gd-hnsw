# 安装指南

本文档介绍gd_hnsw的已验证环境、依赖安装与编译方法。

## 已验证环境

为保证您可以顺利安全地使用gd_hnsw，请确保所使用的环境信息在已验证环境范围内。

**表 1**gd_hnsw已验证环境<a id="gd_hnsw已验证环境"></a>

| 操作系统 | 处理器类型 | 内存 | 编译器 | 其他 |
|----------|------------|------|--------|------|
| openEuler 24.03 LTS SP3 | 鲲鹏950处理器 | 24*64GB | GCC 12.3.1 | CMake>=3.18，需NUMA |

>![表示说明的图片](./public_sys-resources/icon-note.gif)**说明：**
>
> - gd_hnsw依赖NUMA（`libnuma`），当前仅支持Linux+NUMA，非Linux或非NUMA环境编译会报错。
> - 距离核使用ARM NEON指令，目标架构须为AArch64。编译时建议使用`-march=native`（CMake已默认开启）。

## 依赖说明

gd_hnsw编译依赖如下组件。

| 依赖 | 说明 | 获取方式 |
|------|------|----------|
| CMake>=3.18 | 构建系统 | 系统包管理器 |
| GCC>=12 版本12.3.1 | C++17编译器 | 系统包管理器 |
| MPI | 节点同步与集合通信（如OpenMPI/MPICH） 版本4.1.4| yum install openmpi-devel等 |
| OpenMP | 线程级并行| 随GCC提供 |
| HDF5（CXX组件） | HDF5数据集加载 | yum install hdf5 hdf5-devel |
| libnuma | NUMA 探测 | yum install numactl numactl-devel |
| UB相关组件与驱动 | 跨节点通信 | 请参见下文"[安装UB相关组件与驱动](./installation_guide.md#安装ub相关组件与驱动)" |
| Faiss | HNSW 索引构建 ，1.13.2版本| 下载[Faiss](https://github.com/facebookresearch/faiss.git)  |
| OpenBLAS（可选） | Faiss数学库依赖 | 请参见下文"[获取OpenBLAS](./installation_guide.md#获取openblas)" |

## 编译安装

### 安装系统依赖

以openEuler 24.03 LTS SP3为例。

```bash
yum install make cmake hdf5 hdf5-devel numactl numactl-devel openmpi-devel gcc g++ blas-devel lapack-devel
```

### 获取OpenBLAS

1. Faiss依赖数学库，从GitHub仓下载开源OpenBLAS源代码，标签为v0.3.29。保存在编译机器可访问的路径中，假设位于“/path/to/OpenBLAS-0.3.29”。

   ```bash
   git clone --branch v0.3.29 --single-branch https://github.com/OpenMathLib/OpenBLAS.git
   ```

2. 编译源代码获取libopenblas.so。

   ```bash
   cd /path/to/OpenBLAS-0.3.29/OpenBLAS
   make
   make install
   ```

### 安装UB相关组件与驱动

相关版本依赖如下。

| 软件名 | 版本 | 获取路径 |
|------|------|----------|
| kernel | 6.6.0-145.3.24.155.oe2403sp3 | 跟随openEular 24.03 LTS sp3最新版本发布，通过yum install安装 |
| umdk | 25.12.0-B122.oe2403sp3 | 跟随openEular 24.03 LTS sp3最新版本发布，通过yum install安装 |
| obmm | 1.0-3oe2403sp3 | 跟随openEular 24.03 LTS sp3最新版本发布，通过yum install安装 |
| libummu | 1.0.3-3oe2403sp3 | 跟随openEular 24.03 LTS sp3最新版本发布，通过yum install安装 |
| libcdma | 1.0.4-4oe2403sp3 | 跟随openEular 24.03 LTS sp3最新版本发布，通过yum install安装 |
| ubutils | 1.0.2-6oe2403sp3 | 跟随openEular 24.03 LTS sp3最新版本发布，通过yum install安装 |
| ubctl | 1.0.3-4oe2403sp3 | 跟随openEular 24.03 LTS sp3最新版本发布，通过yum install安装 |
| ubs-engine | 1.0.2-6oe2403sp3 | 跟随openEular 24.03 LTS sp3最新版本发布，通过yum install安装 |
| ubs-engine-client | 1.0.2-6oe2403sp3 | 跟随openEular 24.03 LTS sp3最新版本发布，通过yum install安装 |
| ubs-comm-lib | 1.0.1-2oe2403sp3 | 跟随openEular 24.03 LTS sp3最新版本发布，通过yum install安装 |
| ubs-mem-shmem | 1.0.2-3oe2403sp3 | 跟随openEular 24.03 LTS sp3最新版本发布，通过yum install安装 |
| UBM | 2.0.0.B023-1 | [sopport网站获取](https://support.huawei.com) |
| UBE | 1.1.10.0.b142-openeular24.03 | [sopport网站获取](https://support.huawei.com) |
| BIOS | 2.16.0.B013 | [sopport网站获取](https://support.huawei.com) |

### 编译gd_hnsw

1. 下载当前算法源代码。

   ```bash
   git clone https://gitcode.com/boostkit/gd-hnsw.git
   ```

2. 编译。

   ```bash
   cd gd-hnsw
   sh scripts/build.sh -t release --ubs_mem /path/to/ubs_mem
   ```

>![表示说明的图片](./public_sys-resources/icon-note.gif)**说明：**
> `--ubs_mem`必须指向包含`include/ubs_mem.h`与`lib/libubsm_sdk.so`的目录；未指定时CMake会在系统默认路径查找，找不到则报错。

### 编译产物

| 产物 | 类型 | 说明 |
|------|------|------|
| gd_hnsw_core | 静态库 | 纯算法库（layout/search/io，不含MPI/ubs-mem/faiss/HDF5） |
| gd_hnsw_faiss | 静态库 | faiss抽取层 |
| gd_hnsw_api | 静态库 | **对外库**（Context，链core+faiss+MPI+ubs-mem+HDF5） |
| gd_hnsw_bench | 可执行 | 最小示例程序 |
| build_index | 可执行 | 命令行建库工具 |

## 修订记录

| 文档版本 | 发布日期 | 修改说明 |
|------|----------|------------|
| 01 | 2026-09-30 | 第一次正式发布。 |

# 安装指南

本文档介绍gd_hnsw的环境要求、依赖安装与编译方法。

## 环境要求

以下为本版本的设计典配环境。各组件的验证报告与精确驱动配套版本以发布版本的配套表为准。

**表 1** gd_hnsw设计典配环境

| 操作系统 | 处理器类型 | 编译器 | 其他 |
|----------|------------|--------|------|
| openEuler 24.03 LTS（Long Term Support，长期支持） SP3 | 鲲鹏950处理器（AArch64） | GCC（GNU Compiler Collection，GNU编译器套件） 12.3 | CMake（Cross-platform Make，跨平台构建工具） 3.22及以上版本，C++20编译环境；跨节点数据面需UMDK（Unified Memory Development Kit，统一内存开发工具包）/URMA（Unified Remote Memory Access，统一远端内存访问） |

>![表示说明的图片](./public_sys-resources/icon-note.gif) **说明：**
>
> - gd_hnsw核心代码使用C++20（`std::jthread`、`std::span`、`std::stop_token`等），编译器与标准库须支持C++20。
> - 跨节点数据面依赖UMDK/URMA，编译时需开启`ENABLE_URMA`；以`ENABLE_URMA=OFF`（默认）构建的库不含URMA数据面，启用分布式检索与检索接口将返回`kInvalidArgument`。

## 依赖说明

gd_hnsw算法库的编译依赖如下组件。

**表 2** gd_hnsw依赖组件

| 依赖 | 说明 | 获取方式 |
|------|------|----------|
| CMake 3.22及以上版本 | 构建系统 | 系统包管理器或pip |
| GCC 12.3 | C++20编译器 | 系统包管理器 |
| fmt | 日志后端（`core/log.hpp`），必需 | 系统包管理器（如fmt-devel）或conan |
| UMDK/URMA及UB（Unified Bus，统一总线）相关组件 | 跨节点数据面（READ/WRITE_NOTIFY），`ENABLE_URMA=ON`时必需 | 请参见下文"[安装UB相关组件与驱动](./installation_guide.md#安装ub相关组件与驱动)" |

索引构建依赖（Faiss/OpenBLAS（Open Basic Linear Algebra Subprograms，开放基础线性代数子程序库）等）由宿主侧构建组件承担，不属于本库构建范围。

## 编译安装

### 安装系统依赖

以openEuler 24.03 LTS SP3为例。

```bash
yum install make cmake gcc g++ fmt-devel
```

如系统源无fmt开发包，可经conan或源码安装fmt，并将其CMake包路径加入`CMAKE_PREFIX_PATH`。

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
| BIOS（Basic Input Output System，基本输入输出系统） | 2.16.0.B013 | [support网站获取](https://support.huawei.com) |

>![表示说明的图片](./public_sys-resources/icon-note.gif) **说明：**
>UB相关组件的具体配套版本以发布版本的版本配套表为准。

### 获取源码

```bash
git clone https://gitcode.com/boostkit/gd-hnsw.git
```

### 编译gd_hnsw

算法库以独立CMake工程交付（`gd_hnsw_urma/`目录）。当前工程未自行初始化GNUInstallDirs，独立构建时需显式补齐安装目录变量。以下示例统一使用`lib`目录，在仓库根目录执行：

```bash
install_prefix=/path/to/install
cmake -S gd_hnsw_urma -B build \
    -DCMAKE_BUILD_TYPE=Release -DENABLE_URMA=ON \
    -DCMAKE_INSTALL_PREFIX="$install_prefix" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_INSTALL_BINDIR=bin \
    -DCMAKE_INSTALL_FULL_LIBDIR="$install_prefix/lib" \
    -DCMAKE_INSTALL_FULL_INCLUDEDIR="$install_prefix/include"
cmake --build build --parallel $(nproc)
cmake --install build
```

>![表示说明的图片](./public_sys-resources/icon-note.gif) **说明：**
>
> - `ENABLE_URMA`默认为`OFF`；开启后编译需能找到UMDK/URMA头文件与库（默认查找系统路径，亦可通过`UMDK_ROOT`环境变量或`CMAKE_PREFIX_PATH`指定安装路径）。
> - `ENABLE_URMA=OFF`构建的库不含跨节点数据面，仅供编译与便携验证，无法启用分布式检索。
> - 安装前缀须在**配置阶段**经`-DCMAKE_INSTALL_PREFIX`指定：pkg-config与CMake包配置文件在配置阶段固化安装路径，仅在`cmake --install`阶段使用`--prefix`会导致包内路径与实际安装位置不一致。
> - 当前版本安装规则未初始化GNUInstallDirs。若外部也未定义相关变量，`CMAKE_INSTALL_LIBDIR`为空会使部分安装目标展开为`/pkgconfig`、`/cmake/ub_hnsw`等绝对路径；`CMAKE_INSTALL_FULL_LIBDIR`或`CMAKE_INSTALL_FULL_INCLUDEDIR`为空还会使包配置缺失库或头文件路径。因此独立构建须按示例显式提供这些变量。若使用`lib64`，须同步调整`CMAKE_INSTALL_LIBDIR`与`CMAKE_INSTALL_FULL_LIBDIR`；安装前缀应使用绝对路径。
> - 安装后检查`lib/pkgconfig/ub_hnsw.pc`中的`libdir`、`includedir`及`lib/cmake/ub_hnsw/ub_hnsw-config.cmake`中的头文件路径，确认其与实际安装位置一致；使用`lib64`时相应替换目录名。宿主接入前仍需在目标Linux/URMA环境完成构建与链接验证。

Milvus侧经`find_package(ub_hnsw)`引用本库（导入目标`ub_hnsw::ub_hnsw`），接入细节请参见Milvus侧资料。

### 编译产物

**表 4** gd_hnsw编译产物

| 产物 | 类型 | 说明 |
|------|------|------|
| libub_hnsw.so | 库 | 检索运行时（core/index/compute/search/transport/runtime/protocol/control）与对外C ABI、C++接口 |
| 公共头文件 | 头文件 | `include/hnsw_urma/`（算法接口）与`include/ub_hnsw/ub_c_api.h`（C ABI） |
| ub_hnsw.pc | pkg-config文件 | 供CGO等经pkg-config链接（`-lub_hnsw`） |
| ub_hnsw CMake包 | 包配置 | `find_package(ub_hnsw)`可用，暴露导入目标`ub_hnsw::ub_hnsw` |

## 常见问题

1. **CMake报"ENABLE_URMA requires the URMA library, headers and bonding header"**

   确认UB相关组件与驱动已安装（见"安装UB相关组件与驱动"）；如UMDK安装在非系统默认路径，请通过`UMDK_ROOT`环境变量或`CMAKE_PREFIX_PATH`指定。不需要跨节点数据面时可改以`ENABLE_URMA=OFF`构建。

2. **CMake报"ub_hnsw requires fmt"**

   确认fmt开发包已安装（如`fmt-devel`）；如经conan或源码安装，请将其CMake包路径加入`CMAKE_PREFIX_PATH`。

3. **编译报C++20特性缺失（如`std::jthread`）**

   编译器或标准库过旧。请使用GCC 12.3及以上版本，并确认以C++20或更高标准编译。

4. **运行时无跨节点流量**

   确认构建时`ENABLE_URMA=ON`，各节点URMA设备可用（`ubctl`等工具枚举）、传输模式与设备能力匹配（当前驱动要求`rm`可靠消息模式），且对端资源已在资源目录中发布并完成导入。

## 缩略语

| 缩略语 | 英文全称 | 中文全称 |
|--------|----------|----------|
| BIOS | Basic Input Output System | 基本输入输出系统 |
| CMake | Cross-platform Make | 跨平台构建工具 |
| GCC | GNU Compiler Collection | GNU编译器套件 |
| LTS | Long Term Support | 长期支持 |
| OpenBLAS | Open Basic Linear Algebra Subprograms | 开放基础线性代数子程序库 |
| UB | Unified Bus | 统一总线 |
| UMDK | Unified Memory Development Kit | 统一内存开发工具包 |
| URMA | Unified Remote Memory Access | 统一远端内存访问 |

## 修订记录

| 文档版本 | 发布日期 | 修改说明 |
|------|----------|------------|
| 01 | 2026-12-30 | 第一次正式发布。 |

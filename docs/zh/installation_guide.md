# 安装指南

本文档介绍gd_hnsw的环境要求、依赖安装与编译方法。

## 环境要求

以下列出本版本发布时的验证环境及构建工具要求。UB相关组件与驱动的具体版本见下文“[安装UB相关组件与驱动](#安装ub相关组件与驱动)”。

**表 1** 硬件环境要求

| 项目 | 要求 |
| --- | --- |
| 处理器 | 鲲鹏950处理器（AArch64） |
| 通信设备 | 支持URMA的UB设备，设备与链路已完成配置；检查步骤见《[快速入门](./quick_start.md#检查urma设备和链路)》 |

**表 2** 操作系统和软件要求

| 项目 | 要求 |
| --- | --- |
| 操作系统 | openEuler 24.03 LTS SP3 |
| 编译器 | GCC 12.3，编译器与标准库支持C++20 |
| 构建工具 | CMake 3.22及以上版本 |

>![表示说明的图片](./public_sys-resources/icon-note.gif) **说明：**
>
> - gd_hnsw核心代码使用C++20（`std::jthread`、`std::span`、`std::stop_token`等），编译器与标准库须支持C++20。
> - 跨节点数据面依赖UMDK（Unified Memory Development Kit，统一内存开发工具包）/URMA（Unified Remote Memory Access，统一远端内存访问），编译时需开启`ENABLE_URMA`。以`ENABLE_URMA=OFF`（默认）构建的库不含URMA数据面：调用启用分布式检索接口将返回`kInvalidArgument`，分布式检索不可用。

## 依赖组件

gd_hnsw算法库的编译依赖组件如下，编译器与构建工具版本要求见“[环境要求](#环境要求)”。

**表 3** gd_hnsw依赖组件

| 依赖 | 说明 | 获取方式 |
| --- | --- | --- |
| fmt | 日志后端（`core/log.hpp`），必需 | 系统包管理器（如fmt-devel）或conan |
| UMDK/URMA及UB（Unified Bus，统一总线）相关组件 | 跨节点数据面（READ/WRITE_NOTIFY），`ENABLE_URMA=ON`时必需 | 请参见下文"[安装UB相关组件与驱动](./installation_guide.md#安装ub相关组件与驱动)" |

索引构建依赖（Faiss/OpenBLAS（Open Basic Linear Algebra Subprograms，开放基础线性代数子程序库）等）由宿主侧构建组件承担，不属于本库构建范围。

## 编译安装

1. ### 安装系统依赖

   以openEuler 24.03 LTS SP3为例。

   ```bash
   yum install make cmake gcc g++ fmt-devel
   ```

   如系统源无fmt开发包，可经conan或源码安装fmt，并将其CMake包路径加入`CMAKE_PREFIX_PATH`。

2. ### 安装UB相关组件与驱动

   相关版本依赖如下。

   **表 4** UB相关组件与驱动版本

   | 软件名 | 版本 | 获取路径 |
   | --- | --- | --- |
   | kernel | 6.6.0-145.3.24.155.oe2403sp3 | 跟随openEuler 24.03 LTS SP3最新版本发布，通过yum install安装 |
   | umdk | 25.12.0-B122.oe2403sp3 | 跟随openEuler 24.03 LTS SP3最新版本发布，通过yum install安装 |
   | obmm | 1.0-3oe2403sp3 | 跟随openEuler 24.03 LTS SP3最新版本发布，通过yum install安装 |
   | libummu | 1.0.3-3oe2403sp3 | 跟随openEuler 24.03 LTS SP3最新版本发布，通过yum install安装 |
   | libcdma | 1.0.4-4oe2403sp3 | 跟随openEuler 24.03 LTS SP3最新版本发布，通过yum install安装 |
   | ubutils | 1.0.2-6oe2403sp3 | 跟随openEuler 24.03 LTS SP3最新版本发布，通过yum install安装 |
   | ubctl | 1.0.3-4oe2403sp3 | 跟随openEuler 24.03 LTS SP3最新版本发布，通过yum install安装 |
   | UBM | 2.0.0.B023-1 | [support网站获取](https://support.huawei.com/enterprise/zh/doc/EDOC1100621906/5e967bc6?idPath=23710424%7C251364417%7C251364851%7C254884027%7C269670789) |
   | UBE | 1.1.10.0.b142-openeuler24.03 | [support网站获取](https://support.huawei.com/enterprise/zh/doc/EDOC1100621906/5e967bc6?idPath=23710424%7C251364417%7C251364851%7C254884027%7C269670789) |
   | BIOS（Basic Input Output System，基本输入输出系统） | 2.16.0.B013 | [support网站获取](https://support.huawei.com/enterprise/zh/doc/EDOC1100621906/5e967bc6?idPath=23710424%7C251364417%7C251364851%7C254884027%7C269670789) |

   >![表示说明的图片](./public_sys-resources/icon-note.gif) **说明：**
   >UB相关组件的具体配套版本以发布版本的版本配套表为准。

3. ### 获取源码

   ```bash
   git clone https://gitcode.com/boostkit/gd-hnsw.git
   cd gd-hnsw
   ```

4. ### 编译gd_hnsw

   算法库以独立CMake工程交付（`gd_hnsw_urma/`目录）。当前工程未自行初始化GNUInstallDirs，独立构建时需显式补齐安装目录变量。以下示例统一使用`lib`目录，在仓库根目录执行：

   ```bash
   install_prefix="$PWD/install"
   cmake -S gd_hnsw_urma -B build \
       -DCMAKE_BUILD_TYPE=Release -DENABLE_URMA=ON \
       -DCMAKE_INSTALL_PREFIX="$install_prefix" \
       -DCMAKE_INSTALL_LIBDIR=lib \
       -DCMAKE_INSTALL_BINDIR=bin \
       -DCMAKE_INSTALL_FULL_LIBDIR="$install_prefix/lib" \
       -DCMAKE_INSTALL_FULL_INCLUDEDIR="$install_prefix/include" || exit 1
   cmake --build build --parallel "$(nproc)" || exit 1
   test -s build/libub_hnsw.so || exit 1
   printf 'BUILD_OK\n'

   cmake --install build || exit 1
   test -s "$install_prefix/lib/libub_hnsw.so" || exit 1
   test -s "$install_prefix/include/ub_hnsw/ub_c_api.h" || exit 1
   test -s "$install_prefix/lib/pkgconfig/ub_hnsw.pc" || exit 1
   test -s "$install_prefix/lib/cmake/ub_hnsw/ub_hnsw-config.cmake" || exit 1
   printf 'INSTALL_OK\n'
   ```

   以上命令在配置、编译或安装失败时立即退出。除CMake自身日志外，成功时会依次输出以下检查标记：

   ```text
   BUILD_OK
   INSTALL_OK
   ```

   `BUILD_OK`表示编译命令成功且构建目录中的动态库非空；`INSTALL_OK`表示安装命令成功，安装目录中的动态库、C ABI头文件、pkg-config文件及CMake包配置文件均存在且非空。未看到对应标记时，先处理该步骤的错误日志，再重新执行；这些检查不代替目标环境中的链接与检索验证。

   >![表示说明的图片](./public_sys-resources/icon-note.gif) **说明：**
   >
   > - `ENABLE_URMA`默认为`OFF`；开启后编译需能找到UMDK/URMA头文件与库（默认查找系统路径，亦可通过`UMDK_ROOT`环境变量或`CMAKE_PREFIX_PATH`指定安装路径）。
   > - `ENABLE_URMA=OFF`构建的库不含跨节点数据面，仅供编译与便捷验证，无法启用分布式检索。
   > - 安装前缀须在**配置阶段**经`-DCMAKE_INSTALL_PREFIX`指定：pkg-config与CMake包配置文件在配置阶段固化安装路径，仅在`cmake --install`阶段使用`--prefix`会导致包内路径与实际安装位置不一致。
   > - 当前版本安装规则未初始化GNUInstallDirs。独立构建时，须按示例显式提供安装目录变量。
   > - `CMAKE_INSTALL_LIBDIR`为空时，部分安装目标会展开为`/pkgconfig`、`/cmake/ub_hnsw`等绝对路径。
   > - `CMAKE_INSTALL_FULL_LIBDIR`为空时，生成的pkg-config文件会缺失库目录。`CMAKE_INSTALL_FULL_INCLUDEDIR`为空时，该文件会缺失头文件目录。
   > - 若使用`lib64`，须同步调整`CMAKE_INSTALL_LIBDIR`与`CMAKE_INSTALL_FULL_LIBDIR`；安装前缀应使用绝对路径。
   > - 安装后检查`lib/pkgconfig/ub_hnsw.pc`中的`libdir`、`includedir`及`lib/cmake/ub_hnsw/ub_hnsw-config.cmake`中的头文件路径，确认其与实际安装位置一致；使用`lib64`时相应替换目录名。宿主接入前仍需在目标Linux/URMA环境完成构建与链接验证。

   Milvus侧经`find_package(ub_hnsw)`引用本库（导入目标`ub_hnsw::ub_hnsw`），接入细节请参见Milvus侧资料。

### 编译产物

**表 5** gd_hnsw编译产物

| 产物 | 类型 | 说明 |
| --- | --- | --- |
| libub_hnsw.so | 库 | 检索运行时（core/index/compute/search/transport/runtime/protocol/control）与对外C ABI、C++接口 |
| 公共头文件 | 头文件 | `include/hnsw_urma/`（算法接口）与`include/ub_hnsw/ub_c_api.h`（C ABI） |
| ub_hnsw.pc | pkg-config文件 | 供CGO等经pkg-config链接（`-lub_hnsw`） |
| ub_hnsw CMake包 | 包配置 | `find_package(ub_hnsw)`可用，暴露导入目标`ub_hnsw::ub_hnsw` |

## 缩略语

| 缩略语 | 英文全称 | 中文全称 |
| --- | --- | --- |
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
| --- | --- | --- |
| 01 | 2026-12-30 | 第一次正式发布。 |

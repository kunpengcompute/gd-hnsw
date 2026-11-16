# 安装指南

本文介绍 gd-hnsw 的环境要求、依赖准备与构建步骤。

## 1. 环境要求

### 1.1 硬件与操作系统

| 项目 | 要求 |
| --- | --- |
| 服务器 | 鲲鹏 KP950 超节点，节点间 URMA 电组网互连 |
| OS | openEuler 24.03 LTS SP3 |
| 编译器 | GCC 12.3 |
| 构建工具 | CMake ≥ 3.16，make |
| 运行时依赖 | MPI（OpenMPI 或等价实现）、UMDK/URMA 用户态驱动、ubs-mem |

> 说明：无 URMA 设备的单机环境仅可用于编译验证与单分片调试，无法获得分布式检索能力。

### 1.2 第三方依赖（随仓子模块，无需单独安装）

| 依赖 | 版本 | 用途 |
| --- | --- | --- |
| faiss | v1.13.2 | 离线索引构建（HNSW 图构建与提取） |
| googletest | v1.14.0 | 单元测试 |

两个依赖均为 git 子模块，构建脚本会自动检出、编译并安装到 `dist/3rdparty/`。

## 2. 获取源码

```bash
git clone <仓库地址> gd-hnsw
cd gd-hnsw
git submodule update --init --recursive
```

> 若使用 `scripts/build.sh`，子模块初始化会自动完成，可跳过上面第二条命令。

## 3. 构建

一键构建脚本：

```bash
./scripts/build.sh
```

脚本依次完成：

1. glibc 版本检查（兼容性校验）；
2. 子模块初始化与第三方依赖构建（faiss / googletest，安装到 `dist/3rdparty/`）；
3. CMake 配置与编译（分层构建：`gd_hnsw_core` → 提取/构建层 → `gd_hnsw_api`）；
4. 单元测试（gtest）。

构建产物：

- 库文件：`libgd_hnsw_api.so` 等（公共头文件位于 `include/`）；
- 工具程序：`build_index`（离线索引构建）、`gd_hnsw_bench`（性能基准）；
- 第三方依赖安装目录：`dist/3rdparty/`。

如需手动分步构建，参考 `CMakeLists.txt` 顶层的分层条件编译选项，并使用 `-DCMAKE_PREFIX_PATH=dist/3rdparty` 指向依赖安装目录。

## 4. 验证安装

```bash
# 运行单元测试（构建脚本已默认执行）
cd build && ctest --output-on-failure
```

单元测试覆盖：SuperBlockV1 布局与 CRC 校验、HNSW 搜索算法正确性、L2 / dot+norm 距离核、minimax heap、索引读写、并发安全、doorbell 同步原语、pushdown 协议与搜索流程等。

构建通过后，请继续阅读 [快速入门](quick_start.md) 完成第一次索引构建与检索。

## 5. 常见问题

**Q：构建脚本报 glibc 版本不满足？**
A：gd-hnsw 依赖较新 glibc 特性，请确认 OS 为 openEuler 24.03 LTS SP3 或等价版本，不要混用旧发行版的工具链。

**Q：子模块拉取失败？**
A：检查网络代理配置后重新执行 `git submodule update --init --recursive`；faiss 仓库较大，首次拉取耗时较长属正常现象。

**Q：运行时报 URMA 相关错误？**
A：确认 UMDK 驱动已加载、节点间 URMA 组网连通，且每个进程启动环境具备内存注册权限（锁页内存 ulimit 配置）。

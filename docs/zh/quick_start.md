# 快速入门

本文介绍gd_hnsw与Milvus集成的功能验证流程，以及通过C ABI集成算法库的接入流程。

## 集成Milvus

### 特性简介

gd_hnsw是算法侧的全局分布式HNSW检索库，通过URMA提供跨分片图搜索与距离计算能力。Milvus是向量数据库，负责向量数据管理、索引构建与加载，并向用户提供检索接口。两者集成后，用户通过Milvus完成数据导入、建索引、加载和检索，由gd_hnsw执行全局图检索。

### 环境部署

本示例采用直接运行Milvus进程的方式，在一台具备URMA设备的鲲鹏服务器上启动8个工作QueryNode，使用8个全局图分片。参考环境及配套软件如下。

**表 1** 参考环境及配套软件

| 项目 | 本示例环境及软件要求 |
| ---- | -------------------- |
| 服务器 | 鲲鹏950处理器，AArch64（ARM64）架构，具备配套URMA设备 |
| CPU与NUMA分配 | 示例使用4个NUMA节点，每个工作QueryNode分配48个逻辑CPU；分配方法见“[规划QueryNode的CPU和NUMA](#规划querynode的cpu和numa)” |
| 操作系统 | openEuler 24.03 LTS SP3 |
| 编译器 | GCC 12.3，编译器与标准库支持C++20 |
| 构建工具 | CMake 3.22及以上版本 |
| 数据库 | Milvus v3.0.0，应用配套URMA补丁并与gd_hnsw联合编译 |
| 容器运行方式 | 使用Docker和Docker Compose运行etcd、MinIO、Pulsar；Milvus各组件在宿主机直接以进程方式运行 |
| etcd镜像 | `milvusdb/etcd:3.5.5-r2` |
| MinIO镜像 | `minio/minio:RELEASE.2024-05-28T17-19-04Z` |
| Pulsar镜像 | `apachepulsar/pulsar:2.8.2` |
| 验证客户端 | Python 3.9及以上、PyMilvus 3.0.1、h5py和NumPy |
| 其他工具 | Bash、Git、make、numactl和curl；完整数据库编译依赖请参见[数据库侧配套说明](https://gitcode.com/boostkit/milvus/tree/master) |

本示例在上述鲲鹏950、openEuler 24.03 LTS SP3环境中运行，仅作为功能验证参考。实际操作不绑定特定服务器名称或业务IP；使用满足配套要求的服务器，并根据本机CPU/NUMA拓扑、部署目录和服务地址填写后续参数。表中的CPU分配是本示例的配置，不作为所有部署场景的固定最低要求。中间件镜像取自本示例配套的Compose配置。

URMA设备及软件配套要求请参见《安装指南》中的“[环境要求](./installation_guide.md#环境要求)”和“[安装UB相关组件与驱动](./installation_guide.md#安装ub相关组件与驱动)”。

### 功能验证流程

gd_hnsw与Milvus集成后的功能验证流程如下。

1. [准备源码和运行环境](#准备源码和运行环境)。
2. [准备数据库侧中间件](#准备数据库侧中间件)。
3. [设置Milvus进程环境](#设置milvus进程环境)。
4. [按顺序启动集群](#按顺序启动集群)。
5. [准备GIST数据及客户端](#准备gist数据及客户端)。
6. [检查算法实际生效](#检查算法实际生效)。
7. [常见问题](#常见问题)。

#### 准备源码和运行环境

##### 获取源码及补丁

**表 2** 源码及配套材料

| 材料 | 获取方式 |
| ---- | -------- |
| Milvus官方源码 | [Milvus v3.0.0](https://github.com/milvus-io/milvus/releases/tag/v3.0.0) |
| Milvus URMA补丁 | [0001-milvus-add-urma-global-hnsw-search.patch](https://gitcode.com/boostkit/milvus/blob/master/0001-milvus-add-urma-global-hnsw-search.patch) |
| gd_hnsw算法源码 | [gd-hnsw仓库](https://gitcode.com/boostkit/gd-hnsw)，`master`分支的`gd_hnsw_urma/`目录 |
| Milvus编译环境及完整依赖 | 请参见[数据库侧配套说明](https://gitcode.com/boostkit/milvus/tree/master)中的编译环境与依赖说明 |
| 数据库运行环境 | 与补丁配套的完整`configs/milvus.yaml`、Docker Compose部署配置及其环境变量、覆盖文件；请参见[数据库侧配套说明](https://gitcode.com/boostkit/milvus/tree/master) |

本文将数据库团队提供、与所用补丁版本匹配的说明统称为“[数据库侧配套说明](https://gitcode.com/boostkit/milvus/tree/master)”。请从该仓库获取编译环境与依赖说明、数据库及中间件部署配置。使用前需核对完整`milvus.yaml`、Compose配置、环境变量、覆盖文件、数据卷与运行用户权限，以及实际服务入口和端口映射，不能仅凭Compose基础文件推定已具备完整部署配置。

在已具备配套编译环境的终端设置工作目录并获取源码。

```bash
export QS_ROOT=/opt/milvus-urma
export MILVUS_HOME="$QS_ROOT/milvus"
export ALGO_REPO="$QS_ROOT/gd-hnsw"
mkdir -p "$QS_ROOT/patches"

git clone --branch v3.0.0 --single-branch \
  https://github.com/milvus-io/milvus.git "$MILVUS_HOME"
git clone --branch master --single-branch \
  https://gitcode.com/boostkit/gd-hnsw.git "$ALGO_REPO"
test -f "$ALGO_REPO/gd_hnsw_urma/CMakeLists.txt"
```

从上表补丁页面下载原始补丁文件，保存为`$QS_ROOT/patches/0001-milvus-add-urma-global-hnsw-search.patch`，不要将网页HTML保存成补丁。

1. 进入Milvus源码目录，检查补丁是否可应用。只有检查成功时才执行下一步。

   ```bash
   cd "$MILVUS_HOME"
   git apply --check "$QS_ROOT/patches/0001-milvus-add-urma-global-hnsw-search.patch" && \
     printf 'PATCH_CHECK_OK\n'
   ```

   `git apply --check`成功时默认不输出内容，上述命令在检查成功后输出以下标记。

   ```output
   PATCH_CHECK_OK
   ```

2. 确认出现`PATCH_CHECK_OK`后，应用补丁并记录版本。

   ```bash
   git apply "$QS_ROOT/patches/0001-milvus-add-urma-global-hnsw-search.patch" || exit 1
   git rev-parse HEAD
   git -C "$ALGO_REPO" rev-parse HEAD
   sha256sum "$QS_ROOT/patches/0001-milvus-add-urma-global-hnsw-search.patch"
   ```

保留两个仓库的Commit及补丁校验值，便于追溯本次运行版本。补丁检查失败的处理请参见《常见问题和故障案例》中的“[Milvus URMA补丁检查失败](./FAQ.md#milvus-urma补丁检查失败)”。

##### 放置算法源码并联合编译

补丁工程使用`internal/core/thirdparty/ub_hnsw`作为算法源码目录，该目录下应直接包含算法的`CMakeLists.txt`、`include/`和`src/`，避免多嵌套一层`gd_hnsw_urma`。

对于尚未放入算法源码的新工作目录，执行以下命令。

```bash
export ALGO_TARGET="$MILVUS_HOME/internal/core/thirdparty/ub_hnsw"
mkdir -p "$ALGO_TARGET"
cp -a "$ALGO_REPO/gd_hnsw_urma/." "$ALGO_TARGET/"
test -f "$ALGO_TARGET/CMakeLists.txt"
test -d "$ALGO_TARGET/include"
test -d "$ALGO_TARGET/src"

cd "$MILVUS_HOME"
ENABLE_URMA=ON make
```

若补丁或配套工程已经放入算法代码，先核对该目录与选定算法版本一致，再编译。独立编译算法库的操作见《[安装指南](./installation_guide.md#编译gd_hnsw)》；本流程后续运行的是联合编译后的Milvus。

检查运行产物。

```bash
test -x "$MILVUS_HOME/bin/milvus"
test -f "$MILVUS_HOME/internal/core/output/lib/libmilvus_core.so"
test -f "$MILVUS_HOME/internal/core/output/lib/libjemalloc.so"
LD_LIBRARY_PATH="$MILVUS_HOME/internal/core/output/lib:${LD_LIBRARY_PATH:-}" \
  ldd "$MILVUS_HOME/internal/core/output/lib/libmilvus_core.so"
```

**通过标准**：构建成功，程序及动态库存在，动态链接依赖没有`not found`。Milvus编译环境、头文件或其他数据库编译依赖缺失时，请参见[数据库侧配套说明](https://gitcode.com/boostkit/milvus/tree/master)中的编译环境与依赖说明；URMA组件及算法独立编译依赖请分别参见《安装指南》中的“[安装UB相关组件与驱动](./installation_guide.md#安装ub相关组件与驱动)”和“[编译gd_hnsw](./installation_guide.md#编译gd_hnsw)”。

##### 检查URMA设备和链路

先按《[安装指南](./installation_guide.md#安装ub相关组件与驱动)》安装配套驱动及UMDK/URMA组件。在每个运行QueryNode的节点上，以后续启动Milvus的用户执行以下检查。本例8个工作QueryNode部署在同一台服务器上，在该服务器执行即可；跨服务器部署时，每台均需检查。

1. 检查工具是否可用，并枚举设备。

   ```bash
   command -v urma_admin || exit 1
   command -v urma_perftest || exit 1
   urma_admin show
   ```

   **通过标准**。命令正常返回且能看到计划使用的URMA设备。未找到工具时补齐配套工具；未列出设备时，先检查驱动安装、设备配置及当前用户的访问权限。

2. 查看目标设备的完整属性，核对端口状态、有效EID及传输模式。将`URMA_DEVICE`设置为上一步输出的实际设备名，以下以`bonding_dev_0`为例。

   ```bash
   export URMA_DEVICE=bonding_dev_0
   urma_admin show --dev "$URMA_DEVICE" --whole
   ```

   **通过标准**。目标设备存在，所用端口处于激活状态，计划使用的EID索引对应有效EID，且设备支持RM（可靠消息）模式。将设备名和EID索引分别填入后文`queryNode.urma.device`、`queryNode.urma.eidIdx`，不要仅因设备名称相同就默认索引为0。`trans_mode`若以位图输出，最低位为1表示支持RM，例如`0x1`和`0x7`均包含RM能力。命令及字段说明见[openEuler URMA用户指南](https://docs.openeuler.org/zh/docs/24.03_LTS_SP3/unifiedbus/unifiedbus/urma/URMA%20User%20Guide.ch.html)。

3. 跨服务器部署时，在两端执行URMA收发测试。

   1. 先在服务端终端执行以下命令。`URMA_DEVICE`和`URMA_EID_IDX`分别填写服务端实际设备名及有效EID索引，以下以`bonding_dev_0`和`0`为例。

      ```bash
      export URMA_DEVICE=bonding_dev_0
      export URMA_EID_IDX=0
      urma_perftest send_lat -d "$URMA_DEVICE" --eid_idx "$URMA_EID_IDX" \
        -p 0 -n 1000 -I 64 --ctp
      ```

   2. 再在客户端终端执行以下命令。设备名和EID索引填写客户端本地的实际值，`SERVER_URMA_IP`填写服务端实际的URMA通信IP。

      ```bash
      export URMA_DEVICE=bonding_dev_0
      export URMA_EID_IDX=0
      SERVER_URMA_IP="请替换为服务端URMA通信IP"
      urma_perftest send_lat -d "$URMA_DEVICE" --eid_idx "$URMA_EID_IDX" \
        -p 0 -n 1000 -I 64 --ctp -S "$SERVER_URMA_IP"
      ```

   这里`-p 0`指定RM模式，`-I 64`指定inline大小，EID索引由`--eid_idx`单独指定。

   **通过标准**：两端完成1000次收发测试，正常输出时延结果并成功退出，没有设备打开、建链或超时错误。按实际通信关系覆盖各节点间链路；失败时先修复设备、链路或权限问题，再启动Milvus。本章节仅验证基本通信，算法成功接入仍需完成后续章节。

##### 核对数据库连接和算法配置

将数据库侧配套的完整配置放在`$MILVUS_HOME/configs/milvus.yaml`。数据库服务地址、凭据、数据卷及其他数据库参数请参见[数据库侧配套说明](https://gitcode.com/boostkit/milvus/tree/master)中的部署配置说明。本文仅列与后续操作直接相关的连接信息和算法字段。

已跑通配置使用的访问端口如下，地址替换为本环境实际可访问的中间件地址。

**表 3** 数据库及中间件访问端口

| 字段 | 本例访问端口 |
| ---- | ------------ |
| `etcd.endpoints` | 31114 |
| `minio.address`、`minio.port` | 9008 |
| `pulsar.address`、`pulsar.port` | 6659 |
| `proxy.port` | 16745，后续`MILVUS_URI`与此一致 |

这些是Milvus访问服务的端口，不一定等于容器内部端口。需逐项核对完整Compose配置中的宿主机端口映射或现有服务的实际入口，使其与`milvus.yaml`一致；端口映射及服务入口的配置请参见[数据库侧配套说明](https://gitcode.com/boostkit/milvus/tree/master)。凭据、鉴权及数据库存储配置保留配套文件的实际值。

算法配置至少核对以下字段：下列是完整`milvus.yaml`中的局部内容，合并到对应位置，不要替换整个配置文件。

```yaml
queryNode:
  urma:
    enabled: true
    device: "bonding_dev_0"
    eidIdx: 0
    transportMode: rm
    globalIndex:
      enabled: true
      builder: faiss
  ub:
    enabled: true
    queryTimeoutMs: 120000
    operationTimeoutMs: 120000
    runtimeOptions: "--distance-policy pushdown --use-dot-norm --layer0-sort local-only --request-ring-depth 32 --profile"
```

`bonding_dev_0`为配套配置中的设备名，执行时按实际设备替换；EID索引也需对应实际设备。两个超时字段单位为毫秒。保留本版本已经验证的其他资源参数，修改后重新启动相关进程使其生效。

后面的QueryNode启动命令会覆盖部分YAML字段：配置文件中的Distance/RPC线程数为15/7，本流程运行时使用14/5；`cpuLayout`由`contiguous`覆盖为`llc-spread`，CPU来源由`cgroup`覆盖为显式CPU列表。`expandBatch=42`及`maxActiveQueries=32`同样由启动环境设置。运行选项中的`--profile`用于计时信息输出，功能是否成功仍通过四个验证模块和服务日志判断。

#### 准备数据库侧中间件

请参见[数据库侧配套说明](https://gitcode.com/boostkit/milvus/tree/master)中的部署配置说明，准备完整Compose部署配置、环境变量及需要的覆盖文件，并完成数据卷和运行用户权限配置。以下以这些文件放在Milvus工程根目录为例，实际目录不同则修改`COMPOSE_DIR`。

```bash
export COMPOSE_DIR="$MILVUS_HOME"
cd "$COMPOSE_DIR"
docker compose config --services
docker compose up -d etcd minio pulsar
docker compose ps etcd minio pulsar
docker compose logs --tail=80 etcd minio pulsar
```

**通过标准**：3项中间件运行正常，实际服务入口与前表及`milvus.yaml`一致。若运行环境已提供就绪中间件，核对连接即可继续。

本示例配套的Compose基础文件还包含编译容器定义，且未直接列出etcd、MinIO和Pulsar的宿主机端口映射。启动前需取得完整部署配置，并确认端口映射、环境变量、覆盖文件、数据卷和运行用户权限均已配置，具体请参见[数据库侧配套说明](https://gitcode.com/boostkit/milvus/tree/master)。本文继续使用已经准备好的数据库环境，不展开数据库组件内部配置。

#### 设置Milvus进程环境

后续启动命令在同一Bash终端依次执行。路径使用当前工作目录，日志和PID单独保存。

```bash
export MILVUS_BIN="$MILVUS_HOME/bin/milvus"
export MILVUS_YAML="$MILVUS_HOME/configs/milvus.yaml"
export INDEX_DIR="$QS_ROOT/index/gist"
export LOG_DIR="$MILVUS_HOME/logs"
export PID_DIR="$MILVUS_HOME/run"
mkdir -p "$INDEX_DIR" "$LOG_DIR" "$PID_DIR"

export LD_LIBRARY_PATH="$MILVUS_HOME/internal/core/output/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$MILVUS_HOME/internal/core/output/lib/libjemalloc.so"
export MALLOC_CONF=background_thread:true
export GODEBUG=madvdontneed=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY ftp_proxy PROXY_IP
export no_proxy=127.0.0.1,localhost

cd "$MILVUS_HOME"
ps -eo pid,args | grep -F "$MILVUS_BIN run" | grep -v grep || true
ss -lntp | grep -E ':(16745|183[0-9]|184[01])\b' || true
```

`INDEX_DIR`沿用配套启动环境中的本地索引目录参数。目录名称不代表实际分片数量；本例全局图由后面的`create_index`构建，目标分片数明确设置为8。应使用匹配补丁的全局索引配置，避免误用其他测试遗留的离线索引。

##### 规划QueryNode的CPU和NUMA

启动参数使用8个QueryNode，每进程分配48个逻辑CPU。下面是已验证启动命令采用的分配方式，执行前先核对实际拓扑。

```bash
lscpu
numactl --hardware
```

**表 4** QueryNode的CPU与NUMA分配示例

| QueryNode编号 | 业务端口 | CPU范围示例 | NUMA节点示例 |
| ------------- | -------- | ----------- | ------------ |
| 0 | 1832 | 0～47 | 0 |
| 1 | 1833 | 48～95 | 0 |
| 2 | 1834 | 96～143 | 1 |
| 3 | 1835 | 144～191 | 1 |
| 4 | 1836 | 192～239 | 2 |
| 5 | 1837 | 240～287 | 2 |
| 6 | 1838 | 288～335 | 3 |
| 7 | 1839 | 336～383 | 3 |

表中CPU编号必须存在、在进程允许范围内，并与所选NUMA节点对应。其他CPU拓扑应重新填写下面的数组；这套48核分配是示例配置，不能当作算法在所有场景下的固定资源要求。

```bash
QN_CPU_RANGES=(0-47 48-95 96-143 144-191 192-239 240-287 288-335 336-383)
QN_NUMA_NODES=(0 0 1 1 2 2 3 3)
```

#### 按顺序启动集群

以下顺序沿用本次跑通的进程启动方式。前面启动的进程可能在等待协调服务期间重试连接，因此每步先检查进程和日志，全部组件启动后再统一验证服务可用性。等待时间用于复现启动节奏，不能单独作为健康判据。

1. 启动streamingnode。

   ```bash
   cd "$MILVUS_HOME"
   METRICS_PORT=1830 nohup "$MILVUS_BIN" run streamingnode --run-with-subprocess \
     > "$LOG_DIR/streamingnode.log" 2>&1 &
   echo "$!" > "$PID_DIR/streamingnode.pid"
   sleep 15
   kill -0 "$(cat "$PID_DIR/streamingnode.pid")"
   tail -n 30 "$LOG_DIR/streamingnode.log"
   ```

2. 启动datanode。

   ```bash
   METRICS_PORT=1831 nohup "$MILVUS_BIN" run datanode --run-with-subprocess \
     > "$LOG_DIR/datanode.log" 2>&1 &
   echo "$!" > "$PID_DIR/datanode.pid"
   sleep 10
   kill -0 "$(cat "$PID_DIR/datanode.pid")"
   tail -n 30 "$LOG_DIR/datanode.log"
   ```

3. 启动8个工作QueryNode。

   每个进程使用独立业务端口及CPU集合，保留已跑通命令中的算法参数：`expandBatch=42`、`maxActiveQueries=32`，Search/Distance/READ/RPC工作线程分别为10/14/2/5。

   ```bash
   for n in 0 1 2 3 4 5 6 7; do
       port=$((1832 + n))
       cpu_range="${QN_CPU_RANGES[$n]}"
       numa_node="${QN_NUMA_NODES[$n]}"
       nohup numactl --physcpubind="$cpu_range" --membind="$numa_node" \
         env MILVUS_CONF_querynodeport="$port" \
         MILVUS_CONF_querynodeUbSeedDir="$INDEX_DIR" \
         MILVUS_CONF_querynodeUbExpandBatch=42 \
         MILVUS_CONF_querynodeUbMaxActiveQueries=32 \
         MILVUS_CONF_querynodeUbThreadsSearch=10 \
         MILVUS_CONF_querynodeUbThreadsDistance=14 \
         MILVUS_CONF_querynodeUbThreadsReadProgress=2 \
         MILVUS_CONF_querynodeUbThreadsRpcProgress=5 \
         MILVUS_CONF_querynodeUbCpuBindMode=auto \
         MILVUS_CONF_querynodeUbCpuLayout=llc-spread \
         MILVUS_CONF_querynodeUbAllowedCpusSource=yaml \
         MILVUS_CONF_querynodeUbAllowedCpus="$cpu_range" \
         METRICS_PORT="$port" \
         "$MILVUS_BIN" run querynode --run-with-subprocess \
         > "$LOG_DIR/querynode_$port.log" 2>&1 &
       echo "$!" > "$PID_DIR/querynode_$port.pid"
       sleep 2
       kill -0 "$(cat "$PID_DIR/querynode_$port.pid")" || exit 1
   done
   sleep 20
   ```

   端口用于区分进程，不能直接作为全局图分片编号。实际分片归属由本版本接入层的装载和分配逻辑决定，在后续Load及算法日志中核对8个全局分片与参与进程的对应关系。

4. 启动mixcoord。

   ```bash
   METRICS_PORT=1840 nohup "$MILVUS_BIN" run mixcoord --run-with-subprocess \
     > "$LOG_DIR/mixcoord.log" 2>&1 &
   echo "$!" > "$PID_DIR/mixcoord.pid"
   sleep 20
   kill -0 "$(cat "$PID_DIR/mixcoord.pid")"
   tail -n 30 "$LOG_DIR/mixcoord.log"
   ```

5. 启动proxy。

   ```bash
   METRICS_PORT=1841 nohup "$MILVUS_BIN" run proxy --run-with-subprocess \
     > "$LOG_DIR/proxy.log" 2>&1 &
   echo "$!" > "$PID_DIR/proxy.pid"
   sleep 30
   kill -0 "$(cat "$PID_DIR/proxy.pid")"
   tail -n 30 "$LOG_DIR/proxy.log"
   ```

6. 检查进程和日志。

   ```bash
   for pid_file in "$PID_DIR"/*.pid; do
       pid=$(cat "$pid_file")
       kill -0 "$pid" || { printf '进程已退出: %s\n' "$pid_file"; exit 1; }
   done
   ps -eo pid,args | grep -F "$MILVUS_BIN run" | grep -v grep
   ss -lntp | grep -E ':(16745|183[0-9]|184[01])\b' || true
   tail -n 50 "$LOG_DIR/mixcoord.log" "$LOG_DIR/proxy.log"
   ```

   **通过标准**。本次启动的各组件及8个QueryNode持续运行，没有持续的连接、设备或资源错误，Proxy侦听端口与配置一致。由于采用子进程启动，`ps`行数不等于逻辑组件数；还需继续用客户端连接和四个功能模块验证整个链路。

#### 准备GIST数据及客户端

本步骤准备客户端与GIST数据，完成后按顺序执行下文的数据导入、全局索引构建、加载和检索四个验证模块。

##### 安装Python依赖

本示例使用Milvus v3.0.0和PyMilvus 3.0.1。PyMilvus是Milvus的Python SDK，客户端与服务端版本号不要求完全相同；官方[Milvus v3.0.0发布说明](https://github.com/milvus-io/milvus/releases/tag/v3.0.0)的配套表列出的Python SDK版本为3.0.1，[PyMilvus v3.0.1发布说明](https://github.com/milvus-io/pymilvus/releases/tag/v3.0.1)也将其推荐用于Milvus 3.0。因此下方安装命令固定为`pymilvus==3.0.1`，与环境表保持一致。该SDK要求Python 3.9及以上；配套补丁版本若另有要求，以其说明为准。

```bash
python3 -m venv "$QS_ROOT/.venv"
source "$QS_ROOT/.venv/bin/activate"
python -m pip install 'pymilvus==3.0.1' h5py numpy
python -c 'import pymilvus, h5py, numpy; print("pymilvus", pymilvus.__version__)'
```

##### 获取数据并设置公共参数

本例直接读取HDF5格式的GIST数据，包含`train`、`test`和`neighbors`，不需要转换为VectorDBBench格式。数据来源及格式说明见[ANN-Benchmarks数据集](https://github.com/erikbern/ann-benchmarks#data-sets)。

```bash
export MILVUS_HDF5="$QS_ROOT/datasets/gist-960-euclidean.hdf5"
mkdir -p "$(dirname "$MILVUS_HDF5")"
# 已有同一数据文件时，跳过下载并修改上面的路径
curl -fL https://ann-benchmarks.com/gist-960-euclidean.hdf5 -o "$MILVUS_HDF5"

export MILVUS_URI=http://127.0.0.1:16745
export MILVUS_COLLECTION="gist_smoke_$(date +%Y%m%d_%H%M%S)_$RANDOM"
export MILVUS_DIM=960
export MILVUS_TOPK=10
export MILVUS_NQ=1
export MILVUS_EF=200
export MILVUS_M=64
export MILVUS_EFC=40
export MILVUS_SHARDS=8
export MILVUS_GLOBAL_SHARDS=8
export MILVUS_INSERT_BATCH=10000
export MILVUS_RPC_TIMEOUT=600
export MILVUS_TIMEOUT=86400
printf '本次集合: %s\n' "$MILVUS_COLLECTION"
```

`MILVUS_URI`使用当前Proxy实际地址和端口。此连接示例沿用未启用鉴权的隔离功能测试环境。与本补丁配套的鉴权配置请参见[数据库侧配套说明](https://gitcode.com/boostkit/milvus/tree/master)，启用鉴权及配置连接凭据的方法请参见Milvus官方文档“[Authenticate User Access](https://milvus.io/docs/authenticate.md#Connect-to-Milvus-with-authentication)”。启用后，下面各模块的`connections.connect`均需按实际凭据配置，连接示例请参见官方“[connect()](https://milvus.io/api-reference/pymilvus/v3.0.x/ORM/Connections/connect.md)”的`token`、`user`和`password`参数说明。

接下来的4个模块在同一已激活虚拟环境的终端中顺序执行，始终使用同一个`MILVUS_COLLECTION`。更换终端时重新设置相同值，不要在步骤之间重新生成集合名。每个模块单独连接Milvus，便于定位失败阶段。

#### 模块一：创建集合并导入GIST

本模块导入全部100万条训练向量，将训练行号作为主键，保证后面能与HDF5真值ID对应。每批10000条，每批约36.6 MiB原始float32向量数据，仍需考虑RPC消息和序列化开销。

```bash
python - <<'PY'
import os
import h5py
import numpy as np
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

uri = os.environ["MILVUS_URI"]
name = os.environ["MILVUS_COLLECTION"]
rpc_timeout = float(os.environ["MILVUS_RPC_TIMEOUT"])
stage_timeout = float(os.environ["MILVUS_TIMEOUT"])
dim = int(os.environ["MILVUS_DIM"])
batch_size = int(os.environ["MILVUS_INSERT_BATCH"])
assert batch_size > 0
connections.connect(uri=uri, timeout=rpc_timeout)
if utility.has_collection(name):
    raise RuntimeError(f"集合 {name} 已存在，请确认状态，不要重复导入")

with h5py.File(os.environ["MILVUS_HDF5"], "r") as h5:
    for key in ("train", "test", "neighbors"):
        assert key in h5, f"HDF5缺少{key}"
    rows, actual_dim = h5["train"].shape
    assert rows == 1_000_000 and actual_dim == dim == 960
    schema = CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ], description="GIST global HNSW functional verification")
    col = Collection(name=name, schema=schema, consistency_level="Strong",
                     shards_num=int(os.environ["MILVUS_SHARDS"]), timeout=rpc_timeout)
    for begin in range(0, rows, batch_size):
        end = min(begin + batch_size, rows)
        vectors = np.asarray(h5["train"][begin:end], dtype=np.float32)
        ids = np.arange(begin, end, dtype=np.int64)
        result = col.insert([ids.tolist(), vectors.tolist()], timeout=stage_timeout)
        assert result.insert_count == end - begin
        if end % 100000 == 0 or end == rows:
            print(f"inserted {end}/{rows}", flush=True)
    col.flush(timeout=stage_timeout)
    assert col.num_entities == rows, f"行数不符: {col.num_entities}"
    print("IMPORT_OK", name, "rows=", col.num_entities)
connections.disconnect("default")
PY
```

**通过标准**：最后输出`IMPORT_OK`，行数为1000000。失败时先排查连接、消息大小、存储和写入状态，不要重复执行导入导致数据状态混淆。

#### 模块二：构建8分片全局HNSW索引

`index_type`使用`HNSW`，在索引参数中传入`global_index_shard_count`，由配套Milvus接入层创建全局图。集合写入分片数与全局图分片数是不同参数，本例都取8，全局图参与进程数也应匹配8。

```bash
python - <<'PY'
import os
from pymilvus import Collection, connections, utility

name = os.environ["MILVUS_COLLECTION"]
rpc_timeout = float(os.environ["MILVUS_RPC_TIMEOUT"])
stage_timeout = float(os.environ["MILVUS_TIMEOUT"])
connections.connect(uri=os.environ["MILVUS_URI"], timeout=rpc_timeout)
col = Collection(name)
assert col.num_entities == 1_000_000
col.create_index(
    field_name="vector", index_name="idx_vector",
    index_params={
        "index_type": "HNSW",
        "metric_type": "L2",
        "params": {
            "M": int(os.environ["MILVUS_M"]),
            "efConstruction": int(os.environ["MILVUS_EFC"]),
            "global_index_shard_count": str(int(os.environ["MILVUS_GLOBAL_SHARDS"])),
        },
    },
    timeout=stage_timeout,
)
utility.wait_for_index_building_complete(name, index_name="idx_vector", timeout=stage_timeout)
progress = utility.index_building_progress(name, index_name="idx_vector", timeout=rpc_timeout)
print("index progress:", progress)
assert int(progress.get("indexed_rows", -1)) >= 1_000_000
print("INDEX_OK")
connections.disconnect("default")
PY
```

**通过标准**：等待索引构建完成，已索引行数覆盖100万条，输出`INDEX_OK`。构建超时时查看datanode及相关构建日志，不能把提交索引请求成功视为构建完成。

#### 模块三：加载集合

加载刚才构建的索引，等待数据就绪，并打印副本和QueryNode信息供核对。

```bash
python - <<'PY'
import os
from pymilvus import Collection, connections, utility

name = os.environ["MILVUS_COLLECTION"]
rpc_timeout = float(os.environ["MILVUS_RPC_TIMEOUT"])
stage_timeout = float(os.environ["MILVUS_TIMEOUT"])
connections.connect(uri=os.environ["MILVUS_URI"], timeout=rpc_timeout)
col = Collection(name)
col.load(replica_number=1, timeout=stage_timeout)
utility.wait_for_loading_complete(name, timeout=stage_timeout)
print("loading progress:", utility.loading_progress(name, timeout=rpc_timeout))
print("load state:", utility.load_state(name))
try:
    replicas = col.get_replicas()
    groups = getattr(replicas, "groups", [replicas]) if not isinstance(replicas, (list, tuple)) else replicas
    for group in groups:
        node_ids = getattr(group, "node_ids", getattr(group, "group_nodes", []))
        print("replica group nodes:", sorted(int(x) for x in node_ids))
except Exception as exc:
    print("replica information unavailable:", exc)
print("LOAD_OK")
connections.disconnect("default")
PY
```

**通过标准**：装载等待完成，进度为100%，集合Loaded并输出`LOAD_OK`。结合QueryNode日志核对全局图8个分片的实际归属和资源发布；客户端副本信息只能辅助检查，不能单独证明算法链路已启用。

#### 模块四：检索并计算样本召回率

读取HDF5中的前`MILVUS_NQ`条查询向量及对应真值，以`ef=200`、top-K为10执行裸向量检索。默认只查1条，属于功能冒烟测试，不统计QPS。

```bash
python - <<'PY'
import os
import h5py
import numpy as np
from pymilvus import Collection, connections

name = os.environ["MILVUS_COLLECTION"]
topk = int(os.environ["MILVUS_TOPK"])
nq = int(os.environ["MILVUS_NQ"])
ef = int(os.environ["MILVUS_EF"])
rpc_timeout = float(os.environ["MILVUS_RPC_TIMEOUT"])
assert 1 <= nq <= 100
assert 0 < topk <= ef
with h5py.File(os.environ["MILVUS_HDF5"], "r") as h5:
    rows = h5["train"].shape[0]
    assert topk <= h5["neighbors"].shape[1]
    queries = min(nq, h5["test"].shape[0])
    assert queries > 0
    vectors = np.asarray(h5["test"][:queries], dtype=np.float32)
    ground_truth = np.asarray(h5["neighbors"][:queries, :topk], dtype=np.int64)

connections.connect(uri=os.environ["MILVUS_URI"], timeout=rpc_timeout)
col = Collection(name)
results = col.search(
    data=vectors.tolist(), anns_field="vector",
    param={"metric_type": "L2", "params": {"ef": ef}},
    limit=topk, consistency_level="Strong", timeout=rpc_timeout,
)
assert len(results) == queries
matches = 0
for i, hits in enumerate(results):
    ids = [int(hit.id) for hit in hits]
    distances = [float(hit.distance) for hit in hits]
    assert len(ids) == topk and len(set(ids)) == topk
    assert all(0 <= value < rows for value in ids)
    assert np.isfinite(distances).all()
    truth = set(int(value) for value in ground_truth[i])
    assert len(truth) == topk and all(0 <= value < rows for value in truth)
    count = len(set(ids) & truth)
    matches += count
    print(f"query={i} recall@{topk}={count / topk:.4f}")
    print("ids:", ids)
    print("distances:", distances)
recall = matches / (queries * topk)
print(f"SEARCH_COMPLETED recall@{topk}={recall:.4f} nq={queries}")
connections.disconnect("default")
PY
```

**通过标准**：每条查询返回10个有效且不重复的ID及有限距离，输出`SEARCH_COMPLETED`和样本Recall@10，无检索异常。样本召回率用于检查本次查询，不能用单条查询的结果代表整个数据集的召回率，也不使用任意低阈值直接宣称质量达标。

#### 检查算法实际生效

四个功能模块完成后，再检查本次集合对应的服务日志。

```bash
for port in 1832 1833 1834 1835 1836 1837 1838 1839; do
    printf '\n=== querynode %s ===\n' "$port"
    grep -Ei 'URMA|global.*shard|UB effective|UB worker path search done|mailbox|UbEnableDistributedSearch' \
      "$LOG_DIR/querynode_$port.log" | tail -50
done
```

核对实际运行版本及配置、8个全局分片与参与进程的归属、URMA资源发布及对端导入，以及本次查询的算法完成记录。单纯返回top-K或日志里出现`URMA`字样不能代替链路核验；日志名称以所部署版本为准。

本次流程完成后，保留集合、索引和日志，便于复查。重复测试时不要自动删除同名前缀的所有集合；先确认旧集合和图状态，按配套数据库流程释放或重置后再开始新一轮。

#### 常见问题

**表 5** 集成Milvus常见问题及检查方向

| 现象 | 检查方向 |
| ---- | -------- |
| 补丁无法应用或联合编译失败 | Milvus版本、补丁、算法目录及接口是否匹配，完整编译依赖是否已安装 |
| Pulsar无法启动 | Compose实际数据卷、运行UID/GID和目录权限，查看容器日志 |
| Milvus进程刚启动就退出 | 动态库、配置文件、端口、URMA设备，检查对应组件日志 |
| numactl或CPU绑定失败 | CPU编号是否存在且允许使用，CPU范围与NUMA节点是否匹配 |
| 客户端连不上 | Proxy实际侦听端口、地址、代理设置及鉴权配置 |
| 导入或建索引超时 | 写入批大小、中间件状态、存储、构建日志及当前阶段进度 |
| Load或检索等待、报资源未就绪 | 8个工作QueryNode、实际全局分片数、URMA链路和图资源发布状态 |
| 召回异常 | 是否导入全量数据，主键与训练行号是否一致，查询及真值是否来自同一HDF5，实际算法及ef是否生效 |

集成Milvus过程中的常见问题以及检查办法请参见《常见问题和故障案例》中的“[跑通路径验证失败排查](./FAQ.md#跑通路径验证失败排查)”，更多算法接口语义请参见《API参考》中的“[调用约定](./api_reference.md#概述)”、“[配置接口](./api_reference.md#配置接口)”和“[启用与检索接口](./api_reference.md#启用与检索接口)”。

## 集成C ABI

C ABI指遵循C语言调用约定的应用二进制接口，其中ABI是Application Binary Interface（应用二进制接口）的缩略语。gd_hnsw通过公共头文件`ub_c_api.h`提供C兼容接口，供Milvus等宿主调用算法库，完成运行时配置、分片装载、资源交换和检索。

本例由Milvus接入层完成算法库调用，部署和测试过程中无需手工调用C接口。自行开发宿主时，按以下生命周期接入。各接口的参数和前置条件请参见《API参考》中的“[配置接口](./api_reference.md#配置接口)”、“[装载接口](./api_reference.md#装载接口)”、“[资源发布与导入接口](./api_reference.md#资源发布与导入接口)”、“[启用与检索接口](./api_reference.md#启用与检索接口)”和“[释放接口](./api_reference.md#释放接口)”；调用约定和错误处理请参见“[概述](./api_reference.md#概述)”、“[状态码](./api_reference.md#状态码)”及“[UbLastError](./api_reference.md#ublasterror)”。

1. 通过`UbSetUrmaContext`注入宿主持有的URMA上下文。
2. 通过`UbConfigure`配置运行时，调用`UbEstimateRuntimeMemory`检查配置并估算内存。
3. 通过`UbLoadShardFromDir`装载本进程的分片，通过`UbSetLocalShard`设置本地分片身份。
4. 通过`UbBindGlobalGraphIdentity`绑定全局图身份。
5. 通过`UbEnsureRpcMailbox`创建距离请求、响应和通知邮箱。
6. 通过`UbGetGraphPublishSnapshot`取得图和向量快照，由宿主发布本分片的五类注册内存资源。
7. 由宿主发现对端记录并激活链路，通过`UbImportPeerHandles`导入对端资源句柄。
8. 资源就绪后调用`UbEnableDistributedSearch`启用分布式检索。
9. 调用`UbSearch`或`UbSearchWithTimeout`发起查询。
10. 通过`UbStopDistributedSearch`停止运行时，通过`UbShutdown`释放状态。

## 缩略语

**表 6** 缩略语

| 缩略语 | 英文全称 | 中文全称 |
| ------ | -------- | -------- |
| ABI | Application Binary Interface | 应用二进制接口 |
| HNSW | Hierarchical Navigable Small World | 分层可导航小世界 |
| NUMA | Non-Uniform Memory Access | 非一致内存访问 |
| QN | QueryNode | 查询节点 |
| URMA | Unified Remote Memory Access | 统一远端内存访问 |

## 修订记录

**表 7** 修订记录

| 文档版本 | 发布日期 | 修改说明 |
| -------- | -------- | -------- |
| 01 | 2026-12-30 | 第一次正式发布。 |

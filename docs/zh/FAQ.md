# 常见问题和故障案例（FAQ）

本文提供gd_hnsw常见问题和故障案例，涵盖安装编译、部署环境、原理与使用等类别内容。（条目持续补充）

## 安装编译相关

### Milvus URMA补丁检查失败

**问题**

按《快速入门》中的“[获取源码及补丁](./quick_start.md#获取源码及补丁)”执行`git apply --check`失败，或没有输出`PATCH_CHECK_OK`。

**回答**

检查命令仅验证补丁能否应用，不会修改源码。检查失败时停止后续应用和构建，按以下顺序排查。

1. 确认当前目录是选定的Milvus源码仓库，核对`git describe --tags --always`和`git rev-parse HEAD`，本示例要求Milvus v3.0.0及其配套URMA补丁。
2. 检查补丁文件是否完整，确认下载的是原始补丁内容而非HTML网页，并记录文件的SHA-256校验值。提示`No valid patches in input`时，优先核对文件内容及下载方式。
3. 执行`git status --short`检查工作区。出现`patch does not apply`时，核对源码版本、已有修改以及补丁是否已应用。可用`git apply --reverse --check <补丁路径>`辅助检查能否反向应用，但不要据此直接回退源码或丢弃现有修改。
4. 确认源码与补丁配套且工作区状态符合要求后，重新执行快速入门中的检查命令；出现`PATCH_CHECK_OK`后再应用补丁。仍失败时，保留版本、补丁校验值及完整错误信息，交由配套补丁提供方确认。

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
2. 各节点URMA设备可用，通过`urma_admin show`枚举设备，并按《[快速入门](./quick_start.md#检查urma设备和链路)》核对设备属性、EID及节点间收发测试；
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

排查建议：按《[快速入门](./quick_start.md#检查urma设备和链路)》确认设备、EID及RM能力，确认库以`ENABLE_URMA=ON`编译，并检查接入流程已注入上下文。

### 跑通路径验证失败排查

**问题**

按《[快速入门](./quick_start.md)》跑通路径逐步验证时，某环节验证不通过，如何定位？

**回答**

先定位失败环节，再按下表对照排查。算法库接口返回非0时，以本次返回码为准，并读取`UbLastError`中可用的错误详情。部分接口不会更新该字符串，详情为空或残留旧值时，应结合本次调用阶段和宿主日志定位。

**日志查看入口**

算法库默认将带`[ub_hnsw INFO]`等前缀的日志写入进程标准错误，不使用固定日志文件；宿主可接管日志输出。Milvus的`log.file.rootPath`为空时，日志默认输出到标准输出和标准错误；配置该项后，到配置目录查看日志。

快速入门通过Docker Compose运行etcd、MinIO和Pulsar，Milvus各组件直接以进程方式启动。启动命令将标准输出和标准错误重定向到`$LOG_DIR`下的组件日志，并把PID记录到`$PID_DIR`；优先检查这些日志和PID，以及Compose中间件状态。

采用其他部署方式时，按实际环境补充日志检查：Docker使用`docker logs <容器名> --tail 200`，Kubernetes使用`kubectl logs -n <命名空间> <Pod名> -c <容器名> --tail=200`；容器重启后可加`--previous`查看上一次容器日志。

| 环节 | 常见失败原因 | 自检命令/接口 |
| --- | --- | --- |
| 环境准备 | URMA设备未识别、EID无效、传输模式不匹配、节点间通信失败 | 按快速入门检查URMA设备和链路；另用`nc`/`wget`检查数据库服务的TCP/HTTP连通性 |
| 编译/获取算法库 | 未找到UMDK/URMA头文件与库、fmt缺失、编译器过旧 | 见本文"安装编译相关"条目 |
| 部署集群 | 组件退出、中间件连接失败、工作QueryNode数量与全局图分片数不匹配 | 检查PID、组件日志和Compose状态；本例8个工作QueryNode对应8个全局图分片，不计Proxy、协调组件及其他数据处理组件 |
| 导入数据 | 连接失败、写入失败、数据状态或行数不符 | 检查客户端输出、datanode及中间件日志，确认`IMPORT_OK`与100万条数据 |
| 构建全局索引 | 构建任务未完成、参数或构建组件异常 | 检查`create_index`结果、索引构建进度与datanode相关日志，确认`INDEX_OK`及已索引行数 |
| 装载并启用 | 分片未装载、资源未发布、对端未导入、URMA上下文未注入 | 检查加载进度及QueryNode日志，核对8个全局分片归属、资源发布和对端导入；客户端Load完成不能单独证明算法已启用 |
| 查询验证 | 查询参数越界、集合未加载、运行时未启用 | 检查客户端错误、`SEARCH_COMPLETED`及对应QueryNode算法完成日志；核对距离、top-K和ef |

自行开发宿主、直接调用C ABI时，再按《[API参考](./api_reference.md)》核对`global.meta`、`upper_graph.index`及本地`shard_N.index`，并使用`UbGetShardStatus`、`UbEnableDistributedSearch`和`UbLastError`辅助定位。`UbRpcDistanceReady`仅表示运行时对象是否存在，`UbPing`仅验证接口调用连通，均不能代替实际检索验证。快速入门中的索引由Milvus接入层构建和装载，无需手工向`INDEX_DIR`放置索引三元组。

**常用自检命令**

以下目录沿用快速入门。根据实际配置修改路径及服务地址；本例Proxy端口为16745、etcd访问端口为31114，不一定等于容器内部端口。`nc`检查TCP端口，`wget`只用于HTTP服务；不要用HTTP探测命令测试仅提供gRPC的接口。

```bash
export QS_ROOT=/opt/milvus-urma
export MILVUS_HOME="$QS_ROOT/milvus"
export LOG_DIR="$MILVUS_HOME/logs"
export PID_DIR="$MILVUS_HOME/run"
export COMPOSE_DIR="$MILVUS_HOME"  # 完整Compose配置所在目录
PROXY_HOST=127.0.0.1              # 按Proxy实际地址修改
ETCD_HOST=请替换为milvus.yaml中etcd的实际地址

# 查看URMA设备，完整检查步骤见快速入门。
urma_admin show

# 检查客户端到Milvus Proxy，以及节点到etcd的TCP连通性。
nc -vz "$PROXY_HOST" 16745
nc -vz "$ETCD_HOST" 31114

# 检查本机监听端口和时间同步状态。
ss -lntp
timedatectl status

# 检查本次启动的Milvus进程及日志。
for pid_file in "$PID_DIR"/*.pid; do
    test -s "$pid_file" || { printf '请核对PID目录: %s\n' "$PID_DIR"; exit 1; }
    pid=$(cat "$pid_file")
    if kill -0 "$pid" 2>/dev/null; then
        ps -p "$pid" -o pid,args
    else
        printf '进程已退出: %s\n' "$pid_file"
    fi
done
tail -n 50 "$LOG_DIR/mixcoord.log" "$LOG_DIR/proxy.log" "$LOG_DIR/datanode.log"
tail -n 50 "$LOG_DIR"/querynode_*.log

# 检查Compose中间件；目录应包含与启动时相同的完整部署配置。
(cd "$COMPOSE_DIR" && docker compose ps && \
  docker compose logs --tail=50 etcd minio pulsar)
```

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
| --- | --- | --- |
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
| --- | --- | --- |
| 01 | 2026-12-30 | 第一次正式发布。 |

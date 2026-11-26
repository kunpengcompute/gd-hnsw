# 最佳实践

本文档提供gd_hnsw与Milvus集成的实践案例，旨在帮助用户实现在具体场景中调用gd_hnsw算法。

## 约束与限制

- 数据库编译依赖、中间件部署、完整数据库配置请参见[Milvus侧配套说明](https://gitcode.com/boostkit/milvus/tree/master)。
- 本文档关注全局分布式检索算法，不限定具体服务器名称、业务IP或个人工作目录。
- 快速入门默认只查询1条向量，用于验证功能流程。本实践扩大查询样本并重复测量，查询参数和运行时起点与快速入门一致；实际质量和时延以当前版本、数据及运行环境下的结果为准。

## 集成milvus

### 环境部署

#### 保存环境信息

下面使用与快速入门相同的目录。`MILVUS_COLLECTION`必须填写快速入门已经创建并加载的实际集合名，不要重新生成名字。

```bash
export QS_ROOT=/opt/milvus-urma
export MILVUS_HOME="$QS_ROOT/milvus"
export MILVUS_BIN="$MILVUS_HOME/bin/milvus"
export LOG_DIR="$MILVUS_HOME/logs"
export PID_DIR="$MILVUS_HOME/run"
export MILVUS_URI=http://127.0.0.1:16745
export MILVUS_HDF5="$QS_ROOT/datasets/gist-960-euclidean.hdf5"
export MILVUS_COLLECTION=请填写快速入门保留的集合名
export RESULT_DIR="$QS_ROOT/results/gist"
source "$QS_ROOT/.venv/bin/activate"
mkdir -p "$RESULT_DIR"

uname -a
lscpu
numactl --hardware
free -h
df -h / /var
sha256sum "$MILVUS_BIN" \
  "$MILVUS_HOME/internal/core/output/lib/libmilvus_core.so"
```

记录实际软件版本、设备与驱动、CPU/NUMA分配、配置文件和环境变量。配置和连接凭据按数据库侧规则保管；测试报告中只摘录必要字段。

### 前提条件

#### 部署Checklist

1. **功能流程完成**：快速入门四个模块均成功，集合包含100万条数据，索引构建和Load完成，客户端能检索。
2. **版本可追溯**：记录Milvus和算法Commit、URMA补丁校验值，以及实际运行的`bin/milvus`和动态库校验值。
3. **数据库连接正确**：使用数据库侧完整配置，etcd、MinIO、Pulsar入口与`milvus.yaml`一致；本例Proxy客户端端口为16745。
4. **算法链路生效**：URMA及全局索引开关启用，8个全局分片的装载、资源发布、对端导入和实际UB检索记录可核对，方法见《[快速入门](./quick_start.md#检查算法实际生效)》。
5. **CPU/NUMA分配有效**：进程允许CPU范围、`numactl`绑定与算法运行时CPU列表一致；为Milvus其他组件、中间件和系统留出资源。
6. **容量充足**：同时考虑索引、向量、算法运行时、Milvus和中间件的内存，以及数据、日志和临时文件的磁盘空间。
7. **测量条件固定**：使用相同数据文件、查询样本、真值、距离、top-K和统计方法，避免其他测试负载干扰。

队列和RPC注册内存预算属于配置上限，不能直接相加后视为进程固定占用。算法接口`UbEstimateRuntimeMemory`提供的是运行时开销估算，不包含索引和宿主的全部内存，详细语义见《[API参考](./api_reference.md)》。

### 操作步骤

#### 设置场景与参数起点

| 项目 | 本实践设置 |
| ---- | ---------- |
| 数据 | GIST HDF5，训练集1000000条、960维，含查询向量及近邻真值 |
| 字段与距离 | 主键`id`为训练行号，向量字段`vector`，L2距离 |
| 运行进程 | 8个工作QueryNode，业务端口1832～1839 |
| 全局图分片 | `global_index_shard_count=8`，与参与该图的工作进程匹配 |
| 集合写入分片 | `shards_num=8`，与全局图分片含义不同 |
| 建图参数 | `M=64`、`efConstruction=40` |
| 查询起点 | `ef=200`、top-K为10 |
| 运行时起点 | `expandBatch=42`、`maxActiveQueries=32` |
| 工作线程 | Search/Distance/READ/RPC分别为10/14/2/5 |
| 客户端 | 与快速入门相同的PyMilvus、h5py及NumPy环境 |

先使用上述配置完成全量导入、建索引、加载及检索，再开展调优。数据库环境准备、配置及进程启动的具体操作请参见[Milvus侧配套说明](https://gitcode.com/boostkit/milvus/tree/master)。本文不预设通用的召回或性能承诺；测试前根据业务要求约定Recall@10、时延、错误和资源占用目标。

#### 区分YAML与进程环境覆盖值

实际跑通环境中的部分值来自`milvus.yaml`，部分由QueryNode启动命令覆盖。

启动命令仍含同名覆盖变量时，仅修改YAML不会改变这些字段；命令内显式设置的值也会覆盖终端导出的同名环境变量。`runtimeOptions`中的同名参数也可能覆盖独立字段，应同时检查。配置修改、进程环境覆盖及重启的具体操作请参见[Milvus侧配套说明](https://gitcode.com/boostkit/milvus/tree/master)，修改后核对实际生效值。

#### 标定召回率和串行响应

##### 复用已加载的集合

`ef`属于每次检索参数，可以在同一个索引上调整，无需为每个`ef`重新导入、建索引或重启进程。保持`M=64`、`efConstruction=40`和top-K为10，先围绕起始值200选择候选查询参数。

本例选择`128,200,400`作为待测点。候选值不是预先验证的最优值，应根据测量结果扩大或缩小范围，并保持`ef ≥ top-K`。

下面一次请求只查询1条向量，默认使用HDF5中的前1000条查询向量，按顺序发请求，每个候选值重复3轮。测得的是当前客户端到Milvus的端到端响应及串行观测吞吐，不代表系统最大并发QPS。

```bash
export MILVUS_EF_LIST=128,200,400
export MILVUS_EVAL_NQ=1000
export MILVUS_EVAL_REPEATS=3
export MILVUS_EVAL_WARMUP=10
export MILVUS_TOPK=10
export MILVUS_RPC_TIMEOUT=600
```

#### 执行评估并保存结果

保持上一节的变量和Python虚拟环境，执行下面代码。它只读取已加载集合并发起检索，不重新导入、创建索引或删除数据。遇到请求、结果或数据校验错误时，当前评估终止，应先排查再重测。

```bash
python - <<'PY'
import csv
import os
import time
from pathlib import Path
import h5py
import numpy as np
from pymilvus import Collection, connections, utility

name = os.environ["MILVUS_COLLECTION"]
topk = int(os.environ["MILVUS_TOPK"])
nq = int(os.environ["MILVUS_EVAL_NQ"])
repeats = int(os.environ["MILVUS_EVAL_REPEATS"])
warmup = int(os.environ["MILVUS_EVAL_WARMUP"])
efs = [int(value) for value in os.environ["MILVUS_EF_LIST"].split(",")]
timeout = float(os.environ["MILVUS_RPC_TIMEOUT"])
assert topk > 0 and nq > 0 and repeats > 0 and warmup >= 0
assert efs and all(ef >= topk for ef in efs)

with h5py.File(os.environ["MILVUS_HDF5"], "r") as h5:
    assert h5["train"].shape == (1_000_000, 960)
    assert nq <= h5["test"].shape[0]
    assert nq <= h5["neighbors"].shape[0]
    assert topk <= h5["neighbors"].shape[1]
    vectors = np.asarray(h5["test"][:nq], dtype=np.float32)
    truth = np.asarray(h5["neighbors"][:nq, :topk], dtype=np.int64)
truth_sets = [set(int(value) for value in row) for row in truth]
assert all(len(row) == topk and all(0 <= value < 1_000_000 for value in row)
           for row in truth_sets)

connections.connect(uri=os.environ["MILVUS_URI"], timeout=timeout)
try:
    col = Collection(name)
    assert col.num_entities == 1_000_000
    print("collection:", name, "load_state:", utility.load_state(name))
    print("indexes:", [index.params for index in col.indexes])
    output = Path(os.environ["RESULT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    filename = output / f"gist_serial_eval_{time.time_ns()}.csv"
    columns = ["collection", "ef", "repeat", "queries", "recall_at_k",
               "serial_observed_qps", "p50_ms", "p95_ms", "p99_ms", "total_seconds"]
    with filename.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for ef in efs:
            for repeat in range(1, repeats + 1):
                for i in range(warmup):
                    col.search(data=[vectors[i % nq].tolist()], anns_field="vector",
                               param={"metric_type": "L2", "params": {"ef": ef}},
                               limit=topk, consistency_level="Strong", timeout=timeout)
                matches = 0
                latencies = []
                begin = time.perf_counter()
                for i, vector in enumerate(vectors):
                    request_begin = time.perf_counter()
                    result = col.search(
                        data=[vector.tolist()], anns_field="vector",
                        param={"metric_type": "L2", "params": {"ef": ef}},
                        limit=topk, consistency_level="Strong", timeout=timeout,
                    )
                    latencies.append((time.perf_counter() - request_begin) * 1000)
                    assert len(result) == 1 and len(result[0]) == topk
                    ids = [int(hit.id) for hit in result[0]]
                    assert len(set(ids)) == topk
                    assert all(0 <= value < 1_000_000 for value in ids)
                    assert np.isfinite([float(hit.distance) for hit in result[0]]).all()
                    matches += len(set(ids) & truth_sets[i])
                elapsed = time.perf_counter() - begin
                p50, p95, p99 = np.percentile(latencies, [50, 95, 99])
                row = dict(collection=name, ef=ef, repeat=repeat, queries=nq,
                           recall_at_k=matches / (nq * topk),
                           serial_observed_qps=nq / elapsed,
                           p50_ms=float(p50), p95_ms=float(p95), p99_ms=float(p99),
                           total_seconds=elapsed)
                writer.writerow(row)
                file.flush()
                print(row, flush=True)
    print("RESULT_CSV:", filename)
finally:
    connections.disconnect("default")
PY
```

#### 调整运行时与资源配置

固定已选择的索引参数、`ef`、查询样本和负载，每次只改变一个运行时因素，修改后重启并确认生效，再重复相同评估。

| 参数 | 当前起始值 | 观察和调整方向 |
| ---- | ---------- | -------------- |
| `expandBatch` | 42 | 每轮展开工作量；同时观察召回、时延、通信和资源需求 |
| `maxActiveQueries` | 32 | 算法进程的在途查询上限；调大前检查槽位、队列及通信资源开销 |
| Search线程 | 10 | 本地图搜索工作量与CPU占用 |
| Distance线程 | 14 | 对端距离下推任务处理压力 |
| READ Progress线程 | 2 | 远端读完成推进与等待情况 |
| RPC Progress线程 | 5 | 邮箱收发与RPC推进压力 |
| CPU/NUMA | 每进程显式CPU集合，`llc-spread` | 实际拓扑、允许CPU列表、内存位置及系统资源余量 |

线程总数应适合各进程实际允许的CPU集合，还需计入Control线程、Milvus和系统其他线程。快速入门中的48核分配是该启动配置的示例，不是所有环境的固定最低要求。

客户端并发、每次请求的`nq`与`maxActiveQueries`作用不同。当前串行评估每次只有一个请求，不能单凭这组结果判断提高在途上限能否改善并发吞吐。

`runtimeOptions`中的`--profile`用于计时信息输出，比较不同参数时应保持该开关一致。操作超时、查询超时和客户端RPC超时属于不同层次，调整时核对单位和实际生效值；放宽超时不能代替查明排队、链路或资源问题。

#### 配置变化和复测状态处理

##### 哪些变化需要重启或重建

| 变化 | 处理方法 |
| ---- | -------- |
| 仅修改检索`ef`或top-K | 在现有已加载索引上重新检索，保持`0 < top-K ≤ ef` |
| 修改线程、expandBatch、CPU布局或其他启动期运行时参数 | 修改实际生效入口，正常停止并重新启动相关Milvus进程，核对资源和配置 |
| 修改M、efConstruction或换数据、建立新全局图 | 释放原集合、重置持有旧图的进程，按快速入门重新构建并加载 |

当前运行时每进程持有一张全局图。调优查询参数应优先复用当前集合；更换索引或图时，避免旧图仍驻留影响下一轮。

##### 释放原集合

停止客户端查询后，使用原集合名执行。该操作释放装载状态，不删除数据或索引。

```bash
python - <<'PY'
import os
from pymilvus import Collection, connections, utility
connections.connect(uri=os.environ["MILVUS_URI"], timeout=600)
try:
    name = os.environ["MILVUS_COLLECTION"]
    if utility.has_collection(name):
        Collection(name).release(timeout=600)
        print("RELEASED:", name)
finally:
    connections.disconnect("default")
PY
```

##### 正常停止并按原顺序启动

数据库侧负责组件退出和数据持久化的完整维护流程。以下适用于快速入门用PID文件记录的本次进程组，先停止Proxy及查询，再停止数据和协调组件，中间件保持原有部署。

确认PID目录属于本次部署，逐个核对命令行后发送SIGTERM并等待退出；PID不匹配或退出超时时停止操作并排查，不自动强制杀进程。

```bash
components=(proxy querynode_1832 querynode_1833 querynode_1834 querynode_1835
            querynode_1836 querynode_1837 querynode_1838 querynode_1839
            datanode streamingnode mixcoord)
for component in "${components[@]}"; do
    pid_file="$PID_DIR/$component.pid"
    test -s "$pid_file" || { printf '请核对PID文件: %s\n' "$pid_file"; exit 1; }
    pid=$(cat "$pid_file")
    if ! kill -0 "$pid" 2>/dev/null; then
        printf '进程已停止: %s\n' "$component"
        continue
    fi
    command_line=$(ps -p "$pid" -o args=)
    role="$component"
    [[ "$component" == querynode_* ]] && role=querynode
    [[ "$command_line" == *"$MILVUS_BIN run $role"* ]] || \
      { printf 'PID命令不匹配，请人工核对: %s\n' "$pid"; exit 1; }
    kill -TERM "$pid" || exit 1
    deadline=$((SECONDS + 300))
    while kill -0 "$pid" 2>/dev/null; do
        (( SECONDS < deadline )) || { printf '进程退出超时: %s\n' "$component"; exit 1; }
        sleep 1
    done
done
```

随后重新执行《[快速入门](./quick_start.md#设置milvus进程环境)》中的环境设置、CPU数组和《[逐项启动命令](./quick_start.md#按顺序启动集群)》，使更新后的参数生效。更换图时使用新的集合名并完成导入、建索引、加载和检索；单纯重测`ef`时无需执行本节停止流程。

### 测试结果

#### 查看测试输出与指标

字段解释：

- `recall_at_k`为本轮全部查询的命中数除以`查询数 × top-K`，需要使用全量训练数据及相同主键映射。
- P50/P95/P99来自客户端逐请求计时，包含RPC和服务处理等端到端开销，不等于算法内部搜索耗时。
- `serial_observed_qps`为本轮查询数除以总耗时，包含客户端串行处理及结果核对开销，不能作为最大吞吐报告。
- 预热请求不计入正式统计。不同轮次应保持预热、查询顺序、缓存条件和负载来源一致，并记录资源状态。

默认参数共发出9000次正式检索及90次预热检索，应预留相应时间。测试失败时保留已完成轮次及错误日志，不将未完成轮次视为合格结果。

#### 根据结果选择参数

先筛选满足业务召回率目标的候选值，再检查时延及资源占用。每个候选值至少保留3轮原始结果，汇总中位值和波动范围，不能只摘取一次最优值。

若增大`ef`仍无法达到召回目标，核对数据、真值、主键映射、实际算法链路及建图参数。需要调整`M`或`efConstruction`时，按“[配置变化和复测状态处理](#配置变化和复测状态处理)”中的流程重新构建索引，再标定查询参数。

若要评估并发最大吞吐，应另行使用已验证的并发压测方法，固定并发请求数、每请求向量数、查询总量、统计时长和错误口径。本例串行结果用于同一测试口径下的参数观察，不沿用其他部署规模的历史QPS或P99结果。

#### 结果记录和验收Checklist

本实践测试gd_hnsw与Milvus集成后的全局分布式检索能力：通过Milvus完成GIST数据导入、全局索引构建、加载和检索，结合QueryNode日志确认查询实际进入gd_hnsw算法路径；再使用GIST近邻真值核对Recall@10，测量客户端到Milvus的端到端时延和串行观测吞吐，并检查不同参数下重复运行的结果及资源状态。

每轮记录集合与数据文件版本、Milvus和算法版本、补丁、二进制及配置校验信息，以及实际全局分片数、CPU/NUMA、建图参数、请求参数和日志位置。

- **功能**：8个工作QueryNode和目标全局图资源正常，100万条数据、索引和Load状态正确，查询进入gd_hnsw路径。
- **质量**：使用明确的查询样本数和对应真值，Recall@10满足测试前约定的目标。
- **响应**：按明确的串行或并发口径记录时延，目标与测量方法一致。
- **稳定性**：至少3轮完整结果，无请求错误、异常重启或持续资源压力；保留波动和失败信息。
- **可追溯性**：参数修改、生效检查、原始CSV和服务日志完整，功能冒烟结果与扩大样本评估分别标明。

遇到数据、设备、装载或运行时问题时，先恢复快速入门的功能验证，再继续调优。接口语义见《[API参考](./api_reference.md)》，排查入口见《[常见问题和故障案例](./FAQ.md)》。

## 缩略语

| 缩略语 | 英文全称 | 中文全称 |
| ------ | -------- | -------- |
| HNSW | Hierarchical Navigable Small World | 分层可导航小世界 |
| NUMA | Non-Uniform Memory Access | 非一致内存访问 |
| QPS | Queries Per Second | 每秒查询数 |
| RPC | Remote Procedure Call | 远程过程调用 |
| URMA | Unified Remote Memory Access | 统一远端内存访问 |

## 修订记录

| 文档版本 | 发布日期 | 修改说明 |
| -------- | -------- | -------- |
| 01 | 2026-12-30 | 第一次正式发布。 |

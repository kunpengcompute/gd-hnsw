# 最佳实践

本文档提供gd_hnsw与Milvus集成的实践案例，旨在帮助用户实现在具体场景中调用gd_hnsw算法

## 约束与限制

- 本例沿用快速入门的GIST数据、8个工作QueryNode和8个全局图分片。
- 数据库编译、中间件部署和完整配置请参见[Milvus侧配套说明](https://gitcode.com/boostkit/milvus/tree/master)。路径、地址和CPU分配填写自己环境中的实际值。
- 先用快速入门的参数跑通，再一次修改一个参数，方便看出是哪项修改影响了结果。

## 集成Milvus

### 环境部署

使用《快速入门》中的“[环境部署](./quick_start.md#环境部署)”。中间件由Docker Compose运行，Milvus各组件直接以进程方式启动。

在同一个Bash终端中操作，保持快速入门使用的Python虚拟环境和环境变量。更换终端时，先恢复原来的目录、服务地址、数据路径和集合名。`MILVUS_COLLECTION`填写已经创建的集合名；复用原索引时，不要重新生成集合名。

### 前提条件

开始调整参数前，先确认以下内容。

- 快速入门的导入、构建索引、加载和检索四个模块已经完成，集合中有100万条GIST数据。
- 8个工作QueryNode运行正常，URMA设备和链路可用。
- 按“[检查算法实际生效](./quick_start.md#检查算法实际生效)”核对日志，确认本次查询确实使用gd_hnsw。
- 记下当前的集合名和参数，后面用相同数据比较修改前后的结果。

### 操作步骤

先分清下面三类参数，再决定改完后执行哪些步骤。

| 参数 | 在哪里修改 | 改完后怎么做 |
| ---- | ---------- | ------------ |
| 构图参数，如`M`、`efConstruction` | 创建索引时的`create_index`参数 | 重新构图，再load和search |
| 启动参数，如线程数、CPU分配、`expandBatch` | `milvus.yaml`或QueryNode启动命令 | 重启相关进程，再load和search |
| 查询参数，如`ef` | 每次调用`search`时的参数 | 直接重新search |

#### 设置构图参数

构图参数决定索引怎样建立，只有创建索引时才会使用。先沿用快速入门的设置。

| 参数 | 起始值 | 说明 |
| ---- | ------ | ---- |
| `M` | 64 | 控制图中节点的连接数量 |
| `efConstruction` | 40 | 控制构图时的候选搜索范围 |
| `global_index_shard_count` | 8 | 将全局图分成8个分片，与本例参与检索的8个工作QueryNode对应 |

快速入门通过下面的环境变量将参数传给`create_index`。

```bash
export MILVUS_M=64
export MILVUS_EFC=40
export MILVUS_GLOBAL_SHARDS=8
```

具体调用位置请参见“[模块二：构建8分片全局HNSW索引](./quick_start.md#模块二构建8分片全局hnsw索引)”。例如，想比较`efConstruction=40`与`80`，可以将`MILVUS_EFC`改为`80`，再重新构图。这里的80只是对比示例，是否适合当前数据要看实际结果。

只改环境变量不会改变已经建好的索引。需要换一组构图参数时，先按“[重启进程，再重新load](#重启进程再重新load)”释放旧集合并重启持有旧图的进程，再使用新的集合名，依次执行快速入门的“[创建集合并导入GIST](./quick_start.md#模块一创建集合并导入gist)”和“[构建8分片全局HNSW索引](./quick_start.md#模块二构建8分片全局hnsw索引)”。随后重新load和search。原集合保留，方便后面核对。

如果只是调整启动参数或查询`ef`，继续使用原集合和原索引，不需要重新导入或构图。

#### 设置进程启动前的参数

下面这些参数在QueryNode启动时读取。先保持起始值，确认能跑通后，再按需要调整。

| 参数 | 起始值 | 说明 |
| ---- | ------ | ---- |
| `expandBatch` | 42 | 控制每轮展开的节点数 |
| `maxActiveQueries` | 32 | 控制算法进程最多同时处理多少个查询 |
| Search线程 | 10 | 处理图搜索 |
| Distance线程 | 14 | 处理距离计算任务 |
| READ Progress线程 | 2 | 推进远端读取 |
| RPC Progress线程 | 5 | 推进请求和响应收发 |
| CPU/NUMA分配 | 沿用快速入门中的分配方式 | 按本机可用CPU和NUMA节点填写 |

快速入门中，这些值由“[启动8个工作QueryNode](./quick_start.md#按顺序启动集群)”的命令设置。例如，调整Distance线程数时，修改启动命令中的`MILVUS_CONF_querynodeUbThreadsDistance=14`。

启动命令中的同名变量会覆盖YAML配置。如果只改`milvus.yaml`，启动命令仍写着旧值，最终使用的仍是旧值。也不要只在另一个终端里`export`后就认为运行中的进程已经更新。

超时时间和`runtimeOptions`等YAML字段请参见“[核对数据库连接和算法配置](./quick_start.md#核对数据库连接和算法配置)”。具体配置及进程维护操作请参见[Milvus侧配套说明](https://gitcode.com/boostkit/milvus/tree/master)。

每次先改一项，例如只改Distance线程数，其余参数保持原样。改好后执行下一步。

#### 重启进程，再重新load

如果改了构图参数或启动参数，按下面的顺序操作。只改查询`ef`时，直接跳到“[调整ef，重新search](#调整ef重新search)”。

1. 停止当前查询，释放旧集合的加载状态。这一步不会删除集合、数据或索引。

   ```bash
   python - <<'PY'
   import os
   from pymilvus import Collection, connections

   connections.connect(uri=os.environ["MILVUS_URI"], timeout=600)
   try:
       Collection(os.environ["MILVUS_COLLECTION"]).release(timeout=600)
       print("RELEASE_OK")
   finally:
       connections.disconnect("default")
   PY
   ```

2. 按[Milvus侧配套说明](https://gitcode.com/boostkit/milvus/tree/master)正常停止本次部署中需要重启的Milvus进程，确认旧进程已经退出，再使用修改后的配置启动。启动顺序请参见《快速入门》中的“[设置Milvus进程环境](./quick_start.md#设置milvus进程环境)”、“[规划QueryNode的CPU和NUMA](./quick_start.md#规划querynode的cpu和numa)”和“[按顺序启动集群](./quick_start.md#按顺序启动集群)”。本例8个工作QueryNode均按更新后的配置启动，中间件保持原部署。

3. 如果更改了构图参数，按“[设置构图参数](#设置构图参数)”使用新集合重新导入和构图，等待`INDEX_OK`。如果只改启动参数，保留原来的`MILVUS_COLLECTION`，跳过重新导入和构图。

4. 对本轮要测试的集合重新执行load，等待加载完成。

   ```bash
   python - <<'PY'
   import os
   from pymilvus import Collection, connections, utility

   name = os.environ["MILVUS_COLLECTION"]
   connections.connect(uri=os.environ["MILVUS_URI"], timeout=600)
   try:
       Collection(name).load(replica_number=1, timeout=86400)
       utility.wait_for_loading_complete(name, timeout=86400)
       print("loading progress:", utility.loading_progress(name))
       print("LOAD_OK")
   finally:
       connections.disconnect("default")
   PY
   ```

看到加载进度为100%和`LOAD_OK`后，再进行检索，并检查QueryNode日志中的参数和全局图资源是否正确。

#### 调整ef，重新search

`ef`是每次查询时设置的参数。可以把它理解成“这次查询要搜索多大的候选范围”。索引不变时，修改`ef`后直接重新search，不需要重启、重新load或重新构图。

先用`ef=200`检查查询是否正常，再试`128、200、400`进行比较。下面使用同一批1000条查询，top-K固定为10。每次发一个查询，收到结果后再发下一个，并统计召回率、平均查询耗时和QPS。

```bash
export MILVUS_EF_LIST=200
export MILVUS_EVAL_NQ=1000
export MILVUS_TOPK=10
export MILVUS_RPC_TIMEOUT=600
```

执行下面的检索代码。

```bash
python - <<'PY'
import os
import time
import h5py
import numpy as np
from pymilvus import Collection, connections

nq = int(os.environ["MILVUS_EVAL_NQ"])
topk = int(os.environ["MILVUS_TOPK"])
efs = [int(value) for value in os.environ["MILVUS_EF_LIST"].split(",")]
timeout = float(os.environ["MILVUS_RPC_TIMEOUT"])
assert nq > 0 and topk > 0 and efs and all(ef >= topk for ef in efs)
with h5py.File(os.environ["MILVUS_HDF5"], "r") as h5:
    assert h5["train"].shape == (1_000_000, 960)
    assert nq <= h5["test"].shape[0] and nq <= h5["neighbors"].shape[0]
    assert topk <= h5["neighbors"].shape[1]
    queries = np.asarray(h5["test"][:nq], dtype=np.float32)
    truth = [set(int(value) for value in row)
             for row in h5["neighbors"][:nq, :topk]]
assert all(len(row) == topk and all(0 <= value < 1_000_000 for value in row)
           for row in truth)

connections.connect(uri=os.environ["MILVUS_URI"], timeout=timeout)
try:
    col = Collection(os.environ["MILVUS_COLLECTION"])
    assert col.num_entities == 1_000_000
    for ef in efs:
        matches = 0
        query_seconds = 0.0
        begin = time.perf_counter()
        for i, query in enumerate(queries):
            query_begin = time.perf_counter()
            result = col.search(
                data=[query.tolist()], anns_field="vector",
                param={"metric_type": "L2", "params": {"ef": ef}},
                limit=topk, consistency_level="Strong", timeout=timeout,
            )
            query_seconds += time.perf_counter() - query_begin
            assert len(result) == 1 and len(result[0]) == topk
            ids = [int(hit.id) for hit in result[0]]
            assert len(set(ids)) == topk and all(0 <= value < 1_000_000 for value in ids)
            assert np.isfinite([float(hit.distance) for hit in result[0]]).all()
            matches += len(set(ids) & truth[i])
        elapsed = time.perf_counter() - begin
        print(f"ef={ef} queries={nq} Recall@{topk}={matches / (nq * topk):.4f} "
              f"avg_ms={query_seconds * 1000 / nq:.3f} QPS={nq / elapsed:.2f}", flush=True)
finally:
    connections.disconnect("default")
PY
```

单个`ef`跑通后，修改下面的参数，再执行同一段检索代码，就能比较三组结果。

```bash
export MILVUS_EF_LIST=128,200,400
```

`MILVUS_EF_LIST`只是这个客户端示例中要依次测试的`ef`列表，不是Milvus进程的启动参数。各值都要满足`ef ≥ top-K`。比较时保持数据、集合、查询条数和其他参数不变。

### 测试结果

#### 输出结果

每个`ef`会输出一行结果，重点看下面几项。

- **Recall@10**：召回率，表示返回结果命中GIST近邻真值的比例，越接近1说明本次查询的结果越完整。
- **avg_ms**：平均每次查询的耗时，单位毫秒，从客户端发出请求计时到收到结果，包含通信和服务处理时间。
- **QPS**：查询条数除以本组测试总时间，包含客户端处理和结果核对时间。本例一次只发一个请求，结果用于比较相同测试方式下的参数，不代表系统最大并发吞吐。

先检查有没有报错、数据条数和加载状态是否正确，再看召回率是否达到业务要求，最后比较耗时和QPS。

如果想提高召回率，可以先尝试增大`ef`。如果查询变慢，或者QPS下降，就结合召回率要求选择合适的值。若增加`ef`后仍不满足要求，再检查数据、真值和算法链路，确认无误后再考虑调整`M`或`efConstruction`重新构图。参数效果以实际测试为准。

#### 结果记录和验收Checklist

这套流程检查的是：Milvus能否完成数据导入、构图、load和search，查询是否真正进入gd_hnsw，以及调整参数后召回率、耗时和QPS有什么变化。

- **功能**：集合有100万条数据，索引构建和加载完成，8个工作QueryNode及全局图资源正常，search能返回结果，日志确认使用gd_hnsw。
- **质量**：用同一批查询和对应真值计算Recall@10，检查是否达到业务要求。
- **响应**：记录查询条数、`ef`、平均耗时和QPS，使用相同方式比较修改前后的结果。
- **稳定性**：参数初步选定后，重复运行至少3轮，观察是否有报错、异常重启或明显波动。
- **可追溯性**：保留集合名、构图参数、启动参数、`ef`、终端结果和对应日志，方便复查。

有问题时，先回到《快速入门》的对应步骤检查，排查方法请参见《常见问题和故障案例》中的“[跑通路径验证失败排查](./FAQ.md#跑通路径验证失败排查)”。

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

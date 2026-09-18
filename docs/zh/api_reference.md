# API参考

## 概述

gd_hnsw对外暴露头文件`gd_hnsw_api.h`。所有接口位于`namespace gd_hnsw`，状态封装在单一有状态的`Context`对象中，无全局变量。

调用时序：`initialize`，`build`/`load`，`node_init_and_sync`，`search`，`finalize`。

## 状态码

```cpp
enum class Status : int {
    Ok             = 0,
    InvalidArg     = -1,
    MpiError       = -2,
    UbsemError     = -3,
    IoError        = -4,
    ValidationErr  = -5,
    NotInitialized = -6,
    AlreadyInit    = -7,
};
```

`Status`由`Context`方法及各`*Result`结构返回。库不抛异常，常规错误由调用方根据返回码决定如何处理。

## Context

`Context`是全部状态的容器（pimpl，不可拷贝、不可移动）。

```cpp
class Context {
public:
    Context();
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&) = delete;
    Context& operator=(Context&&) = delete;

    Status       initialize(int* argc = nullptr, char*** argv = nullptr,
                            const InitOptions& opts = {});
    BuildResult  build(const BuildOptions& opts);
    LoadResult   load(uint32_t node_id, const std::string& shard_base_path);
    Status       node_init_and_sync(uint32_t node_id, const DeployOptions& opts = {});
    SearchResult search(const float* queries, uint64_t nq,
                        const SearchParams& params,
                        int32_t* out_ids, float* out_dists);
    static std::vector<uint32_t> route_queries(const Context& ctx,
                                               const float* queries, uint64_t nq);
    Status       finalize();

    // accessors
    bool     initialized() const;
    int      rank() const;
    int      size() const;
    uint32_t num_shards() const;
    uint32_t dim() const;
    uint64_t ntotal() const;
    bool     has_centroids() const;
    bool     has_idmap() const;
    const std::vector<uint32_t>& idmap() const;
    Status   last_status() const;
};
```

### initialize

`Context`中定义的对外接口。

**接口定义**

```cpp
Status initialize(int* argc = nullptr, char*** argv = nullptr,
                  const InitOptions& opts = {});
```

**接口用途**

算法初始化。

**参数说明**

| 参数名称 | 参数类型 | 参数说明 | 取值范围 |
|----------|----------|------|----------|
| argc | int* | 命令行参数的个数 | 默认0 |
| argv | char*** | 命令行参数  | — |
| opts | const InitOptions& | 初始化参数 | 非空 |

**InitOptions参数说明**

| 参数名称 | 参数类型 | 参数说明 | 取值范围 |
|----------|----------|------|----------|
| num_threads | int | 容器进程可用于算法的线程数 | 整数，默认 0 |

### build

`Context`中定义的对外接口。

**接口定义**

```cpp
BuildResult build(const BuildOptions& opts);
```

**接口用途**

读数据集（HDF5/.bin），构建全局分片索引并落盘。

**参数说明**

| 参数名称 | 参数类型 | 参数说明 | 取值范围 |
|----------|----------|------|----------|
| opts | const BuildOptions& | 构建索引参数 | 非空 |

**BuildOptions参数说明**

| 参数名称 | 参数类型 | 参数说明 | 取值范围 |
|----------|----------|------|----------|
| dataset_path | std::string | 数据集路径（HDF5文件或 .bin目录/文件） | 有效路径 |
| output_path | std::string | 输出前缀，写output_path.shard_0 .. shard_N-1 | 有效路径 |
| num_shards | uint32_t | 分片数 | ≥1，默认1 |
| M | uint32_t | HNSW 的M参数 | 正整数，默认16 |
| ef_construction | uint32_t | 建图ef_construction | 正整数，默认200 |
| ef_search | uint32_t | 写入索引的默认ef_search | 正整数，默认64 |
| build_threads | uint32_t | 构建线程数；0=全核 | 默认0 |
| cluster_partition | bool | 是否用K-Means聚类分片（同时写 idmap + centroids） | 默认false |
| dataset_format | std::string | 数据集格式 | auto/hdf5/bin，默认auto |
| bin_base_path | std::string | .bin模式下覆盖base向量文件 | 可选 |

**BuildResult返回值参数说明**

| 字段 | 类型 |参数说明 |
|------|------|------|
| status | Status | 执行状态 |
| ntotal | uint64_t | 总向量数 |
| dim | uint32_t | 向量维度 |
| num_shards | uint32_t | 实际分片数 |
| shard_bytes | std::vector<uint64_t> | 各分片文件大小 |
| error | std::string | 错误描述（status!=Ok时） |

### load

`Context`中定义的对外接口。

**接口定义**

```cpp
LoadResult load(uint32_t node_id, const std::string& shard_base_path);
```

**接口用途**

加载分片索引进算法。

**参数说明**

| 参数名称 | 参数类型 | 参数说明 | 取值范围 |
|----------|----------|------|----------|
| node_id | uint32_t | 本节点分片号 | [0, num_shards) |
| shard_base_path | std::string | 分片文件前缀，实际读取path.shard_node_id | 有效路径 |

**LoadResult返回值参数说明**

| 字段 | 类型 | 参数说明 |
|------|------|------|
| status | Status | 执行状态 |
| ntotal | uint64_t | 全局总向量数 |
| dim | uint32_t | 向量维度 |
| M | uint32_t | HNSW M |
| ef_search | uint32_t | 索引内ef_search |
| num_shards | uint32_t | 分片数 |
| shard_id | uint32_t | 本节点分片号 |
| ntotal_local | uint64_t | 本分片向量数 |
| global_id_begin | uint64_t | 本分片全局id起始 |
| error | std::string | 错误描述 |

### node_init_and_sync

`Context`中定义的对外接口。

**接口定义**

```cpp
Status node_init_and_sync(uint32_t node_id, const DeployOptions& opts = {});
```

**接口用途**

节点初始化与同步。

**参数说明**

| 参数名称 | 参数类型 | 参数说明 | 取值范围 |
|----------|----------|------|----------|
| node_id | uint32_t | 节点id | 默认0 |
| opts | const DeployOptions& | 部署参数 | 非空 |

**DeployOptions参数说明**

| 参数名称 | 参数类型 | 参数说明 | 取值范围 |
|----------|----------|------|----------|
| search_threads | uint32_t | 搜索线程数；0 = 自动（num_threads-service_threads） | 默认0 |
| service_threads | uint32_t | service 线程数；0 = 自动（cores/4，且≤num_shards-1） | 默认0 |
| expand_batch | uint32_t | multi-pop扩展批量；1=单候选（默认），>1合并远端小批量 | ≥1，默认1 |
| use_dot_norm | bool | 是否启用dot+norm距离核 | 默认false |

**返回值**：`Status`。

>![表示说明的图片](./public_sys-resources/icon-note.gif)**说明：**pushdown channel是跨节点成对的，**所有节点都必须调用**`node_init_and_sync`，任一节点不参与会导致其它节点映射channel失败。

### search

`Context`中定义的对外接口。

**接口定义**

```cpp
SearchResult search(const float* queries, uint64_t nq,
                    const SearchParams& params,
                    int32_t* out_ids, float* out_dists);
```

**接口用途**

对本节点query做全局检索，返回完整top-K。

**参数说明**

| 参数名称 | 参数类型 | | 取值范围 |
|----------|----------|------|----------|
| queries | const float* | 查询向量 | 非空 |
| nq | uint64_t | 查询向量数 | ≥0 |
| params | const SearchParams& | 检索参数（见下） | — |
| out_ids | int32_t* | 输出id | 非空 |
| out_dists | float* | 输出距离 | 非空 |

**SearchParams参数说明**

| 参数名称 | 参数类型 | 参数说明 | 取值范围 |
|----------|----------|------|----------|
| k | int32_t | top-K 结果数 | >0，默认10 |
| ef_search | int32_t | 查询ef_search；0=用索引内ef_search | ≥0，默认0 |

**SearchResult返回值参数说明**

| 字段 | 类型 | 参数说明 |
|------|------|------|
| status | Status | 执行状态 |
| error | std::string | 错误描述 |

**输出约定**：`out_ids[nq*k]`/`out_dists[nq*k]`由调用方分配；未填满的位置`id=-1, dist=FLT_MAX`。若`idmap`存在，结果id默认是**内部 gid**；如需原始gid，调用方用`Context::idmap()`自行重映射。

>![表示说明的图片](./public_sys-resources/icon-note.gif)**说明：**`search`无副作用，可反复调用，`k`/`ef_search`每次可变。

### route_queries

`Context`中定义的对外接口。

**接口定义**

```cpp
static std::vector<uint32_t> route_queries(const Context& ctx,
                                           const float* queries, uint64_t nq);
```

**接口用途**

按质心/均匀把query分到各节点，返回**本rank** 应处理的query下标列表。有centroids时对每条query找最近质心；无centroids时按rank均匀切片。

**参数说明**

| 参数名称 | 参数类型 | 参数说明 |
|----------|----------|------|
| ctx | const Context& | 已load过的Context（centroids/idmap须已加载，否则退化为均匀切片） |
| queries | const float* | 全量查询向量，nq*dim |
| nq | uint64_t | 查询向量数 |

**返回值**：本rank应处理的query下标列表（`std::vector<uint32_t>`）。

### finalize

`Context`中定义的对外接口。

**接口定义**

```cpp
Status finalize();
```

**接口用途**

释放资源。

>![表示说明的图片](./public_sys-resources/icon-note.gif)**说明：**`finalize`必须所有节点一起调，内部有Barrier与deallocate顺序依赖。

## 访问器

| 接口 | 返回类型 | 说明 |
|------|----------|------|
| initialized() | bool | 是否已initialize |
| rank() | int | MPI rank（==node_id==shard_id），未init返回-1 |
| size() | int | MPI world size |
| num_shards() | uint32_t | 分片数 |
| dim() | uint32_t | 向量维度 |
| ntotal() | uint64_t | 全局总向量数 |
| has_centroids() | bool | 是否已加载centroids（cluster_partition build） |
| has_idmap() | bool | 是否已加载idmap |
| idmap() | const std::vector<uint32_t>& | idmap（新gid映射到原始gid），无则为空；引用Context内部状态，finalize前有效 |
| last_status() | Status | 最近一次状态 |

## 修订记录

| 文档版本 | 发布日期 | 修改说明 |
|------|----------|------------|
| 01 | 2026-09-30 | 第一次正式发布。 |

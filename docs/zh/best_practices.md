# 最佳实践

本文档提供使用gd_hnsw的部署Checklist、典型场景实践与调优建议。

## 部署Checklist

部署分布式检索前，请逐项确认。

1. **每个容器能读本节点分片文件**：`/path/idx.shard_<rank>`必须可读。
2. **ub 环境**：所有节点必须成功安装ub环境。
3. **NUMA**：当前仅支持Linux+NUMA。非Linux编译会报错。

## 典型场景实践

1. 离线构建索引。

   - 使用hdf5格式数据集构建索引示例。

      ```bash
      ./build/benchmarks/build_index -d /path/to/data/sift-128-euclidean.hdf5 \
                 -o /path/to/data/idx \
                 -M 16 -c 200 -e 64 -s 8 -t 32
      ```

   - 使用bin格式数据集构建索引示例。

      ```bash
      ./build/benchmarks/build_index -d /path/to/data/gist_1M_960/base.bin \
                 -o /path/to/data/idx \
                 -M 64 -c 40 -e 355 -s 8 -t 32
      ```

   - 使用config文件的形式构建索引示例。

      ```bash
      ./build/benchmarks/build_index --config configs/gist-960-euclidean.config
      ```

      结果：`/path/to/data/idx.shard_0 .. idx.shard_7`。部署前分发到各节点（共享存储或本地副本）。

2. 在线检索（8节点）。

   1. 首先，先进入当前算法configs目录下，配置对应数据集的相关配置参数。

   2. 然后运行以下算法。

      ```bash
      mpirun --allow-run-as-root -np 8 --hostfile hostfile --map-by ppr:1:numa --bind-to numa ./build/benchmarks/gd_hnsw_bench --config configs/gist-960-euclidean.config
      ```

      >![表示说明的图片](./public_sys-resources/icon-note.gif) **说明：**
      >`hostfile`中为服务器IP，形如：`xxx.xxx.xxx.xxx`。

## 调优建议

### HNSW参数

| 参数名称 | 参数说明 |
|------|------|
| M | 越大图越稠密、召回越高、内存与建图时间增加 |
| ef_construction | 建图候选宽度，越大图质量越高、构建越慢 |
| ef_search | 查询候选宽度，越大召回越高、时延越高；可每查可变 |

>![表示说明的图片](./public_sys-resources/icon-note.gif) **说明：**
>召回率与时延的权衡建议先用`ef_search`调，再回退调`M`。`build`一次性成本高，`search`时`ef_search`可在线调整。

### 并发与线程

`search_threads`/`service_threads`默认0（自动）：service=cores/4且≤num_shards-1，search=剩余。

### pushdown批量

`expand_batch`（`DeployOptions`）控制每轮弹出候选数。`1`为单候选（默认），`>1`合并远端小批量、摊薄通道往返。远端距离请求零碎、通道往返成为瓶颈时可尝试调大。

## 修订记录

| 文档版本 | 发布日期 | 修改说明 |
|------|----------|------------|
| 01 | 2026-09-30 | 第一次正式发布。 |

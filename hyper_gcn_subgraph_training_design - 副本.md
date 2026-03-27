# 改进版 HyperGCN 训练管道设计提纲

## 🎯 目标

将 HyperGCN 模型训练流程替换为基于 Proposal 中提出的“大小子图循环采样训练策略”，以实现更好的训练效率和扩展性。

---

## 1️⃣ 超参数定义

| 参数 | 含义           | 示例值 |
| -- | ------------ | --- |
| T  | 大循环轮数        | 100 |
| R  | 每轮大循环中子图数量   | 10  |
| m  | 每个子图的最大节点数   | 100 |
| L  | 每个子图训练的小循环次数 | 10  |

---

## 2️⃣ 总体训练流程框架

```python
for t in range(T):  # 大循环
    subsample_graphs = [sample_subgraph(train_data, m, strategy="TIHS") for _ in range(R)]
    for subgraph in subsample_graphs:
        for l in range(L):  # 小循环
            train_on_subgraph(model, subgraph)
```

---

## 3️⃣ 子图采样函数设计（支持多策略切换）

采样函数统一接口：

```python
def sample_subgraph(data, m, strategy="TIHS"):
    ...
    return subgraph_data  # 必须是超边集合
```

- **输入**：

  - `data`: 超图数据结构（如 `{eid: [v1, v2, ...], ...}`）
  - `m`: 子图最大节点数
  - `strategy`: 子图采样策略，可选 `"TIHS"` 或 `"snowball"`

- **输出**：

  - 子图数据（超边集合），如 `{eid: [v1, v2, ...]}`，可直接用于训练函数。

### 🎲 方法一：Totally-Induced Hyperedge Sampling (TIHS)

```python
def sample_subgraph_tihs(hyperedges_dict, max_nodes):
    import random
    selected_edges = set()
    selected_nodes = set()

    all_edges = list(hyperedges_dict.items())
    random.shuffle(all_edges)

    for eid, nodes in all_edges:
        selected_edges.add(eid)
        selected_nodes.update(nodes)
        for eid2, nodes2 in hyperedges_dict.items():
            if eid2 not in selected_edges and set(nodes2).issubset(selected_nodes):
                selected_edges.add(eid2)
        if len(selected_nodes) >= max_nodes:
            break

    return {eid: hyperedges_dict[eid] for eid in selected_edges}
```

### 🔁 方法二：Snowball Sampling for Hypergraphs

```python
def sample_subgraph_snowball(hyperedges_dict, max_nodes):
    import random
    from collections import deque, defaultdict

    node2edges = defaultdict(set)
    for eid, nodes in hyperedges_dict.items():
        for v in nodes:
            node2edges[v].add(eid)

    all_nodes = list(node2edges.keys())
    start_node = random.choice(all_nodes)

    visited_nodes = set([start_node])
    selected_edges = set()
    queue = deque([start_node])

    while queue and len(visited_nodes) < max_nodes:
        current = queue.popleft()
        for eid in node2edges[current]:
            if eid not in selected_edges:
                selected_edges.add(eid)
                for v in hyperedges_dict[eid]:
                    if v not in visited_nodes:
                        visited_nodes.add(v)
                        queue.append(v)
        if len(visited_nodes) >= max_nodes:
            break

    return {eid: hyperedges_dict[eid] for eid in selected_edges}
```

### 🧩 总调度接口

```python
def sample_subgraph(data, m, strategy="TIHS"):
    if strategy == "TIHS":
        return sample_subgraph_tihs(data, m)
    elif strategy == "snowball":
        return sample_subgraph_snowball(data, m)
    else:
        raise ValueError(f"Unsupported sampling strategy: {strategy}")
```

---

## 4️⃣ 子图训练函数设计

函数：`train_on_subgraph(model, subgraph_data)`

- 本函数与全图训练函数相同，因子图训练等价于在主模型上进行“数据分片式”训练。
- 所有子图共享主模型的权重参数，确保 GCN 权重矩阵全局一致。
- 每个子图训练 `L` 次小循环，相当于局部更新主模型权重。
- 模型参数在每个子图内逐步收敛，并在大循环中融合。

---

## 5️⃣ 子图融合策略

- 每轮大循环结束后，收集所有子图训练后的模型权重，执行**平均聚合**。

  - 示例：对每个子图的更新权重矩阵张量，令：
    ```python
    theta_global = sum(theta_sub_i for i in range(R)) / R
    ```
  - 将该聚合后的权重作为这个大循环模型的更新权重矩阵用于随后的测试集评估，以及用于下一轮子图初始化的共享模型参数。


---

## 6️⃣ 评估与实验记录

- 记录每个大循环下每个子图训练的平均时间
- 每个大循环更新模型参数后模型在验证集和测试集上的评估（直接搬用原新项目的评估）
- 其他原项目的实验记录建议保留


---

## 🔗 建议函数命名接口

| 函数名                                       | 功能           |
| ----------------------------------------- | ------------ |
| `sample_subgraph(data, m, strategy)`      | 子图采样（支持策略切换） |
| `train_on_subgraph(model, subgraph_data)` | 子图训练         |


---

## 📌 推荐参数设置（参考 Proposal）

| 数据集 | T   | R   | m   | L   |
| --- | --- | --- | --- | --- |
| 自定义 | 100 | 16  | 300 | 5   |



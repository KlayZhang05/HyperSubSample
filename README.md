# HyperSubSample

基于子图采样与参数聚合的超边预测实验代码仓库。

本项目聚焦于超图上的超边预测问题，目标是在大规模场景下缓解全图训练对显存和时间的压力。整体思路是在原始超图上反复进行子图采样，在多个子图上分别训练 HyperGCN 风格模型，再通过参数聚合逼近全图训练效果，并在真实业务数据上验证效率与预测表现。

## 项目背景

传统图神经网络或超图神经网络训练通常依赖全图加载。当网络规模较大时，显存占用和训练时间会迅速增长，甚至使训练不可行。为此，本项目尝试将子图采样与分布式训练思路迁移到超图场景：

- 将完整超图拆分为多个可控规模的子图
- 在每个子图上进行局部训练
- 对多个子图上的参数估计进行聚合
- 比较全图训练与子图训练在效率和效果上的差异

根据项目 proposal 和中期材料，当前实验主要围绕以下问题展开：

- 超边预测任务建模
- 子图采样策略设计
- 多轮局部训练与全局参数聚合
- 在真实推荐业务数据上的实证分析

## 方法概述

当前代码实现的核心模块包括：

1. `ModifiedHyperGCN`
   将训练超图映射为 HyperGCN 风格的节点表示学习模块。
2. `HyperedgeAggregator`
   将超边内节点嵌入聚合为超边表示。
3. `HyperedgePredictionModel`
   使用 `HyperGCN + Hyperedge Aggregator + MLP` 完成超边二分类。
4. `SNSNegativeSampler`
   基于超边大小分布进行负采样，构造正负样本。
5. `SubgraphHyperedgeTrainer`
   支持子图采样训练，包括 TIHS 和 Snowball 两种采样策略。

子图训练的大致流程为：

1. 从原始训练超图中采样出多个子图
2. 使用同一组全局参数作为各子图模型的初始化
3. 在每个子图上进行若干轮局部训练
4. 对各子图模型参数进行聚合
5. 重复上述过程直到完成全部大循环

## 仓库结构

```text
hyperedge_prediction/
├── main.py
├── training_pipeline.py
├── end_to_end_model.py
├── modified_hypergcn.py
├── hyperedge_aggregator.py
├── subgraph_sampler.py
├── sns_negative_sampler.py
├── system_check.py
├── README_subgraph_training.md
└── requirements.txt
```

主要文件说明：

- `main.py`: 命令行入口，支持全图训练与子图训练
- `training_pipeline.py`: 数据加载、训练、验证、测试与日志逻辑
- `end_to_end_model.py`: 端到端超边预测模型定义
- `modified_hypergcn.py`: HyperGCN 相关实现
- `subgraph_sampler.py`: 子图采样策略实现
- `sns_negative_sampler.py`: 负采样实现
- `system_check.py`: 结构与语法检查辅助脚本

## 环境依赖

建议使用 Python 3.10 及以上版本。

安装依赖：

```bash
pip install -r requirements.txt
```

如果你需要 GPU 版 PyTorch，请根据本机 CUDA 版本从 PyTorch 官方说明安装对应版本，再补装其余依赖。

## 数据准备

代码当前默认依赖以下几类输入文件：

- 训练集 CSV
- 验证集 CSV
- 测试集 CSV
- 超边大小采样器 `edge_size_sampler.pkl`

从现有代码逻辑看：

- `training_pipeline.py` 读取的超边字段名为 `full_hyperedge`
- 每条超边通常编码为逗号分隔的节点序列
- 超边格式形如 `"user_id,item_id_1,item_id_2,..."`

需要注意：

- 当前仓库中未包含 `edge_size_sampler.py`，但 `sns_negative_sampler.py` 会导入它
- 当前仓库也未包含公开可直接运行的完整数据集
- `main.py` 中的默认数据路径仍保留了本地开发路径，实际运行时建议显式传参覆盖

因此，若你希望在新环境直接复现实验，至少需要自行准备：

- 训练/验证/测试 CSV
- `edge_size_sampler.pkl`
- `edge_size_sampler.py` 或同等功能实现

## 快速开始

### 1. 全图训练

```bash
python main.py \
  --train_csv /path/to/train.csv \
  --val_csv /path/to/val.csv \
  --test_csv /path/to/test.csv \
  --size_sampler /path/to/edge_size_sampler.pkl \
  --num_users 82740 \
  --num_products 38830 \
  --training_strategy full_graph \
  --epochs 100 \
  --lr 0.001 \
  --embedding_dim 64 \
  --output_dir ./outputs
```

### 2. 子图采样训练

```bash
python main.py \
  --train_csv /path/to/train.csv \
  --val_csv /path/to/val.csv \
  --test_csv /path/to/test.csv \
  --size_sampler /path/to/edge_size_sampler.pkl \
  --num_users 82740 \
  --num_products 38830 \
  --training_strategy subgraph \
  --T 50 \
  --R 8 \
  --m 200 \
  --L 3 \
  --sampling_strategy snowball \
  --epochs 100 \
  --lr 0.001 \
  --embedding_dim 64 \
  --output_dir ./outputs
```

### 3. 仅测试已有模型

```bash
python main.py \
  --mode test \
  --training_strategy subgraph \
  --output_dir ./outputs
```

## 关键参数

- `--training_strategy`: `full_graph` 或 `subgraph`
- `--T`: 大循环次数
- `--R`: 每轮采样的子图数量
- `--m`: 单个子图允许的最大节点数
- `--L`: 每个子图上的局部训练轮数
- `--sampling_strategy`: `TIHS` 或 `snowball`
- `--embedding_dim`: 节点嵌入维度
- `--lr`: 学习率
- `--output_dir`: 模型与日志输出目录

## 输出内容

训练完成后，输出目录和日志目录中通常会包含：

- `best_full_graph_model.pth` 或 `best_subgraph_model.pth`
- `config.json`
- 训练历史图像
- `logs/training_*.log`

此外，项目会缓存验证集和测试集负样本，例如：

- `val_negatives_*.pkl`
- `test_negatives_*.pkl`

这些缓存文件通常不需要纳入版本控制。

## 已实现的采样策略

### TIHS

TIHS 即 Totally-Induced Hyperedge Sampling。其思路是先选择部分超边，再递归加入由当前节点集合所完全诱导出的其他超边，以尽量保留子图内部的高阶结构完整性。

### Snowball

Snowball 采样从随机种子节点出发，逐步扩展与其相邻的超边和节点，能够更快覆盖局部邻域结构，适合构造局部性更强的训练子图。

## 当前限制

当前仓库仍然更接近研究型实验代码，而不是开箱即用的完整工程版本，主要限制包括：

- 部分默认路径仍为作者本地路径
- 外部依赖 `edge_size_sampler.py` 不在当前仓库中
- 数据预处理流程未单独整理为可复现脚本
- 若在不同平台或不同编码环境下使用，部分历史中文注释可能出现显示异常

如果后续继续完善，优先建议补齐：

- 数据预处理脚本
- `edge_size_sampler.py`
- 更标准的数据样例与字段说明
- 实验配置文件
- 结果复现实验表格

## 参考背景

本仓库 README 参考了项目 proposal 与中期报告中对子图采样超图训练框架的描述整理而成。相关方法背景涉及：

- HyperGCN
- Hypergraph Neural Networks
- 超图子采样与分布式训练
- 超边预测中的负采样构造

## License

当前仓库未单独声明开源协议。如需公开发布，建议补充 `LICENSE` 文件。

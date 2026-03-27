# HyperSubSample

本仓库为超边预测任务的研究代码实现，核心内容是基于子图采样与参数聚合的超图训练方法。当前实现以 HyperGCN 风格模型为基础，同时保留完整图训练与子图采样训练两种实验设置，用于比较不同训练策略在预测效果与计算开销上的差异。

## 1. 研究问题

超图神经网络训练通常依赖完整超图结构。当节点数和超边数较大时，训练过程在显存占用和时间开销上都较为昂贵。针对这一问题，本项目将超边预测建模为二分类任务，并引入子图采样训练框架：

- 在原始训练超图上采样多个子图
- 在各子图上进行局部训练
- 对局部模型参数进行聚合
- 将聚合后的结果与完整图训练进行比较

## 2. 方法框架

当前模型由以下三部分组成：

1. `ModifiedHyperGCN`
   用于在训练超图上学习节点表示。

2. `HyperedgeAggregator`
   用于将超边内部节点嵌入聚合为超边表示。

3. `MLP`
   用于完成候选超边的二分类预测。

训练模式包括：

- `full_graph`：在完整训练超图上直接训练
- `subgraph`：对子图重复采样，在局部训练后进行参数聚合

当前实现的子图采样策略包括：

- `TIHS`
- `snowball`

## 3. 仓库结构

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

主要文件说明如下：

- `main.py`：命令行入口
- `training_pipeline.py`：训练、验证、测试与日志流程
- `end_to_end_model.py`：端到端超边预测模型
- `modified_hypergcn.py`：HyperGCN 相关实现
- `hyperedge_aggregator.py`：超边表示聚合模块
- `subgraph_sampler.py`：子图采样策略实现
- `sns_negative_sampler.py`：负采样模块

## 4. 环境依赖

建议使用 Python 3.10 及以上版本。

依赖安装方式如下：

```bash
pip install -r requirements.txt
```

如需使用 GPU，请根据本机 CUDA 环境安装兼容版本的 PyTorch。

## 5. 数据格式

运行代码时需要提供：

- 训练集 CSV
- 验证集 CSV
- 测试集 CSV
- `edge_size_sampler.pkl`

从当前实现看，CSV 文件中应包含 `full_hyperedge` 字段。每条超边表示为逗号分隔的节点序列，例如：

```text
user_id,item_id_1,item_id_2,item_id_3
```

当前仓库未包含完整数据文件。运行实验时建议通过命令行参数显式指定数据路径。

## 6. 运行方式

### 6.1 完整图训练

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

### 6.2 子图采样训练

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

### 6.3 仅测试

```bash
python main.py \
  --mode test \
  --training_strategy subgraph \
  --output_dir ./outputs
```

## 7. 主要参数

- `--training_strategy`：`full_graph` 或 `subgraph`
- `--T`：外层迭代次数
- `--R`：每轮采样的子图数量
- `--m`：单个子图允许的最大节点数
- `--L`：每个子图上的局部训练轮数
- `--sampling_strategy`：`TIHS` 或 `snowball`
- `--embedding_dim`：节点嵌入维度
- `--lr`：学习率
- `--output_dir`：模型与日志输出目录

## 8. 输出内容

训练完成后，通常会生成以下文件：

- `best_full_graph_model.pth` 或 `best_subgraph_model.pth`
- `config.json`
- 训练过程图像
- `logs/` 目录下的日志文件

程序还会缓存验证集和测试集负样本，例如：

- `val_negatives_*.pkl`
- `test_negatives_*.pkl`

## 9. 当前状态

本仓库为研究阶段代码，重点在于实验逻辑实现。若需在全新环境中严格复现实验，仍需结合具体数据、路径设置和本地依赖环境进行调整。

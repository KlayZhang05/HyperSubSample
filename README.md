# HyperSubSample

本仓库保存的是一套面向超边预测任务的研究型实验代码。整体工作围绕“如何在较大规模超图上降低训练成本”这一问题展开，核心思路是将全图训练改写为“子图采样 + 局部训练 + 参数聚合”的训练框架，并将其与传统全图训练方式进行比较。

当前仓库中的实现主要基于 HyperGCN 风格的超图表示学习方法，将超边预测建模为二分类问题：给定一个候选超边，判断其是否为真实存在的超边。代码同时保留了完整图训练和子图采样训练两条实验路径，便于后续做消融、对照和效率分析。

## 一、问题背景

在图神经网络和超图神经网络的训练过程中，一个直接的困难在于：模型通常依赖整张图或整张超图的结构信息。当图规模继续增大时，显存占用和训练时间都会迅速上升，进而限制方法在真实业务数据上的可用性。

本项目关注的是超边预测任务。与普通图中的边预测不同，超边本身由多个节点共同组成，因此模型不仅需要学习节点表示，还需要刻画多个节点之间的高阶组合关系。若仍然采用全图加载和全图训练的方式，在大规模场景下代价较高。

基于这一点，当前代码实现了一套子图采样训练框架。其基本思路可以概括为：

- 不直接在完整超图上反复训练
- 而是在原始训练超图上采样出多个规模可控的子图
- 在每个子图上进行若干轮局部训练
- 再对多个子图训练得到的模型参数进行聚合

这一设计的目标不是完全替代全图训练，而是在尽量保留预测性能的前提下，降低单次训练的资源消耗，并为后续分布式或并行扩展提供一个清晰的实验基础。

## 二、方法概述

当前实现的整体模型可以概括为三层：

1. 用 `ModifiedHyperGCN` 在训练超图上学习节点嵌入
2. 用 `HyperedgeAggregator` 将超边内节点嵌入聚合为超边表示
3. 用 `MLP` 对超边表示进行二分类，输出候选超边的预测结果

从训练机制上看，仓库中支持两种模式：

- `full_graph`：在完整训练超图上直接训练
- `subgraph`：对训练超图做重复采样，在子图上局部训练，再进行参数聚合

子图训练的大致流程如下：

1. 从原始训练超图中采样出若干个子图
2. 使用当前全局模型参数初始化每一个子图模型
3. 在每个子图上进行若干轮局部优化
4. 对多个子图模型的参数进行聚合，更新全局模型
5. 重复上述过程直到完成全部外层迭代

目前实现了两种子图采样策略：

- `TIHS`：Totally-Induced Hyperedge Sampling
- `snowball`：基于局部扩展的雪球采样

其中，`TIHS` 更强调保留由当前节点集合所诱导出的超边结构；`snowball` 更偏向于从种子节点出发逐步扩展局部邻域。这两种策略的差异，可以直接通过训练时间、验证集表现和稳定性进行比较。

## 三、仓库结构

```text
hyperedge_prediction/
├── main.py
├── parallel_subgraph_trainer.py
├── setup_native_extensions.py
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

- `main.py`
  命令行入口，负责组织训练、测试、参数解析和输出目录管理。

- `parallel_subgraph_trainer.py`
  可选并行子图训练入口。在 `subgraph` 模式下，当并行子图数量大于 1 时使用，用于并行执行同一轮中的多个子图局部训练并完成参数聚合。

- `setup_native_extensions.py`
  用于编译可选的 C++ 扩展模块。扩展编译完成后，子图采样与子图样本筛选会优先使用 native 实现。

- `training_pipeline.py`
  训练主流程所在文件，包括数据加载、负采样、验证、测试、日志记录以及完整图训练 / 子图训练的组织逻辑。

- `end_to_end_model.py`
  端到端超边预测模型定义，负责将 HyperGCN、超边聚合和分类器组合起来。

- `modified_hypergcn.py`
  HyperGCN 相关实现，是当前节点表示学习部分的核心。

- `hyperedge_aggregator.py`
  负责将同一条候选超边中的多个节点嵌入聚合为一个固定维度表示。

- `subgraph_sampler.py`
  实现 `TIHS` 和 `snowball` 两种子图采样策略。

- `sns_negative_sampler.py`
  负采样模块。当前实现基于超边大小分布进行负样本生成。

- `system_check.py`
  一个简单的结构检查脚本，用来快速检查若干核心文件是否存在、是否可以通过基础语法编译。

## 四、环境依赖

建议使用较新的 Python 环境。当前依赖整理在 `requirements.txt` 中，可以直接安装：

```bash
pip install -r requirements.txt
```

如果需要在 GPU 上训练，建议先根据本机 CUDA 环境单独安装合适版本的 PyTorch，再补装其余依赖。

如果希望在 CPU 并行子图训练时尽量减少 Python 解释器和 GIL 带来的影响，可以额外编译仓库中的 native 扩展：

```bash
python setup_native_extensions.py build_ext --inplace
```

编译成功后，`subgraph_sampler.py` 会自动优先调用 `subgraph_sampler_native`，无需额外修改训练命令。

## 五、数据说明

代码默认需要以下几类输入：

- 训练集 CSV
- 验证集 CSV
- 测试集 CSV
- 超边大小采样器 `edge_size_sampler.pkl`

从当前代码逻辑来看，训练、验证和测试 CSV 应至少包含超边字段。`training_pipeline.py` 中读取的是 `full_hyperedge` 列，该列中的一条超边通常被表示为逗号分隔的节点序列，例如：

```text
user_id,item_id_1,item_id_2,item_id_3
```

也就是说，这里的一个超边一般对应“一个用户 + 若干商品”的组合。

需要额外说明两点：

第一，当前仓库没有包含完整的公开数据文件，因此克隆仓库后不能直接无参运行。

第二，负采样部分依赖超边大小分布采样器，因此除了三份数据切分文件外，还需要准备 `edge_size_sampler.pkl`。如果在新的实验环境中运行，建议直接通过命令行参数显式传入数据路径，而不要依赖代码中的默认路径。

## 六、运行方式

### 1. 完整图训练

如果希望直接在完整训练超图上训练模型，可以使用：

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

如果希望使用当前仓库的重点实验设置，即“子图采样 + 参数聚合”，可以使用：

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

如果已经完成训练，只希望加载已有权重并在测试集上评估，可以使用：

```bash
python main.py \
  --mode test \
  --training_strategy subgraph \
  --output_dir ./outputs
```

### 4. GPU 训练

当前入口支持通过 `--device` 显式指定运行设备，可选值为 `auto`、`cpu` 和 `cuda`。

- `auto`：自动选择可用 GPU，否则退回 CPU
- `cpu`：强制使用 CPU
- `cuda`：强制使用 GPU；若当前环境不可用则直接报错

例如，在单卡 GPU 上运行子图训练时可以写为：

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
  --device cuda \
  --output_dir ./outputs_gpu
```

### 5. 并行子图训练

当前版本支持可选并行子图训练，通过 `--parallel_subgraphs` 控制同一轮中并行训练的子图数量。默认值为 `1`，即串行执行；当该值大于 `1` 时，会启用并行子图训练器。

例如：

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
  --device cpu \
  --parallel_subgraphs 4 \
  --output_dir ./outputs_parallel
```

当前实现中需要注意两点：

- 并行子图训练仅在 CPU 模式下启用，原因是同一进程内多个子图模型同时占用单张 GPU 容易产生资源竞争
- 当设置 `--parallel_subgraphs > 1` 且设备为 `cuda` 时，程序会自动退回串行子图训练，但仍保留相同的训练入口与输出格式

如果已经编译 `subgraph_sampler_native`，则子图采样与子图样本筛选会在 C++ 扩展层执行，这一部分可以减少 GIL 对 CPU 并行的限制；模型前向、反向传播和参数聚合仍然沿用现有的 PyTorch 训练流程。

## 七、主要参数说明

为了便于后续实验记录，这里将几个核心参数单独列出：

- `--training_strategy`
  训练模式。`full_graph` 表示全图训练，`subgraph` 表示子图采样训练。

- `--T`
  外层迭代次数，即大循环次数。

- `--R`
  每次外层迭代中采样的子图数量。

- `--m`
  单个子图允许的最大节点数。

- `--L`
  每个子图上的局部训练轮数，即小循环次数。

- `--sampling_strategy`
  子图采样策略，目前支持 `TIHS` 和 `snowball`。

- `--device`
  运行设备选择，支持 `auto`、`cpu` 和 `cuda`。

- `--parallel_subgraphs`
  并行子图数量，仅在 `subgraph` 模式下生效。设为 `1` 时为串行训练；大于 `1` 时尝试启用 CPU 并行子图训练。

- `--embedding_dim`
  节点嵌入维度。

- `--lr`
  学习率。

- `--output_dir`
  模型权重、配置文件和训练过程输出的保存目录。

## 八、输出结果

训练完成后，通常会在输出目录中看到以下内容：

- `best_full_graph_model.pth` 或 `best_subgraph_model.pth`
- `config.json`
- 训练历史图像
- 训练日志文件

此外，程序会缓存验证集和测试集的负样本，例如：

- `val_negatives_*.pkl`
- `test_negatives_*.pkl`

这样做主要是为了避免每次评估时重复生成负样本，减少实验中的额外开销。

## 九、当前实现状态

这个仓库目前更接近研究阶段的实验代码，而不是一个已经完全整理好的工程化项目。因此，在使用时需要注意以下几点：

- 代码重点在于实验逻辑本身，而不是完整封装
- 数据预处理流程没有完全独立整理成公共脚本
- 部分默认路径带有历史开发环境痕迹
- 如果需要严格复现实验，仍然需要结合具体数据文件和本地环境进一步调整

从研究实验的角度看，当前版本已经能够支撑以下工作：

- 完整图训练与子图采样训练的对照实验
- 不同子图采样策略的比较
- 训练时间、验证精度和测试性能的记录
- 负采样方案与参数设置的进一步扩展

# HyperGCN 子图采样训练系统

根据 MD 文档中提出的"大小子图循环采样训练策略"，我们已经成功实现了子图采样训练功能。

## 🎯 实现的功能

### ✅ 已完成的组件

1. **子图采样器** (`subgraph_sampler.py`)
   - 支持 TIHS (Totally-Induced Hyperedge Sampling) 策略
   - 支持 Snowball Sampling 策略  
   - 统一的 `sample_subgraph()` 接口

2. **子图训练器** (`SubgraphHyperedgeTrainer` 类)
   - 实现了完整的大小循环训练策略
   - 支持模型权重融合（平均聚合）
   - 详细的训练日志和统计记录
   - 可视化训练历史

3. **主入口增强** (`main.py`)
   - 添加了 `--training_strategy` 参数，支持 `full_graph` 和 `subgraph` 两种模式
   - 完整的子图训练参数支持 (T, R, m, L, sampling_strategy)

## 🚀 使用方法

### 全图训练（原有方法）
```bash
python main.py --training_strategy full_graph --epochs 100
```

### 子图采样训练（新方法）
```bash
python main.py --training_strategy subgraph \
  --T 100 \          # 大循环轮数
  --R 16 \           # 每轮子图数量  
  --m 300 \          # 子图最大节点数
  --L 5 \            # 子图训练小循环次数
  --sampling_strategy TIHS    # 采样策略：TIHS 或 snowball
```

### 完整参数示例
```bash
python main.py \
  --train_csv /path/to/train.csv \
  --val_csv /path/to/val.csv \
  --test_csv /path/to/test.csv \
  --size_sampler /path/to/edge_size_sampler.pkl \
  --num_users 82740 \
  --num_products 38830 \
  --training_strategy subgraph \
  --T 50 --R 8 --m 200 --L 3 \
  --sampling_strategy snowball \
  --lr 0.001 \
  --embedding_dim 64 \
  --output_dir ./subgraph_outputs
```

## 📊 训练流程详解

### 子图采样训练策略
根据 MD 文档的设计：

```python
for t in range(T):  # 大循环
    subsample_graphs = [sample_subgraph(train_data, m, strategy) for _ in range(R)]
    for subgraph in subsample_graphs:
        for l in range(L):  # 小循环
            train_on_subgraph(model, subgraph)
    # 模型权重融合
    aggregate_model_weights()
```

### 超参数说明
| 参数 | 含义 | 推荐值 | 
|------|------|--------|
| T | 大循环轮数 | 100 |
| R | 每轮大循环中子图数量 | 16 |
| m | 每个子图的最大节点数 | 300 |
| L | 每个子图训练的小循环次数 | 5 |

### 采样策略对比
- **TIHS**: 完全诱导超边采样，确保子图包含所有被选中节点完全包含的超边
- **Snowball**: 雪球采样，从随机起点开始扩展邻接超边

## 📈 输出文件

### 训练过程文件
- `logs/training_YYYYMMDD_HHMMSS.log` - 完整训练日志
- `subgraph_training_history.png` - 子图训练历史可视化
- `subgraph_training_history_time_stats.json` - 详细时间统计

### 模型文件  
- `best_subgraph_model.pth` - 最佳子图训练模型
- `config.json` - 训练配置信息

### 训练历史可视化
子图训练会生成4个图表：
1. 验证准确率 vs 大循环轮数
2. 大循环训练时间历史
3. 子图训练时间分布  
4. 模型融合时间历史

## 🔧 技术实现细节

### 模型权重融合
每个大循环结束后，使用平均聚合策略：
```python
theta_global = sum(theta_sub_i for i in range(R)) / R
```

### 子图训练流程
1. 从完整训练超图中采样R个子图
2. 每个子图创建独立的模型实例
3. 复制主模型权重到子图模型
4. 在子图上进行L次小循环训练
5. 融合所有子图模型权重到主模型
6. 在验证集上评估融合后的模型

### 负样本策略
- 验证集和测试集：使用固定负样本（缓存）
- 训练集：第1个epoch动态负采样，后续epoch使用固定负样本
- 子图训练：从全图负样本中筛选属于子图的样本

## 🧪 测试验证

运行集成测试验证系统完整性：
```bash
python test_integration.py
```

该测试会验证：
- 所有模块导入
- 子图采样器功能
- 训练器创建
- 参数解析

## 📝 与原有系统的兼容性

- ✅ 完全向后兼容，原有的全图训练功能不受影响
- ✅ 共享相同的数据加载和预处理流程  
- ✅ 统一的模型架构和评估方法
- ✅ 一致的日志和输出格式

## 🎉 系统优势

### 相比全图训练的优势
1. **内存效率**: 子图训练减少内存占用
2. **训练速度**: 并行子图训练可能加速收敛
3. **可扩展性**: 更好地处理大规模超图
4. **鲁棒性**: 多样化的子图训练提高模型泛化能力

### 实验对比
现在可以轻松比较两种训练策略：
- 运行相同参数的全图训练和子图训练
- 比较训练时间、内存使用和最终准确率  
- 分析不同子图采样策略的效果

---

**注意**: 请确保您的数据集文件路径正确，且已安装所有必要依赖（torch, pandas, numpy, matplotlib, tqdm 等）。

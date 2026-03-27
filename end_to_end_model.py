#!/usr/bin/env python3
"""
端到端超边预测模型
HyperGCN + 聚合器 + MLP分类器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any

# 导入我们实现的组件
from modified_hypergcn import ModifiedHyperGCN
from hyperedge_aggregator import HyperedgeAggregator


class HyperedgePredictionModel(nn.Module):
    """
    端到端超边预测模型
    
    架构: HyperGCN → 超边聚合器 → MLP分类器
    """
    
    def __init__(self, num_nodes: int, hypergraph: Dict, node_features: torch.Tensor, 
                 embedding_dim: int = 400, hidden_dims: List[int] = [256, 128], 
                 hypergcn_args: Any = None):
        """
        初始化端到端模型
        
        Args:
            num_nodes: 节点总数（用户数 + 商品数）
            hypergraph: 超图结构（训练集超边）
            node_features: 节点特征矩阵
            embedding_dim: 节点嵌入维度
            hidden_dims: MLP隐藏层维度列表
            hypergcn_args: HyperGCN参数
        """
        super(HyperedgePredictionModel, self).__init__()
        
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        # 1. HyperGCN部分（真正的HyperGCN实现）
        # 确保设备一致性 - 如果配置存在则使用配置的设备
        if hypergcn_args and hasattr(hypergcn_args, 'cuda'):
            use_cuda = hypergcn_args.cuda
        else:
            use_cuda = False  # 默认使用CPU以避免设备不匹配
        
        device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        
        print(f"  - 正在初始化真正的HyperGCN...")
        print(f"  - 设备: {device}")
        print(f"  - 超边数量: {len(hypergraph)}")
        
        # 确保节点特征在正确的设备上
        if use_cuda and torch.cuda.is_available():
            node_features = node_features.cuda()
        else:
            node_features = node_features.cpu()
        
        self.hypergcn = ModifiedHyperGCN(
            V=num_nodes,
            E=hypergraph, 
            X=node_features,
            embedding_dim=embedding_dim,
            depth=2,
            dropout=0.3,
            mediators=True,
            fast=True,
            cuda=use_cuda and torch.cuda.is_available()
        )
        
        # 2. 超边聚合器
        self.aggregator = HyperedgeAggregator(aggregation_method='mean')
        
        # 3. MLP分类器
        mlp_layers = []
        input_dim = embedding_dim
        
        for hidden_dim in hidden_dims:
            mlp_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            input_dim = hidden_dim
            
        # 输出层：二分类
        mlp_layers.append(nn.Linear(input_dim, 2))
        
        self.mlp_classifier = nn.Sequential(*mlp_layers)
        
        print(f"✓ 端到端模型初始化完成")
        print(f"  - 节点数: {num_nodes}")
        print(f"  - 嵌入维度: {embedding_dim}")
        print(f"  - MLP隐藏层: {hidden_dims}")
        
    def forward(self, candidate_hyperedges: List[List[int]]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            candidate_hyperedges: 候选超边列表 [[node1, node2, ...], ...]
            
        Returns:
            分类logits，形状: (batch_size, 2)
        """
        # 1. HyperGCN获取节点嵌入
        node_embeddings = self.hypergcn.get_node_embeddings()  # 形状: (num_nodes, embedding_dim)
        
        # 2. 聚合候选超边的节点嵌入
        hyperedge_embeddings = self.aggregator.forward_batch(node_embeddings, candidate_hyperedges)  # 形状: (batch_size, embedding_dim)
        
        # 3. 确保hyperedge_embeddings在正确的设备上（强制设备一致性）
        device = next(self.parameters()).device
        hyperedge_embeddings = hyperedge_embeddings.to(device)
        
        # 4. MLP分类器
        logits = self.mlp_classifier(hyperedge_embeddings)  # 形状: (batch_size, 2)
        
        return logits
    
    def predict_probabilities(self, candidate_hyperedges: List[List[int]]) -> torch.Tensor:
        """
        预测超边概率
        
        Args:
            candidate_hyperedges: 候选超边列表
            
        Returns:
            超边概率，形状: (batch_size,)
        """
        logits = self.forward(candidate_hyperedges)
        probabilities = F.softmax(logits, dim=1)
        return probabilities[:, 1]  # 返回正类（超边）的概率
    
    def get_node_embeddings(self) -> torch.Tensor:
        """
        获取节点嵌入（用于分析和可视化）
        
        Returns:
            节点嵌入矩阵，形状: (num_nodes, embedding_dim)
        """
        return self.hypergcn.get_node_embeddings()


class HyperedgePredictionConfig:
    """
    超边预测模型配置类
    """
    
    def __init__(self):
        # 模型参数
        self.embedding_dim = 400
        self.mlp_hidden_dims = [256, 128]  # MLP隐藏层维度，处理batch_size x embedding_dim的输入
        
        # HyperGCN参数
        self.hypergcn_depth = 2
        self.hypergcn_dropout = 0.3
        self.use_mediators = True
        self.fast_mode = True
        
        # 训练参数
        self.learning_rate = 0.01
        self.weight_decay = 0.0005
        self.epochs = 100
        
        # 设备 - 智能选择设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"✓ 模型配置初始化完成")
        print(f"  - 设备: {self.device}")
        print(f"  - 嵌入维度: {self.embedding_dim}")


def create_mock_hypergcn_args(config: HyperedgePredictionConfig, num_features: int, num_classes: int = 2):
    """
    创建HyperGCN参数对象
    
    Args:
        config: 模型配置
        num_features: 节点特征维度
        num_classes: 分类类别数
        
    Returns:
        HyperGCN参数对象
    """
    class Args:
        def __init__(self):
            self.d = num_features  # 节点特征维度
            self.c = num_classes   # 分类类别数
            self.depth = config.hypergcn_depth
            self.dropout = config.hypergcn_dropout
            self.mediators = config.use_mediators
            self.fast = config.fast_mode
            self.cuda = config.device.type == 'cuda'
            
    return Args()


# 测试代码
if __name__ == "__main__":
    print("=== 端到端模型测试 ===")
    
    try:
        # 1. 创建配置
        config = HyperedgePredictionConfig()
        # 强制使用CPU进行测试
        config.device = torch.device('cpu')
        
        # 2. 创建模拟数据
        num_users = 100
        num_products = 200
        num_nodes = num_users + num_products
        feature_dim = 64
        
        # 节点特征（随机初始化）
        node_features = torch.randn(num_nodes, feature_dim)
        
        # 模拟超图（几个简单的超边）
        hypergraph = {
            0: [0, 100, 101, 102],      # 用户0购买商品100,101,102
            1: [1, 103, 104],           # 用户1购买商品103,104
            2: [2, 100, 105, 106, 107], # 用户2购买商品100,105,106,107
        }
        
        # 候选超边（正负样本）
        candidate_hyperedges = [
            [0, 100, 101],           # 可能的超边
            [1, 200, 250],           # 另一个可能的超边
            [50, 150, 180, 190],     # 第三个候选超边
        ]
        
        print(f"✓ 模拟数据创建完成")
        print(f"  - 节点数: {num_nodes} (用户: {num_users}, 商品: {num_products})")
        print(f"  - 特征维度: {feature_dim}")
        print(f"  - 训练超边数: {len(hypergraph)}")
        print(f"  - 候选超边数: {len(candidate_hyperedges)}")
        
        # 3. 创建HyperGCN参数
        hypergcn_args = create_mock_hypergcn_args(config, feature_dim)
        
        # 4. 创建端到端模型
        model = HyperedgePredictionModel(
            num_nodes=num_nodes,
            hypergraph=hypergraph,
            node_features=node_features,
            embedding_dim=config.embedding_dim,
            hidden_dims=config.mlp_hidden_dims,
            hypergcn_args=hypergcn_args
        )
        
        # 5. 测试前向传播
        print(f"\n=== 测试前向传播 ===")
        with torch.no_grad():
            logits = model(candidate_hyperedges)
            probabilities = model.predict_probabilities(candidate_hyperedges)
            
        print(f"✓ 前向传播成功")
        print(f"  - 输入候选超边数: {len(candidate_hyperedges)}")
        print(f"  - 输出logits形状: {logits.shape}")
        print(f"  - 输出概率形状: {probabilities.shape}")
        print(f"  - 超边概率: {probabilities.tolist()}")
        
        # 6. 测试节点嵌入
        node_embeddings = model.get_node_embeddings()
        print(f"  - 节点嵌入形状: {node_embeddings.shape}")
        
        print(f"\n🎉 端到端模型测试完成!")
        print(f"✓ 真正的HyperGCN已成功集成到端到端模型中！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

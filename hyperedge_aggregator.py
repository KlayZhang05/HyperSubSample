"""
Hyperedge Aggregator
将节点嵌入聚合为超边表示
"""

import torch
import torch.nn as nn
import numpy as np


class HyperedgeAggregator(nn.Module):
    """
    超边聚合器：将超边中节点的嵌入聚合为超边表示
    """
    
    def __init__(self, aggregation_method='mean'):
        """
        Args:
            aggregation_method: 聚合方法 ('mean', 'sum', 'max')
        """
        super(HyperedgeAggregator, self).__init__()
        self.aggregation_method = aggregation_method
    
    def forward(self, node_embeddings, hyperedge_indices):
        """
        聚合超边中节点的嵌入
        
        Args:
            node_embeddings: 节点嵌入矩阵 [num_nodes, embedding_dim] 
                            或者直接的超边节点嵌入 [hyperedge_size, embedding_dim]
            hyperedge_indices: 超边节点索引列表 [hyperedge_size]
                              如果node_embeddings已经是超边节点嵌入，则应为连续索引
            
        Returns:
            超边表示向量 [embedding_dim]
        """
        # 兼容两种模式：
        # 1. 标准模式：从完整嵌入矩阵中索引
        # 2. 直接模式：node_embeddings已经是超边节点嵌入
        if len(node_embeddings.shape) == 2 and node_embeddings.shape[0] == len(hyperedge_indices):
            # 直接模式：node_embeddings已经是选中的超边节点嵌入
            hyperedge_node_embeddings = node_embeddings
        else:
            # 标准模式：从完整嵌入矩阵中索引
            hyperedge_node_embeddings = node_embeddings[hyperedge_indices]  # [hyperedge_size, embedding_dim]
        
        if self.aggregation_method == 'mean':
            return torch.mean(hyperedge_node_embeddings, dim=0)
        elif self.aggregation_method == 'sum':
            return torch.sum(hyperedge_node_embeddings, dim=0)
        elif self.aggregation_method == 'max':
            return torch.max(hyperedge_node_embeddings, dim=0)[0]
        else:
            raise ValueError(f"Unsupported aggregation method: {self.aggregation_method}")
    
    def forward_batch(self, node_embeddings, batch_hyperedges):
        """
        批量聚合多个超边
        
        Args:
            node_embeddings: 节点嵌入矩阵 [num_nodes, embedding_dim]
            batch_hyperedges: 批量超边，每个超边是节点索引列表
                              [[edge1_nodes], [edge2_nodes], ...]
            
        Returns:
            批量超边表示 [batch_size, embedding_dim]
        """
        batch_representations = []
        
        for hyperedge_indices in batch_hyperedges:
            hyperedge_repr = self.forward(node_embeddings, hyperedge_indices)
            batch_representations.append(hyperedge_repr)
        
        # 确保返回的张量在与输入相同的设备上
        result = torch.stack(batch_representations)
        if node_embeddings.is_cuda:
            result = result.cuda()
        return result


class PaddedHyperedgeAggregator(nn.Module):
    """
    支持填充的超边聚合器，可以处理不同大小的超边批次
    """
    
    def __init__(self, embedding_dim, aggregation_method='mean', pad_value=-1):
        """
        Args:
            embedding_dim: 节点嵌入维度
            aggregation_method: 聚合方法
            pad_value: 填充值，用于标识无效节点
        """
        super(PaddedHyperedgeAggregator, self).__init__()
        self.embedding_dim = embedding_dim
        self.aggregation_method = aggregation_method
        self.pad_value = pad_value
    
    def forward(self, node_embeddings, padded_hyperedges, hyperedge_lengths):
        """
        处理填充后的超边批次
        
        Args:
            node_embeddings: 节点嵌入矩阵 [num_nodes, embedding_dim]
            padded_hyperedges: 填充后的超边张量 [batch_size, max_hyperedge_size]
            hyperedge_lengths: 每个超边的实际长度 [batch_size]
            
        Returns:
            批量超边表示 [batch_size, embedding_dim]
        """
        batch_size, max_size = padded_hyperedges.shape
        batch_representations = []
        
        for i in range(batch_size):
            length = hyperedge_lengths[i]
            valid_indices = padded_hyperedges[i, :length]
            
            # 获取有效节点的嵌入
            valid_embeddings = node_embeddings[valid_indices]  # [length, embedding_dim]
            
            # 聚合
            if self.aggregation_method == 'mean':
                hyperedge_repr = torch.mean(valid_embeddings, dim=0)
            elif self.aggregation_method == 'sum':
                hyperedge_repr = torch.sum(valid_embeddings, dim=0)
            elif self.aggregation_method == 'max':
                hyperedge_repr = torch.max(valid_embeddings, dim=0)[0]
            else:
                raise ValueError(f"Unsupported aggregation method: {self.aggregation_method}")
            
            batch_representations.append(hyperedge_repr)
        
        return torch.stack(batch_representations)


def pad_hyperedges(hyperedges, pad_value=-1):
    """
    将不同大小的超边填充为相同大小
    
    Args:
        hyperedges: 超边列表，每个超边是节点索引列表
        pad_value: 填充值
        
    Returns:
        padded_hyperedges: 填充后的张量 [batch_size, max_hyperedge_size]
        hyperedge_lengths: 每个超边的实际长度 [batch_size]
    """
    hyperedge_lengths = [len(edge) for edge in hyperedges]
    max_length = max(hyperedge_lengths)
    
    padded_hyperedges = []
    for edge in hyperedges:
        padded_edge = edge + [pad_value] * (max_length - len(edge))
        padded_hyperedges.append(padded_edge)
    
    return torch.tensor(padded_hyperedges), torch.tensor(hyperedge_lengths)


# 工具函数
def test_aggregator():
    """测试聚合器功能"""
    print("测试超边聚合器...")
    
    # 创建测试数据
    num_nodes = 10
    embedding_dim = 400
    node_embeddings = torch.randn(num_nodes, embedding_dim)
    
    # 测试单个超边聚合
    aggregator = HyperedgeAggregator('mean')
    hyperedge = [0, 1, 2, 3]  # 超边包含节点0,1,2,3
    result = aggregator(node_embeddings, hyperedge)
    print(f"单个超边聚合结果形状: {result.shape}")
    assert result.shape == (embedding_dim,), f"期望形状({embedding_dim},), 实际{result.shape}"
    
    # 测试批量超边聚合
    batch_hyperedges = [[0, 1, 2], [3, 4, 5, 6], [7, 8]]
    batch_result = aggregator.forward_batch(node_embeddings, batch_hyperedges)
    print(f"批量超边聚合结果形状: {batch_result.shape}")
    assert batch_result.shape == (3, embedding_dim), f"期望形状(3, {embedding_dim}), 实际{batch_result.shape}"
    
    # 测试填充聚合器
    padded_aggregator = PaddedHyperedgeAggregator(embedding_dim, 'mean')
    padded_edges, lengths = pad_hyperedges(batch_hyperedges)
    padded_result = padded_aggregator(node_embeddings, padded_edges, lengths)
    print(f"填充聚合结果形状: {padded_result.shape}")
    assert padded_result.shape == (3, embedding_dim), f"期望形状(3, {embedding_dim}), 实际{padded_result.shape}"
    
    print("✅ 所有测试通过！")


if __name__ == "__main__":
    test_aggregator()

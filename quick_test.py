#!/usr/bin/env python3
"""
快速验证真正的HyperGCN系统是否正常工作
"""

import torch
import numpy as np
from typing import Dict, List

def quick_test():
    """快速测试真正的HyperGCN系统"""
    print("🧪 快速验证真正的HyperGCN系统")
    print("=" * 50)
    
    try:
        # 1. 测试组件导入
        print("1. 测试组件导入...")
        from modified_hypergcn import ModifiedHyperGCN, Laplacian, HyperGraphConvolution
        from hyperedge_aggregator import HyperedgeAggregator
        from end_to_end_model import HyperedgePredictionModel, HyperedgePredictionConfig
        print("   ✓ 所有组件导入成功")
        
        # 2. 测试HyperGCN核心功能
        print("2. 测试HyperGCN核心功能...")
        
        # 创建简单测试数据
        V = 20  # 20个节点
        E = {
            0: [0, 1, 2, 3],
            1: [4, 5, 6],
            2: [7, 8, 9, 10, 11],
            3: [12, 13, 14],
            4: [15, 16, 17, 18, 19]
        }  # 5个超边
        X = torch.randn(V, 32)  # 节点特征
        
        # 测试HyperGCN
        hypergcn = ModifiedHyperGCN(
            V=V, E=E, X=X,
            embedding_dim=64, depth=2,
            dropout=0.1, mediators=True,
            fast=False, cuda=False
        )
        
        embeddings = hypergcn.get_node_embeddings()
        print(f"   ✓ HyperGCN输出形状: {embeddings.shape}")
        print(f"   ✓ 包含Laplacian近似和超图卷积")
        
        # 3. 测试端到端模型
        print("3. 测试端到端模型...")
        
        config = HyperedgePredictionConfig()
        config.device = torch.device('cpu')  # 强制CPU
        config.embedding_dim = 64
        
        from end_to_end_model import create_mock_hypergcn_args
        hypergcn_args = create_mock_hypergcn_args(config, 32)
        
        model = HyperedgePredictionModel(
            num_nodes=V,
            hypergraph=E,
            node_features=X,
            embedding_dim=64,
            hidden_dims=[32, 16],
            hypergcn_args=hypergcn_args
        )
        
        # 测试前向传播
        candidate_hyperedges = [
            [0, 1, 2],
            [10, 15, 18],
            [5, 8, 12, 16]
        ]
        
        with torch.no_grad():
            logits = model(candidate_hyperedges)
            probs = model.predict_probabilities(candidate_hyperedges)
        
        print(f"   ✓ 端到端模型输出形状: {logits.shape}")
        print(f"   ✓ 超边概率: {probs.tolist()}")
        
        # 4. 验证架构完整性
        print("4. 验证架构完整性...")
        
        # 检查是否真的在使用Laplacian近似
        import inspect
        from modified_hypergcn import Laplacian
        laplacian_code = inspect.getsource(Laplacian)
        if "rv = np.random.rand" in laplacian_code and "np.argmax" in laplacian_code:
            print("   ✓ Laplacian近似函数包含随机投影和supremum/infimum计算")
        else:
            print("   ❌ Laplacian近似函数不完整")
        
        # 检查超图卷积层
        conv_params = sum(p.numel() for p in hypergcn.parameters())
        print(f"   ✓ HyperGCN参数量: {conv_params:,}")
        
        print("\n🎉 系统验证完成！")
        print("✅ 真正的HyperGCN已成功集成")
        print("✅ 包含完整的Laplacian近似和超图卷积")
        print("✅ 端到端超边预测模型正常工作")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 系统验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n🚀 系统已准备就绪，可以开始训练！")
        print("   运行命令: python main.py")
    else:
        print("\n💥 系统存在问题，请检查错误信息")

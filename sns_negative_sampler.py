#!/usr/bin/env python3
"""
Sized Negative Sampling (SNS) 负采样器
基于保存的超边尺寸分布进行负采样
"""

import numpy as np
import pickle
import json
import random
from typing import List, Tuple, Set
import os
import sys

# 添加上级目录到路径，以便导入hyperedge_sampler
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edge_size_sampler import load_hyperedge_size_sampler


class SNSNegativeSampler:
    """
    Sized Negative Sampling (SNS) 负采样器
    
    功能：
    1. 从保存的尺寸分布中采样超边尺寸
    2. 为每个尺寸均匀随机采样节点组合
    3. 确保负样本不与真实超边重复
    """
    
    def __init__(self, size_sampler_path: str, num_users: int, num_products: int, 
                 real_hyperedges: Set[Tuple] = None):
        """
        初始化SNS负采样器
        
        Args:
            size_sampler_path: 保存的尺寸采样器路径（.pkl文件）
            num_users: 用户总数
            num_products: 商品总数
            real_hyperedges: 真实超边集合，用于避免重复（可选）
        """
        self.size_sampler = load_hyperedge_size_sampler(size_sampler_path)
        self.num_users = num_users
        self.num_products = num_products
        self.real_hyperedges = real_hyperedges or set()
        
        print(f"✓ SNS负采样器初始化完成")
        print(f"  - 用户数: {num_users}")
        print(f"  - 商品数: {num_products}")
        print(f"  - 真实超边数: {len(self.real_hyperedges)}")
        
    def sample_negative_hyperedges(self, num_samples: int, max_attempts: int = 1000) -> List[List[int]]:
        """
        采样负超边
        
        Args:
            num_samples: 需要采样的负超边数量
            max_attempts: 每个超边的最大尝试次数（避免重复）
            
        Returns:
            负超边列表，格式: [[user_idx, product1, product2, ...], ...]
        """
        negative_hyperedges = []
        
        for _ in range(num_samples):
            # 1. 从尺寸分布中采样超边尺寸
            size = self.size_sampler.sample(1)[0]
            
            # 2. 生成负超边（确保不与真实超边重复）
            negative_hyperedge = self._generate_single_negative_hyperedge(size, max_attempts)
            
            if negative_hyperedge is not None:
                negative_hyperedges.append(negative_hyperedge)
            else:
                # 如果无法生成不重复的负样本，生成一个随机的
                negative_hyperedge = self._generate_random_hyperedge(size)
                negative_hyperedges.append(negative_hyperedge)
                
        return negative_hyperedges
    
    def _generate_single_negative_hyperedge(self, size: int, max_attempts: int) -> List[int]:
        """
        生成单个负超边，确保不与真实超边重复
        
        Args:
            size: 超边尺寸
            max_attempts: 最大尝试次数
            
        Returns:
            负超边 [user_idx, product1, product2, ...]，如果失败返回None
        """
        for _ in range(max_attempts):
            hyperedge = self._generate_random_hyperedge(size)
            hyperedge_tuple = tuple(hyperedge)
            
            if hyperedge_tuple not in self.real_hyperedges:
                return hyperedge
                
        return None  # 超过最大尝试次数
    
    def _generate_random_hyperedge(self, size: int) -> List[int]:
        """
        生成随机超边
        
        Args:
            size: 超边尺寸
            
        Returns:
            超边 [user_idx, product1, product2, ...]
        """
        # 随机选择一个用户
        user_idx = random.randint(0, self.num_users - 1)
        
        # 随机选择 (size-1) 个不重复的商品
        product_indices = random.sample(range(self.num_users, self.num_users + self.num_products), 
                                      size - 1)
        
        return [user_idx] + product_indices
    
    def update_real_hyperedges(self, new_hyperedges: Set[Tuple]):
        """
        更新真实超边集合
        
        Args:
            new_hyperedges: 新的真实超边集合
        """
        self.real_hyperedges = new_hyperedges
        print(f"✓ 真实超边集合已更新，共 {len(self.real_hyperedges)} 条")


def load_real_hyperedges_from_csv(csv_path: str) -> Set[Tuple]:
    """
    从CSV文件加载真实超边
    
    Args:
        csv_path: CSV文件路径
        
    Returns:
        真实超边集合
    """
    import pandas as pd
    
    real_hyperedges = set()
    
    try:
        df = pd.read_csv(csv_path)
        
        for _, row in df.iterrows():
            # 解析超边格式: "0,1001,1002,1003"
            hyperedge = [int(x) for x in row['hyperedge'].split(',')]
            real_hyperedges.add(tuple(hyperedge))
            
        print(f"✓ 从 {csv_path} 加载了 {len(real_hyperedges)} 条真实超边")
        
    except Exception as e:
        print(f"❌ 加载真实超边失败: {e}")
        
    return real_hyperedges


# 测试代码
if __name__ == "__main__":
    print("=== SNS负采样器测试 ===")
    
    # 测试参数
    size_sampler_path = "../edge_size_sampler.pkl"
    num_users = 1000
    num_products = 2000
    
    try:
        # 1. 加载真实超边（可选）
        real_hyperedges_path = "../hyperedges_7days_reformatted_train.csv"
        if os.path.exists(real_hyperedges_path):
            real_hyperedges = load_real_hyperedges_from_csv(real_hyperedges_path)
        else:
            real_hyperedges = set()
            print("⚠️ 未找到真实超边文件，将生成可能重复的负样本")
        
        # 2. 创建SNS负采样器
        sampler = SNSNegativeSampler(
            size_sampler_path=size_sampler_path,
            num_users=num_users,
            num_products=num_products,
            real_hyperedges=real_hyperedges
        )
        
        # 3. 采样负超边
        num_negative_samples = 10
        negative_hyperedges = sampler.sample_negative_hyperedges(num_negative_samples)
        
        print(f"\n✓ 成功采样 {len(negative_hyperedges)} 个负超边:")
        for i, hyperedge in enumerate(negative_hyperedges[:5]):  # 只显示前5个
            print(f"  负样本 {i+1}: {hyperedge} (尺寸: {len(hyperedge)})")
        
        if len(negative_hyperedges) > 5:
            print(f"  ... 还有 {len(negative_hyperedges) - 5} 个负样本")
            
        # 4. 统计尺寸分布
        sizes = [len(he) for he in negative_hyperedges]
        print(f"\n负样本尺寸分布:")
        from collections import Counter
        size_counts = Counter(sizes)
        for size, count in sorted(size_counts.items()):
            print(f"  尺寸 {size}: {count} 个")
            
        print("\n🎉 SNS负采样器测试完成!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

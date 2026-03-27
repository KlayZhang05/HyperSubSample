#!/usr/bin/env python3
"""
子图采样器实现
支持 TIHS 和 Snowball 采样策略
"""

import random
import numpy as np
from typing import Dict, List, Set, Any
from collections import deque, defaultdict


def sample_subgraph_tihs(hyperedges_dict: Dict[Any, List[int]], max_nodes: int) -> Dict[Any, List[int]]:
    """
    Totally-Induced Hyperedge Sampling (TIHS)
    
    Args:
        hyperedges_dict: 超边字典 {eid: [node_list]}
        max_nodes: 子图最大节点数
        
    Returns:
        子图超边字典
    """
    selected_edges = set()
    selected_nodes = set()
    
    all_edges = list(hyperedges_dict.items())
    random.shuffle(all_edges)
    
    for eid, nodes in all_edges:
        selected_edges.add(eid)
        selected_nodes.update(nodes)
        
        # 添加所有被当前节点完全包含的超边
        for eid2, nodes2 in hyperedges_dict.items():
            if eid2 not in selected_edges and set(nodes2).issubset(selected_nodes):
                selected_edges.add(eid2)
                
        if len(selected_nodes) >= max_nodes:
            break
    
    return {eid: hyperedges_dict[eid] for eid in selected_edges}


def sample_subgraph_snowball(hyperedges_dict: Dict[Any, List[int]], max_nodes: int) -> Dict[Any, List[int]]:
    """
    Snowball Sampling for Hypergraphs
    
    Args:
        hyperedges_dict: 超边字典 {eid: [node_list]}
        max_nodes: 子图最大节点数
        
    Returns:
        子图超边字典
    """
    if not hyperedges_dict:
        return {}
    
    # 构建节点到超边的映射
    node2edges = defaultdict(set)
    for eid, nodes in hyperedges_dict.items():
        for v in nodes:
            node2edges[v].add(eid)
    
    if not node2edges:
        return {}
    
    # 随机选择起始节点
    all_nodes = list(node2edges.keys())
    start_node = random.choice(all_nodes)
    
    visited_nodes = set([start_node])
    selected_edges = set()
    queue = deque([start_node])
    
    while queue and len(visited_nodes) < max_nodes:
        current = queue.popleft()
        
        # 添加当前节点相关的所有超边
        for eid in node2edges[current]:
            if eid not in selected_edges:
                selected_edges.add(eid)
                
                # 添加该超边中的所有节点
                for v in hyperedges_dict[eid]:
                    if v not in visited_nodes:
                        visited_nodes.add(v)
                        queue.append(v)
                        
        if len(visited_nodes) >= max_nodes:
            break
    
    return {eid: hyperedges_dict[eid] for eid in selected_edges}


def sample_subgraph(data: Dict[Any, List[int]], m: int, strategy: str = "TIHS") -> Dict[Any, List[int]]:
    """
    子图采样统一接口
    
    Args:
        data: 超图数据结构 {eid: [v1, v2, ...]}
        m: 子图最大节点数
        strategy: 采样策略 "TIHS" 或 "snowball"
        
    Returns:
        子图数据 {eid: [v1, v2, ...]}
    """
    if strategy == "TIHS":
        return sample_subgraph_tihs(data, m)
    elif strategy == "snowball":
        return sample_subgraph_snowball(data, m)
    else:
        raise ValueError(f"不支持的采样策略: {strategy}")


def validate_subgraph(subgraph: Dict[Any, List[int]], original_graph: Dict[Any, List[int]]) -> bool:
    """
    验证子图的有效性
    
    Args:
        subgraph: 子图
        original_graph: 原始超图
        
    Returns:
        是否有效
    """
    # 检查所有超边是否在原图中
    for eid, nodes in subgraph.items():
        if eid not in original_graph:
            return False
        if nodes != original_graph[eid]:
            return False
    
    return True


# 测试代码
if __name__ == "__main__":
    print("=== 子图采样器测试 ===")
    
    # 创建测试超图
    test_hypergraph = {
        0: [0, 1, 2, 3],
        1: [4, 5, 6],
        2: [7, 8, 9, 10, 11],
        3: [12, 13, 14],
        4: [15, 16, 17, 18, 19],
        5: [0, 5, 10],
        6: [1, 6, 11, 16],
        7: [2, 7, 12, 17],
        8: [3, 8, 13, 18],
        9: [4, 9, 14, 19]
    }
    
    print(f"原始超图: {len(test_hypergraph)} 个超边")
    all_nodes = set()
    for nodes in test_hypergraph.values():
        all_nodes.update(nodes)
    print(f"总节点数: {len(all_nodes)}")
    
    # 测试TIHS采样
    print("\n--- 测试 TIHS 采样 ---")
    for max_nodes in [5, 10, 15]:
        subgraph = sample_subgraph(test_hypergraph, max_nodes, "TIHS")
        subgraph_nodes = set()
        for nodes in subgraph.values():
            subgraph_nodes.update(nodes)
        
        print(f"max_nodes={max_nodes}: 得到 {len(subgraph)} 个超边, {len(subgraph_nodes)} 个节点")
        print(f"  有效性检查: {validate_subgraph(subgraph, test_hypergraph)}")
    
    # 测试Snowball采样
    print("\n--- 测试 Snowball 采样 ---")
    for max_nodes in [5, 10, 15]:
        subgraph = sample_subgraph(test_hypergraph, max_nodes, "snowball")
        subgraph_nodes = set()
        for nodes in subgraph.values():
            subgraph_nodes.update(nodes)
        
        print(f"max_nodes={max_nodes}: 得到 {len(subgraph)} 个超边, {len(subgraph_nodes)} 个节点")
        print(f"  有效性检查: {validate_subgraph(subgraph, test_hypergraph)}")
    
    print("\n✓ 子图采样器测试完成")

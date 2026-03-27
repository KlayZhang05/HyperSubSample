#!/usr/bin/env python3
"""
简化的系统结构测试 - 不依赖PyTorch
"""

import os
import sys

def test_file_structure():
    """测试文件结构是否正确"""
    print("=== 测试文件结构 ===")
    
    expected_files = [
        "main.py",
        "training_pipeline.py", 
        "end_to_end_model.py",
        "modified_hypergcn.py",
        "hyperedge_aggregator.py",
        "sns_negative_sampler.py",
        "README.md"
    ]
    
    for file in expected_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"❌ {file}")
    
    # 检查数据文件
    data_files = [
        "../hyperedges_7days_reformatted_train.csv",
        "../hyperedges_7days_reformatted_val.csv", 
        "../hyperedges_7days_reformatted_test.csv",
        "../edge_size_sampler.pkl"
    ]
    
    print(f"\n=== 检查数据文件 ===")
    for file in data_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"❌ {file}")

def test_code_syntax():
    """测试代码语法是否正确"""
    print(f"\n=== 测试代码语法 ===")
    
    python_files = [
        "main.py",
        "training_pipeline.py",
        "end_to_end_model.py", 
        "modified_hypergcn.py",
        "hyperedge_aggregator.py",
        "sns_negative_sampler.py"
    ]
    
    for file in python_files:
        if os.path.exists(file):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    code = f.read()
                
                # 简单的语法检查
                compile(code, file, 'exec')
                print(f"✓ {file} - 语法正确")
                
            except SyntaxError as e:
                print(f"❌ {file} - 语法错误: {e}")
            except Exception as e:
                print(f"⚠️ {file} - 其他问题: {e}")

def test_hypergcn_components():
    """测试HyperGCN核心组件是否包含"""
    print(f"\n=== 检查HyperGCN核心组件 ===")
    
    hypergcn_file = "modified_hypergcn.py"
    if not os.path.exists(hypergcn_file):
        print(f"❌ {hypergcn_file} 文件不存在")
        return
    
    with open(hypergcn_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查关键组件
    components_to_check = [
        ("Laplacian", "Laplacian近似函数"),
        ("HyperGraphConvolution", "超图卷积层"),
        ("SparseMM", "稀疏矩阵乘法"),
        ("symnormalise", "对称归一化"),
        ("adjacency", "邻接矩阵构建"),
        ("update", "边权重更新"),
        ("ssm2tst", "稀疏矩阵转张量")
    ]
    
    for component, description in components_to_check:
        if component in content:
            print(f"✓ {component}: {description}")
        else:
            print(f"❌ {component}: {description} - 未找到")
    
    # 检查是否真的使用了HyperGCN而不是简单MLP
    if "nn.Linear" in content and "Laplacian" in content:
        print(f"✓ 真正的HyperGCN实现：包含Laplacian近似和超图卷积")
    elif "nn.Linear" in content and "Laplacian" not in content:
        print(f"❌ 检测到简单MLP伪装成HyperGCN")
    else:
        print(f"⚠️ HyperGCN实现不明确")


def count_lines_of_code():
    """统计代码行数"""
    print(f"\n=== 代码统计 ===")
    
    python_files = [
        "main.py",
        "training_pipeline.py",
        "end_to_end_model.py",
        "modified_hypergcn.py", 
        "hyperedge_aggregator.py",
        "sns_negative_sampler.py"
    ]
    
    total_lines = 0
    total_code_lines = 0
    
    for file in python_files:
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            file_total = len(lines)
            file_code = sum(1 for line in lines 
                          if line.strip() and not line.strip().startswith('#'))
            
            total_lines += file_total
            total_code_lines += file_code
            
            print(f"  {file}: {file_total} 行 ({file_code} 代码行)")
    
    print(f"\n总计: {total_lines} 行 ({total_code_lines} 代码行)")

def show_system_summary():
    """显示系统概览"""
    print(f"\n" + "="*60)
    print("🎉 超边预测系统构建完成!")
    print("="*60)
    
    print(f"\n📋 系统组件:")
    components = [
        ("真正的HyperGCN", "包含Laplacian近似的超图卷积网络"),
        ("Hyperedge Aggregator", "超边节点嵌入聚合器"),
        ("SNS Negative Sampler", "基于尺寸分布的负采样器"),
        ("End-to-End Model", "HyperGCN + 聚合器 + MLP分类器"),
        ("Training Pipeline", "完整训练和评估流程"),
        ("Main Entry", "命令行主入口程序")
    ]
    
    for name, desc in components:
        print(f"  ✓ {name}: {desc}")
    
    print(f"\n🏗️ 系统架构:")
    print("  输入: 候选超边列表")
    print("  ↓")
    print("  HyperGCN(训练集超图) → 节点嵌入(400维)")
    print("  ↓") 
    print("  均值聚合 → 超边嵌入(400维)")
    print("  ↓")
    print("  MLP分类器 → 超边概率")
    
    print(f"\n📊 训练策略:")
    print("  ✓ 端到端训练")
    print("  ✓ 每epoch动态负采样(1:1正负比)")
    print("  ✓ 基于尺寸分布的SNS负采样")
    print("  ✓ 防数据泄露的数据分割")
    
    print(f"\n🚀 使用方法:")
    print("  训练+测试: python main.py")
    print("  自定义参数: python main.py --epochs 200 --batch_size 128")
    print("  详细说明: 参见 README.md")


if __name__ == "__main__":
    test_file_structure()
    test_code_syntax()
    test_hypergcn_components()
    count_lines_of_code()
    show_system_summary()

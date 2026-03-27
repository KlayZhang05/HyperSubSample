#!/usr/bin/env python3
"""
超边预测系统主入口脚本
整合所有组件，提供完整的训练和测试流程
"""

import os
import sys
import argparse
import torch
import json
import numpy as np
from datetime import datetime

# 导入我们的组件
from training_pipeline import HyperedgeTrainer, SubgraphHyperedgeTrainer, HyperedgePredictionConfig
from end_to_end_model import HyperedgePredictionModel


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="超边预测系统")
    parser.add_argument("--train_csv", type=str, default="/mnt/workspace/推荐算法/Hypersubsample/data/hyperedges_7days_reformatted_train.csv",
                       help="训练集CSV文件路径")
    parser.add_argument("--val_csv", type=str, default="/mnt/workspace/推荐算法/Hypersubsample/data/hyperedges_7days_reformatted_val.csv",
                       help="验证集CSV文件路径")
    parser.add_argument("--test_csv", type=str, default="/mnt/workspace/推荐算法/Hypersubsample/data/hyperedges_7days_reformatted_test.csv",
                       help="测试集CSV文件路径")
    parser.add_argument("--size_sampler", type=str, default="/mnt/workspace/推荐算法/Hypersubsample/data/edge_size_sampler.pkl",
                       help="尺寸采样器文件路径")
    parser.add_argument("--num_users", type=int, default=82740,
                       help="用户总数")
    parser.add_argument("--num_products", type=int, default=38830,
                       help="商品总数")
    parser.add_argument("--epochs", type=int, default=100,
                       help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="批次大小")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="学习率")
    parser.add_argument("--embedding_dim", type=int, default=64,
                       help="嵌入维度")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="输出目录")
    parser.add_argument("--mode", type=str, choices=["train", "test", "both"], default="both",
                       help="运行模式: train(仅训练), test(仅测试), both(训练+测试)")
    
    # 训练策略参数
    parser.add_argument("--training_strategy", type=str, choices=["full_graph", "subgraph"], default="full_graph",
                       help="训练策略: full_graph(全图训练), subgraph(子图采样训练)")
    
    # 子图训练参数（仅在 --training_strategy=subgraph 时使用）
    parser.add_argument("--T", type=int, default=100,
                       help="大循环轮数（子图训练）")
    parser.add_argument("--R", type=int, default=16,
                       help="每轮大循环中子图数量（子图训练）")
    parser.add_argument("--m", type=int, default=300,
                       help="每个子图的最大节点数（子图训练）")
    parser.add_argument("--L", type=int, default=5,
                       help="每个子图训练的小循环次数（子图训练）")
    parser.add_argument("--sampling_strategy", type=str, choices=["TIHS", "snowball"], default="TIHS",
                       help="子图采样策略（子图训练）")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("🚀 超边预测系统启动")
    print(f"📋 训练策略: {args.training_strategy}")
    print("=" * 60)
    
    # 检查必要文件
    required_files = [args.train_csv, args.val_csv, args.test_csv, args.size_sampler]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ 缺少必要文件: {missing_files}")
        print("请确保数据预处理管道已运行并生成了所需文件")
        return
    
    # 配置模型
    config = HyperedgePredictionConfig()
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.embedding_dim = args.embedding_dim
    
    print(f"✓ 模型配置:")
    print(f"  - 设备: {config.device}")
    print(f"  - 训练轮数: {config.epochs}")
    print(f"  - 批次大小: {config.batch_size}")
    print(f"  - 学习率: {config.learning_rate}")
    print(f"  - 嵌入维度: {config.embedding_dim}")
    
    if args.training_strategy == "subgraph":
        print(f"✓ 子图训练配置:")
        print(f"  - 大循环轮数 (T): {args.T}")
        print(f"  - 每轮子图数量 (R): {args.R}")
        print(f"  - 子图最大节点数 (m): {args.m}")
        print(f"  - 子图训练次数 (L): {args.L}")
        print(f"  - 采样策略: {args.sampling_strategy}")
    
    # 创建训练器（根据策略选择）
    if args.training_strategy == "full_graph":
        trainer = HyperedgeTrainer(config)
        print("✓ 使用全图训练器")
    else:  # subgraph
        trainer = SubgraphHyperedgeTrainer(
            config=config,
            T=args.T,
            R=args.R,
            m=args.m,
            L=args.L,
            sampling_strategy=args.sampling_strategy
        )
        print("✓ 使用子图采样训练器")
    
    try:
        if args.mode in ["train", "both"]:
            print(f"\n🔥 开始{'子图' if args.training_strategy == 'subgraph' else '全图'}训练阶段")
            print("-" * 40)
            
            # 加载数据
            val_data, test_data, train_hypergraph = trainer.load_data(
                train_csv=args.train_csv,
                val_csv=args.val_csv,
                test_csv=args.test_csv,
                size_sampler_path=args.size_sampler,
                num_users=args.num_users,
                num_products=args.num_products
            )
            
            # 创建模型
            num_nodes = args.num_users + args.num_products
            model = trainer.create_model(num_nodes, train_hypergraph)
            
            # 训练模型
            model_save_path = os.path.join(args.output_dir, f"best_{'subgraph' if args.training_strategy == 'subgraph' else 'full_graph'}_model.pth")
            model = trainer.train(model, val_data, model_save_path)
              # 绘制训练历史（包含时间统计）
            if args.training_strategy == "full_graph":
                history_plot_path = os.path.join(args.output_dir, "training_history.png")
                trainer.plot_training_history(history_plot_path)
            else:  # subgraph
                # 子图训练的历史绘制（如果实现了的话）
                if hasattr(trainer, 'plot_subgraph_training_history'):
                    history_plot_path = os.path.join(args.output_dir, "subgraph_training_history.png")
                    trainer.plot_subgraph_training_history(history_plot_path)
            
            # 保存配置（包含时间信息）
            config_path = os.path.join(args.output_dir, "config.json")
            config_dict = {
                "embedding_dim": config.embedding_dim,
                "mlp_hidden_dims": config.mlp_hidden_dims,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "epochs": config.epochs,
                "num_users": args.num_users,
                "num_products": args.num_products,
                "train_time": datetime.now().isoformat(),
                "training_completed": True,
                "training_strategy": args.training_strategy,
                "best_validation_accuracy": trainer.best_val_accuracy
            }
            
            # 添加训练时间统计
            if args.training_strategy == "full_graph":
                config_dict.update({
                    "total_training_time_seconds": sum(trainer.epoch_times) if trainer.epoch_times else 0,
                    "average_epoch_time_seconds": np.mean(trainer.epoch_times) if trainer.epoch_times else 0,
                })
            else:  # subgraph
                config_dict.update({
                    "T": args.T,
                    "R": args.R,
                    "m": args.m,
                    "L": args.L,
                    "sampling_strategy": args.sampling_strategy,
                    "total_training_time_seconds": sum(trainer.major_cycle_times) if hasattr(trainer, 'major_cycle_times') and trainer.major_cycle_times else 0,
                    "average_major_cycle_time_seconds": np.mean(trainer.major_cycle_times) if hasattr(trainer, 'major_cycle_times') and trainer.major_cycle_times else 0,
                    "average_subgraph_time_seconds": np.mean(trainer.subgraph_times) if hasattr(trainer, 'subgraph_times') and trainer.subgraph_times else 0,
                    "total_subgraphs_trained": len(trainer.subgraph_times) if hasattr(trainer, 'subgraph_times') else 0,
                })
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            print(f"✓ 训练阶段完成")
            
        if args.mode in ["test", "both"]:
            print(f"\n📊 开始测试阶段")
            print("-" * 40)
            
            # 如果是仅测试模式，需要重新加载数据和模型
            if args.mode == "test":
                # 加载数据
                val_data, test_data, train_hypergraph = trainer.load_data(
                    train_csv=args.train_csv,
                    val_csv=args.val_csv,
                    test_csv=args.test_csv,
                    size_sampler_path=args.size_sampler,
                    num_users=args.num_users,
                    num_products=args.num_products
                )
                
                # 重新创建模型
                num_nodes = args.num_users + args.num_products
                model = trainer.create_model(num_nodes, train_hypergraph)
                
                # 加载训练好的权重
                model_save_path = os.path.join(args.output_dir, f"best_{'subgraph' if args.training_strategy == 'subgraph' else 'full_graph'}_model.pth")
                if os.path.exists(model_save_path):
                    model.load_state_dict(torch.load(model_save_path))
                    print(f"✓ 从 {model_save_path} 加载模型权重")
                else:
                    print(f"❌ 未找到训练好的模型: {model_save_path}")
                    return
            
            # 在测试集上评估
            test_accuracy = trainer.evaluate(model, test_data)
            
            print(f"✅ 测试结果:")
            print(f"  - 训练策略: {args.training_strategy}")
            print(f"  - 测试集准确率: {test_accuracy:.4f}")
            
            # 保存测试结果
            results_path = os.path.join(args.output_dir, "test_results.json")
            results = {
                "test_accuracy": test_accuracy,
                "test_time": datetime.now().isoformat(),
                "training_strategy": args.training_strategy
            }
            
            # 添加策略特定信息
            if args.training_strategy == "subgraph":
                results.update({
                    "T": args.T,
                    "R": args.R,
                    "m": args.m,
                    "L": args.L,
                    "sampling_strategy": args.sampling_strategy
                })
            
            if args.mode == "both":
                results["best_val_accuracy"] = trainer.best_val_accuracy
                
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"✓ 测试结果已保存到: {results_path}")
        
        print(f"\n🎉 超边预测系统运行完成!")
        print(f"📁 所有输出文件保存在: {args.output_dir}")
        
    except Exception as e:
        print(f"\n❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 保存错误信息
        error_path = os.path.join(args.output_dir, "error_log.txt")
        with open(error_path, 'w') as f:
            f.write(f"错误时间: {datetime.now().isoformat()}\n")
            f.write(f"错误信息: {str(e)}\n")
            f.write(f"详细堆栈:\n{traceback.format_exc()}")
        print(f"❌ 错误信息已保存到: {error_path}")


if __name__ == "__main__":
    main()

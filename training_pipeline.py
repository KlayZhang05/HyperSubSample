#!/usr/bin/env python3
"""
超边预测模型训练流程
包含动态负采样、验证和测试
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Set
import os
import pickle
import json
import glob
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
import sys
from contextlib import contextmanager

# 导入我们的组件
from end_to_end_model import HyperedgePredictionModel, HyperedgePredictionConfig, create_mock_hypergcn_args
from sns_negative_sampler import SNSNegativeSampler, load_real_hyperedges_from_csv
from subgraph_sampler import sample_subgraph

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")


class TeeOutput:
    """
    双输出类：同时输出到控制台和文件
    """
    def __init__(self, file_path, mode='w'):
        self.file = open(file_path, mode, encoding='utf-8')
        self.terminal = sys.stdout
    
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()  # 确保实时写入文件
    
    def flush(self):
        self.terminal.flush()
        self.file.flush()
    
    def close(self):
        self.file.close()


@contextmanager
def dual_output(log_file_path):
    """
    上下文管理器：在指定范围内同时输出到控制台和日志文件
    """
    # 保存原始stdout
    original_stdout = sys.stdout
    
    try:
        # 创建双输出对象
        tee = TeeOutput(log_file_path)
        sys.stdout = tee
        yield tee
    finally:
        # 恢复原始stdout
        sys.stdout = original_stdout
        if 'tee' in locals():
            tee.close()


class HyperedgeTrainer:
    """
    超边预测模型训练器
    """
    
    def __init__(self, config: HyperedgePredictionConfig):
        """
        初始化训练器
        
        Args:
            config: 模型配置
        """
        self.config = config
        self.device = config.device
        
        # 训练历史
        self.train_losses = []
        self.val_accuracies = []
        self.epoch_times = []  # 每个epoch的耗时
        self.best_val_accuracy = 0.0
        self.best_model_state = None
        
        # 设置日志
        self.setup_logging()
        
        # 训练开始时间
        self.training_start_time = None
        
        print(f"✓ 训练器初始化完成，使用设备: {self.device}")
    
    def setup_logging(self):
        """设置简化的日志系统 - 只生成完整输出日志"""
        # 创建日志目录
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # 生成日志文件名（包含时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.full_output_log = os.path.join(log_dir, f"training_{timestamp}.log")
        
        # 不再使用复杂的logging系统，只使用简单的print输出
        print(f"� 日志将保存到: {self.full_output_log}")
    
    def load_data(self, train_csv: str, val_csv: str, test_csv: str, 
                  size_sampler_path: str, num_users: int, num_products: int, 
                  force_regenerate_negatives: bool = False) -> Tuple:
        """
        加载训练、验证和测试数据 - 混合策略：第一个epoch动态负采样，后续epoch使用固定负样本
        
        Args:
            train_csv: 训练集CSV文件路径
            val_csv: 验证集CSV文件路径  
            test_csv: 测试集CSV文件路径
            size_sampler_path: 尺寸采样器路径
            num_users: 用户总数
            num_products: 商品总数
            force_regenerate_negatives: 是否强制重新生成负样本（默认False，优先使用缓存）
            
        Returns:
            (val_data, test_data, train_hypergraph)
        """
        print("=== 加载数据（混合负采样策略）===")
        
        # 1. 加载正样本
        train_positives = self._load_hyperedges_from_csv(train_csv)
        val_positives = self._load_hyperedges_from_csv(val_csv)
        test_positives = self._load_hyperedges_from_csv(test_csv)
        
        print(f"✓ 正样本加载完成:")
        print(f"  - 训练集: {len(train_positives)} 条")
        print(f"  - 验证集: {len(val_positives)} 条")
        print(f"  - 测试集: {len(test_positives)} 条")
        
        # 2. 创建负采样器
        train_hyperedges_set = set(tuple(he) for he in train_positives)
        self.negative_sampler = SNSNegativeSampler(
            size_sampler_path=size_sampler_path,
            num_users=num_users,
            num_products=num_products,
            real_hyperedges=train_hyperedges_set
        )
        
        # 3. 为验证集和测试集生成或加载固定的负样本（训练集采用混合策略）
        val_negatives_cache_path = f"val_negatives_{len(val_positives)}.pkl"
        test_negatives_cache_path = f"test_negatives_{len(test_positives)}.pkl"
        
        # 如果强制重新生成，删除现有缓存
        if force_regenerate_negatives:
            for cache_path in [val_negatives_cache_path, test_negatives_cache_path]:
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                    print(f"🗑️  删除旧缓存: {cache_path}")
        
        # 加载验证集负样本
        if os.path.exists(val_negatives_cache_path):
            print(f"🔄 加载已有验证集负样本: {val_negatives_cache_path}")
            with open(val_negatives_cache_path, 'rb') as f:
                val_negatives = pickle.load(f)
        else:
            print(f"🔄 生成新的验证集负样本...")
            val_negatives = self.negative_sampler.sample_negative_hyperedges(len(val_positives))
            # 保存到文件
            with open(val_negatives_cache_path, 'wb') as f:
                pickle.dump(val_negatives, f)
            print(f"✓ 验证集负样本已保存到: {val_negatives_cache_path}")
        
        # 加载测试集负样本
        if os.path.exists(test_negatives_cache_path):
            print(f"🔄 加载已有测试集负样本: {test_negatives_cache_path}")
            with open(test_negatives_cache_path, 'rb') as f:
                test_negatives = pickle.load(f)
        else:
            print(f"🔄 生成新的测试集负样本...")
            test_negatives = self.negative_sampler.sample_negative_hyperedges(len(test_positives))
            # 保存到文件
            with open(test_negatives_cache_path, 'wb') as f:
                pickle.dump(test_negatives, f)
            print(f"✓ 测试集负样本已保存到: {test_negatives_cache_path}")
        
        print(f"✓ 验证/测试集负样本准备完成:")
        print(f"  - 验证集负样本: {len(val_negatives)} 条 ({'复用缓存' if os.path.exists(val_negatives_cache_path) and not force_regenerate_negatives else '新生成'})")
        print(f"  - 测试集负样本: {len(test_negatives)} 条 ({'复用缓存' if os.path.exists(test_negatives_cache_path) and not force_regenerate_negatives else '新生成'})")
        
        # 4. 训练集采用混合策略
        print(f"🎯 训练集采用混合负采样策略:")
        print(f"  - 第1个epoch: 动态负采样 (每batch 1:1 正负比例)")
        print(f"  - 第2+个epoch: 使用第1个epoch收集的固定负样本")
        
        # 保存正样本用于混合策略
        self.train_positives = train_positives
        self.train_negatives = []  # 初始为空，第一个epoch后会填充
        
        val_data = (val_positives, val_negatives)
        test_data = (test_positives, test_negatives)
        
        # 5. 构建训练超图（仅使用训练集超边）
        train_hypergraph = {i: hyperedge for i, hyperedge in enumerate(train_positives)}
        
        return val_data, test_data, train_hypergraph
    
    def _load_hyperedges_from_csv(self, csv_path: str) -> List[List[int]]:
        """从CSV文件加载超边"""
        hyperedges = []
        df = pd.read_csv(csv_path)
        
        for _, row in df.iterrows():
            # 使用 'full_hyperedge' 列，并去掉首尾引号
            hyperedge_str = row['full_hyperedge'].strip('"')
            hyperedge = [int(x) for x in hyperedge_str.split(',')]
            hyperedges.append(hyperedge)
            
        return hyperedges
    
    @staticmethod
    def clear_negative_sample_cache():
        """清理所有负样本缓存文件"""
        import glob
        cache_files = (glob.glob("train_negatives_*.pkl") + 
                      glob.glob("val_negatives_*.pkl") + 
                      glob.glob("test_negatives_*.pkl"))
        for cache_file in cache_files:
            try:
                os.remove(cache_file)
                print(f"🗑️  已删除缓存文件: {cache_file}")
            except Exception as e:
                print(f"⚠️  删除缓存文件失败 {cache_file}: {e}")
        
        if cache_files:
            print(f"✓ 已清理 {len(cache_files)} 个负样本缓存文件")
        else:
            print("ℹ️  没有找到负样本缓存文件")
    
    @staticmethod
    def list_negative_sample_cache():
        """列出所有负样本缓存文件"""
        import glob
        cache_files = (glob.glob("train_negatives_*.pkl") + 
                      glob.glob("val_negatives_*.pkl") + 
                      glob.glob("test_negatives_*.pkl"))
        if cache_files:
            print("📂 找到的负样本缓存文件:")
            for cache_file in cache_files:
                file_size = os.path.getsize(cache_file) / 1024 / 1024  # MB
                mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                if "train_" in cache_file:
                    cache_type = "训练(收集)" if "collected" in cache_file else "训练"
                elif "val_" in cache_file:
                    cache_type = "验证"
                else:
                    cache_type = "测试"
                print(f"  - {cache_file} ({cache_type}集, {file_size:.2f}MB, 修改时间: {mod_time})")
        else:
            print("ℹ️  没有找到负样本缓存文件")
    
    def create_model(self, num_nodes: int, train_hypergraph: Dict, feature_dim: int = 64) -> HyperedgePredictionModel:
        """
        创建模型
        
        Args:
            num_nodes: 节点总数
            train_hypergraph: 训练超图
            feature_dim: 节点特征维度
            
        Returns:
            初始化的模型
        """
        print("=== 创建模型 ===")
        
        # 1. 创建节点特征（随机初始化）
        node_features = torch.randn(num_nodes, feature_dim)
        
        # 2. 创建HyperGCN参数
        hypergcn_args = create_mock_hypergcn_args(self.config, feature_dim)
        
        # 3. 创建模型
        model = HyperedgePredictionModel(
            num_nodes=num_nodes,
            hypergraph=train_hypergraph,
            node_features=node_features,
            embedding_dim=self.config.embedding_dim,
            hidden_dims=self.config.mlp_hidden_dims,
            hypergcn_args=hypergcn_args
        )
        
        model = model.to(self.device)
        
        print(f"✓ 模型创建完成，参数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  - HyperGCN参数: {sum(p.numel() for p in model.hypergcn.parameters()):,}")
        print(f"  - 下游任务参数: {sum(p.numel() for p in model.aggregator.parameters()) + sum(p.numel() for p in model.mlp_classifier.parameters()):,}")
        
        return model
    
    def train_epoch(self, model: HyperedgePredictionModel, epoch_num: int) -> Tuple[float, torch.Tensor]:
        """
        训练一个epoch - 端到端训练策略（全图训练，无批处理）
        
        Args:
            model: 模型
            epoch_num: 当前epoch编号（从0开始）
            
        Returns:
            (平均损失, 最终节点嵌入) - 返回嵌入供验证复用
        """
        epoch_start_time = time.time()
        
        model.train()
        
        # 准备全图训练数据（不使用DataLoader）
        if epoch_num == 0:
            # 第一个epoch：动态负采样
            print("🎯 第1个epoch - 使用动态负采样策略（全图训练）")
            positive_count = len(self.train_positives)
            negative_samples = self.negative_sampler.sample_negative_hyperedges(positive_count)
            
            # 收集负样本用于后续epoch
            self.train_negatives = negative_samples
            print(f"✓ 收集到 {len(negative_samples)} 个负样本用于后续epoch")
            
            # 可选：保存到缓存文件
            cache_path = f"train_negatives_{len(self.train_positives)}_collected.pkl"
            with open(cache_path, 'wb') as f:
                pickle.dump(negative_samples, f)
            print(f"✓ 收集的负样本已缓存到: {cache_path}")
            
            use_dynamic_sampling = True
        else:
            # 第二个及以后的epoch：使用固定负样本
            if not self.train_negatives:
                print("❌ 错误：应该在第一个epoch后收集到负样本，但负样本列表为空")
                # 降级到动态采样
                print("🔄 降级到动态负采样...")
                positive_count = len(self.train_positives)
                negative_samples = self.negative_sampler.sample_negative_hyperedges(positive_count)
                use_dynamic_sampling = True
            else:
                print(f"🎯 第{epoch_num+1}个epoch - 使用固定负样本策略（全图训练）")
                print(f"  - 使用第1个epoch收集的 {len(self.train_negatives)} 个负样本")
                negative_samples = self.train_negatives
                use_dynamic_sampling = False
        
        # 🚀 全图训练：组合所有正负样本
        all_hyperedges = self.train_positives + negative_samples
        all_labels = [1] * len(self.train_positives) + [0] * len(negative_samples)
        all_labels = torch.tensor(all_labels, dtype=torch.long).to(self.device)
        
        print(f"📊 全图训练数据统计:")
        print(f"  - 正样本: {len(self.train_positives)} 个")
        print(f"  - 负样本: {len(negative_samples)} 个")
        print(f"  - 总样本: {len(all_hyperedges)} 个")
        
        # 🚀 端到端训练：创建统一的优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        
        # 🚀 全图前向传播（一次性处理所有数据）
        optimizer.zero_grad()
        
        print("� 开始全图前向传播...")
        forward_start_time = time.time()
        
        # 完整的前向传播（包含HyperGCN全图Laplacian近似）
        logits = model(all_hyperedges)
        
        forward_time = time.time() - forward_start_time
        print(f"✓ 全图前向传播完成: {forward_time:.2f}s")
        
        # 计算损失
        loss = F.cross_entropy(logits, all_labels)
        
        print(f"📉 损失值: {loss.item():.4f}")
        print("🔄 开始反向传播...")
        backward_start_time = time.time()
        
        # 反向传播（更新所有参数：HyperGCN + 聚合器 + MLP）
        loss.backward()
        
        # 更新所有参数
        optimizer.step()
        
        backward_time = time.time() - backward_start_time
        print(f"✓ 反向传播完成: {backward_time:.2f}s")
        
        # 🚀 获取最终节点嵌入用于验证
        with torch.no_grad():
            final_node_embeddings = model.hypergcn.get_node_embeddings().clone()
        
        epoch_time = time.time() - epoch_start_time
        self.epoch_times.append(epoch_time)
        
        # 详细日志记录
        sampling_strategy = "动态负采样" if use_dynamic_sampling else "固定负样本"
        print(f"✓ Epoch训练完成 - 总耗时: {epoch_time:.2f}s")
        print(f"  - 采样策略: {sampling_strategy}")
        print(f"  - 前向传播: {forward_time:.2f}s")
        print(f"  - 反向传播: {backward_time:.2f}s")
        print(f"  - 训练样本: {len(all_hyperedges)} 个（全图训练）")
        print(f"  - 全图HyperGCN: Laplacian近似覆盖整个超图")
        
        return loss.item(), final_node_embeddings
    
    
    def evaluate(self, model: HyperedgePredictionModel, data: Tuple, cached_embeddings=None) -> float:
        """
        评估模型（全图评估，无批处理）
        
        Args:
            model: 模型
            data: (正样本, 负样本) 元组
            cached_embeddings: 可选的缓存节点嵌入，避免重复计算HyperGCN
            
        Returns:
            准确率
        """
        model.eval()
        
        positives, negatives = data
        all_hyperedges = positives + negatives
        all_labels = [1] * len(positives) + [0] * len(negatives)
        
        print(f"� 评估数据统计:")
        print(f"  - 正样本: {len(positives)} 个")
        print(f"  - 负样本: {len(negatives)} 个")
        print(f"  - 总样本: {len(all_hyperedges)} 个")
        
        with torch.no_grad():
            eval_start_time = time.time()
            
            if cached_embeddings is not None:
                print("✓ 使用缓存嵌入进行全图评估")
                # 🚀 手动进行聚合和分类（使用缓存嵌入）
                all_embeddings = []
                for hyperedge in all_hyperedges:
                    hedge_indices = torch.tensor(hyperedge, device=self.device)
                    hedge_node_embeddings = cached_embeddings[hedge_indices]
                    
                    # 直接聚合
                    aggregated_embedding = model.aggregator.forward(hedge_node_embeddings, torch.arange(len(hyperedge), device=self.device))
                    all_embeddings.append(aggregated_embedding)
                
                all_embeddings = torch.stack(all_embeddings)
                # MLP分类
                logits = model.mlp_classifier(all_embeddings)
            else:
                print("🔄 使用完整模型进行全图评估（包含HyperGCN重计算）")
                # 🚀 使用完整的模型forward（全图HyperGCN + 聚合 + 分类）
                logits = model(all_hyperedges)
            
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            
            eval_time = time.time() - eval_start_time
            print(f"✓ 全图评估完成: {eval_time:.2f}s")
        
        # 计算准确率
        print("预测结果的和：", predictions.sum())
        correct = sum(1 for pred, label in zip(predictions, all_labels) if pred == label)
        accuracy = correct / len(all_labels)
        
        return accuracy
    def train(self, model: HyperedgePredictionModel, val_data: Tuple, 
              save_path: str = "best_model.pth") -> HyperedgePredictionModel:
        """
        完整训练流程（支持完整输出日志记录）
        
        Args:
            model: 模型
            val_data: 验证数据
            save_path: 模型保存路径
            
        Returns:
            训练好的模型
        """
        self.training_start_time = time.time()
        
        # 🎯 开启完整输出日志记录
        with dual_output(self.full_output_log):
            print("=" * 60)
            print("📝 完整输出日志记录已启动")
            print(f"📄 日志文件: {self.full_output_log}")
            print("=" * 60)
            
            print("=" * 50)
            print("🔥 开始训练")
            print("=" * 50)
            print(f"训练配置:")
            print(f"  - 总轮数: {self.config.epochs}")
            print(f"  - 批次大小: {self.config.batch_size}")
            print(f"  - 学习率: {self.config.learning_rate}")
            print(f"  - 训练样本数: {len(self.train_positives)} 正样本 + {len(self.train_negatives)} 负样本")
            print(f"  - 验证样本数: {len(val_data[0])}")
            print(f"  - 设备: {self.device}")
            
            best_epoch = 0
            
            for epoch in tqdm(range(self.config.epochs), desc="训练进度"):
                epoch_start_time = time.time()
                
                print(f"\n{'='*50}")
                print(f"🚀 开始第 {epoch+1}/{self.config.epochs} 轮训练")
                print(f"{'='*50}")
                
                # 训练一个epoch，获取最终嵌入
                train_loss, final_embeddings = self.train_epoch(model, epoch)
                
                # 🚀 优化：验证时直接使用训练结束时的嵌入（无需重复计算）
                val_start_time = time.time()
                print("✓ 使用训练时的最终嵌入进行验证")
                
                val_accuracy = self.evaluate(model, val_data, final_embeddings)
                val_time = time.time() - val_start_time
                
                epoch_total_time = time.time() - epoch_start_time
                
                # 记录历史
                self.train_losses.append(train_loss)
                self.val_accuracies.append(val_accuracy)
                
                # 保存最佳模型
                if val_accuracy > self.best_val_accuracy:
                    self.best_val_accuracy = val_accuracy
                    self.best_model_state = model.state_dict().copy()
                    best_epoch = epoch + 1
                    print(f"🏆 发现新的最佳模型！验证准确率: {val_accuracy:.4f}")
                    
                # 详细日志记录
                elapsed_time = time.time() - self.training_start_time
                avg_epoch_time = elapsed_time / (epoch + 1)
                remaining_epochs = self.config.epochs - (epoch + 1)
                estimated_remaining_time = avg_epoch_time * remaining_epochs
                
                # 格式化时间显示
                elapsed_str = self._format_time(elapsed_time)
                remaining_str = self._format_time(estimated_remaining_time)
                
                print(f"\n📊 第 {epoch+1} 轮训练结果:")
                print(f"  📉 训练损失: {train_loss:.4f}")
                print(f"  🎯 验证准确率: {val_accuracy:.4f} {'✨ (历史最佳!)' if val_accuracy == self.best_val_accuracy else ''}")
                print(f"  ⏱️  验证耗时: {val_time:.2f}s")
                print(f"  ⏱️  本轮总耗时: {epoch_total_time:.2f}s")
                print(f"  🕐 累计训练时间: {elapsed_str}")
                print(f"  🕑 预计剩余时间: {remaining_str}")
                print(f"  📈 平均每轮时间: {avg_epoch_time:.2f}s")
                
                # 每10个epoch打印详细统计
                if (epoch + 1) % 10 == 0:
                    self._log_training_statistics(epoch + 1)
                    
                print("-" * 50)
            
            # 训练完成统计
            print(f"\n{'='*60}")
            print("🎉 训练完成！")
            print(f"{'='*60}")
            self._log_final_statistics(best_epoch, save_path)
            
            print(f"\n📝 完整输出日志已保存到: {self.full_output_log}")
        
        # 恢复最佳模型
        model.load_state_dict(self.best_model_state)
        
        # 保存模型
        torch.save(self.best_model_state, save_path)
        
        return model
    def plot_training_history(self, save_path: str = "training_history.png"):
        """绘制训练历史，包含时间信息"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # 损失曲线
        ax1.plot(epochs, self.train_losses, 'b-', label='train loss', linewidth=2)
        ax1.set_title('train loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('loss')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 验证准确率曲线
        ax2.plot(epochs, self.val_accuracies, 'r-', label='val acc', linewidth=2)
        best_epoch = np.argmax(self.val_accuracies) + 1
        best_acc = np.max(self.val_accuracies)
        ax2.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'best: Epoch {best_epoch}')
        ax2.set_title(f'val acc (best: {best_acc:.4f})', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('acc')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Epoch时间曲线
        if len(self.epoch_times) > 0:
            ax3.plot(epochs[:len(self.epoch_times)], self.epoch_times, 'g-', label='Epoch_time', linewidth=2)
            avg_time = np.mean(self.epoch_times)
            ax3.axhline(y=avg_time, color='orange', linestyle='--', alpha=0.7, label=f'平均: {avg_time:.2f}s')
            ax3.set_title(f'train_time/epoch (平均: {avg_time:.2f}s)', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('time (s)')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        
        # 累积时间曲线
        if len(self.epoch_times) > 0:
            cumulative_times = np.cumsum(self.epoch_times)
            ax4.plot(epochs[:len(cumulative_times)], cumulative_times / 60, 'purple', linewidth=2, label='累积时间')
            total_time = cumulative_times[-1]
            ax4.set_title(f'cummlative time (总计: {self._format_time(total_time)})', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('cummlative time (min)')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        print(f"📊 训练历史图表已保存到: {save_path}")
          # 同时保存详细的时间统计到JSON
        time_stats = {
            "total_epochs": len(self.epoch_times),
            "total_training_time_seconds": float(sum(self.epoch_times)) if self.epoch_times else 0.0,
            "average_epoch_time_seconds": float(np.mean(self.epoch_times)) if self.epoch_times else 0.0,
            "min_epoch_time_seconds": float(np.min(self.epoch_times)) if self.epoch_times else 0.0,
            "max_epoch_time_seconds": float(np.max(self.epoch_times)) if self.epoch_times else 0.0,
            "epoch_times": [float(t) for t in self.epoch_times],
            "best_validation_accuracy": float(self.best_val_accuracy) if self.best_val_accuracy is not None else 0.0,
            "best_epoch": int(np.argmax(self.val_accuracies) + 1) if self.val_accuracies else 0
        }
        
        time_stats_path = save_path.replace('.png', '_time_stats.json')
        with open(time_stats_path, 'w', encoding='utf-8') as f:
            json.dump(time_stats, f, indent=2, ensure_ascii=False)
        
        print(f"⏱️  时间统计已保存到: {time_stats_path}")
    
    def _format_time(self, seconds: float) -> str:
        """格式化时间显示"""
        if seconds < 60:
            return f"{seconds:.1f}秒"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}分钟"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}小时"
    
    def _log_training_statistics(self, current_epoch: int):
        """记录训练统计信息"""
        if len(self.epoch_times) > 0:
            avg_epoch_time = np.mean(self.epoch_times)
            min_epoch_time = np.min(self.epoch_times)
            max_epoch_time = np.max(self.epoch_times)
            
            print(f"📊 前{current_epoch}轮统计:")
            print(f"  ⏱️  平均Epoch时间: {avg_epoch_time:.2f}s")
            print(f"  ⏱️  最快Epoch时间: {min_epoch_time:.2f}s")
            print(f"  ⏱️  最慢Epoch时间: {max_epoch_time:.2f}s")
            
        if len(self.val_accuracies) > 0:
            max_val_acc = np.max(self.val_accuracies)
            current_val_acc = self.val_accuracies[-1]
            print(f"  🎯 当前验证准确率: {current_val_acc:.4f}")
            print(f"  🏆 最佳验证准确率: {max_val_acc:.4f}")
    
    def _log_final_statistics(self, best_epoch: int, save_path: str):
        """记录最终训练统计"""
        total_training_time = time.time() - self.training_start_time
        
        print("=" * 60)
        print("🎉 训练完成！")
        print("=" * 60)
        print(f"📈 训练结果总结:")
        print(f"  🏆 最佳验证准确率: {self.best_val_accuracy:.4f}")
        print(f"  🎯 最佳模型出现在: Epoch {best_epoch}")
        print(f"  ⏱️  总训练时间: {self._format_time(total_training_time)}")
        
        if len(self.epoch_times) > 0:
            avg_epoch_time = np.mean(self.epoch_times)
            total_epoch_time = sum(self.epoch_times)
            print(f"  ⏱️  平均Epoch时间: {avg_epoch_time:.2f}s")
            print(f"  ⏱️  纯训练时间: {self._format_time(total_epoch_time)}")
            print(f"  📊 总Epoch数: {len(self.epoch_times)}")
            
        print(f"  💾 模型已保存到: {save_path}")
        print(f"  📝 完整日志已保存到: {self.full_output_log}")
        print("=" * 60)


# 测试代码
if __name__ == "__main__":
    print("=== 训练流程测试 ===")
    
    try:
        # 配置
        config = HyperedgePredictionConfig()
        config.epochs = 5  # 测试用少量epoch
        config.batch_size = 8
        
        # 数据路径
        train_csv = os.path.join(DEFAULT_DATA_DIR, "hyperedges_7days_reformatted_train.csv")
        val_csv = os.path.join(DEFAULT_DATA_DIR, "hyperedges_7days_reformatted_val.csv")
        test_csv = os.path.join(DEFAULT_DATA_DIR, "hyperedges_7days_reformatted_test.csv")
        size_sampler_path = os.path.join(DEFAULT_DATA_DIR, "edge_size_sampler.pkl")
        
        # 检查文件是否存在
        required_files = [train_csv, val_csv, test_csv, size_sampler_path]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            print(f"❌ 缺少必要文件: {missing_files}")
            print("请确保数据预处理管道已运行并生成了所需文件")
        else:
            print("✓ 所有必要文件都存在，开始测试...")
            
            # 创建训练器
            trainer = HyperedgeTrainer(config)
            
            # 这里只是框架测试，实际数据加载需要真实文件
            print("🎉 训练流程框架测试完成!")
            print("在真实数据上运行时，将执行完整的训练流程")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()



class SubgraphHyperedgeTrainer(HyperedgeTrainer):
    """
    基于子图采样的超边预测模型训练器
    实现"大小子图循环采样训练策略"
    """
    
    def __init__(self, config: HyperedgePredictionConfig, 
                 T: int = 100, R: int = 16, m: int = 300, L: int = 5, 
                 sampling_strategy: str = "TIHS"):
        """
        初始化子图训练器
        
        Args:
            config: 模型配置
            T: 大循环轮数
            R: 每轮大循环中子图数量
            m: 每个子图的最大节点数
            L: 每个子图训练的小循环次数
            sampling_strategy: 子图采样策略 ("TIHS" 或 "snowball")
        """
        super().__init__(config)
        
        # 子图训练超参数
        self.T = T  # 大循环轮数
        self.R = R  # 每轮子图数量
        self.m = m  # 子图最大节点数
        self.L = L  # 子图训练小循环次数
        self.sampling_strategy = sampling_strategy
        
        # 子图训练历史记录
        self.big_cycle_times = []  # 每个大循环的耗时
        self.subgraph_train_times = []  # 所有子图训练的耗时
        self.model_fusion_times = []  # 模型融合的耗时
        self.big_cycle_val_accuracies = []  # 每个大循环后的验证准确率
        
        print(f"✓ 子图训练器初始化完成")
        print(f"  - 大循环轮数 (T): {self.T}")
        print(f"  - 每轮子图数量 (R): {self.R}")
        print(f"  - 子图最大节点数 (m): {self.m}")
        print(f"  - 子图训练小循环次数 (L): {self.L}")
        print(f"  - 采样策略: {self.sampling_strategy}")
    
    def train_on_subgraph(self, model: HyperedgePredictionModel, subgraph_data: Dict, 
                         subgraph_positives: List, subgraph_negatives: List) -> HyperedgePredictionModel:
        """
        在子图上训练模型 L 次小循环
        
        Args:
            model: 主模型（用于获取权重和配置）
            subgraph_data: 子图超边数据 {eid: [node_list]}
            subgraph_positives: 子图正样本超边
            subgraph_negatives: 子图负样本超边
            
        Returns:
            训练后的模型
        """
        print(f"    🔧 开始子图训练 (L={self.L})")
        
        # 创建专门用于子图的模型实例
        subgraph_model = HyperedgePredictionModel(
            num_nodes=model.hypergcn.V,  # 保持节点总数不变
            hypergraph=subgraph_data,    # 使用子图的超边结构
            node_features=model.hypergcn.node_features.clone(),  # 复制节点特征
            embedding_dim=self.config.embedding_dim,
            hidden_dims=self.config.mlp_hidden_dims,
            hypergcn_args=None
        ).to(self.device)
        
        # 复制主模型的权重到子图模型
        subgraph_model.load_state_dict(model.state_dict())
        
        subgraph_model.train()
        optimizer = torch.optim.Adam(subgraph_model.parameters(), 
                                   lr=self.config.learning_rate, 
                                   weight_decay=self.config.weight_decay)
        
        # 组合子图的正负样本
        all_hyperedges = subgraph_positives + subgraph_negatives
        all_labels = [1] * len(subgraph_positives) + [0] * len(subgraph_negatives)
        all_labels = torch.tensor(all_labels, dtype=torch.long).to(self.device)
        
        print(f"      - 子图样本: {len(subgraph_positives)} 正 + {len(subgraph_negatives)} 负")
        
        # L次小循环训练
        for l in range(self.L):
            optimizer.zero_grad()
            
            # 在子图上前向传播
            logits = subgraph_model(all_hyperedges)
            loss = F.cross_entropy(logits, all_labels)
            
            # 反向传播更新参数
            loss.backward()
            optimizer.step()
            
            if l == 0 or (l + 1) % max(1, self.L // 2) == 0:
                print(f"        小循环 {l+1}/{self.L}: 损失 = {loss.item():.4f}")
        
        return subgraph_model
    
    def extract_subgraph_samples(self, subgraph_data: Dict, all_positives: List, all_negatives: List) -> Tuple[List, List]:
        """
        从全图样本中提取属于子图的样本
        
        Args:
            subgraph_data: 子图超边数据 {eid: [node_list]}
            all_positives: 全图正样本
            all_negatives: 全图负样本
            
        Returns:
            (子图正样本, 子图负样本)
        """
        subgraph_edges_set = set()
        for eid, nodes in subgraph_data.items():
            subgraph_edges_set.add(tuple(sorted(nodes)))
        
        # 提取属于子图的正样本
        subgraph_positives = []
        for edge in all_positives:
            if tuple(sorted(edge)) in subgraph_edges_set:
                subgraph_positives.append(edge)
        
        # 为子图生成负样本（简化：直接从全图负样本中筛选）
        subgraph_negatives = []
        for edge in all_negatives:
            # 检查负样本的节点是否都在子图中
            subgraph_nodes = set()
            for nodes in subgraph_data.values():
                subgraph_nodes.update(nodes)
            
            if all(node in subgraph_nodes for node in edge):
                subgraph_negatives.append(edge)
                
            # 限制负样本数量，保持正负比例大致平衡
            if len(subgraph_negatives) >= len(subgraph_positives):
                break
        
        return subgraph_positives, subgraph_negatives
    
    def fuse_model_weights(self, main_model: HyperedgePredictionModel, 
                          subgraph_models: List[HyperedgePredictionModel]) -> HyperedgePredictionModel:
        """
        融合多个子图训练后的模型权重（平均聚合）
        
        Args:
            main_model: 主模型
            subgraph_models: 子图训练后的模型列表
            
        Returns:
            融合后的模型
        """
        print(f"    🔄 融合 {len(subgraph_models)} 个子图模型权重")
        
        # 获取主模型的状态字典
        fused_state_dict = main_model.state_dict()
        
        # 对每个参数进行平均
        for param_name in fused_state_dict.keys():
            # 收集所有子图模型中该参数的值
            param_values = []
            for model in subgraph_models:
                param_values.append(model.state_dict()[param_name].clone())
            
            # 计算平均值
            if param_values:
                avg_param = torch.stack(param_values).mean(dim=0)
                fused_state_dict[param_name] = avg_param
        
        # 加载融合后的权重
        main_model.load_state_dict(fused_state_dict)
        
        return main_model
    
    def train(self, model: HyperedgePredictionModel, val_data: Tuple, 
              save_path: str = "best_subgraph_model.pth") -> HyperedgePredictionModel:
        """
        基于子图的完整训练流程
        
        Args:
            model: 模型
            val_data: 验证数据
            save_path: 模型保存路径
            
        Returns:
            训练好的模型
        """
        self.training_start_time = time.time()
        
        # 🎯 开启完整输出日志记录
        with dual_output(self.full_output_log):
            print("=" * 60)
            print("🔥 开始子图采样训练")
            print("=" * 60)
            print(f"子图训练配置:")
            print(f"  - 大循环轮数 (T): {self.T}")
            print(f"  - 每轮子图数量 (R): {self.R}")
            print(f"  - 子图最大节点数 (m): {self.m}")
            print(f"  - 子图训练小循环次数 (L): {self.L}")
            print(f"  - 采样策略: {self.sampling_strategy}")
            print(f"  - 训练样本数: {len(self.train_positives)} 正样本")
            print(f"  - 验证样本数: {len(val_data[0])}")
            print(f"  - 设备: {self.device}")
            
            # 构建完整的训练超图（用于子图采样）
            train_hypergraph = {i: hyperedge for i, hyperedge in enumerate(self.train_positives)}
            
            # 准备负样本（使用全图负样本）
            if not self.train_negatives:
                print("🔄 生成全图负样本用于子图训练...")
                self.train_negatives = self.negative_sampler.sample_negative_hyperedges(len(self.train_positives))
            
            best_big_cycle = 0
            
            # T轮大循环
            for t in tqdm(range(self.T), desc="大循环进度"):
                big_cycle_start_time = time.time()
                
                print(f"\n{'='*60}")
                print(f"🚀 开始第 {t+1}/{self.T} 轮大循环")
                print(f"{'='*60}")
                
                # R个子图的模型权重收集
                subgraph_models = []
                subgraph_times = []
                
                # 采样R个子图
                print(f"🎲 采样 {self.R} 个子图 (策略: {self.sampling_strategy}, 最大节点: {self.m})")
                
                for r in range(self.R):
                    subgraph_start_time = time.time()
                    
                    print(f"  📊 子图 {r+1}/{self.R}")
                    
                    # 采样子图
                    subgraph_data = sample_subgraph(train_hypergraph, self.m, self.sampling_strategy)
                    
                    if not subgraph_data:
                        print(f"    ⚠️ 子图 {r+1} 采样失败，跳过")
                        continue
                    
                    # 统计子图信息
                    subgraph_nodes = set()
                    for nodes in subgraph_data.values():
                        subgraph_nodes.update(nodes)
                    
                    print(f"    - 子图规模: {len(subgraph_data)} 个超边, {len(subgraph_nodes)} 个节点")
                    
                    # 提取子图样本
                    subgraph_positives, subgraph_negatives = self.extract_subgraph_samples(
                        subgraph_data, self.train_positives, self.train_negatives)
                    
                    if not subgraph_positives:
                        print(f"    ⚠️ 子图 {r+1} 无有效正样本，跳过")
                        continue
                    
                    # 在子图上训练（返回训练后的子图模型）
                    subgraph_model = self.train_on_subgraph(
                        model, subgraph_data, subgraph_positives, subgraph_negatives)
                    
                    subgraph_models.append(subgraph_model)
                    
                    subgraph_time = time.time() - subgraph_start_time
                    subgraph_times.append(subgraph_time)
                    print(f"    ✓ 子图 {r+1} 训练完成: {subgraph_time:.2f}s")
                
                # 融合所有子图模型权重
                if subgraph_models:
                    fusion_start_time = time.time()
                    print(f"🔄 融合 {len(subgraph_models)} 个子图模型...")
                    
                    model = self.fuse_model_weights(model, subgraph_models)
                    
                    fusion_time = time.time() - fusion_start_time
                    self.model_fusion_times.append(fusion_time)
                    print(f"✓ 模型融合完成: {fusion_time:.2f}s")
                else:
                    print("⚠️ 没有有效的子图模型用于融合")
                
                # 在验证集上评估融合后的模型
                val_start_time = time.time()
                val_accuracy = self.evaluate(model, val_data)
                val_time = time.time() - val_start_time
                
                # 记录历史
                self.big_cycle_val_accuracies.append(val_accuracy)
                self.subgraph_train_times.extend(subgraph_times)
                
                # 保存最佳模型
                if val_accuracy > self.best_val_accuracy:
                    self.best_val_accuracy = val_accuracy
                    self.best_model_state = model.state_dict().copy()
                    best_big_cycle = t + 1
                    print(f"🏆 发现新的最佳模型！验证准确率: {val_accuracy:.4f}")
                
                big_cycle_time = time.time() - big_cycle_start_time
                self.big_cycle_times.append(big_cycle_time)
                
                # 详细日志记录
                elapsed_time = time.time() - self.training_start_time
                avg_big_cycle_time = elapsed_time / (t + 1)
                remaining_cycles = self.T - (t + 1)
                estimated_remaining_time = avg_big_cycle_time * remaining_cycles
                
                # 格式化时间显示
                elapsed_str = self._format_time(elapsed_time)
                remaining_str = self._format_time(estimated_remaining_time)
                
                print(f"\n📊 第 {t+1} 轮大循环结果:")
                print(f"  🎯 验证准确率: {val_accuracy:.4f} {'✨ (历史最佳!)' if val_accuracy == self.best_val_accuracy else ''}")
                print(f"  ⏱️ 验证耗时: {val_time:.2f}s")
                print(f"  ⏱️ 大循环总耗时: {big_cycle_time:.2f}s")
                print(f"  ⏱️ 子图训练平均耗时: {np.mean(subgraph_times):.2f}s" if subgraph_times else "  ⏱️ 子图训练平均耗时: N/A")
                print(f"  🕐 累计训练时间: {elapsed_str}")
                print(f"  🕑 预计剩余时间: {remaining_str}")
                print(f"  📈 平均每轮大循环时间: {avg_big_cycle_time:.2f}s")
                
                # 每10个大循环打印详细统计
                if (t + 1) % 10 == 0:
                    self._log_subgraph_training_statistics(t + 1)
                    
                print("-" * 60)
            
            # 恢复最佳模型
            model.load_state_dict(self.best_model_state)
            
            # 保存模型
            torch.save(self.best_model_state, save_path)
            
            # 记录最终统计
            self._log_final_subgraph_statistics(best_big_cycle, save_path)
        
        return model
    
    def _log_subgraph_training_statistics(self, current_cycle: int):
        """记录子图训练统计信息"""
        if len(self.big_cycle_times) > 0:
            avg_big_cycle_time = np.mean(self.big_cycle_times)
            min_big_cycle_time = np.min(self.big_cycle_times)
            max_big_cycle_time = np.max(self.big_cycle_times)
            
            print(f"📊 前{current_cycle}轮大循环统计:")
            print(f"  ⏱️ 平均大循环时间: {avg_big_cycle_time:.2f}s")
            print(f"  ⏱️ 最快大循环时间: {min_big_cycle_time:.2f}s")
            print(f"  ⏱️ 最慢大循环时间: {max_big_cycle_time:.2f}s")
            
        if len(self.subgraph_train_times) > 0:
            avg_subgraph_time = np.mean(self.subgraph_train_times)
            print(f"  ⏱️ 平均子图训练时间: {avg_subgraph_time:.2f}s")
            
        if len(self.model_fusion_times) > 0:
            avg_fusion_time = np.mean(self.model_fusion_times)
            print(f"  ⏱️ 平均模型融合时间: {avg_fusion_time:.2f}s")
            
        if len(self.big_cycle_val_accuracies) > 0:
            max_val_acc = np.max(self.big_cycle_val_accuracies)
            current_val_acc = self.big_cycle_val_accuracies[-1]
            print(f"  🎯 当前验证准确率: {current_val_acc:.4f}")
            print(f"  🏆 最佳验证准确率: {max_val_acc:.4f}")
    
    def _log_final_subgraph_statistics(self, best_cycle: int, save_path: str):
        """记录最终子图训练统计"""
        total_training_time = time.time() - self.training_start_time
        
        print("=" * 60)
        print("🎉 子图训练完成！")
        print("=" * 60)
        print(f"📈 子图训练结果总结:")
        print(f"  🏆 最佳验证准确率: {self.best_val_accuracy:.4f}")
        print(f"  🎯 最佳模型出现在: 第{best_cycle}轮大循环")
        print(f"  ⏱️ 总训练时间: {self._format_time(total_training_time)}")
        
        if len(self.big_cycle_times) > 0:
            avg_big_cycle_time = np.mean(self.big_cycle_times)
            total_big_cycle_time = sum(self.big_cycle_times)
            print(f"  ⏱️ 平均大循环时间: {avg_big_cycle_time:.2f}s")
            print(f"  ⏱️ 总大循环时间: {self._format_time(total_big_cycle_time)}")
            
        if len(self.subgraph_train_times) > 0:
            avg_subgraph_time = np.mean(self.subgraph_train_times)
            total_subgraph_time = sum(self.subgraph_train_times)
            print(f"  ⏱️ 平均子图训练时间: {avg_subgraph_time:.2f}s")
            print(f"  ⏱️ 总子图训练时间: {self._format_time(total_subgraph_time)}")
            
        if len(self.model_fusion_times) > 0:
            avg_fusion_time = np.mean(self.model_fusion_times)
            total_fusion_time = sum(self.model_fusion_times)
            print(f"  ⏱️ 平均模型融合时间: {avg_fusion_time:.2f}s")
            print(f"  ⏱️ 总模型融合时间: {self._format_time(total_fusion_time)}")
        
        print(f"  💾 模型已保存到: {save_path}")
        print("=" * 60)
    
    def plot_subgraph_training_history(self, save_path: str = "subgraph_training_history.png"):
        """绘制子图训练历史，包含时间信息"""
        if not self.big_cycle_val_accuracies:
            print("⚠️ 没有训练历史数据可绘制")
            return
            
        try:
            import matplotlib.pyplot as plt
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # 绘制验证准确率历史
            cycles = range(1, len(self.big_cycle_val_accuracies) + 1)
            ax1.plot(cycles, self.big_cycle_val_accuracies, 'b-', marker='o', linewidth=2, markersize=4)
            ax1.set_xlabel('大循环轮数')
            ax1.set_ylabel('验证准确率')
            ax1.set_title('子图训练验证准确率历史')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
            
            # 添加最佳点标记
            if self.big_cycle_val_accuracies:
                best_cycle = np.argmax(self.big_cycle_val_accuracies) + 1
                best_acc = max(self.big_cycle_val_accuracies)
                ax1.scatter([best_cycle], [best_acc], color='red', s=100, zorder=5)
                ax1.annotate(f'最佳: {best_acc:.4f}', 
                           xy=(best_cycle, best_acc), 
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            # 绘制大循环时间历史
            if self.big_cycle_times:
                ax2.plot(cycles[:len(self.big_cycle_times)], self.big_cycle_times, 'g-', marker='s', linewidth=2, markersize=4)
                ax2.set_xlabel('大循环轮数')
                ax2.set_ylabel('时间 (秒)')
                ax2.set_title('大循环训练时间历史')
                ax2.grid(True, alpha=0.3)
                
                # 添加平均时间线
                avg_time = np.mean(self.big_cycle_times)
                ax2.axhline(y=avg_time, color='red', linestyle='--', alpha=0.7, 
                           label=f'平均: {avg_time:.2f}s')
                ax2.legend()
            
            # 绘制子图训练时间分布
            if self.subgraph_train_times:
                ax3.hist(self.subgraph_train_times, bins=20, alpha=0.7, color='purple', edgecolor='black')
                ax3.set_xlabel('时间 (秒)')
                ax3.set_ylabel('频次')
                ax3.set_title('子图训练时间分布')
                ax3.grid(True, alpha=0.3)
                
                # 添加统计信息
                mean_time = np.mean(self.subgraph_train_times)
                median_time = np.median(self.subgraph_train_times)
                ax3.axvline(mean_time, color='red', linestyle='--', label=f'均值: {mean_time:.2f}s')
                ax3.axvline(median_time, color='orange', linestyle='--', label=f'中位数: {median_time:.2f}s')
                ax3.legend()
            
            # 绘制模型融合时间历史
            if self.model_fusion_times:
                fusion_cycles = range(1, len(self.model_fusion_times) + 1)
                ax4.plot(fusion_cycles, self.model_fusion_times, 'orange', marker='^', linewidth=2, markersize=4)
                ax4.set_xlabel('大循环轮数')
                ax4.set_ylabel('时间 (秒)')
                ax4.set_title('模型融合时间历史')
                ax4.grid(True, alpha=0.3)
                
                # 添加平均融合时间线
                avg_fusion_time = np.mean(self.model_fusion_times)
                ax4.axhline(y=avg_fusion_time, color='red', linestyle='--', alpha=0.7,
                           label=f'平均: {avg_fusion_time:.2f}s')
                ax4.legend()
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📈 子图训练历史图表已保存到: {save_path}")
            
            # 保存详细的时间统计
            time_stats = {
                "big_cycle_times": [float(t) for t in self.big_cycle_times],
                "subgraph_train_times": [float(t) for t in self.subgraph_train_times],
                "model_fusion_times": [float(t) for t in self.model_fusion_times],
                "big_cycle_val_accuracies": [float(a) for a in self.big_cycle_val_accuracies],
                "best_validation_accuracy": float(self.best_val_accuracy) if self.best_val_accuracy is not None else 0.0,
                "best_big_cycle": int(np.argmax(self.big_cycle_val_accuracies) + 1) if self.big_cycle_val_accuracies else 0,
                "total_big_cycles": len(self.big_cycle_times),
                "total_subgraph_trainings": len(self.subgraph_train_times),
                "average_big_cycle_time": float(np.mean(self.big_cycle_times)) if self.big_cycle_times else 0.0,
                "average_subgraph_train_time": float(np.mean(self.subgraph_train_times)) if self.subgraph_train_times else 0.0,
                "average_model_fusion_time": float(np.mean(self.model_fusion_times)) if self.model_fusion_times else 0.0
            }
            
            time_stats_path = save_path.replace('.png', '_time_stats.json')
            with open(time_stats_path, 'w', encoding='utf-8') as f:
                json.dump(time_stats, f, indent=2, ensure_ascii=False)
            
            print(f"⏱️ 时间统计已保存到: {time_stats_path}")
            
        except ImportError:
            print("⚠️ matplotlib未安装，无法绘制图表")
        except Exception as e:
            print(f"⚠️ 绘制图表时出错: {e}")

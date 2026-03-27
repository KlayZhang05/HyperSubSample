#!/usr/bin/env python3
"""
Optional parallel subgraph trainer.

Parallel execution is only enabled on CPU. CUDA mode falls back to serial
execution to avoid multiple subgraph models contending for a single device.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datetime import timedelta
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from end_to_end_model import HyperedgePredictionModel, create_mock_hypergcn_args
from training_pipeline import SubgraphHyperedgeTrainer, dual_output
from subgraph_sampler import sample_subgraph


class ParallelSubgraphHyperedgeTrainer(SubgraphHyperedgeTrainer):
    def __init__(
        self,
        config,
        T: int = 100,
        R: int = 16,
        m: int = 300,
        L: int = 5,
        sampling_strategy: str = "TIHS",
        parallel_subgraphs: int = 1,
    ):
        super().__init__(config, T=T, R=R, m=m, L=L, sampling_strategy=sampling_strategy)
        self.parallel_subgraphs = max(1, parallel_subgraphs)

    def train_on_subgraph(
        self,
        model: HyperedgePredictionModel,
        subgraph_data: Dict,
        subgraph_positives: List,
        subgraph_negatives: List,
    ) -> HyperedgePredictionModel:
        subgraph_model = HyperedgePredictionModel(
            num_nodes=model.num_nodes,
            hypergraph=subgraph_data,
            node_features=model.hypergcn.node_features.clone(),
            embedding_dim=self.config.embedding_dim,
            hidden_dims=self.config.mlp_hidden_dims,
            hypergcn_args=create_mock_hypergcn_args(
                self.config, model.hypergcn.node_features.shape[1]
            ),
        ).to(self.device)
        subgraph_model.load_state_dict(model.state_dict())
        subgraph_model.train()

        optimizer = torch.optim.Adam(
            subgraph_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        all_hyperedges = subgraph_positives + subgraph_negatives
        all_labels = [1] * len(subgraph_positives) + [0] * len(subgraph_negatives)
        all_labels = torch.tensor(all_labels, dtype=torch.long).to(self.device)

        for _ in range(self.L):
            optimizer.zero_grad()
            logits = subgraph_model(all_hyperedges)
            loss = F.cross_entropy(logits, all_labels)
            loss.backward()
            optimizer.step()

        return subgraph_model

    def _train_single_subgraph_job(
        self, model: HyperedgePredictionModel, train_hypergraph: Dict, index: int
    ):
        start_time = time.time()
        subgraph_data = sample_subgraph(train_hypergraph, self.m, self.sampling_strategy)
        if not subgraph_data:
            return None

        subgraph_nodes = set()
        for nodes in subgraph_data.values():
            subgraph_nodes.update(nodes)

        subgraph_positives, subgraph_negatives = self.extract_subgraph_samples(
            subgraph_data, self.train_positives, self.train_negatives
        )
        if not subgraph_positives:
            return None

        trained_model = self.train_on_subgraph(
            model, subgraph_data, subgraph_positives, subgraph_negatives
        )

        return {
            "index": index,
            "model": trained_model,
            "time": time.time() - start_time,
            "num_edges": len(subgraph_data),
            "num_nodes": len(subgraph_nodes),
            "num_positives": len(subgraph_positives),
            "num_negatives": len(subgraph_negatives),
        }

    def _run_subgraph_batch(self, model: HyperedgePredictionModel, train_hypergraph: Dict):
        run_parallel = self.parallel_subgraphs > 1 and self.device.type == "cpu"
        if self.parallel_subgraphs > 1 and self.device.type != "cpu":
            print("当前并行子图训练仅在 CPU 模式下启用，CUDA 模式将回退为串行执行")

        results = []
        if run_parallel:
            print(f"使用 CPU 并行子图训练，worker 数量: {self.parallel_subgraphs}")
            with ThreadPoolExecutor(max_workers=self.parallel_subgraphs) as executor:
                futures = [
                    executor.submit(self._train_single_subgraph_job, model, train_hypergraph, idx)
                    for idx in range(self.R)
                ]
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        results.append(result)
        else:
            for idx in range(self.R):
                result = self._train_single_subgraph_job(model, train_hypergraph, idx)
                if result is not None:
                    results.append(result)

        results.sort(key=lambda item: item["index"])
        return results

    def _format_time(self, seconds: float) -> str:
        return str(timedelta(seconds=max(0, int(seconds))))

    def train(
        self, model: HyperedgePredictionModel, val_data: Tuple, save_path: str = "best_subgraph_model.pth"
    ) -> HyperedgePredictionModel:
        self.training_start_time = time.time()

        with dual_output(self.full_output_log):
            print("=" * 60)
            print("开始子图采样训练")
            print("=" * 60)
            print("子图训练配置:")
            print(f"  - 大循环轮数 (T): {self.T}")
            print(f"  - 每轮子图数量 (R): {self.R}")
            print(f"  - 子图最大节点数 (m): {self.m}")
            print(f"  - 子图训练小循环次数 (L): {self.L}")
            print(f"  - 采样策略: {self.sampling_strategy}")
            print(f"  - 并行子图数量: {self.parallel_subgraphs}")
            print(f"  - 训练样本数: {len(self.train_positives)} 正样本")
            print(f"  - 验证样本数: {len(val_data[0])}")
            print(f"  - 设备: {self.device}")

            train_hypergraph = {i: hyperedge for i, hyperedge in enumerate(self.train_positives)}
            if not self.train_negatives:
                print("生成全图负样本用于子图训练...")
                self.train_negatives = self.negative_sampler.sample_negative_hyperedges(
                    len(self.train_positives)
                )

            best_big_cycle = 0

            for t in tqdm(range(self.T), desc="大循环进度"):
                big_cycle_start_time = time.time()
                print(f"\n{'=' * 60}")
                print(f"开始第 {t + 1}/{self.T} 轮大循环")
                print(f"{'=' * 60}")

                subgraph_results = self._run_subgraph_batch(model, train_hypergraph)
                subgraph_models = [item["model"] for item in subgraph_results]
                subgraph_times = [item["time"] for item in subgraph_results]

                if subgraph_models:
                    fusion_start_time = time.time()
                    model = self.fuse_model_weights(model, subgraph_models)
                    fusion_time = time.time() - fusion_start_time
                    self.model_fusion_times.append(fusion_time)
                    print(f"模型融合完成: {fusion_time:.2f}s")
                else:
                    print("没有有效的子图模型用于融合")

                val_start_time = time.time()
                val_accuracy = self.evaluate(model, val_data)
                val_time = time.time() - val_start_time

                self.big_cycle_val_accuracies.append(val_accuracy)
                self.subgraph_train_times.extend(subgraph_times)

                if val_accuracy > self.best_val_accuracy:
                    self.best_val_accuracy = val_accuracy
                    self.best_model_state = {
                        key: value.detach().clone() for key, value in model.state_dict().items()
                    }
                    best_big_cycle = t + 1
                    print(f"发现新的最佳模型，验证准确率: {val_accuracy:.4f}")

                big_cycle_time = time.time() - big_cycle_start_time
                self.big_cycle_times.append(big_cycle_time)

                elapsed_time = time.time() - self.training_start_time
                avg_big_cycle_time = elapsed_time / (t + 1)
                remaining_time = avg_big_cycle_time * (self.T - t - 1)

                print(f"\n第 {t + 1} 轮大循环结果:")
                print(
                    f"  - 验证准确率: {val_accuracy:.4f}"
                    f"{' (历史最佳)' if val_accuracy == self.best_val_accuracy else ''}"
                )
                print(f"  - 验证耗时: {val_time:.2f}s")
                print(f"  - 大循环总耗时: {big_cycle_time:.2f}s")
                print(
                    f"  - 子图训练平均耗时: {np.mean(subgraph_times):.2f}s"
                    if subgraph_times
                    else "  - 子图训练平均耗时: N/A"
                )
                print(f"  - 累计训练时间: {self._format_time(elapsed_time)}")
                print(f"  - 预计剩余时间: {self._format_time(remaining_time)}")
                print(f"  - 平均每轮大循环时间: {avg_big_cycle_time:.2f}s")

                if (t + 1) % 10 == 0:
                    self._log_subgraph_training_statistics(t + 1)
                print("-" * 60)

            if self.best_model_state is not None:
                model.load_state_dict(self.best_model_state)
                torch.save(self.best_model_state, save_path)

            self._log_final_subgraph_statistics(best_big_cycle, save_path)

        return model

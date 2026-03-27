#!/usr/bin/env python3
"""
Hyperedge prediction entry point.
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch

from parallel_subgraph_trainer import ParallelSubgraphHyperedgeTrainer
from training_pipeline import HyperedgePredictionConfig, HyperedgeTrainer, SubgraphHyperedgeTrainer

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available on this machine.")
        return torch.device("cuda")
    return torch.device("cpu")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="超边预测系统")
    parser.add_argument("--train_csv", type=str, default=os.path.join(DEFAULT_DATA_DIR, "hyperedges_7days_reformatted_train.csv"))
    parser.add_argument("--val_csv", type=str, default=os.path.join(DEFAULT_DATA_DIR, "hyperedges_7days_reformatted_val.csv"))
    parser.add_argument("--test_csv", type=str, default=os.path.join(DEFAULT_DATA_DIR, "hyperedges_7days_reformatted_test.csv"))
    parser.add_argument("--size_sampler", type=str, default=os.path.join(DEFAULT_DATA_DIR, "edge_size_sampler.pkl"))
    parser.add_argument("--num_users", type=int, default=82740)
    parser.add_argument("--num_products", type=int, default=38830)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--mode", choices=["train", "test", "both"], default="both")
    parser.add_argument("--training_strategy", choices=["full_graph", "subgraph"], default="full_graph")
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--R", type=int, default=16)
    parser.add_argument("--m", type=int, default=300)
    parser.add_argument("--L", type=int, default=5)
    parser.add_argument("--sampling_strategy", choices=["TIHS", "snowball"], default="TIHS")
    parser.add_argument("--parallel_subgraphs", type=int, default=1)
    return parser


def create_config(args: argparse.Namespace) -> HyperedgePredictionConfig:
    config = HyperedgePredictionConfig()
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.embedding_dim = args.embedding_dim
    config.device = resolve_device(args.device)
    return config


def create_trainer(args: argparse.Namespace, config: HyperedgePredictionConfig):
    if args.training_strategy == "full_graph":
        return HyperedgeTrainer(config)

    if args.parallel_subgraphs > 1:
        return ParallelSubgraphHyperedgeTrainer(
            config=config,
            T=args.T,
            R=args.R,
            m=args.m,
            L=args.L,
            sampling_strategy=args.sampling_strategy,
            parallel_subgraphs=args.parallel_subgraphs,
        )

    return SubgraphHyperedgeTrainer(
        config=config,
        T=args.T,
        R=args.R,
        m=args.m,
        L=args.L,
        sampling_strategy=args.sampling_strategy,
    )


def main():
    parser = build_parser()
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    required_files = [args.train_csv, args.val_csv, args.test_csv, args.size_sampler]
    missing = [path for path in required_files if not os.path.exists(path)]
    if missing:
        print(f"缺少必要文件: {missing}")
        return

    config = create_config(args)
    trainer = create_trainer(args, config)

    print("=" * 60)
    print("超边预测系统启动")
    print(f"训练策略: {args.training_strategy}")
    print(f"运行设备: {config.device}")
    if args.training_strategy == "subgraph":
        print(f"并行子图数量: {args.parallel_subgraphs}")
    print("=" * 60)

    try:
        if args.mode in ["train", "both"]:
            val_data, test_data, train_hypergraph = trainer.load_data(
                train_csv=args.train_csv,
                val_csv=args.val_csv,
                test_csv=args.test_csv,
                size_sampler_path=args.size_sampler,
                num_users=args.num_users,
                num_products=args.num_products,
            )

            num_nodes = args.num_users + args.num_products
            model = trainer.create_model(num_nodes, train_hypergraph)
            model_save_path = os.path.join(
                args.output_dir,
                f"best_{'subgraph' if args.training_strategy == 'subgraph' else 'full_graph'}_model.pth",
            )
            model = trainer.train(model, val_data, model_save_path)

            if args.training_strategy == "full_graph":
                trainer.plot_training_history(os.path.join(args.output_dir, "training_history.png"))
            elif hasattr(trainer, "plot_subgraph_training_history"):
                trainer.plot_subgraph_training_history(
                    os.path.join(args.output_dir, "subgraph_training_history.png")
                )

            config_payload = {
                "embedding_dim": config.embedding_dim,
                "mlp_hidden_dims": config.mlp_hidden_dims,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "epochs": config.epochs,
                "num_users": args.num_users,
                "num_products": args.num_products,
                "device": str(config.device),
                "train_time": datetime.now().isoformat(),
                "training_strategy": args.training_strategy,
                "best_validation_accuracy": trainer.best_val_accuracy,
            }

            if args.training_strategy == "full_graph":
                config_payload.update(
                    {
                        "total_training_time_seconds": sum(trainer.epoch_times) if trainer.epoch_times else 0,
                        "average_epoch_time_seconds": np.mean(trainer.epoch_times) if trainer.epoch_times else 0,
                    }
                )
            else:
                config_payload.update(
                    {
                        "T": args.T,
                        "R": args.R,
                        "m": args.m,
                        "L": args.L,
                        "sampling_strategy": args.sampling_strategy,
                        "parallel_subgraphs": args.parallel_subgraphs,
                        "total_training_time_seconds": sum(trainer.big_cycle_times) if trainer.big_cycle_times else 0,
                        "average_major_cycle_time_seconds": np.mean(trainer.big_cycle_times) if trainer.big_cycle_times else 0,
                        "average_subgraph_time_seconds": np.mean(trainer.subgraph_train_times) if trainer.subgraph_train_times else 0,
                        "total_subgraphs_trained": len(trainer.subgraph_train_times),
                    }
                )

            with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(config_payload, f, indent=2)

        if args.mode in ["test", "both"]:
            if args.mode == "test":
                val_data, test_data, train_hypergraph = trainer.load_data(
                    train_csv=args.train_csv,
                    val_csv=args.val_csv,
                    test_csv=args.test_csv,
                    size_sampler_path=args.size_sampler,
                    num_users=args.num_users,
                    num_products=args.num_products,
                )

                num_nodes = args.num_users + args.num_products
                model = trainer.create_model(num_nodes, train_hypergraph)
                model_save_path = os.path.join(
                    args.output_dir,
                    f"best_{'subgraph' if args.training_strategy == 'subgraph' else 'full_graph'}_model.pth",
                )
                if not os.path.exists(model_save_path):
                    print(f"未找到训练好的模型: {model_save_path}")
                    return
                model.load_state_dict(torch.load(model_save_path, map_location=config.device))

            test_accuracy = trainer.evaluate(model, test_data)
            results = {
                "test_accuracy": test_accuracy,
                "test_time": datetime.now().isoformat(),
                "training_strategy": args.training_strategy,
            }
            if args.training_strategy == "subgraph":
                results.update(
                    {
                        "T": args.T,
                        "R": args.R,
                        "m": args.m,
                        "L": args.L,
                        "sampling_strategy": args.sampling_strategy,
                        "parallel_subgraphs": args.parallel_subgraphs,
                    }
                )
            if args.mode == "both":
                results["best_val_accuracy"] = trainer.best_val_accuracy

            with open(os.path.join(args.output_dir, "test_results.json"), "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

            print(f"测试集准确率: {test_accuracy:.4f}")

    except Exception as exc:
        print(f"运行失败: {exc}")
        raise


if __name__ == "__main__":
    main()

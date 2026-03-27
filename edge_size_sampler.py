#!/usr/bin/env python3
"""
Utilities for building, saving, loading, and sampling hyperedge-size distributions.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import pickle
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


def build_size_distribution(hyperedges, window_days: int = 7):
    """
    Build an empirical hyperedge-size distribution from hyperedge records.

    Each item in `hyperedges` is expected to follow the historical project format:
    `(user_id, window_id, spu_set)`.
    """
    size_list = [len(spu_set) for _, _, spu_set in hyperedges]
    if not size_list:
        raise ValueError("hyperedges must not be empty")

    size_distribution = Counter(size_list)
    total_hyperedges = len(size_list)
    size_probabilities = {
        size: count / total_hyperedges for size, count in size_distribution.items()
    }

    return size_distribution, size_probabilities, size_list


def sample_size_from_distribution(
    size_probabilities: dict[int, float], n_samples: int = 1, random_seed: int | None = None
):
    """Sample hyperedge sizes according to the empirical probability mass."""
    rng = np.random.default_rng(random_seed)
    sizes = list(size_probabilities.keys())
    probabilities = list(size_probabilities.values())
    sampled_sizes = rng.choice(sizes, size=n_samples, p=probabilities)
    return sampled_sizes.tolist()


def sample_size_uniform(size_list: Sequence[int], n_samples: int = 1, random_seed: int | None = None):
    """Sample hyperedge sizes uniformly from the observed size list."""
    rng = np.random.default_rng(random_seed)
    sampled_indices = rng.choice(len(size_list), size=n_samples, replace=True)
    return [size_list[i] for i in sampled_indices]


class HyperedgeSizeSampler:
    """Serializable wrapper around a hyperedge-size distribution."""

    def __init__(
        self,
        hyperedges=None,
        window_days: int = 7,
        size_distribution=None,
        size_probabilities=None,
        size_list=None,
        stats=None,
    ):
        self.window_days = window_days

        if hyperedges is not None:
            self._build_from_hyperedges(hyperedges)
        elif (
            size_distribution is not None
            and size_probabilities is not None
            and size_list is not None
            and stats is not None
        ):
            self._build_from_data(size_distribution, size_probabilities, size_list, stats)
        else:
            raise ValueError(
                "Provide either `hyperedges` or a complete serialized sampler payload."
            )

    def _build_from_hyperedges(self, hyperedges):
        size_dist, size_probs, size_list = build_size_distribution(hyperedges, self.window_days)
        self.size_distribution = size_dist
        self.size_probabilities = size_probs
        self.size_list = size_list
        self.stats = {
            "total_hyperedges": len(size_list),
            "min_size": min(size_list),
            "max_size": max(size_list),
            "mean_size": float(np.mean(size_list)),
            "median_size": float(np.median(size_list)),
            "std_size": float(np.std(size_list)),
        }

    def _build_from_data(self, size_distribution, size_probabilities, size_list, stats):
        self.size_distribution = Counter(size_distribution)
        self.size_probabilities = dict(size_probabilities)
        self.size_list = list(size_list)
        self.stats = dict(stats)

    def sample(
        self, n_samples: int = 1, method: str = "probability", random_seed: int | None = None
    ):
        if method == "probability":
            return sample_size_from_distribution(
                self.size_probabilities, n_samples=n_samples, random_seed=random_seed
            )
        if method == "uniform":
            return sample_size_uniform(
                self.size_list, n_samples=n_samples, random_seed=random_seed
            )
        raise ValueError("method must be 'probability' or 'uniform'")

    def __call__(self, n_samples: int = 1, method: str = "probability", random_seed: int | None = None):
        return self.sample(n_samples=n_samples, method=method, random_seed=random_seed)

    def get_stats(self):
        return self.stats.copy()

    def plot_distribution(self, figsize=(10, 6), save_path: str | None = None):
        sizes = sorted(self.size_distribution.keys())
        counts = [self.size_distribution[size] for size in sizes]
        probs = [self.size_probabilities[size] for size in sizes]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        ax1.bar(sizes, counts, alpha=0.7, color="skyblue")
        ax1.set_xlabel("Hyperedge size")
        ax1.set_ylabel("Count")
        ax1.set_title("Hyperedge size counts")
        ax1.grid(True, alpha=0.3)

        ax2.bar(sizes, probs, alpha=0.7, color="lightcoral")
        ax2.set_xlabel("Hyperedge size")
        ax2.set_ylabel("Probability")
        ax2.set_title("Hyperedge size probabilities")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()


def get_hyperedge_size_sampler(hyperedges, window_days: int = 7):
    return HyperedgeSizeSampler(hyperedges=hyperedges, window_days=window_days)


def save_hyperedge_size_sampler(sampler, save_path: str = "hyperedge_size_sampler.pkl"):
    """Persist sampler data as both pickle and human-readable JSON metadata."""
    sampler_data = {
        "size_distribution": dict(sampler.size_distribution),
        "size_probabilities": dict(sampler.size_probabilities),
        "size_list": list(sampler.size_list),
        "stats": dict(sampler.stats),
        "window_days": getattr(sampler, "window_days", 7),
    }

    save_path_obj = Path(save_path)
    with save_path_obj.open("wb") as f:
        pickle.dump(sampler_data, f)

    json_path = save_path_obj.with_name(f"{save_path_obj.stem}_info.json")
    json_payload = {
        "stats": sampler_data["stats"],
        "size_distribution": sampler_data["size_distribution"],
        "size_probabilities": sampler_data["size_probabilities"],
        "window_days": sampler_data["window_days"],
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)

    return str(save_path_obj), str(json_path)


def load_hyperedge_size_sampler(load_path: str = "hyperedge_size_sampler.pkl", return_class: bool = True):
    """Load a serialized hyperedge-size sampler."""
    load_path_obj = Path(load_path)
    if not load_path_obj.exists():
        raise FileNotFoundError(f"Sampler file not found: {load_path}")

    with load_path_obj.open("rb") as f:
        sampler_data = pickle.load(f)

    if return_class:
        return HyperedgeSizeSampler(
            window_days=sampler_data.get("window_days", 7),
            size_distribution=sampler_data["size_distribution"],
            size_probabilities=sampler_data["size_probabilities"],
            size_list=sampler_data["size_list"],
            stats=sampler_data["stats"],
        )

    def sampler(n_samples: int = 1, method: str = "probability", random_seed: int | None = None):
        if method == "probability":
            return sample_size_from_distribution(
                sampler_data["size_probabilities"], n_samples=n_samples, random_seed=random_seed
            )
        if method == "uniform":
            return sample_size_uniform(
                sampler_data["size_list"], n_samples=n_samples, random_seed=random_seed
            )
        raise ValueError("method must be 'probability' or 'uniform'")

    sampler.size_distribution = sampler_data["size_distribution"]
    sampler.size_probabilities = sampler_data["size_probabilities"]
    sampler.size_list = sampler_data["size_list"]
    sampler.stats = sampler_data["stats"]
    sampler.window_days = sampler_data.get("window_days", 7)
    return sampler


def create_and_save_sampler(hyperedges, window_days: int = 7, save_path: str = "hyperedge_size_sampler.pkl"):
    sampler = HyperedgeSizeSampler(hyperedges=hyperedges, window_days=window_days)
    save_hyperedge_size_sampler(sampler, save_path)
    return sampler


def quick_sample(
    sampler_path: str, n_samples: int = 1, method: str = "probability", random_seed: int | None = None
):
    sampler = load_hyperedge_size_sampler(sampler_path)
    return sampler.sample(n_samples=n_samples, method=method, random_seed=random_seed)


def list_saved_samplers(directory: str = "."):
    return list(Path(directory).glob("*sampler*.pkl"))


def generate_negative_hyperedge_with_sampled_size(
    sampler_path: str, user_nodes: Sequence[int], product_nodes: Sequence[int], random_seed: int | None = None
):
    rng = np.random.default_rng(random_seed)
    k = quick_sample(sampler_path, n_samples=1)[0]
    user = int(rng.choice(user_nodes, 1)[0])
    products = rng.choice(product_nodes, k - 1, replace=False).tolist()
    return [user] + products, k

#!/usr/bin/env python3
"""
Subgraph sampling utilities with an optional native C++ backend.
"""

from __future__ import annotations

import random
from collections import defaultdict, deque
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

try:
    import subgraph_sampler_native as _native_sampler
except ImportError:
    _native_sampler = None


HyperedgeDict = Dict[Any, List[int]]


def native_sampler_available() -> bool:
    return _native_sampler is not None


def _normalize_edge(edge: Sequence[int]) -> Tuple[int, ...]:
    return tuple(sorted(int(node) for node in edge))


def sample_subgraph_tihs_python(hyperedges_dict: HyperedgeDict, max_nodes: int) -> HyperedgeDict:
    selected_edges = set()
    selected_nodes = set()

    all_edges = list(hyperedges_dict.items())
    random.shuffle(all_edges)

    for eid, nodes in all_edges:
        selected_edges.add(eid)
        selected_nodes.update(nodes)

        for eid2, nodes2 in hyperedges_dict.items():
            if eid2 not in selected_edges and set(nodes2).issubset(selected_nodes):
                selected_edges.add(eid2)

        if len(selected_nodes) >= max_nodes:
            break

    return {eid: list(hyperedges_dict[eid]) for eid in selected_edges}


def sample_subgraph_snowball_python(hyperedges_dict: HyperedgeDict, max_nodes: int) -> HyperedgeDict:
    if not hyperedges_dict:
        return {}

    node2edges = defaultdict(set)
    for eid, nodes in hyperedges_dict.items():
        for node in nodes:
            node2edges[node].add(eid)

    if not node2edges:
        return {}

    start_node = random.choice(list(node2edges.keys()))
    visited_nodes = {start_node}
    selected_edges = set()
    queue = deque([start_node])

    while queue and len(visited_nodes) < max_nodes:
        current = queue.popleft()
        for eid in node2edges[current]:
            if eid in selected_edges:
                continue
            selected_edges.add(eid)
            for node in hyperedges_dict[eid]:
                if node not in visited_nodes:
                    visited_nodes.add(node)
                    queue.append(node)

        if len(visited_nodes) >= max_nodes:
            break

    return {eid: list(hyperedges_dict[eid]) for eid in selected_edges}


def sample_subgraph_tihs(hyperedges_dict: HyperedgeDict, max_nodes: int) -> HyperedgeDict:
    if _native_sampler is not None:
        seed = random.randrange(0, 2**63)
        return dict(_native_sampler.sample_subgraph_tihs(hyperedges_dict, max_nodes, seed))
    return sample_subgraph_tihs_python(hyperedges_dict, max_nodes)


def sample_subgraph_snowball(hyperedges_dict: HyperedgeDict, max_nodes: int) -> HyperedgeDict:
    if _native_sampler is not None:
        seed = random.randrange(0, 2**63)
        return dict(_native_sampler.sample_subgraph_snowball(hyperedges_dict, max_nodes, seed))
    return sample_subgraph_snowball_python(hyperedges_dict, max_nodes)


def sample_subgraph(data: HyperedgeDict, m: int, strategy: str = "TIHS") -> HyperedgeDict:
    if strategy == "TIHS":
        return sample_subgraph_tihs(data, m)
    if strategy == "snowball":
        return sample_subgraph_snowball(data, m)
    raise ValueError(f"Unsupported sampling strategy: {strategy}")


def extract_subgraph_samples_python(
    subgraph_data: HyperedgeDict,
    all_positives: Sequence[Sequence[int]],
    all_negatives: Sequence[Sequence[int]],
) -> Tuple[List[List[int]], List[List[int]]]:
    subgraph_edges_set = {_normalize_edge(nodes) for nodes in subgraph_data.values()}

    subgraph_positives: List[List[int]] = []
    for edge in all_positives:
        if _normalize_edge(edge) in subgraph_edges_set:
            subgraph_positives.append(list(edge))

    subgraph_nodes: Set[int] = set()
    for nodes in subgraph_data.values():
        subgraph_nodes.update(int(node) for node in nodes)

    subgraph_negatives: List[List[int]] = []
    for edge in all_negatives:
        if all(int(node) in subgraph_nodes for node in edge):
            subgraph_negatives.append(list(edge))
        if len(subgraph_negatives) >= len(subgraph_positives):
            break

    return subgraph_positives, subgraph_negatives


def extract_subgraph_samples(
    subgraph_data: HyperedgeDict,
    all_positives: Sequence[Sequence[int]],
    all_negatives: Sequence[Sequence[int]],
) -> Tuple[List[List[int]], List[List[int]]]:
    if _native_sampler is not None:
        positives, negatives = _native_sampler.extract_subgraph_samples(
            subgraph_data, list(all_positives), list(all_negatives)
        )
        return list(positives), list(negatives)

    return extract_subgraph_samples_python(subgraph_data, all_positives, all_negatives)


def validate_subgraph(subgraph: HyperedgeDict, original_graph: HyperedgeDict) -> bool:
    for eid, nodes in subgraph.items():
        if eid not in original_graph:
            return False
        if list(nodes) != list(original_graph[eid]):
            return False
    return True


if __name__ == "__main__":
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
        9: [4, 9, 14, 19],
    }

    print("Native sampler:", native_sampler_available())
    for strategy in ["TIHS", "snowball"]:
        for max_nodes in [5, 10, 15]:
            subgraph = sample_subgraph(test_hypergraph, max_nodes, strategy)
            subgraph_nodes = set()
            for nodes in subgraph.values():
                subgraph_nodes.update(nodes)
            print(
                strategy,
                "max_nodes=",
                max_nodes,
                "edges=",
                len(subgraph),
                "nodes=",
                len(subgraph_nodes),
                "valid=",
                validate_subgraph(subgraph, test_hypergraph),
            )

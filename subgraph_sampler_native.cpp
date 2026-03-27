#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <deque>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;

using EdgeId = std::int64_t;
using NodeId = std::int64_t;
using EdgeNodes = std::vector<NodeId>;
using Hyperedges = std::vector<std::pair<EdgeId, EdgeNodes>>;

struct VectorHash {
    std::size_t operator()(const std::vector<NodeId>& values) const noexcept {
        std::size_t seed = values.size();
        for (NodeId value : values) {
            seed ^= std::hash<NodeId>{}(value) + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
        }
        return seed;
    }
};

static Hyperedges parse_hyperedges(const py::dict& hyperedges_dict) {
    Hyperedges hyperedges;
    hyperedges.reserve(py::len(hyperedges_dict));
    for (auto item : hyperedges_dict) {
        EdgeId edge_id = py::cast<EdgeId>(item.first);
        EdgeNodes nodes = py::cast<EdgeNodes>(item.second);
        hyperedges.emplace_back(edge_id, std::move(nodes));
    }
    return hyperedges;
}

static py::dict to_python_dict(
    const Hyperedges& all_hyperedges,
    const std::vector<EdgeId>& selected_order,
    const std::unordered_set<EdgeId>& selected_lookup
) {
    py::dict result;
    for (EdgeId edge_id : selected_order) {
        if (selected_lookup.find(edge_id) == selected_lookup.end()) {
            continue;
        }
        for (const auto& item : all_hyperedges) {
            if (item.first == edge_id) {
                result[py::int_(edge_id)] = py::cast(item.second);
                break;
            }
        }
    }
    return result;
}

static py::dict sample_subgraph_tihs_impl(
    const py::dict& hyperedges_dict,
    std::int64_t max_nodes,
    std::uint64_t seed
) {
    Hyperedges hyperedges = parse_hyperedges(hyperedges_dict);
    if (hyperedges.empty()) {
        return py::dict();
    }

    std::mt19937_64 rng(seed);
    std::shuffle(hyperedges.begin(), hyperedges.end(), rng);

    std::unordered_set<EdgeId> selected_edges;
    std::unordered_set<NodeId> selected_nodes;
    std::vector<EdgeId> selected_order;
    selected_order.reserve(hyperedges.size());

    {
        py::gil_scoped_release release;

        for (const auto& item : hyperedges) {
            EdgeId edge_id = item.first;
            const EdgeNodes& nodes = item.second;

            if (selected_edges.insert(edge_id).second) {
                selected_order.push_back(edge_id);
            }

            for (NodeId node : nodes) {
                selected_nodes.insert(node);
            }

            for (const auto& candidate : hyperedges) {
                EdgeId candidate_id = candidate.first;
                if (selected_edges.find(candidate_id) != selected_edges.end()) {
                    continue;
                }

                bool is_subset = true;
                for (NodeId node : candidate.second) {
                    if (selected_nodes.find(node) == selected_nodes.end()) {
                        is_subset = false;
                        break;
                    }
                }

                if (is_subset && selected_edges.insert(candidate_id).second) {
                    selected_order.push_back(candidate_id);
                }
            }

            if (static_cast<std::int64_t>(selected_nodes.size()) >= max_nodes) {
                break;
            }
        }
    }

    return to_python_dict(hyperedges, selected_order, selected_edges);
}

static py::dict sample_subgraph_snowball_impl(
    const py::dict& hyperedges_dict,
    std::int64_t max_nodes,
    std::uint64_t seed
) {
    Hyperedges hyperedges = parse_hyperedges(hyperedges_dict);
    if (hyperedges.empty()) {
        return py::dict();
    }

    std::unordered_map<NodeId, std::vector<EdgeId>> node_to_edges;
    std::unordered_map<EdgeId, EdgeNodes> edge_lookup;
    for (const auto& item : hyperedges) {
        edge_lookup.emplace(item.first, item.second);
        for (NodeId node : item.second) {
            node_to_edges[node].push_back(item.first);
        }
    }

    if (node_to_edges.empty()) {
        return py::dict();
    }

    std::vector<NodeId> all_nodes;
    all_nodes.reserve(node_to_edges.size());
    for (const auto& item : node_to_edges) {
        all_nodes.push_back(item.first);
    }

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::size_t> dist(0, all_nodes.size() - 1);
    NodeId start_node = all_nodes[dist(rng)];

    std::unordered_set<NodeId> visited_nodes;
    std::unordered_set<EdgeId> selected_edges;
    std::vector<EdgeId> selected_order;
    std::deque<NodeId> queue;
    visited_nodes.insert(start_node);
    queue.push_back(start_node);

    {
        py::gil_scoped_release release;

        while (!queue.empty() && static_cast<std::int64_t>(visited_nodes.size()) < max_nodes) {
            NodeId current = queue.front();
            queue.pop_front();

            auto edge_iter = node_to_edges.find(current);
            if (edge_iter == node_to_edges.end()) {
                continue;
            }

            for (EdgeId edge_id : edge_iter->second) {
                if (!selected_edges.insert(edge_id).second) {
                    continue;
                }
                selected_order.push_back(edge_id);

                const auto& nodes = edge_lookup.at(edge_id);
                for (NodeId node : nodes) {
                    if (visited_nodes.insert(node).second) {
                        queue.push_back(node);
                    }
                }
            }

            if (static_cast<std::int64_t>(visited_nodes.size()) >= max_nodes) {
                break;
            }
        }
    }

    return to_python_dict(hyperedges, selected_order, selected_edges);
}

static py::tuple extract_subgraph_samples_impl(
    const py::dict& subgraph_data,
    const py::list& all_positives,
    const py::list& all_negatives
) {
    Hyperedges subgraph = parse_hyperedges(subgraph_data);

    std::unordered_set<std::vector<NodeId>, VectorHash> subgraph_edges;
    std::unordered_set<NodeId> subgraph_nodes;
    for (const auto& item : subgraph) {
        EdgeNodes canonical = item.second;
        std::sort(canonical.begin(), canonical.end());
        subgraph_edges.insert(canonical);
        for (NodeId node : item.second) {
            subgraph_nodes.insert(node);
        }
    }

    std::vector<EdgeNodes> positives = py::cast<std::vector<EdgeNodes>>(all_positives);
    std::vector<EdgeNodes> negatives = py::cast<std::vector<EdgeNodes>>(all_negatives);

    std::vector<EdgeNodes> selected_positives;
    std::vector<EdgeNodes> selected_negatives;

    {
        py::gil_scoped_release release;

        for (const auto& edge : positives) {
            EdgeNodes canonical = edge;
            std::sort(canonical.begin(), canonical.end());
            if (subgraph_edges.find(canonical) != subgraph_edges.end()) {
                selected_positives.push_back(edge);
            }
        }

        for (const auto& edge : negatives) {
            bool all_inside = true;
            for (NodeId node : edge) {
                if (subgraph_nodes.find(node) == subgraph_nodes.end()) {
                    all_inside = false;
                    break;
                }
            }
            if (all_inside) {
                selected_negatives.push_back(edge);
            }
            if (selected_negatives.size() >= selected_positives.size()) {
                break;
            }
        }
    }

    return py::make_tuple(py::cast(selected_positives), py::cast(selected_negatives));
}

PYBIND11_MODULE(subgraph_sampler_native, m) {
    m.doc() = "Native subgraph sampling helpers";
    m.def("sample_subgraph_tihs", &sample_subgraph_tihs_impl, py::arg("hyperedges_dict"), py::arg("max_nodes"), py::arg("seed"));
    m.def("sample_subgraph_snowball", &sample_subgraph_snowball_impl, py::arg("hyperedges_dict"), py::arg("max_nodes"), py::arg("seed"));
    m.def("extract_subgraph_samples", &extract_subgraph_samples_impl, py::arg("subgraph_data"), py::arg("all_positives"), py::arg("all_negatives"));
}

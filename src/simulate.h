#pragma once

#include "graph_generators.h"
#include "util.h"
#include <boost/graph/betweenness_centrality.hpp>
#include <boost/graph/graph_utility.hpp>
#include <algorithm>
#include <cmath>
#include <random>
#include <utility>
#include <vector>

enum class DegreeMode { Total, In, Out };

struct SeedFailure {
    std::vector<std::pair<int, int>> rem_list;
    int chunk;
};

template <typename Graph>
SeedFailure seed_failure(const Graph& g, double p_min, int n_pts, std::mt19937& rng) {
    int n_e = static_cast<int>(boost::num_edges(g));
    int max_rem = static_cast<int>(std::ceil((1.0 - p_min) * n_e));

    auto [ei, ei_end] = boost::edges(g);
    std::vector<std::pair<int, int>> all_edges;
    all_edges.reserve(n_e);
    for (auto it = ei; it != ei_end; ++it) {
        int u = static_cast<int>(boost::source(*it, g));
        int v = static_cast<int>(boost::target(*it, g));
        all_edges.emplace_back(u, v);
    }

    std::shuffle(all_edges.begin(), all_edges.end(), rng);

    int n_take = std::min(max_rem, static_cast<int>(all_edges.size()));
    std::vector<std::pair<int, int>> rem_list(all_edges.begin(), all_edges.begin() + n_take);

    int chunk = (n_take > 0) ? (n_take / n_pts) : 0;

    return {rem_list, chunk};
}

template <typename Graph>
void set_theta_weights(Graph& g, double theta, DegreeMode mode) {
    std::vector<int> d;
    switch (mode) {
        case DegreeMode::In:  d = degrees_in(g);  break;
        case DegreeMode::Out: d = degrees_out(g); break;
        default:              d = degrees(g);     break;
    }
    auto [ei, ei_end] = boost::edges(g);
    for (auto it = ei; it != ei_end; ++it) {
        int u = static_cast<int>(boost::source(*it, g));
        int v = static_cast<int>(boost::target(*it, g));
        double w = std::pow(std::max(1e-15, static_cast<double>(d[u]) * d[v]), theta);
        boost::put(boost::edge_weight, g, *it, w);
    }
}

template <typename Graph>
std::pair<Graph, int> cascade(const Graph& g, const std::vector<double>& capacity,
                              double theta, DegreeMode mode) {
    Graph h = g;
    int t_step = 1;

    while (true) {
        set_theta_weights(h, theta, mode);

        int nv = static_cast<int>(boost::num_vertices(h));
        std::vector<double> bc(nv, 0.0);

        try {
            boost::brandes_betweenness_centrality(h,
                boost::centrality_map(
                    boost::make_iterator_property_map(bc.begin(),
                        boost::get(boost::vertex_index, h))
                )
                .weight_map(boost::get(boost::edge_weight, h))
                .vertex_index_map(boost::get(boost::vertex_index, h))
            );
        } catch (...) {
            break;
        }

        std::vector<int> failed;
        for (int v = 0; v < nv; ++v) {
            if (bc[v] > capacity[v]) {
                failed.push_back(v);
            }
        }

        if (failed.empty()) break;

        for (int v : failed) {
            boost::clear_vertex(v, h);
        }

        ++t_step;
    }

    return {h, t_step};
}

#include "simulate.h"
#include "betweenness.h"
#include <algorithm>
#include <cmath>
#include <random>

SeedFailure seed_failure(const Graph& g, double p_min, std::mt19937_64& rng) {
    int n_e = static_cast<int>(boost::num_edges(g));
    int max_rem = static_cast<int>(std::llround((1.0 - p_min) * n_e));
    max_rem = std::clamp(max_rem, 0, n_e);

    auto [ei, ei_end] = boost::edges(g);
    std::vector<std::pair<int, int>> all_edges;
    all_edges.reserve(n_e);
    for (auto it = ei; it != ei_end; ++it) {
        int u = static_cast<int>(boost::source(*it, g));
        int v = static_cast<int>(boost::target(*it, g));
        all_edges.emplace_back(u, v);
    }

    std::shuffle(all_edges.begin(), all_edges.end(), rng);

    all_edges.resize(max_rem);
    return {std::move(all_edges)};
}

std::pair<Graph, int> cascade(Graph h, const std::vector<double>& capacity, double θ) {
    int t_step = 1;
    int nv = static_cast<int>(boost::num_vertices(h));
    BCMWorkspace ws;
    std::vector<int> failed;
    failed.reserve(nv);

    while (true) {
        const auto& bc = betweenness(h, θ, ws);

        failed.clear();
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

    return {std::move(h), t_step};
}

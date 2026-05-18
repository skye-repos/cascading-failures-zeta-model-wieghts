#pragma once

#include "graph_generators.h"
#include <vector>
#include <utility>

struct SeedFailure {
    std::vector<std::pair<int, int>> rem_list;
    int chunk;
};

SeedFailure seed_failure(const Graph& g, double p_min, int n_pts, std::mt19937& rng);

void set_theta_weights(Graph& g, double theta);

std::pair<Graph, int> cascade(const Graph& g, const std::vector<double>& capacity, double theta);

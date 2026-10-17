#pragma once

#include "graph_generators.h"
#include <vector>
#include <utility>

struct SeedFailure {
    std::vector<std::pair<int, int>> rem_list;
};

SeedFailure seed_failure(const Graph& g, double p_min, std::mt19937_64& rng);

std::pair<Graph, int> cascade(Graph g, const std::vector<double>& capacity, double θ);

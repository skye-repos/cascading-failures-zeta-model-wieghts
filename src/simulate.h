#pragma once

#include "graph_generators.h"
#include <vector>
#include <utility>

struct SeedFailure {
    std::vector<std::pair<int, int>> rem_list;
};

SeedFailure seed_failure(const Graph& g, double p_min, std::mt19937_64& rng);

void set_θ_weights(Graph& g, double θ);

std::pair<Graph, int> cascade(const Graph& g, const std::vector<double>& capacity, double θ);

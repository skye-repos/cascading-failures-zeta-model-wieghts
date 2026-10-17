#pragma once

#include "graph_generators.h"
#include <utility>
#include <vector>

struct BCMWorkspace {
    std::vector<int> degree;
    std::vector<int> offset;
    std::vector<int> pos;
    std::vector<int> neighbor;
    std::vector<double> weight;
    std::vector<double> dist;
    std::vector<double> sigma;
    std::vector<double> delta;
    std::vector<double> bc;
    std::vector<int> order;
    std::vector<int> pred_head;
    std::vector<int> pred_vertex;
    std::vector<int> pred_next;
    std::vector<std::pair<double, int>> heap;
};

const std::vector<double>& betweenness(const Graph& g, double θ, BCMWorkspace& ws);

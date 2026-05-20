#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/properties.hpp>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

template <typename Directed>
using GraphBase = boost::adjacency_list<
    boost::vecS, boost::vecS, Directed,
    boost::no_property,
    boost::property<boost::edge_weight_t, double>
>;

using UndirectedGraph = GraphBase<boost::undirectedS>;
using DirectedGraph   = GraphBase<boost::bidirectionalS>;
using Graph = UndirectedGraph;

UndirectedGraph zeta_taurus(int L, double zeta, double k_avg, std::mt19937& rng);

template <typename Directed>
GraphBase<Directed> read_csv(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open graph file: " + path);
    std::string line;
    std::vector<std::pair<int, int>> edges;
    while (std::getline(f, line)) {
        if (line.empty() || !std::isdigit(static_cast<unsigned char>(line[0])))
            continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream ss(line);
        int u, v;
        if (ss >> u >> v)
            edges.emplace_back(u, v);
    }
    if (edges.empty())
        throw std::runtime_error("Empty graph file: " + path);
    int min_v = edges[0].first, max_v = edges[0].first;
    for (auto& e : edges) {
        min_v = std::min(min_v, std::min(e.first, e.second));
        max_v = std::max(max_v, std::max(e.first, e.second));
    }
    int offset = (min_v == 1) ? 1 : 0;
    int nv = max_v - offset + 1;
    GraphBase<Directed> g(nv);
    for (auto& e : edges)
        boost::add_edge(e.first - offset, e.second - offset, g);
    return g;
}

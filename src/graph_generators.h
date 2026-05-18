#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/properties.hpp>
#include <random>

using Graph = boost::adjacency_list<
    boost::vecS,
    boost::vecS,
    boost::undirectedS,
    boost::no_property,
    boost::property<boost::edge_weight_t, double>
>;

Graph zeta_taurus(int L, double zeta, double k_avg, std::mt19937& rng);
Graph barabasi_albert(int m, int N, std::mt19937& rng);
Graph erdos_renyi(int N, double p, std::mt19937& rng);

#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/properties.hpp>
#include <random>
#include <string>

using Graph = boost::adjacency_list<
    boost::vecS,
    boost::vecS,
    boost::undirectedS,
    boost::no_property,
    boost::property<boost::edge_weight_t, double>
>;

enum class Topology { Waxman, ER };

Topology parse_topology(const std::string& name);

std::string topology_name(Topology topology);

Graph zeta_taurus(int L, double ζ, double k_avg, std::mt19937_64& rng);

Graph er_graph(int N, double k_avg, std::mt19937_64& rng);

Graph generate_graph(Topology topology, int L, double ζ, double k_avg,
                     std::mt19937_64& rng);

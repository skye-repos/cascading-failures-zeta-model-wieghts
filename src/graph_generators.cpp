#include "graph_generators.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

Topology parse_topology(const std::string& name) {
    if (name == "waxman" || name == "Waxman" || name == "WAXMAN") {
        return Topology::Waxman;
    }
    if (name == "er" || name == "ER" || name == "Er") {
        return Topology::ER;
    }
    throw std::invalid_argument("unknown topology: " + name);
}

std::string topology_name(Topology topology) {
    return topology == Topology::ER ? "ER" : "waxman";
}

Graph zeta_taurus(int L, double ζ, double k_avg, std::mt19937_64& rng) {
    int N = L * L;
    Graph g(N);

    auto toroidal_dist = [L](int a, int b) {
        int d = std::abs(a - b);
        return std::min(d, L - d);
    };

    double s = 0.0;
    for (int j = 1; j < N; ++j) {
        int rj = j / L, cj = j % L;
        double dx = static_cast<double>(toroidal_dist(0, rj));
        double dy = static_cast<double>(toroidal_dist(0, cj));
        s += std::exp(-std::sqrt(dx * dx + dy * dy) / ζ);
    }
    double coeff = k_avg / s;

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int j = 1; j < N; ++j) {
        int rj = j / L, cj = j % L;
        for (int i = 0; i < j; ++i) {
            int ri = i / L, ci = i % L;
            double dx = static_cast<double>(toroidal_dist(ri, rj));
            double dy = static_cast<double>(toroidal_dist(ci, cj));
            double prob = coeff * std::exp(-std::sqrt(dx * dx + dy * dy) / ζ);
            if (dist(rng) < prob) {
                boost::add_edge(i, j, g);
            }
        }
    }

    return g;
}

Graph er_graph(int N, double k_avg, std::mt19937_64& rng) {
    Graph g(N);
    if (N < 2) {
        return g;
    }

    double q = k_avg / static_cast<double>(N - 1);
    q = std::clamp(q, 0.0, 1.0);

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int j = 1; j < N; ++j) {
        for (int i = 0; i < j; ++i) {
            if (dist(rng) < q) {
                boost::add_edge(i, j, g);
            }
        }
    }

    return g;
}

Graph generate_graph(Topology topology, int L, double ζ, double k_avg,
                     std::mt19937_64& rng) {
    if (topology == Topology::ER) {
        return er_graph(L * L, k_avg, rng);
    }
    return zeta_taurus(L, ζ, k_avg, rng);
}

#include "graph_generators.h"
#include <vector>
#include <cmath>
#include <algorithm>

Graph zeta_taurus(int L, double zeta, double k_avg, std::mt19937& rng) {
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
        s += std::exp(-std::sqrt(dx * dx + dy * dy) / zeta);
    }
    double coeff = k_avg / s;

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int j = 1; j < N; ++j) {
        int rj = j / L, cj = j % L;
        for (int i = 0; i < j; ++i) {
            int ri = i / L, ci = i % L;
            double dx = static_cast<double>(toroidal_dist(ri, rj));
            double dy = static_cast<double>(toroidal_dist(ci, cj));
            double prob = coeff * std::exp(-std::sqrt(dx * dx + dy * dy) / zeta);
            if (dist(rng) < prob) {
                boost::add_edge(i, j, g);
            }
        }
    }

    return g;
}

Graph barabasi_albert(int m, int N, std::mt19937& rng) {
    Graph g(m + 1);
    for (int i = 1; i <= m; ++i) {
        for (int j = 0; j < i; ++j) {
            boost::add_edge(i, j, g);
        }
    }

    std::vector<int> pool;
    for (int i = 0; i <= m; ++i) {
        int d = static_cast<int>(boost::degree(i, g));
        for (int j = 0; j < d; ++j) {
            pool.push_back(i);
        }
    }

    std::uniform_int_distribution<int> pick(0, static_cast<int>(pool.size()) - 1);
    for (int source = m + 2; source <= N; ++source) {
        boost::add_vertex(g);
        std::vector<int> targets;
        while (static_cast<int>(targets.size()) < m) {
            int t = pool[pick(rng)];
            if (std::find(targets.begin(), targets.end(), t) == targets.end()) {
                targets.push_back(t);
                boost::add_edge(source, t, g);
                pool.push_back(t);
                pool.push_back(source);
            }
        }
    }

    return g;
}

Graph erdos_renyi(int N, double p, std::mt19937& rng) {
    Graph g(N);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            if (dist(rng) < p) {
                boost::add_edge(i, j, g);
            }
        }
    }
    return g;
}

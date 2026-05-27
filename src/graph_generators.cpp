#include "graph_generators.h"
#include <vector>
#include <cmath>
#include <algorithm>

Graph zeta_taurus(int L, double ζ, double k_avg, std::mt19937& rng) {
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

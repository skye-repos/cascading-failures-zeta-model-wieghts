#include "util.h"
#include <boost/graph/connected_components.hpp>
#include <boost/graph/graph_utility.hpp>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>

std::vector<int> degrees(const Graph& g) {
    int n = static_cast<int>(boost::num_vertices(g));
    std::vector<int> d(n);
    for (int v = 0; v < n; ++v) {
        d[v] = static_cast<int>(boost::degree(v, g));
    }
    return d;
}

std::pair<std::vector<int>, std::vector<double>> degree_distribution(const Graph& g) {
    auto d = degrees(g);
    auto [min_it, max_it] = std::minmax_element(d.begin(), d.end());
    int k_min = *min_it;
    int k_max = *max_it;
    if (k_min == 0) k_min = 1;
    if (k_max < k_min) {
        return {{k_min}, {1.0}};
    }

    int n = static_cast<int>(d.size());
    int range = k_max - k_min + 1;
    std::vector<double> p(range, 0.0);
    for (int k : d) {
        if (k >= k_min && k <= k_max) {
            p[k - k_min] += 1.0 / n;
        }
    }

    std::vector<int> k_vals(range);
    for (int i = 0; i < range; ++i) {
        k_vals[i] = k_min + i;
    }

    return {k_vals, p};
}

int giant_component_size(const Graph& g) {
    int n = static_cast<int>(boost::num_vertices(g));
    if (n == 0) return 0;

    std::vector<int> comp(n);
    int num = boost::connected_components(g, comp.data());
    std::vector<int> sizes(num, 0);
    for (int c : comp) {
        ++sizes[c];
    }
    return *std::max_element(sizes.begin(), sizes.end());
}

GraphStats graph_stats(const Graph& g) {
    GraphStats stats;
    auto d = degrees(g);
    int n = static_cast<int>(d.size());
    stats.n_vertices = n;
    stats.n_edges = static_cast<int>(boost::num_edges(g));
    if (n == 0) {
        return stats;
    }

    double sum = std::accumulate(d.begin(), d.end(), 0.0);
    stats.k_mean = sum / n;

    double var = 0.0;
    double second = 0.0;
    for (int k : d) {
        double dk = static_cast<double>(k);
        double diff = dk - stats.k_mean;
        var += diff * diff;
        second += dk * dk;
    }
    stats.k_variance = var / n;
    stats.k_second_moment = second / n;
    return stats;
}

void write_manifest_json(const std::string& file_path, const ManifestData& data) {
    std::ofstream f(file_path);
    if (!f.is_open()) return;

    f << std::setprecision(15);
    f << "{\n";
    f << "  \"base_seed\": " << data.base_seed << ",\n";
    f << "  \"realization_id\": " << data.realization_id << ",\n";
    f << "  \"realization_seed\": " << data.realization_seed << ",\n";
    f << "  \"L\": " << data.L << ",\n";
    f << "  \"zeta\": " << data.zeta << ",\n";
    f << "  \"topology\": \"" << data.topology << "\",\n";
    f << "  \"theta_min\": " << data.theta_min << ",\n";
    f << "  \"theta_max\": " << data.theta_max << ",\n";
    f << "  \"N_theta\": " << data.N_theta << ",\n";
    f << "  \"n_theta\": " << data.n_theta << ",\n";
    f << "  \"n_pts\": " << data.n_pts << ",\n";
    f << "  \"k_avg\": " << data.k_avg << ",\n";
    f << "  \"alpha\": " << data.alpha << ",\n";
    f << "  \"p_min\": " << data.p_min << ",\n";
    f << "  \"graph\": {\n";
    f << "    \"n_vertices\": " << data.graph.n_vertices << ",\n";
    f << "    \"n_edges\": " << data.graph.n_edges << ",\n";
    f << "    \"k_mean\": " << data.graph.k_mean << ",\n";
    f << "    \"k_variance\": " << data.graph.k_variance << ",\n";
    f << "    \"k_second_moment\": " << data.graph.k_second_moment << "\n";
    f << "  }\n";
    f << "}\n";
}

void write_csv(const std::vector<double>& matrix, int n_rows, int n_cols,
               const std::string& file_path, const std::vector<std::string>& header) {
    std::ofstream f(file_path);
    if (!f.is_open()) return;

    f << std::setprecision(15);

    for (int c = 0; c < n_cols; ++c) {
        if (c > 0) f << ",";
        f << header[c];
    }
    f << "\n";

    for (int r = 0; r < n_rows; ++r) {
        for (int c = 0; c < n_cols; ++c) {
            if (c > 0) f << ",";
            f << matrix[r * n_cols + c];
        }
        f << "\n";
    }
}

std::vector<double> linspace(double start, double end, int n) {
    std::vector<double> result(n);
    if (n == 1) {
        result[0] = start;
        return result;
    }
    double step = (end - start) / (n - 1);
    for (int i = 0; i < n; ++i) {
        result[i] = start + i * step;
    }
    return result;
}

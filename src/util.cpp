#include "util.h"
#include <boost/graph/connected_components.hpp>
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

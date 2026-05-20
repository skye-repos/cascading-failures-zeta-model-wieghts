#include "util.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cmath>

void write_csv(const std::vector<double>& matrix, int n_rows, int n_cols,
               const std::string& file_path, const std::vector<std::string>& header) {
    std::ofstream f(file_path);
    if (!f.is_open()) {
        std::cerr << "  error: could not write " << file_path << std::endl;
        return;
    }

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
    for (int i = 0; i < n; ++i)
        result[i] = start + i * step;
    return result;
}

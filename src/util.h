#pragma once

#include "graph_generators.h"
#include <cstdint>
#include <vector>
#include <string>
#include <utility>

struct GraphStats {
    int n_vertices = 0;
    int n_edges = 0;
    double k_mean = 0.0;
    double k_variance = 0.0;
    double k_second_moment = 0.0;
};

struct ManifestData {
    std::uint64_t base_seed = 0;
    std::uint64_t realization_seed = 0;
    int realization_id = 0;
    int L = 0;
    double zeta = 0.0;
    std::string topology;
    double theta_min = 0.0;
    double theta_max = 0.0;
    int N_theta = 0;
    int n_theta = 0;
    int n_pts = 0;
    double k_avg = 0.0;
    double alpha = 0.0;
    double p_min = 0.0;
    GraphStats graph;
};

std::vector<int> degrees(const Graph& g);

std::pair<std::vector<int>, std::vector<double>> degree_distribution(const Graph& g);

int giant_component_size(const Graph& g);

GraphStats graph_stats(const Graph& g);

void write_manifest_json(const std::string& file_path, const ManifestData& data);

void write_csv(const std::vector<double>& matrix, int n_rows, int n_cols,
               const std::string& file_path, const std::vector<std::string>& header);

std::vector<double> linspace(double start, double end, int n);

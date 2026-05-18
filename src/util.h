#pragma once

#include "graph_generators.h"
#include <vector>
#include <string>
#include <utility>

std::vector<int> degrees(const Graph& g);

std::pair<std::vector<int>, std::vector<double>> degree_distribution(const Graph& g);

int giant_component_size(const Graph& g);

void write_csv(const std::vector<double>& matrix, int n_rows, int n_cols,
               const std::string& file_path, const std::vector<std::string>& header);

std::vector<double> linspace(double start, double end, int n);

#include "graph_generators.h"
#include "simulate.h"
#include "util.h"
#include <boost/graph/betweenness_centrality.hpp>
#include <omp.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>

namespace fs = std::filesystem;

static std::string fmt_double(double x) {
    std::ostringstream ss;
    ss << x;
    return ss.str();
}

int main(int argc, char* argv[]) {
    int L = 50;
    double ζ = 15.0;
    double θ_min = -1.0;
    double θ_max = 1.0;
    int N_θ = 21;
    int n_pts = 50;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](int i) { return i + 1 < argc ? argv[i + 1] : nullptr; };
        if ((arg == "-L" || arg == "--L") && next(i)) { L = std::stoi(argv[++i]); }
        else if ((arg == "-z" || arg == "--zeta") && next(i)) { ζ = std::stod(argv[++i]); }
        else if ((arg == "--tmin" || arg == "--theta-min") && next(i)) { θ_min = std::stod(argv[++i]); }
        else if ((arg == "--tmax" || arg == "--theta-max") && next(i)) { θ_max = std::stod(argv[++i]); }
        else if ((arg == "-n" || arg == "--N-theta") && next(i)) { N_θ = std::stoi(argv[++i]); }
        else if ((arg == "-p" || arg == "--n-pts") && next(i)) { n_pts = std::stoi(argv[++i]); }
        else {
            std::cerr << "Usage: " << argv[0] << " [options]\n"
                      << "  -L, --L L              system size (default 50)\n"
                      << "  -z, --zeta ζ           characteristic length (default 15)\n"
                      << "  --tmin, --theta-min t  lower θ bound (default -1)\n"
                      << "  --tmax, --theta-max t  upper θ bound (default 1)\n"
                      << "  -n, --N-theta n        number of θ points (default 21)\n"
                      << "  -p, --n-pts n          number of p values (default 50)\n"
                      << "  θ=0 is excluded automatically.\n";
            return 1;
        }
    }
    const double k_avg = 5.0;
    const double α = 2.0;

    auto t0 = std::chrono::steady_clock::now();

    std::random_device rd;
    std::mt19937 rng(rd());

    std::cout << "Generating ζ-model graph (L=" << L
              << ", ζ=" << ζ << ", k_avg=" << k_avg << ")..." << std::endl;
    Graph g = zeta_taurus(L, ζ, k_avg, rng);
    int n_e = static_cast<int>(boost::num_edges(g));
    std::cout << "  Vertices: " << boost::num_vertices(g)
              << ", Edges: " << n_e << std::endl;

    std::vector<double> raw = linspace(θ_min, θ_max, N_θ);
    std::vector<double> θ_list;
    for (double t : raw) {
        if (std::abs(t) > 1e-9) θ_list.push_back(t);
    }
    int n_θ = static_cast<int>(θ_list.size());
    std::cout << "  θ values: " << n_θ << " (";
    for (int i = 0; i < n_θ; ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << θ_list[i];
    }
    std::cout << ")" << std::endl;

    double p_min = (ζ / L < 0.01) ? 0.85 : 0.5;
    std::cout << "  p_min = " << p_min << std::endl;

    auto sf = seed_failure(g, p_min, n_pts, rng);
    const auto& rem_list = sf.rem_list;
    const int chunk = sf.chunk;
    int n_take_edges = static_cast<int>(rem_list.size());
    std::cout << "  seed edges to remove: " << n_take_edges
              << ", chunk=" << chunk << std::endl;

    std::vector<double> p_list(n_pts);
    for (int n = 1; n <= n_pts; ++n) {
        int n_rem = std::min(n * chunk, n_take_edges);
        int row = n_pts - n;
        p_list[row] = static_cast<double>(n_e - n_rem) / n_e;
    }

    // Results (transposed: n_θ x n_pts, row-major flat)
    std::vector<double> S_results(n_θ * n_pts, 0.0);
    std::vector<double> t_results(n_θ * n_pts, 0.0);

    // Degree distributions: per (θ,n) store up to 100 degree values
    int max_k_pre = 100;
    std::vector<double> p_results(n_θ * n_pts * max_k_pre, 0.0);

    std::cout << "Running cascade simulations..." << std::endl;

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n_θ; ++i) {
        double θ = θ_list[i];

        Graph cap_g = g;
        set_θ_weights(cap_g, θ);
        int nv = static_cast<int>(boost::num_vertices(cap_g));
        std::vector<double> capacity(nv, 0.0);
        boost::brandes_betweenness_centrality(cap_g,
            boost::centrality_map(
                boost::make_iterator_property_map(capacity.begin(),
                    boost::get(boost::vertex_index, cap_g))
            )
            .weight_map(boost::get(boost::edge_weight, cap_g))
            .vertex_index_map(boost::get(boost::vertex_index, cap_g))
        );
        for (auto& c : capacity) c *= (1.0 + α);

        for (int n = 1; n <= n_pts; ++n) {
            Graph h = g;
            int n_rem = std::min(n * chunk, static_cast<int>(rem_list.size()));
            for (int r = 0; r < n_rem; ++r) {
                boost::remove_edge(rem_list[r].first, rem_list[r].second, h);
            }

            auto [h_final, t] = cascade(h, capacity, θ);
            int gc = giant_component_size(h_final);
            double S_val = static_cast<double>(gc) / boost::num_vertices(h_final);

            #pragma omp critical
            std::cout << "  θ=" << θ << "  n=" << n << "/" << n_pts
                      << "  t_step=" << t << "  S=" << S_val << std::endl;

            int row = n_pts - n;
            S_results[i * n_pts + row] = S_val;
            t_results[i * n_pts + row] = static_cast<double>(t);

            auto [k_vals, p_vals] = degree_distribution(h_final);
            int nk = std::min(static_cast<int>(p_vals.size()), max_k_pre);
            for (int j = 0; j < nk; ++j) {
                p_results[(i * n_pts + row) * max_k_pre + j] = p_vals[j];
            }
        }
    }

    // Determine global k_max
    int k_max = 0;
    for (int i = 0; i < n_θ * n_pts; ++i) {
        for (int j = max_k_pre - 1; j >= 0; --j) {
            if (p_results[i * max_k_pre + j] != 0.0) {
                k_max = std::max(k_max, j + 1);
                break;
            }
        }
    }
    std::cout << "  k_max = " << k_max << std::endl;

    // Output directories
    std::string base = std::to_string(L);
    std::string ζs = fmt_double(ζ);
    std::string ks = fmt_double(k_avg);
    std::string αs = fmt_double(α);
    std::string dir_suffix = "L = " + base + "/ζ = " + ζs + ", k = " + ks + ", α = " + αs + "/";

    fs::path csv_dir  = fs::path("results") / "CSV" / dir_suffix;
    fs::path plot_dir = fs::path("results") / "plots" / dir_suffix;
    fs::create_directories(csv_dir);
    fs::create_directories(plot_dir);

    // Build string header with θ column
    std::vector<std::string> hdr = {"θ"};
    for (double p : p_list) hdr.push_back(fmt_double(p));

    // Build θ metadata for filenames
    double dθ = (n_θ > 1) ? (θ_list[1] - θ_list[0]) : 0.0;
    double θ_first = θ_list.front();
    double θ_last = θ_list.back();
    std::string θ_meta = " Nθ=" + std::to_string(n_θ)
        + ", dθ=" + fmt_double(dθ)
        + ", θ∈[" + fmt_double(θ_first) + "," + fmt_double(θ_last) + "]";

    // Build S matrix with θ column
    std::vector<double> S_wth(n_θ * (n_pts + 1), 0.0);
    for (int i = 0; i < n_θ; ++i) {
        S_wth[i * (n_pts + 1)] = θ_list[i];
        for (int j = 0; j < n_pts; ++j)
            S_wth[i * (n_pts + 1) + 1 + j] = S_results[i * n_pts + j];
    }

    // Build t matrix with θ column
    std::vector<double> t_wth(n_θ * (n_pts + 1), 0.0);
    for (int i = 0; i < n_θ; ++i) {
        t_wth[i * (n_pts + 1)] = θ_list[i];
        for (int j = 0; j < n_pts; ++j)
            t_wth[i * (n_pts + 1) + 1 + j] = t_results[i * n_pts + j];
    }

    // Write S matrix
    {
        std::string fname = " Gt Component" + θ_meta + " ζ = " + ζs + ".csv";
        write_csv(S_wth, n_θ, n_pts + 1, (csv_dir / fname).string(), hdr);
        std::cout << "  wrote " << fname << std::endl;
    }

    // Write t matrix
    {
        std::string fname = " Cascade Time" + θ_meta + " ζ = " + ζs + ".csv";
        write_csv(t_wth, n_θ, n_pts + 1, (csv_dir / fname).string(), hdr);
        std::cout << "  wrote " << fname << std::endl;
    }

    // Write degree distributions per θ
    for (int i = 0; i < n_θ; ++i) {
        double θ = θ_list[i];
        std::ostringstream ts;
        ts << θ;
        std::string fname = "degree-distribution θ=" + ts.str() + ", ζ=" + ζs + ".csv";

        // n_pts x k_max matrix (row-major flat)
        std::vector<double> deg_mat(n_pts * k_max, 0.0);
        for (int n = 0; n < n_pts; ++n) {
            for (int j = 0; j < k_max; ++j) {
                double val = (j < max_k_pre)
                    ? p_results[(i * n_pts + n) * max_k_pre + j]
                    : 0.0;
                deg_mat[n * k_max + j] = val;
            }
        }

        // transpose to k_max x n_pts then add θ column
        std::vector<double> deg_out(k_max * (n_pts + 1), 0.0);
        for (int r = 0; r < k_max; ++r) {
            deg_out[r * (n_pts + 1)] = θ;
            for (int c = 0; c < n_pts; ++c)
                deg_out[r * (n_pts + 1) + 1 + c] = deg_mat[c * k_max + r];
        }

        write_csv(deg_out, k_max, n_pts + 1, (csv_dir / fname).string(), hdr);
    }

    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = t1 - t0;
    std::cout << "Done. Results in " << csv_dir << std::endl;
    std::cout << "Total time: " << elapsed.count() << " s" << std::endl;
    return 0;
}

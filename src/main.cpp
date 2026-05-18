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
    double zeta = 15.0;
    double theta_min = -1.0;
    double theta_max = 1.0;
    int N_theta = 21;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](int i) { return i + 1 < argc ? argv[i + 1] : nullptr; };
        if ((arg == "-L" || arg == "--L") && next(i)) { L = std::stoi(argv[++i]); }
        else if ((arg == "-z" || arg == "--zeta") && next(i)) { zeta = std::stod(argv[++i]); }
        else if ((arg == "--tmin" || arg == "--theta-min") && next(i)) { theta_min = std::stod(argv[++i]); }
        else if ((arg == "--tmax" || arg == "--theta-max") && next(i)) { theta_max = std::stod(argv[++i]); }
        else if ((arg == "-n" || arg == "--N-theta") && next(i)) { N_theta = std::stoi(argv[++i]); }
        else {
            std::cerr << "Usage: " << argv[0] << " [options]\n"
                      << "  -L, --L L              system size (default 50)\n"
                      << "  -z, --zeta zeta        characteristic length (default 15)\n"
                      << "  --tmin, --theta-min t  lower theta bound (default -1)\n"
                      << "  --tmax, --theta-max t  upper theta bound (default 1)\n"
                      << "  -n, --N-theta n        number of theta points (default 21)\n"
                      << "  theta=0 is excluded automatically.\n";
            return 1;
        }
    }
    const double k_avg = 5.0;
    const double alpha = 2.0;
    const int n_pts = 50;

    std::random_device rd;
    std::mt19937 rng(rd());

    std::cout << "Generating zeta-model graph (L=" << L
              << ", zeta=" << zeta << ", k_avg=" << k_avg << ")..." << std::endl;
    Graph g = zeta_taurus(L, zeta, k_avg, rng);
    std::cout << "  Vertices: " << boost::num_vertices(g)
              << ", Edges: " << boost::num_edges(g) << std::endl;

    std::vector<double> raw = linspace(theta_min, theta_max, N_theta);
    std::vector<double> theta_list;
    for (double t : raw) {
        if (std::abs(t) > 1e-9) theta_list.push_back(t);
    }
    int l_theta = static_cast<int>(theta_list.size());
    std::cout << "  theta values: " << l_theta << " (";
    for (int i = 0; i < l_theta; ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << theta_list[i];
    }
    std::cout << ")" << std::endl;

    double p_min = (zeta / L < 0.01) ? 0.85 : 0.5;
    std::cout << "  p_min = " << p_min << std::endl;

    auto sf = seed_failure(g, p_min, n_pts, rng);
    const auto& rem_list = sf.rem_list;
    const int chunk = sf.chunk;
    std::cout << "  seed edges to remove: " << rem_list.size()
              << ", chunk=" << chunk << std::endl;

    std::vector<double> p_list = linspace(p_min, 1.0, n_pts);

    // Results (transposed: l_theta x n_pts, row-major flat)
    std::vector<double> S_results(l_theta * n_pts, 0.0);
    std::vector<double> t_results(l_theta * n_pts, 0.0);

    // Degree distributions: per (theta,n) store up to 100 degree values
    int max_k_pre = 100;
    std::vector<double> p_results(l_theta * n_pts * max_k_pre, 0.0);

    std::cout << "Running cascade simulations..." << std::endl;

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < l_theta; ++i) {
        double theta = theta_list[i];

        Graph cap_g = g;
        set_theta_weights(cap_g, theta);
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
        for (auto& c : capacity) c *= (1.0 + alpha);

        for (int n = 1; n <= n_pts; ++n) {
            Graph h = g;
            int n_rem = std::min(n * chunk, static_cast<int>(rem_list.size()));
            for (int r = 0; r < n_rem; ++r) {
                boost::remove_edge(rem_list[r].first, rem_list[r].second, h);
            }

            auto [h_final, t] = cascade(h, capacity, theta);
            int gc = giant_component_size(h_final);
            double S_val = static_cast<double>(gc) / boost::num_vertices(h_final);

            #pragma omp critical
            std::cout << "  \u03b8=" << theta << "  n=" << n << "/" << n_pts
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
    for (int i = 0; i < l_theta * n_pts; ++i) {
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
    std::string zs = fmt_double(zeta);
    std::string ks = fmt_double(k_avg);
    std::string as = fmt_double(alpha);
    std::string dir_suffix = "L = " + base + "/zeta = " + zs + ", k = " + ks + ", alpha = " + as + "/";

    fs::path csv_dir  = fs::path("results") / "CSV" / dir_suffix;
    fs::path plot_dir = fs::path("results") / "plots" / dir_suffix;
    fs::create_directories(csv_dir);
    fs::create_directories(plot_dir);

    // Build string header with theta column
    std::vector<std::string> hdr = {"theta"};
    for (double p : p_list) hdr.push_back(fmt_double(p));

    // Build theta string for filenames
    std::string tstr = fmt_double(theta_list[0]);
    for (int i = 1; i < l_theta; ++i)
        tstr += ", " + fmt_double(theta_list[i]);

    // Build S matrix with theta column
    std::vector<double> S_wth(l_theta * (n_pts + 1), 0.0);
    for (int i = 0; i < l_theta; ++i) {
        S_wth[i * (n_pts + 1)] = theta_list[i];
        for (int j = 0; j < n_pts; ++j)
            S_wth[i * (n_pts + 1) + 1 + j] = S_results[i * n_pts + j];
    }

    // Build t matrix with theta column
    std::vector<double> t_wth(l_theta * (n_pts + 1), 0.0);
    for (int i = 0; i < l_theta; ++i) {
        t_wth[i * (n_pts + 1)] = theta_list[i];
        for (int j = 0; j < n_pts; ++j)
            t_wth[i * (n_pts + 1) + 1 + j] = t_results[i * n_pts + j];
    }

    // Write S matrix
    {
        std::string fname = " Gt Component " + tstr + " zeta = " + zs + ".csv";
        write_csv(S_wth, l_theta, n_pts + 1, (csv_dir / fname).string(), hdr);
        std::cout << "  wrote " << fname << std::endl;
    }

    // Write t matrix
    {
        std::string fname = " Cascade Time " + tstr + " zeta = " + zs + ".csv";
        write_csv(t_wth, l_theta, n_pts + 1, (csv_dir / fname).string(), hdr);
        std::cout << "  wrote " << fname << std::endl;
    }

    // Write degree distributions per theta
    for (int i = 0; i < l_theta; ++i) {
        double theta = theta_list[i];
        std::ostringstream ts;
        ts << theta;
        std::string fname = "degree-distribution theta=" + ts.str() + ", zeta=" + zs + ".csv";

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

        // transpose to k_max x n_pts then add theta column
        std::vector<double> deg_out(k_max * (n_pts + 1), 0.0);
        for (int r = 0; r < k_max; ++r) {
            deg_out[r * (n_pts + 1)] = theta;
            for (int c = 0; c < n_pts; ++c)
                deg_out[r * (n_pts + 1) + 1 + c] = deg_mat[c * k_max + r];
        }

        write_csv(deg_out, k_max, n_pts + 1, (csv_dir / fname).string(), hdr);
    }

    std::cout << "Done. Results in " << csv_dir << std::endl;
    return 0;
}

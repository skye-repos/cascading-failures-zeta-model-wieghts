#include "graph_generators.h"
#include "simulate.h"
#include "util.h"
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

template <typename Graph>
static void run_simulation(const Graph& g, int n_pts, double alpha,
                           const std::vector<double>& theta_list,
                           const std::vector<double>& p_list,
                           DegreeMode deg_mode,
                           std::vector<double>& S_results,
                           std::vector<double>& t_results,
                           std::vector<double>& p_results,
                           int& k_max, std::mt19937& rng,
                           const std::string& out_label,
                           const SeedFailure& sf)
{
    const auto& rem_list = sf.rem_list;
    const int chunk = sf.chunk;
    int l_theta = static_cast<int>(theta_list.size());
    int max_k_pre = 100;

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < l_theta; ++i) {
        double theta = theta_list[i];

        Graph cap_g = g;
        set_theta_weights(cap_g, theta, deg_mode);
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
            for (int r = 0; r < n_rem; ++r)
                boost::remove_edge(rem_list[r].first, rem_list[r].second, h);

            auto [h_final, t] = cascade(h, capacity, theta, deg_mode);
            int gc = giant_component_size(h_final);
            double S_val = static_cast<double>(gc) / boost::num_vertices(h_final);

            #pragma omp critical
            std::cout << "  " << out_label << " \u03b8=" << theta
                      << "  n=" << n << "/" << n_pts
                      << "  t_step=" << t << "  S=" << S_val << std::endl;

            int row = n_pts - n;
            S_results[i * n_pts + row] = S_val;
            t_results[i * n_pts + row] = static_cast<double>(t);

            auto [k_vals, p_vals] = degree_distribution(h_final);
            int nk = std::min(static_cast<int>(p_vals.size()), max_k_pre);
            for (int j = 0; j < nk; ++j)
                p_results[(i * n_pts + row) * max_k_pre + j] = p_vals[j];
        }
    }

    k_max = 0;
    for (int i = 0; i < l_theta * n_pts; ++i) {
        for (int j = max_k_pre - 1; j >= 0; --j) {
            if (p_results[i * max_k_pre + j] != 0.0) {
                k_max = std::max(k_max, j + 1);
                break;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    int L = 50;
    double zeta = 15.0;
    double theta_min = -1.0;
    double theta_max = 1.0;
    int N_theta = 21;
    int n_pts = 50;
    std::string graph_path;
    bool directed = false;
    bool use_in_degree = false, use_out_degree = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](int i) { return i + 1 < argc ? argv[i + 1] : nullptr; };
        if ((arg == "-L" || arg == "--L") && next(i)) { L = std::stoi(argv[++i]); }
        else if ((arg == "-z" || arg == "--zeta") && next(i)) { zeta = std::stod(argv[++i]); }
        else if ((arg == "--tmin" || arg == "--theta-min") && next(i)) { theta_min = std::stod(argv[++i]); }
        else if ((arg == "--tmax" || arg == "--theta-max") && next(i)) { theta_max = std::stod(argv[++i]); }
        else if ((arg == "-n" || arg == "--N-theta") && next(i)) { N_theta = std::stoi(argv[++i]); }
        else if ((arg == "-p" || arg == "--n-pts") && next(i)) { n_pts = std::stoi(argv[++i]); }
        else if ((arg == "--graph") && next(i)) { graph_path = argv[++i]; }
        else if (arg == "--directed") { directed = true; }
        else if (arg == "--in-degree") { use_in_degree = true; }
        else if (arg == "--out-degree") { use_out_degree = true; }
        else {
            std::cerr << "Usage: " << argv[0] << " [options]\n"
                      << "  -L, --L L              system size (default 50)\n"
                      << "  -z, --zeta zeta        characteristic length (default 15)\n"
                      << "  --tmin, --theta-min t  lower theta bound (default -1)\n"
                      << "  --tmax, --theta-max t  upper theta bound (default 1)\n"
                      << "  -n, --N-theta n        number of theta points (default 21)\n"
                      << "  -p, --n-pts n          number of p values (default 50)\n"
                      << "  --graph <path>         load edge list from CSV (from,to)\n"
                      << "  --directed             treat graph as directed\n"
                      << "  --in-degree            use in-degree for theta (directed)\n"
                      << "  --out-degree           use out-degree for theta (directed)\n"
                      << "  theta=0 is excluded automatically.\n";
            return 1;
        }
    }

    if (use_in_degree && use_out_degree) {
        std::cerr << "--in-degree and --out-degree are exclusive\n";
        return 1;
    }
    if (!directed && (use_in_degree || use_out_degree))
        std::cout << "  warning: --in-degree/--out-degree ignored for undirected graph\n";

    const double k_avg = 5.0;
    const double alpha = 2.0;

    auto t0 = std::chrono::steady_clock::now();

    std::random_device rd;
    std::mt19937 rng(rd());

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

    double p_min = graph_path.empty() ? ((zeta / L < 0.01) ? 0.85 : 0.5) : 0.5;
    std::cout << "  p_min = " << p_min << std::endl;
    std::vector<double> p_list = linspace(p_min, 1.0, n_pts);

    int max_k_pre = 100;
    std::vector<double> S_results(l_theta * n_pts, 0.0);
    std::vector<double> t_results(l_theta * n_pts, 0.0);
    std::vector<double> p_results(l_theta * n_pts * max_k_pre, 0.0);
    int k_max = 0;

    DegreeMode deg_mode = DegreeMode::Total;
    if (use_in_degree)      deg_mode = DegreeMode::In;
    else if (use_out_degree) deg_mode = DegreeMode::Out;

    std::string out_label;
    std::string dir_suffix;

    auto do_run = [&]<typename G>(G& g, const std::string& label) {
        auto sf = seed_failure(g, p_min, n_pts, rng);
        std::cout << "  seed edges to remove: " << sf.rem_list.size()
                  << ", chunk=" << sf.chunk << std::endl;
        out_label = label;
        run_simulation(g, n_pts, alpha, theta_list, p_list, deg_mode,
                       S_results, t_results, p_results, k_max, rng, out_label, sf);
    };

    if (!graph_path.empty()) {
        auto stem = fs::path(graph_path).stem().string();
        if (directed) {
            std::cout << "Loading directed graph from " << graph_path << "..." << std::endl;
            DirectedGraph g = read_csv<boost::bidirectionalS>(graph_path);
            std::cout << "  Vertices: " << boost::num_vertices(g)
                      << ", Edges: " << boost::num_edges(g) << std::endl;
            dir_suffix = stem + "/directed/";
            do_run(g, stem + "_d");
        } else {
            std::cout << "Loading undirected graph from " << graph_path << "..." << std::endl;
            Graph g = read_csv<boost::undirectedS>(graph_path);
            std::cout << "  Vertices: " << boost::num_vertices(g)
                      << ", Edges: " << boost::num_edges(g) << std::endl;
            dir_suffix = stem + "/undirected/";
            do_run(g, stem + "_u");
        }
    } else {
        std::cout << "Generating zeta-model graph (L=" << L
                  << ", zeta=" << zeta << ", k_avg=" << k_avg << ")..." << std::endl;
        Graph g = zeta_taurus(L, zeta, k_avg, rng);
        std::cout << "  Vertices: " << boost::num_vertices(g)
                  << ", Edges: " << boost::num_edges(g) << std::endl;
        std::string base = std::to_string(L);
        std::string zs = fmt_double(zeta);
        std::string ks = fmt_double(k_avg);
        std::string as = fmt_double(alpha);
        dir_suffix = "L = " + base + "/zeta = " + zs + ", k = " + ks + ", alpha = " + as + "/";
        do_run(g, "zeta = " + zs);
    }

    fs::path csv_dir  = fs::path("results") / "CSV" / dir_suffix;
    fs::path plot_dir = fs::path("results") / "plots" / dir_suffix;
    fs::create_directories(csv_dir);
    fs::create_directories(plot_dir);

    std::vector<std::string> hdr = {"theta"};
    for (double p : p_list) hdr.push_back(fmt_double(p));

    std::string tstr = (l_theta > 1)
        ? fmt_double(theta_list[0]) + "_to_" + fmt_double(theta_list[l_theta - 1])
        : fmt_double(theta_list[0]);

    std::vector<double> S_wth(l_theta * (n_pts + 1), 0.0);
    for (int i = 0; i < l_theta; ++i) {
        S_wth[i * (n_pts + 1)] = theta_list[i];
        for (int j = 0; j < n_pts; ++j)
            S_wth[i * (n_pts + 1) + 1 + j] = S_results[i * n_pts + j];
    }

    std::vector<double> t_wth(l_theta * (n_pts + 1), 0.0);
    for (int i = 0; i < l_theta; ++i) {
        t_wth[i * (n_pts + 1)] = theta_list[i];
        for (int j = 0; j < n_pts; ++j)
            t_wth[i * (n_pts + 1) + 1 + j] = t_results[i * n_pts + j];
    }

    {
        std::string fname = " Gt Component " + tstr + " " + out_label + ".csv";
        write_csv(S_wth, l_theta, n_pts + 1, (csv_dir / fname).string(), hdr);
        std::cout << "  wrote " << fname << std::endl;
    }

    {
        std::string fname = " Cascade Time " + tstr + " " + out_label + ".csv";
        write_csv(t_wth, l_theta, n_pts + 1, (csv_dir / fname).string(), hdr);
        std::cout << "  wrote " << fname << std::endl;
    }

    for (int i = 0; i < l_theta; ++i) {
        double theta = theta_list[i];
        std::ostringstream ts;
        ts << theta;
        std::string fname = "degree-distribution theta=" + ts.str() + ", " + out_label + ".csv";

        std::vector<double> deg_mat(n_pts * k_max, 0.0);
        for (int n = 0; n < n_pts; ++n) {
            for (int j = 0; j < k_max; ++j) {
                double val = (j < max_k_pre)
                    ? p_results[(i * n_pts + n) * max_k_pre + j]
                    : 0.0;
                deg_mat[n * k_max + j] = val;
            }
        }

        std::vector<double> deg_out(k_max * (n_pts + 1), 0.0);
        for (int r = 0; r < k_max; ++r) {
            deg_out[r * (n_pts + 1)] = theta;
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

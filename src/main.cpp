#include "betweenness.h"
#include "graph_generators.h"
#include "simulate.h"
#include "util.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

static std::string fmt_double(double x) {
  std::ostringstream ss;
  ss << x;
  return ss.str();
}

static std::uint64_t splitmix64(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

int main(int argc, char *argv[]) {
  int L = 50;
  double ζ = 15.0;
  double θ_min = -1.0;
  double θ_max = 1.0;
  int N_θ = 21;
  int n_pts = 50;
  std::uint64_t base_seed = 12345;
  int n_realizations = 5;
  Topology topology = Topology::Waxman;
  bool verbose = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&](int i) { return i + 1 < argc ? argv[i + 1] : nullptr; };
    if ((arg == "-L" || arg == "--L") && next(i)) {
      L = std::stoi(argv[++i]);
    } else if ((arg == "-z" || arg == "--zeta") && next(i)) {
      ζ = std::stod(argv[++i]);
    } else if ((arg == "--tmin" || arg == "--theta-min") && next(i)) {
      θ_min = std::stod(argv[++i]);
    } else if ((arg == "--tmax" || arg == "--theta-max") && next(i)) {
      θ_max = std::stod(argv[++i]);
    } else if ((arg == "-n" || arg == "--N-theta") && next(i)) {
      N_θ = std::stoi(argv[++i]);
    } else if ((arg == "-p" || arg == "--n-pts") && next(i)) {
      n_pts = std::stoi(argv[++i]);
    } else if (arg == "--seed" && next(i)) {
      base_seed = std::stoull(argv[++i]);
    } else if (arg == "--realizations" && next(i)) {
      n_realizations = std::stoi(argv[++i]);
    } else if (arg == "--topology" && next(i)) {
      try {
        topology = parse_topology(argv[++i]);
      } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
      }
    } else if (arg == "--verbose") {
      verbose = true;
    } else {
      std::cerr
          << "Usage: " << argv[0] << " [options]\n"
          << "  -L, --L L              system size (default 50)\n"
          << "  -z, --zeta ζ           characteristic length (default 15)\n"
          << "  --tmin, --theta-min t  lower θ bound (default -1)\n"
          << "  --tmax, --theta-max t  upper θ bound (default 1)\n"
          << "  -n, --N-theta n        number of θ points (default 21)\n"
          << "  -p, --n-pts n          number of p values (default 50)\n"
          << "  --seed n               base seed (default 12345); realization r\n"
          << "                         uses splitmix64(base_seed + r + 1)\n"
          << "  --realizations n       number of realizations (default 5)\n"
          << "  --topology t           waxman or er (default waxman)\n"
          << "  --verbose              print per-(θ,p) progress\n"
          << "  θ=0 is excluded automatically.\n";
      return 1;
    }
  }
  const double k_avg = 5.0;
  const double α = 2.0;

  if (L <= 0 || N_θ <= 0 || n_pts <= 0 || n_realizations <= 0) {
    std::cerr << "L, N-theta, n-pts and realizations must be positive.\n";
    return 1;
  }

  auto t0 = std::chrono::steady_clock::now();

  std::vector<double> raw = linspace(θ_min, θ_max, N_θ);
  std::vector<double> θ_list;
  for (double t : raw) {
    if (std::abs(t) > 1e-9)
      θ_list.push_back(t);
  }
  int n_θ = static_cast<int>(θ_list.size());
  if (n_θ == 0) {
    std::cerr << "θ grid contains no non-zero values.\n";
    return 1;
  }

  double p_min = 0.5;
  if (topology == Topology::Waxman && ζ / L < 0.01) {
    p_min = 0.85;
  }

  std::vector<double> p_list = linspace(p_min, 1.0, n_pts);

  std::cout << "Topology: " << topology_name(topology) << ", L=" << L
            << ", ζ=" << ζ << ", k_avg=" << k_avg
            << ", realizations=" << n_realizations
            << ", base_seed=" << base_seed << std::endl;
  std::cout << "  θ values: " << n_θ << std::endl;
  std::cout << "  p_min = " << p_min << ", n_pts = " << n_pts << std::endl;

  std::string base = std::to_string(L);
  std::string ζs = fmt_double(ζ);
  std::string ks = fmt_double(k_avg);
  std::string αs = fmt_double(α);
  std::string topo_seg =
      (topology == Topology::ER) ? ", topology = ER" : "";
  std::string dir_suffix =
      "L = " + base + "/ζ = " + ζs + topo_seg + ", k = " + ks +
      ", α = " + αs + "/";

  fs::path csv_root = fs::path("results") / "CSV" / dir_suffix;
  fs::path plot_root = fs::path("results") / "plots" / dir_suffix;
  fs::create_directories(csv_root);
  fs::create_directories(plot_root);

  double dθ = (n_θ > 1) ? (θ_list[1] - θ_list[0]) : 0.0;
  double θ_first = θ_list.front();
  double θ_last = θ_list.back();
  std::string θ_meta = " Nθ=" + std::to_string(n_θ) + ", dθ=" + fmt_double(dθ) +
                       ", θ∈[" + fmt_double(θ_first) + "," +
                       fmt_double(θ_last) + "]";

  for (int r = 0; r < n_realizations; ++r) {
    std::uint64_t seed_r =
        splitmix64(base_seed + static_cast<std::uint64_t>(r) + 1);
    std::mt19937_64 rng(seed_r);

    std::cout << "Realization " << r << " (seed " << seed_r << ")..."
              << std::endl;

    Graph g = generate_graph(topology, L, ζ, k_avg, rng);
    int nv = static_cast<int>(boost::num_vertices(g));
    int n_e = static_cast<int>(boost::num_edges(g));
    if (n_e == 0) {
      std::cerr << "Generated graph has no edges; aborting realization.\n";
      continue;
    }
    std::cout << "  Vertices: " << nv << ", Edges: " << n_e << std::endl;

    SeedFailure sf = seed_failure(g, p_min, rng);
    const auto &rem_list = sf.rem_list;
    std::cout << "  seed edges available for removal: " << rem_list.size()
              << std::endl;

    std::vector<std::vector<double>> capacity(
        n_θ, std::vector<double>(nv, 0.0));

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n_θ; ++i) {
      BCMWorkspace ws;
      betweenness(g, θ_list[i], ws);
      capacity[i] = std::move(ws.bc);
      for (auto &c : capacity[i])
        c *= (1.0 + α);
    }

    std::vector<double> S_results(n_θ * n_pts, 0.0);
    std::vector<double> t_results(n_θ * n_pts, 0.0);

    int max_k_pre = 100;
    std::vector<double> p_results(n_θ * n_pts * max_k_pre, 0.0);

    std::cout << "  Running cascade simulations..." << std::endl;

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < n_θ; ++i) {
      for (int j = 0; j < n_pts; ++j) {
        double θ = θ_list[i];
        double p = p_list[j];
        int n_rem = static_cast<int>(std::llround((1.0 - p) * n_e));
        n_rem = std::clamp(n_rem, 0, static_cast<int>(rem_list.size()));

        Graph h = g;
        for (int k = 0; k < n_rem; ++k) {
          boost::remove_edge(rem_list[k].first, rem_list[k].second, h);
        }

        auto [h_final, t] = cascade(std::move(h), capacity[i], θ);
        int gc = giant_component_size(h_final);
        double S_val = static_cast<double>(gc) / boost::num_vertices(h_final);

        S_results[i * n_pts + j] = S_val;
        t_results[i * n_pts + j] = static_cast<double>(t);

        auto [k_vals, p_vals] = degree_distribution(h_final);
        int nk = std::min(static_cast<int>(p_vals.size()), max_k_pre);
        for (int k = 0; k < nk; ++k) {
          p_results[(i * n_pts + j) * max_k_pre + k] = p_vals[k];
        }

        if (verbose) {
#pragma omp critical
          std::cout << "  r=" << r << " θ=" << θ << "  p=" << p
                    << "  t_step=" << t << "  S=" << S_val << std::endl;
        }
      }
    }

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

    fs::path run_dir = csv_root / ("realization = " + std::to_string(r));
    fs::path plot_dir = plot_root / ("realization = " + std::to_string(r));
    fs::create_directories(run_dir);
    fs::create_directories(plot_dir);

    std::vector<std::string> hdr = {"θ"};
    for (double p : p_list)
      hdr.push_back(fmt_double(p));

    std::vector<double> S_wth(n_θ * (n_pts + 1), 0.0);
    for (int i = 0; i < n_θ; ++i) {
      S_wth[i * (n_pts + 1)] = θ_list[i];
      for (int j = 0; j < n_pts; ++j)
        S_wth[i * (n_pts + 1) + 1 + j] = S_results[i * n_pts + j];
    }

    std::vector<double> t_wth(n_θ * (n_pts + 1), 0.0);
    for (int i = 0; i < n_θ; ++i) {
      t_wth[i * (n_pts + 1)] = θ_list[i];
      for (int j = 0; j < n_pts; ++j)
        t_wth[i * (n_pts + 1) + 1 + j] = t_results[i * n_pts + j];
    }

    {
      std::string fname = " Gt Component" + θ_meta + " ζ = " + ζs + ".csv";
      write_csv(S_wth, n_θ, n_pts + 1, (run_dir / fname).string(), hdr);
      std::cout << "  wrote " << fname << std::endl;
    }

    {
      std::string fname = " Cascade Time" + θ_meta + " ζ = " + ζs + ".csv";
      write_csv(t_wth, n_θ, n_pts + 1, (run_dir / fname).string(), hdr);
      std::cout << "  wrote " << fname << std::endl;
    }

    for (int i = 0; i < n_θ; ++i) {
      double θ = θ_list[i];
      std::ostringstream ts;
      ts << θ;
      std::string fname =
          "degree-distribution θ=" + ts.str() + ", ζ=" + ζs + ".csv";

      std::vector<double> deg_mat(n_pts * k_max, 0.0);
      for (int j = 0; j < n_pts; ++j) {
        for (int k = 0; k < k_max; ++k) {
          double val = (k < max_k_pre)
                           ? p_results[(i * n_pts + j) * max_k_pre + k]
                           : 0.0;
          deg_mat[j * k_max + k] = val;
        }
      }

      std::vector<double> deg_out(k_max * (n_pts + 1), 0.0);
      for (int rr = 0; rr < k_max; ++rr) {
        deg_out[rr * (n_pts + 1)] = θ;
        for (int c = 0; c < n_pts; ++c)
          deg_out[rr * (n_pts + 1) + 1 + c] = deg_mat[c * k_max + rr];
      }

      write_csv(deg_out, k_max, n_pts + 1, (run_dir / fname).string(), hdr);
    }

    ManifestData manifest;
    manifest.base_seed = base_seed;
    manifest.realization_id = r;
    manifest.realization_seed = seed_r;
    manifest.L = L;
    manifest.zeta = ζ;
    manifest.topology = topology_name(topology);
    manifest.theta_min = θ_min;
    manifest.theta_max = θ_max;
    manifest.N_theta = N_θ;
    manifest.n_theta = n_θ;
    manifest.n_pts = n_pts;
    manifest.k_avg = k_avg;
    manifest.alpha = α;
    manifest.p_min = p_min;
    manifest.graph = graph_stats(g);
    write_manifest_json((run_dir / "run_manifest.json").string(), manifest);

    std::cout << "  done; results in " << run_dir << std::endl;
  }

  auto t1 = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed = t1 - t0;
  std::cout << "Done. Results in " << csv_root << std::endl;
  std::cout << "Total time: " << elapsed.count() << " s" << std::endl;
  return 0;
}

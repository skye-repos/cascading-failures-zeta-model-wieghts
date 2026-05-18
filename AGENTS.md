# AGENTS.md

## Project
C++ port of a Julia cascading-failure simulation (Motter–Lai model on ζ-model
spatial networks).  Research code — no test suite, no CI.

## Build
```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/cascade
```

Requires **Boost** (graph) and **OpenMP**.

macOS:
```sh
brew install boost libomp cmake
cmake -B build -DCMAKE_BUILD_TYPE=Release -DOpenMP_ROOT="$(brew --prefix libomp)"
```

Linux (x86_64):
```sh
# Option A — apt (Ubuntu/Debian)
sudo apt install build-essential cmake libboost-dev-all
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Option B — curl (any distro)
sudo apt install build-essential cmake
curl -L https://archives.boost.io/release/1.87.0/source/boost_1_87_0.tar.bz2 | tar -xj
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBOOST_ROOT=$PWD/boost_1_87_0
cmake --build build
```

## Source layout
| File | Role |
|------|------|
| `src/main.cpp` | Constants, parallel dispatch over θ, CSV + plot output |
| `src/graph_generators.h/cpp` | Graph type (`adjacency_list<vecS,vecS,undirectedS>` with `edge_weight_t`) + ζ_taurus, BA, ER generators |
| `src/simulate.h/cpp` | `seed_failure`, `set_theta_weights`, `cascade` (Motter–Lai with Brandes betweenness) |
| `src/util.h/cpp` | `degrees`, `degree_distribution`, `giant_component_size`, `write_csv` |
| `plot_results.py` | Standalone matplotlib post-processor — generates comparison plots + heatmaps from CSVs |

## Key differences from the original Julia code
- **Boost.Graph** replaces the custom `Graph`, Dijkstra, Brandes, and connected-components implementations
- **OpenMP** replaces Julia's `@distributed SharedArrays` for parallel θ scanning
- **`clear_vertex`** replaces the buggy inline neighbor-iteration loop in the cascade (Julia skipped ~half the neighbors)
- **Cascade step** recomputes Brandes betweenness each iteration (same as Julia — no incremental update)

## Output
CSVs are written to `results/CSV/L = …/zeta = …, k = …, alpha = …/`.
Plots are generated separately with `python plot_results.py`.

## Distributed caveats
- **Do not** use `vecS` for edge storage with undirected graphs if edge-descriptor stability across mutation matters. For this code, edge descriptors are only used ephemerally during the cascade where the graph is rebuilt each iteration — `vecS` is fine and gives better cache locality.
- Apple Clang does not support capturing structured bindings in OpenMP parallel regions.  Use explicit variables instead of `auto [a,b] = …`.
- `brandes_betweenness_centrality` throws on degenerate graphs.  The cascade wraps it in try/catch as a safety net.

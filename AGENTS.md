# AGENTS.md — Cascading Failures (C++ simulation)

This file onboards an AI coding agent to this repository. Its immediate purpose is
to implement the simulation changes required by the manuscript revision for
**EU13024** ("Controlling Robustness in Weighted Spatial Networks"). Read this
whole file before editing code.

---

## 1. What this project is

We simulate overload-induced cascading failures in spatially embedded networks
using the **Motter–Lai (ML)** model on **Waxman-ζ** graphs drawn on an `L × L`
torus, with a pure Erdős–Rényi (ER) graph as a non-scale-free control.
Link weights are degree-dependent:

    w_ij = (k_i * k_j)^theta        (theta > 0 => regime A, theta < 0 => regime B)

The order parameter is the giant-component fraction `S(p)`, where `p` is the
fraction of links surviving the initial random attack. Key observables are the
collapse threshold `p_c` and (under re-examination) an exponent `beta`.

The paper is under revision. The revision removes all transition-order claims
and requires **multiple realizations** and **finite-size scaling (FSS)**. Those
are the changes to implement here.

---

## 2. Repository layout

```
src/
  main.cpp              driver: arg parsing, realizations, θ/p sweep, CSV/manifest output (2-space indent)
  simulate.cpp/.h       seed_failure, set_θ_weights, cascade (4-space indent)
  graph_generators.cpp  zeta_taurus + pure ER builders (4-space indent)
  util.cpp/.h           degrees, degree_distribution, graph stats, manifest, CSV, linspace
aggregate.jl            post-processing: mean/SEM S(p), p_c, ΔS (Julia)
fss.jl                  finite-size scaling: β(L), ΔS(L), saturation report (Julia)
plot_results.jl         post-processing: comparison plots + heatmaps (Julia)
pc_utils.jl             shared p_c smoothing/jump routine (Julia)
test_pc.jl              synthetic-sigmoid unit test for p_c (Julia)
Project.toml            Julia environment for the post-processing scripts
Manifest.toml            resolved Julia package versions
run_tiers.sh            Tier 1/2/3 + ER orchestration (explicit seeds)
run.sh                  legacy example nohup launch lines
CMakeLists.txt          C++20, Boost headers, OpenMP
build/                  cmake build dir (gitignored)
nohup/                  captured stdout logs
results/CSV/...         simulation output (gitignored)
results/analysis/...    aggregated CSVs and FSS outputs (gitignored)
```

Use `Graph = boost::adjacency_list<vecS, vecS, undirectedS, no_property,
property<edge_weight_t, double>>` (defined in `graph_generators.h`). Shortest
paths / betweenness use these edge weights via Dijkstra underneath Brandes.

---

## 3. Build and run

```bash
cmake -B build
cmake --build build -j
./build/cascade -L 100 -z 15.0 -n 21 --seed 12345 --realizations 5
```

- Requires Boost headers and OpenMP.
- macOS deployment target is set to 14.0 in `CMakeLists.txt`.
- `--help` / bad flag prints usage.
- Julia post-processing uses `Project.toml` (run `julia --project=. -e
  'using Pkg; Pkg.instantiate()'` once):
  `julia --project=. test_pc.jl`,
  `julia --project=. aggregate.jl`, `julia --project=. fss.jl`,
  `julia --project=. plot_results.jl`.

### Current CLI flags (see `src/main.cpp`)
| Flag | Meaning | Default |
|---|---|---|
| `-L, --L` | lattice side, N = L² | 50 |
| `-z, --zeta` | characteristic link length | 15.0 |
| `--tmin/--tmax` | θ bounds | -1.0 / 1.0 |
| `-n, --N-theta` | number of θ points (θ=0 auto-excluded) | 21 |
| `-p, --n-pts` | number of p values | 50 |
| `--seed` | base seed; realization `r` uses `splitmix64(base_seed + r + 1)` | 12345 |
| `--realizations` | number of realizations | 5 |
| `--topology` | `waxman` or `er` | waxman |
| `--verbose` | print per-(θ,p) progress | off |

Fixed constants: `k_avg = 5.0`, `alpha = 2.0`.

### Current output
`results/CSV/L = {L}/ζ = {ζ}[, topology = ER], k = 5, α = 2/realization = {r}/`
containing:
- ` Gt Component Nθ=.., dθ=.., θ∈[..] ζ = ..csv` (rows = θ, cols = p; first col θ)
- ` Cascade Time Nθ=...csv`
- `degree-distribution θ=.., ζ=...csv` (one per θ)
- `run_manifest.json` (seeds, topology, θ/p grids, graph degree statistics)

The p header is the target surviving fraction `linspace(p_min, 1, n_pts)`;
`round((1-p)*n_e)` links are removed for each target p.
`julia --project=. plot_results.jl <csv_dir>` renders comparison/heatmap PNGs
into `results/plots/`.

---

## 4. Code conventions

- **Match local indentation per file** — `main.cpp` and `util.cpp` use 2 spaces;
  `simulate.cpp` and `graph_generators.cpp` use 4 spaces. Do not reformat files.
- Greek identifiers are intentional: `ζ`, `θ`, `α`, `dθ`. Keep them.
- No comments unless they already exist in the surrounding style.
- Free functions in `.h`/`.cpp` pairs; no classes.
- OpenMP parallelism is over independent `(θ, p)` cascade tasks
  (`#pragma omp parallel for collapse(2)`); capacities are precomputed per θ.
  No RNG is used inside parallel regions.
- Output filenames and directory names intentionally contain spaces and unicode;
  preserve the existing naming scheme so downstream tooling keeps working.

---

## 5. Current limitations (why this work is needed)

1. **Single realization.** `main.cpp` builds exactly one graph per invocation,
   seeded from `std::random_device` (`rng(rd())`). Runs are **not reproducible**
   and carry no error bars.
2. **No seed control.** There is no way to reproduce or to vary realizations.
3. **Single system size per study.** Existing data is `L = 100` only; no FSS.
4. **No topology control.** Only the Waxman-ζ generator exists; Referee 1 asks
   for degree-distribution statistics and a non-scale-free control.
5. Originally the pipeline was `(L, ζ) -> one CSV per θ` with no aggregation
   across realizations and no FSS/β stage. The Julia scripts `aggregate.jl` and
   `fss.jl` now provide both.

---

## 6. Required changes (task list)

Implement the simulation side of the revision plan. Keep changes minimal and
backward compatible where possible.

### Task 0 — Reproducible seeding
- Add CLI flags: `--seed <uint64>` (base seed) and `--realizations <int>`
  (default 5). Each realization uses
  `seed_r = splitmix64(base_seed + r + 1)`, so a run is fully reproducible.
- Thread the base seed into every stochastic step (graph generation and
  `seed_failure`). Currently these share one `rng`; keep that, but make it
  deterministic.
- Record `base_seed`, `realization_id`, `L`, `ζ`, `theta range`, `N_theta`,
  `n_pts`, `k_avg`, `alpha` into a machine-readable manifest (e.g.
  `run_manifest.json` or a `.txt` sidecar) in each output directory.

### Task 1 — Multiple realizations (Tier 1)
- Loop over `n_realizations`; for each, generate a fresh graph and run the full
  (ζ, θ, p) sweep.
- Write per-realization outputs (do **not** silently average in C++). Suggested
  layout: insert a `realization = <r>` path segment, e.g.
  `results/CSV/L = 100/ζ = .., k = 5, α = 2/realization = 0/...`. Keep the
  existing filename scheme inside.
- Tier 1 configuration: **L = 100**, reduced ζ set of ~8–10 values, θ ∈ [-1, 1]
  with Δθ = 0.1 (`-n 21`), **5 realizations**.

### Task 2 — Finite-size scaling (Tier 2)
- Add a mode/script to run **fixed ζ/L ratios** at ratios `{0.1, 0.25, 0.5}`
  for `L ∈ {25, 50, 75, 100}`, 5 realizations. Run Tier 2 at L = 25, 50, 75
  with θ ∈ {-0.4, -0.2, 0.2, 0.4}; the L = 100 points are reused from Tier 1,
  whose 21-point θ grid contains those values. This avoids overwriting the
  Tier-1 L = 100 outputs.
- Derive `ζ = ratio * L` in the orchestration layer (no code change needed if
  the driver already takes `-z`).
- Because ζ/L is fixed, ζ scales with L. Confirm the graph generator handles
  `ζ/L = 0.5` at L = 25 (ζ = 12.5) and small-`L` p_min logic
  (`p_min = (ζ/L < 0.01 for Waxman) ? 0.85 : 0.5`; ER forces 0.5).
- Restrict the FSS comparison to the common θ values and ratios:
  `julia --project=. fss.jl --ratios 0.1,0.25,0.5 --thetas -0.4,-0.2,0.2,0.4`.
- Provide an **orchestration script** (`run_tiers.sh`) that enumerates
  Tier 1/2/3 runs with explicit `--seed`/`--realizations` and writes
  `nohup/`-style logs.

### Task 3 — Regression stability (Tier 3)
- Repeat the full ζ scan at **L = 100** (same reduced ζ set as Tier 1), 5
  realizations, and confirm the CSVs are byte-identical to Tier 1.
- This repeat is opt-in (`./run_tiers.sh 3`) and is not part of `all`, since
  L = 100 is already the Tier-1 maximum.

### Task 4 — Degree-distribution / topology control (Referee 1 #5)
- Ensure `degree_distribution` output is retained, and add summary statistics
  (mean, variance, and optionally the second moment) to the manifest or a
  dedicated CSV.
- Add at least one **non-scale-free control** generator in
  `graph_generators.cpp`: a pure ER (Poisson) control. Add its selector to the
  CLI (`--topology waxman|er`) and document it. The Gnutella scale-free case
  lives in the separate real-data repo (`1d - cpp-real-data-final`) — do not
  duplicate it here.

### Task 5 — Aggregation and β/FSS analysis (Julia, new)
- Add `aggregate.jl` that reads per-realization CSVs and produces, per
  (L, ζ, θ):
  - `S(p)` mean and standard error across realizations,
  - `p_c` estimate with uncertainty,
  - the jump `ΔS(L)`.
- `p_c` is the point where the second-order finite difference has the largest
  jump. Smooth `S(p)` along the p axis first with a Gaussian kernel whose width
  is `sigma_pts * Δp` (default 2 grid points), then compute second differences
  and select the largest sign-change jump. Bootstrap over realizations for the
  uncertainty. `test_pc.jl` validates this routine on a synthetic sigmoid with
  a known transition.
- Add `fss.jl` that:
  - fits `S(p) - S(p_c) = k (p - p_c)^beta` **only in the fixed window
    `p_c < p ≤ p_c + 0.1`** (never over all `[p_c, 1]`); report the window and
    fit quality,
  - computes `beta(L)` and `ΔS(L)` across the five sizes and reports whether each
    saturates with L.
- **Decision rule:** if `beta(L)` is constant within error, report the single
  universal value and treat any ζ,θ dependence as a finite-size artifact; if it
  drifts without saturation, report β as unresolved. Never force a β(ζ,θ) form.
- All post-processing is Julia; `plot_results.jl` replaces the old Python
  plotting script. Aggregated CSVs and FSS outputs go under `results/analysis/`.

---

## 7. Definition of done

- [ ] Builds clean with `cmake --build build -j`.
- [ ] Same CLI invocation with the same `--seed` reproduces byte-identical CSVs.
- [ ] `--realizations 5` produces 5 distinct, correctly separated output sets.
- [ ] Tier 1/2/3 orchestration script exists and its parameters match Section 6.
- [ ] Aggregation produces `S(p)` mean ± SE and `p_c ± err` for every point.
- [ ] FSS script reports `beta(L)` and `ΔS(L)` for `L ∈ {25,50,75,100}`.
- [ ] Manifest records seed, L, ζ, θ grid, n_pts, k_avg, alpha, realization id.
- [ ] No change to the on-disk CSV column format consumed by `plot_results.jl`
      and the downstream fitting code (see `0 - simulations/2a - fitting`).
- [ ] `julia --project=. test_pc.jl` passes.

## 8. Verification

- Reproduce a tiny run: `./build/cascade -L 25 -z 2.5 -n 3 -p 5 --seed 1
  --realizations 2` and confirm the two realizations differ but rerunning with
  `--seed 1` reproduces them.
- Sanity-check `S(p)` is monotonically increasing in `p` on average and that
  `p_c` estimates move smoothly with ζ.
- Run `julia --project=. test_pc.jl`, `julia --project=. aggregate.jl`, and
  `julia --project=. fss.jl --ratios 0.1,0.25,0.5 --thetas -0.4,-0.2,0.2,0.4`
  on the results, then `julia --project=. plot_results.jl <csv_dir>` on a fresh
  output to confirm the analysis pipeline works.

## 9. Do / Don't

- **Do** keep the `(θ, p)` CSV layout and unicode directory names unchanged.
- **Do** keep per-realization data; aggregate in Julia, not in C++.
- **Do** use `std::mt19937_64` consistently.
- **Don't** refactor unrelated code or reformat files.
- **Don't** add C++ dependencies beyond what CMake already finds
  (Boost and OpenMP). Julia post-processing uses the packages in `Project.toml`
  (DataFrames, CSV, Plots).
- **Don't** remove the degree-distribution output.

## 10. Cost notes

Betweenness centrality dominates runtime (O(NM) per cascade step, repeated per
θ). L = 100 is the configured ceiling (N = 10,000). The tiered design (full
sweep at one L, FSS at few ζ/L ratios) exists to keep the total cost bounded.
Each OpenMP thread copies the graph per (θ, p) — watch peak memory.

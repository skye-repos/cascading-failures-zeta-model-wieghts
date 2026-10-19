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
  simulate.cpp/.h       seed_failure, cascade (4-space indent)
  betweenness.cpp/.h    CSR Brandes BC with reusable workspace (4-space indent)
  graph_generators.cpp  zeta_taurus + pure ER builders (4-space indent)
  util.cpp/.h           degrees, degree_distribution, graph stats, manifest, CSV, linspace
analysis/
  main.jl               single post-processing entrypoint / CLI driver
  Aggregate.jl          mean/SEM S(p), p_c, ΔS, degree summaries (Julia)
  FSS.jl                finite-size scaling: β(L), ΔS(L), saturation (Julia)
  PlotResults.jl        per-realization + summary plots (Julia)
  PCUtils.jl            shared p_c smoothing/jump routine (Julia)
  test_pc.jl            synthetic-sigmoid unit test for p_c (Julia)
  Project.toml          Julia environment for post-processing
  Manifest.toml         resolved Julia package versions
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
cmake --build build --parallel "$(nproc)"
OMP_NUM_THREADS="$(nproc)" ./build/cascade -L 100 -z 15.0 -n 21 \
    --seed 12345 --realizations 10
```

- Requires Boost headers and OpenMP.
- CMake defaults to a `Release` build, enables native CPU tuning
  (`-march=native`/`-mcpu=native`) and LTO. Disable them with
  `-DNATIVE_OPT=OFF` or `-DENABLE_LTO=OFF` if portability is required.
- `run_tiers.sh` defaults `OMP_NUM_THREADS` to the physical-core count and sets
  `OMP_PROC_BIND=spread`, `OMP_DYNAMIC=false`, and `OMP_WAIT_POLICY=active`;
  override these environment variables as needed (e.g. use logical cores to
  test hyperthreading).
- On multi-socket Xeons, set the performance CPU governor and consider
  `numactl --interleave=all ./build/cascade ...` to spread the shared graph
  across memory nodes.
- macOS deployment target is set to 14.0 in `CMakeLists.txt`.
- `--help` / bad flag prints usage.
- Julia post-processing lives in `analysis/` and uses `analysis/Project.toml`
  (run `julia --project=analysis -e 'using Pkg; Pkg.instantiate()'` once):
  - `julia --project=analysis analysis/main.jl test`
  - `julia --project=analysis analysis/main.jl aggregate`
  - `julia --project=analysis analysis/main.jl fss`
  - `julia --project=analysis analysis/main.jl plots`
  - `julia --project=analysis analysis/main.jl beta-zeta`
  - `julia --project=analysis analysis/main.jl all` (aggregate + fss + summary plots + β(ζ) test)
  - Run `all` (or `aggregate`) after new Tier 2 results; `fss` and `plots`
    reuse existing aggregate outputs when present.

### Current CLI flags (see `src/main.cpp`)
| Flag | Meaning | Default |
|---|---|---|
| `-L, --L` | lattice side, N = L² | 50 |
| `-z, --zeta` | characteristic link length | 15.0 |
| `--tmin/--tmax` | θ bounds | -1.0 / 1.0 |
| `-n, --N-theta` | number of θ points (θ=0 auto-excluded) | 21 |
| `-p, --n-pts` | number of p values | 50 |
| `--seed` | base seed; realization `r` uses `splitmix64(base_seed + r + 1)` | 12345 |
| `--realizations` | number of realizations | 10 |
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
`julia --project=analysis analysis/main.jl plots --plots first` renders
per-realization comparison/heatmap PNGs into `results/plots/`; summary plots
go to `results/plots/summary/`.

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
   across realizations and no FSS/β stage. The `analysis/` Julia modules now
   provide both through `analysis/main.jl`.

---

## 6. Required changes (task list)

Implement the simulation side of the revision plan. Keep changes minimal and
backward compatible where possible.

### Task 0 — Reproducible seeding
- Add CLI flags: `--seed <uint64>` (base seed) and `--realizations <int>`
  (default 10). Each realization uses
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
  with Δθ = 0.1 (`-n 21`), **10 realizations**.

### Task 2 — Finite-size scaling (Tier 2)
- Add a mode/script to run **fixed ζ/L ratios** at 10 ratios
  `{0.05, 0.10, …, 0.50}` for `L ∈ {20, 40, 60, 80, 100}`, 10 realizations.
  Run Tier 2 at L = 20, 40, 60, 80 with the same 21-point θ grid as L = 100;
  the L = 100 points are reused from Tier 1. This matches Tier 1's 10 ζ values
  at L = 100 and avoids overwriting the Tier-1 outputs.
- Derive `ζ = ratio * L` in the orchestration layer (no code change needed if
  the driver already takes `-z`).
- Because ζ/L is fixed, ζ scales with L. Confirm the graph generator handles
  `ζ/L = 0.5` at L = 20 (ζ = 10) and small-`L` p_min logic
  (`p_min = (ζ/L < 0.01 for Waxman) ? 0.85 : 0.5`; ER forces 0.5).
- The analysis defaults to `L = 20,40,60,80,100` and uses all available
  ratios/θ values; use `--L all` or `--ratios`/`--thetas` to override.
- Provide an **orchestration script** (`run_tiers.sh`) that enumerates
  Tier 1/2/3 runs with explicit `--seed`/`--realizations` and writes
  `nohup/`-style logs.

### Task 3 — Regression stability (Tier 3)
- Repeat the full ζ scan at **L = 100** (same reduced ζ set as Tier 1), 10
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
- `analysis/main.jl` is the single entrypoint. It calls:
  - `analysis/Aggregate.jl`, which reads per-realization CSVs and produces, per
    (L, ζ, θ): `S(p)` mean and standard error, `p_c` with uncertainty, and
    `ΔS(L)`;
  - `analysis/FSS.jl`, which fits `S(p) - S(p_c) = k (p - p_c)^beta` only in
    the fixed window `p_c < p ≤ p_c + 0.1` (never over all `[p_c, 1]`), reports
    the window and fit quality, and computes `beta(L)` and `ΔS(L)` separately
    for the θ < 0 and θ > 0 regimes;
  - `analysis/PlotResults.jl`, which produces per-realization plots and summary
    plots.
- `p_c` is the point where the second-order finite difference has the largest
  jump. Smooth `S(p)` along the p axis first with a Gaussian kernel whose width
  is `sigma_pts * Δp` (default 2 grid points), then compute second differences
  and select the largest sign-change jump. Bootstrap over realizations for the
  uncertainty. `analysis/test_pc.jl` validates this routine on a synthetic
  sigmoid with a known transition.
- **Decision rule:** if `beta(L)` is constant within error, report the single
  universal value and treat any ζ,θ dependence as a finite-size artifact; if it
  drifts without saturation, report β as unresolved. Never force a β(ζ,θ) form.
- All post-processing is Julia and lives in `analysis/`. Aggregated CSVs and
  FSS outputs go under `results/analysis/`; summary plots go under
  `results/plots/summary/`. FSS outputs carry a `regime` column and are also
  written as `beta_vs_L_negative.csv`, `beta_vs_L_positive.csv`, and the
  corresponding `delta_S_vs_L_*` files; `fss_report.txt` contains separate
  θ < 0 and θ > 0 sections. Plotting also writes `delta_S_over_sqrtL.csv`,
  ΔS/√L plots, a fitted ΔS scaling collapse (`delta_S_scaling_fit.csv`,
  `delta_S_scaling.csv`, `delta_S_scaling_{negative,positive}.png`), and a
  ζ-dependent power-law fit (`delta_S_scaling_zeta_fit.csv`,
  `delta_S_scaling_zeta.csv`, `delta_S_scaling_zeta_{negative,positive}.png`).
- `beta-zeta` refits all available (ζ, θ) pairs and tests β against ζ at fixed
  L with a weighted slope and χ² heterogeneity test. Outputs:
  `beta_vs_zeta.csv`, `beta_vs_ratio.csv`, `beta_zeta_test.csv`,
  `beta_vs_zeta_{negative,positive}.png`,
  `beta_vs_zeta_over_L_{negative,positive}.png`,
  `beta_vs_L_over_zeta_{negative,positive}.png`, and per-ζ negative-regime
  plots `beta_scaling_zeta=<ζ>_negative.png`.

---

## 7. Definition of done

- [ ] Builds clean with `cmake --build build -j`.
- [ ] Same CLI invocation with the same `--seed` reproduces byte-identical CSVs.
- [ ] `--realizations 10` produces 10 distinct, correctly separated output sets.
- [ ] Tier 1/2/3 orchestration script exists and its parameters match Section 6.
- [ ] Aggregation produces `S(p)` mean ± SE and `p_c ± err` for every point.
- [ ] FSS script reports `beta(L)` and `ΔS(L)` for `L ∈ {20,40,60,80,100}`.
- [ ] Manifest records seed, L, ζ, θ grid, n_pts, k_avg, alpha, realization id.
- [ ] No change to the on-disk CSV column format consumed by
      `analysis/PlotResults.jl` and the downstream fitting code
      (see `0 - simulations/2a - fitting`).
- [ ] `julia --project=analysis analysis/main.jl test` passes.

## 8. Verification

- Reproduce a tiny run: `./build/cascade -L 25 -z 2.5 -n 3 -p 5 --seed 1
  --realizations 10` and confirm the realizations differ but rerunning with
  `--seed 1` reproduces them.
- Sanity-check `S(p)` is monotonically increasing in `p` on average and that
  `p_c` estimates move smoothly with ζ.
- Run `julia --project=analysis analysis/main.jl test`,
  `julia --project=analysis analysis/main.jl aggregate`, and
  `julia --project=analysis analysis/main.jl fss`
  on the results, then `julia --project=analysis analysis/main.jl plots --plots first`
  and `julia --project=analysis analysis/main.jl beta-zeta` to confirm the
  analysis pipeline works. Confirm `fss_report.txt` has separate θ < 0 and
  θ > 0 sections, that the per-regime β/ΔS CSVs and plots exist, and that
  `beta_zeta_test.csv` reports the β(ζ) heterogeneity test.

## 9. Do / Don't

- **Do** keep the `(θ, p)` CSV layout and unicode directory names unchanged.
- **Do** keep per-realization data; aggregate in Julia, not in C++.
- **Do** use `std::mt19937_64` consistently.
- **Don't** refactor unrelated code or reformat files.
- **Don't** add C++ dependencies beyond what CMake already finds
  (Boost and OpenMP). Julia post-processing uses the packages in
  `analysis/Project.toml` (DataFrames, CSV, Plots).
- **Don't** remove the degree-distribution output.

## 10. Cost notes

Betweenness centrality dominates runtime (O(NM) per cascade step, repeated per
θ). `betweenness.cpp` implements Brandes directly on a CSR view with reusable
workspace buffers; it is byte-identical to Boost's implementation on the
regression cases. `betweenness()` recomputes degrees and weights from the
current graph on every call, and `cascade()` calls it once per time step. L = 100
is the configured ceiling (N = 10,000). The tiered design (full sweep at one L,
FSS at 10 ζ/L ratios with the full 21-point θ grid) keeps the total cost bounded
but is still substantially larger than the earlier 3-ratio/4-θ Tier 2 design.
Each OpenMP thread copies the graph per (θ, p) — watch peak memory.

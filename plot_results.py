#!/usr/bin/env python3
"""
Post-processing script for cascading failure simulation results.

Reads CSV files produced by cascade and generates comparison plots and
heatmaps matching the Julia output format.

Usage:
    python plot_results.py <results_dir>

The results_dir should be the parameter-specific subdirectory, e.g.:
    results/CSV/L = 50/zeta = 15, k = 5, alpha = 2/
"""
import sys
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_results(results_dir):
    """Read all CSV files in results_dir. Returns dict of
    {filename: (theta_col, data_matrix)} where data_matrix excludes
    the theta column."""
    data = {}
    for fpath in sorted(glob.glob(os.path.join(results_dir, "*.csv"))):
        fname = os.path.basename(fpath)
        with open(fpath) as f:
            header = f.readline().strip().split(",")
        arr = np.loadtxt(fpath, delimiter=",", skiprows=1)
        # First column is theta
        theta_col = arr[:, 0] if arr.ndim > 1 else np.array([arr[0]])
        data_mat = arr[:, 1:] if arr.ndim > 1 else arr[1:]
        data[fname] = (theta_col, data_mat, header)
    return data


def plot_comparison(mat, theta_list, p_list, ylabel, title, outpath):
    """Line plot: each line is a theta value."""
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.viridis
    colors = [cmap(z) for z in np.linspace(0, 1, len(theta_list))]

    for i, (theta, color) in enumerate(zip(theta_list, colors)):
        ax.plot(p_list, mat[i, :], color=color, lw=0.75,
                label=f"\u03b8={theta:.2f}")

    ax.set_xlabel("p")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="lower right", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_heatmap(mat, theta_list, p_list, title, outpath, xlabel="p", ylabel="\u03b8"):
    """Heatmap of S over (p, theta)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(mat, aspect="auto", origin="lower",
                   extent=[p_list[0], p_list[-1], theta_list[0], theta_list[-1]],
                   cmap="viridis")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_degree_heatmaps(data, p_list, plots_dir):
    """For each theta, plot degree distribution heatmap."""
    for key in data:
        if "degree-distribution" not in key:
            continue
        theta_col, mat, header = data[key]
        # mat is k_max x n_pts (rows=degree, cols=p)
        theta = theta_col[0]
        k_vals = np.arange(1, mat.shape[0] + 1)
        pk_mat = mat.T  # n_pts x k_max

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(pk_mat, aspect="auto", origin="lower",
                       extent=[k_vals[0], k_vals[-1], p_list[0], p_list[-1]],
                       cmap="viridis")
        ax.set_xlabel("degree k")
        ax.set_ylabel("p")
        ax.set_title(f"\u03b8={theta:.2f}, Heatmap of final p\u2096")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()

        safe_name = f"degree-heatmap_theta={theta:.2f}.png"
        fig.savefig(os.path.join(plots_dir, safe_name), dpi=150)
        plt.close(fig)


def process_one_dir(csv_dir):
    """Generate plots for a single results directory."""
    plots_dir = csv_dir.replace("/CSV/", "/plots/", 1)
    os.makedirs(plots_dir, exist_ok=True)

    data = parse_results(csv_dir)
    if not data:
        print(f"  No CSV files found in {csv_dir}")
        return

    # Identify combined CSVs by filename
    S_key = [k for k in data if "Gt Component" in k]
    t_key = [k for k in data if "Cascade Time" in k]
    if not S_key:
        print(f"  No giant component CSV found in {csv_dir}")
        return

    # Read header for p_list (skip "theta" column)
    _, _, header = data[S_key[0]]
    p_list = np.array([float(h) for h in header[1:]])

    # Theta and data from combined CSVs
    theta_col_S, S_mat, _ = data[S_key[0]]
    theta_list = theta_col_S

    t_mat = None
    if t_key:
        _, t_mat, _ = data[t_key[0]]

    # -- Comparison plots --
    plot_comparison(S_mat, theta_list, p_list,
                    ylabel="S",
                    title="Giant component",
                    outpath=os.path.join(plots_dir, "comparison_giant_component.png"))

    if t_mat is not None:
        plot_comparison(t_mat, theta_list, p_list,
                        ylabel="Cascade Time",
                        title="Cascade Time",
                        outpath=os.path.join(plots_dir, "comparison_cascade_time.png"))

    # -- Heatmaps --
    plot_heatmap(S_mat, theta_list, p_list,
                 title="GC heatmap",
                 outpath=os.path.join(plots_dir, "heatmap_giant_component.png"))

    # -- Degree distribution heatmaps --
    plot_degree_heatmaps(data, p_list, plots_dir)

    print(f"  Plots written to {plots_dir}")


def main():
    if len(sys.argv) < 2:
        base = "results/CSV"
        available = sorted(glob.glob(os.path.join(base, "*", "*")))
        if not available:
            print("No results found in results/CSV/.")
            print("Run ./build/cascade first, or specify a results directory:")
            print("  e.g. results/CSV/L = .../zeta = ..., k = ..., alpha = .../")
            print("  e.g. results/CSV/<graph_name>/directed/")
            sys.exit(1)
        for csv_dir in available:
            process_one_dir(csv_dir)
    else:
        csv_dir = sys.argv[1]
        if not os.path.isdir(csv_dir):
            print(f"Directory not found: {csv_dir}")
            sys.exit(1)
        process_one_dir(csv_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot cascade duration vs probability p from CSV outputs.

Scans subdirectories under the CSV directory for files named "Cascade Time*.csv". Each file
has first row as p values and subsequent rows as cascade durations (one trial per row).

Produces:
 - plots/cascade_duration_vs_p_all_zeta.png  (all zeta curves)
 - plots/<sanitized_zeta>_cascade_duration_vs_p.png (per-zeta plot)

Usage:
    python3 scripts/plot_cascade_duration.py --csv-dir CSV --out-dir plots

"""

import argparse
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sanitize(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_.-]", "_", name)


def read_cascade_csv(path: Path):
    # no header, first row is p-values
    df = pd.read_csv(path, header=None)
    p = df.iloc[0].astype(float).values
    durations = df.iloc[1:].astype(float).values
    return p, durations


def parse_theta_list_from_filename(path: Path):
    """Extract theta values (floats) in order from the filename.

    e.g. "Cascade Time 0.05, 0.2, 0.35.csv" -> [0.05, 0.2, 0.35]
    Falls back to extracting any floats from the basename if the exact
    pattern isn't present.
    """
    name = path.name
    # try to find substring after 'Cascade Time'
    m = re.search(r'Cascade\s*Time(.*)\.csv', name, flags=re.IGNORECASE)
    fragment = m.group(1) if m else name
    nums = re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', fragment)
    return [float(x) for x in nums]


def plot_csv_by_theta(path: Path, out_dir: Path, mapping='auto'):
    """Plot a single CSV file with one curve per theta parsed from filename.

    mapping: 'auto'|'contiguous'|'interleaved'
      - auto: if n_rows == n_theta -> direct mapping; elif n_rows % n_theta ==0 -> contiguous grouping; else -> interleaved
      - contiguous: group rows into contiguous blocks for each theta
      - interleaved: group rows by interleaving (rows i, i+n_theta, ...)
    """
    p, durations = read_cascade_csv(path)
    theta_list = parse_theta_list_from_filename(path)
    n_theta = len(theta_list)
    n_rows = durations.shape[0]

    if n_theta == 0:
        raise RuntimeError(f"No theta values parsed from filename: {path.name}")

    # Decide grouping strategy
    if mapping == 'auto':
        if n_rows == n_theta:
            strategy = 'direct'
        elif n_rows % n_theta == 0:
            strategy = 'contiguous'
        else:
            strategy = 'interleaved'
    else:
        strategy = mapping

    groups = []
    if strategy == 'direct':
        for i in range(n_theta):
            groups.append(durations[i:i+1])
    elif strategy == 'contiguous':
        trials_per = n_rows // n_theta
        for i in range(n_theta):
            groups.append(durations[i*trials_per:(i+1)*trials_per])
    elif strategy == 'interleaved':
        for i in range(n_theta):
            groups.append(durations[i::n_theta])
    else:
        raise ValueError(f"Unknown mapping strategy: {mapping}")

    fig, ax = plt.subplots(figsize=(8, 6))
    for theta, grp in zip(theta_list, groups):
        mean = np.nanmean(grp, axis=0)
        std = np.nanstd(grp, axis=0)
        label = f"θ={theta}"
        ax.plot(p, mean, marker='o', label=label)
        if np.any(std > 0):
            ax.fill_between(p, mean - std, mean + std, alpha=0.15)

    ax.set_xlabel('Probability p')
    ax.set_ylabel('Cascade duration (time steps)')
    ax.set_title(f'{path.name}')
    ax.grid(True)
    ax.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{sanitize(path.stem)}_time_vs_p.png"
    fig.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    return out_file


def plot_series(p, mean, std, label=None, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.plot(p, mean, marker='o', label=label)
    ax.fill_between(p, mean - std, mean + std, alpha=0.2)
    return ax


def main(csv_dir: Path, out_dir: Path, save_individual=True, plot_per_csv=True, row_mapping='auto'):
    csv_dir = Path(csv_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list(csv_dir.rglob('Cascade Time*.csv'))
    if not files:
        print(f"No Cascade Time CSV files found under {csv_dir}")
        return 1

    combined_fig, combined_ax = plt.subplots(figsize=(8, 6))

    for f in sorted(files):
        try:
            p, durations = read_cascade_csv(f)
        except Exception as e:
            print(f"Failed to read {f}: {e}")
            continue

        mean = np.nanmean(durations, axis=0)
        std = np.nanstd(durations, axis=0)

        # try extract a label (e.g., folder name which contains zeta value)
        label = f.parent.name
        plot_series(p, mean, std, label=label, ax=combined_ax)

        if save_individual:
            fig, ax = plt.subplots(figsize=(8, 6))
            plot_series(p, mean, std, label=label, ax=ax)
            ax.set_xlabel('Probability p')
            ax.set_ylabel('Mean cascade duration (time steps)')
            ax.set_title(f'Cascade duration vs p — {label}')
            ax.grid(True)
            ax.legend()
            out_file = out_dir / f"{sanitize(label)}_cascade_duration_vs_p.png"
            fig.tight_layout()
            fig.savefig(out_file, dpi=200)
            plt.close(fig)

        if plot_per_csv:
            per_csv_out = out_dir / 'per_csv'
            try:
                out_file = plot_csv_by_theta(f, per_csv_out, mapping=row_mapping)
                print(f"Saved per-file plot: {out_file}")
            except Exception as e:
                print(f"Failed to plot per-csv for {f}: {e}")

    combined_ax.set_xlabel('Probability p')
    combined_ax.set_ylabel('Mean cascade duration (time steps)')
    combined_ax.set_title('Cascade duration vs p (multiple ζ)')
    combined_ax.grid(True)
    combined_ax.legend()
    combined_fig.tight_layout()
    combined_fig.savefig(out_dir / 'cascade_duration_vs_p_all_zeta.png', dpi=200)
    plt.close(combined_fig)
    print(f"Saved plots to {out_dir}")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-dir', default='CSV', help='Directory containing CSV subfolders')
    parser.add_argument('--out-dir', default='plots', help='Directory to save plots')
    parser.add_argument('--no-individual', dest='save_individual', action='store_false', help='Do not save per-folder plots')
    parser.add_argument('--csv-file', dest='csv_file', default=None, help='Path to a single Cascade Time CSV to plot per-theta')
    parser.add_argument('--no-per-csv', dest='plot_per_csv', action='store_false', help='Do not save per-CSV theta plots')
    parser.add_argument('--row-mapping', choices=['auto', 'contiguous', 'interleaved'], default='auto', help='How to map rows to theta values when counts differ')
    args = parser.parse_args()

    # If a single csv file is provided, just plot it and exit
    if args.csv_file:
        out = plot_csv_by_theta(Path(args.csv_file), Path(args.out_dir), mapping=args.row_mapping)
        print(f"Saved plot for {args.csv_file} -> {out}")
        raise SystemExit(0)

    raise SystemExit(main(args.csv_dir, args.out_dir, save_individual=args.save_individual, plot_per_csv=args.plot_per_csv, row_mapping=args.row_mapping))

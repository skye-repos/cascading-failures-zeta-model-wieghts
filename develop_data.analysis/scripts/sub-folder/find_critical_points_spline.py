#!/usr/bin/env python3
"""For each "Gt Component" CSV under the CSV directory this script: (original docstring trimmed)"""

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar


def sanitize(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_.=-]", "_", name)


def parse_theta_list_from_filename(path: Path):
    name = path.name
    m = re.search(r'Gt Component(.*)\.csv', name, flags=re.IGNORECASE)
    fragment = m.group(1) if m else name
    nums = re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', fragment)
    return [float(x) for x in nums]


def extract_L_zeta_from_parent(parent_name: str):
    mL = re.search(r'L\s*=\s*([0-9]+)', parent_name)
    mz = re.search(r'ζ\s*=\s*([0-9\.]+)|zeta\s*=\s*([0-9\.]+)', parent_name)
    L = mL.group(1) if mL else parent_name
    z = (mz.group(1) if mz and mz.group(1) is not None else (mz.group(2) if mz and mz.group(2) is not None else 'unknown'))
    return f"L={L}", f"zeta={z}"


def read_gt_csv(path: Path):
    df = pd.read_csv(path, header=None)
    p = df.iloc[0].astype(float).values
    rows = df.iloc[1:].astype(float).values
    return p, rows


def find_peak_of_derivative_spline(p, y, s=None, grid_res=1000):
    # Fit smoothing spline
    spl = UnivariateSpline(p, y, s=s)
    deriv = spl.derivative(n=1)

    # coarse grid to find initial maximum
    grid = np.linspace(p.min(), p.max(), grid_res)
    dvals = deriv(grid)
    i0 = np.nanargmax(dvals)
    p0 = grid[i0]

    # refine by bounded scalar minimization of -deriv
    res = minimize_scalar(lambda x: -deriv(x), bounds=(p.min(), p.max()), method='bounded', options={'xatol':1e-8})
    p_peak = float(res.x)
    deriv_peak = float(deriv(p_peak))
    fval = float(spl(p_peak))
    return p_peak, deriv_peak, fval, spl


def process_file(path: Path, out_base: Path, s=None, grid_res=1000):
    theta_list = parse_theta_list_from_filename(path)
    p, rows = read_gt_csv(path)
    n_theta = len(theta_list)
    n_rows = rows.shape[0]

    if n_rows < n_theta:
        print(f"Warning: file {path} has fewer rows ({n_rows}) than the parsed theta count ({n_theta}). Using min rows.")
    use_n = min(n_theta, n_rows)

    Lname, zname = extract_L_zeta_from_parent(path.parent.name)
    out_dir = out_base / 'spline' / Lname / zname
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    fig, axes = plt.subplots(use_n, 1, figsize=(6, 2.8 * use_n), constrained_layout=True)
    if use_n == 1:
        axes = [axes]

    for i in range(use_n):
        theta = theta_list[i]
        y = rows[i]
        try:
            p_peak, deriv_peak, fval, spl = find_peak_of_derivative_spline(p, y, s=s, grid_res=grid_res)
        except Exception as e:
            print(f"Failed to find peak for {path.name} theta {theta}: {e}")
            p_peak, deriv_peak, fval = np.nan, np.nan, np.nan

        results.append({'file': path.name, 'parent': path.parent.name, 'L': Lname, 'zeta': zname, 'theta': theta, 'p_peak': p_peak, 'deriv_peak': deriv_peak, 'f_at_peak': fval})

        ax = axes[i]
        ax.plot(p, y, 'o-', label=f'theta={theta}')
        xgrid = np.linspace(p.min(), p.max(), 1000)
        ax.plot(xgrid, spl(xgrid), '-', alpha=0.8, label='spline')
        ax.axvline(p_peak, color='C3', linestyle='--', label=f'p*={p_peak:.5f}')
        ax.set_ylabel('Gt Component')
        ax.grid(True)
        ax.legend()

    # save results text file and plot
    txtfile = out_dir / f"{sanitize(path.stem)}_critical_points.txt"
    with open(txtfile, 'w') as fh:
        fh.write('# theta,p_peak,deriv_peak,f_at_peak\n')
        for r in results:
            fh.write(f"{r['theta']},{r['p_peak']},{r['deriv_peak']},{r['f_at_peak']}\n")

    plotfile = out_dir / f"{sanitize(path.stem)}_critical_points.png"
    fig.suptitle(f"{path.parent.name}  -- {path.name}")
    fig.savefig(plotfile, dpi=200)
    plt.close(fig)

    return results, str(txtfile), str(plotfile)


def main(csv_dir: Path, out_base: Path, smoothing=None, grid_res=1000):
    csv_dir = Path(csv_dir)
    out_base = Path(out_base)
    files = list(csv_dir.rglob('Gt Component*.csv'))
    if not files:
        print('No Gt Component files found')
        return 1

    all_results = []
    for f in sorted(files):
        res, txt, plot = process_file(f, out_base, s=smoothing, grid_res=grid_res)
        all_results.extend(res)
        print(f"Processed {f} -> {txt}, {plot}")

    df = pd.DataFrame(all_results)
    summary = out_base / 'spline_critical_points_summary.csv'
    df.to_csv(summary, index=False)
    print(f"Wrote summary to {summary}")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-dir', default='CSV', help='Directory containing CSV files')
    parser.add_argument('--out-dir', default='plots/critical_points', help='Directory to save plots and summaries')
    parser.add_argument('--smoothing', type=float, default=None, help='Smoothing factor s for UnivariateSpline (default None -> interpolation)')
    parser.add_argument('--grid-res', type=int, default=1000, help='Grid resolution for coarse search')
    args = parser.parse_args()
    raise SystemExit(main(args.csv_dir, args.out_dir, smoothing=args.smoothing, grid_res=args.grid_res))

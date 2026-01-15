#!/usr/bin/env python3
"""For each "Gt Component" CSV under the CSV directory this script: (original docstring trimmed)"""

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
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


def find_peak_of_derivative_savgol(p, y, window_length=7, polyorder=3):
    # Ensure window_length is odd and <= len(p)
    wl = int(window_length)
    if wl % 2 == 0:
        wl += 1
    wl = max(3, wl)
    wl = min(wl, len(p) if len(p)%2==1 else len(p)-1)

    # smooth and compute derivative (delta = median spacing)
    delta = np.median(np.diff(p))
    y_s = savgol_filter(y, wl, polyorder)
    dy = savgol_filter(y, wl, polyorder, deriv=1, delta=delta)

    # coarse maximum at sample points
    i0 = np.nanargmax(dy)
    p0 = p[i0]

    # refine by fitting spline to derivative and maximize
    spl = UnivariateSpline(p, dy, s=0)
    res = minimize_scalar(lambda x: -spl(x), bounds=(p.min(), p.max()), method='bounded', options={'xatol':1e-8})
    p_peak = float(res.x)
    deriv_peak = float(spl(p_peak))

    # also get f(p_peak) by fitting spline to original y
    spl_y = UnivariateSpline(p, y_s, s=0)
    fval = float(spl_y(p_peak))
    return p_peak, deriv_peak, fval, y_s, dy


def process_file(path: Path, out_base: Path, window_length=7, polyorder=3):
    theta_list = parse_theta_list_from_filename(path)
    p, rows = read_gt_csv(path)
    n_theta = len(theta_list)
    n_rows = rows.shape[0]

    if n_rows < n_theta:
        print(f"Warning: file {path} has fewer rows ({n_rows}) than the parsed theta count ({n_theta}). Using min rows.")
    use_n = min(n_theta, n_rows)

    Lname, zname = extract_L_zeta_from_parent(path.parent.name)
    out_dir = out_base / 'savgol' / Lname / zname
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    fig, axes = plt.subplots(use_n, 1, figsize=(6, 2.8 * use_n), constrained_layout=True)
    if use_n == 1:
        axes = [axes]

    for i in range(use_n):
        theta = theta_list[i]
        y = rows[i]
        try:
            p_peak, deriv_peak, fval, y_s, dy = find_peak_of_derivative_savgol(p, y, window_length=window_length, polyorder=polyorder)
        except Exception as e:
            print(f"Failed to find peak for {path.name} theta {theta}: {e}")
            p_peak, deriv_peak, fval = np.nan, np.nan, np.nan
            y_s = y
            dy = np.full_like(y, np.nan)

        results.append({'file': path.name, 'parent': path.parent.name, 'L': Lname, 'zeta': zname, 'theta': theta, 'p_peak': p_peak, 'deriv_peak': deriv_peak, 'f_at_peak': fval})

        ax = axes[i]
        ax.plot(p, y, 'o-', label=f'theta={theta}')
        ax.plot(p, y_s, '-', alpha=0.8, label='savgol')
        ax2 = ax.twinx()
        ax2.plot(p, dy, color='0.4', linestyle=':', label='dy/dp')
        ax.axvline(p_peak, color='C3', linestyle='--', label=f'p*={p_peak:.5f}')
        ax.set_ylabel('Gt Component')
        ax2.set_ylabel('derivative', color='0.4')
        ax.grid(True)
        ax.legend(loc='upper left')

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


def main(csv_dir: Path, out_base: Path, window_length=7, polyorder=3):
    csv_dir = Path(csv_dir)
    out_base = Path(out_base)
    files = list(csv_dir.rglob('Gt Component*.csv'))
    if not files:
        print('No Gt Component files found')
        return 1

    all_results = []
    for f in sorted(files):
        res, txt, plot = process_file(f, out_base, window_length=window_length, polyorder=polyorder)
        all_results.extend(res)
        print(f"Processed {f} -> {txt}, {plot}")

    df = pd.DataFrame(all_results)
    summary = out_base / 'savgol_critical_points_summary.csv'
    df.to_csv(summary, index=False)
    print(f"Wrote summary to {summary}")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-dir', default='CSV', help='Directory containing CSV files')
    parser.add_argument('--out-dir', default='plots/critical_points', help='Directory to save plots and summaries')
    parser.add_argument('--window-length', type=int, default=7, help='Savitzky-Golay window length (odd)')
    parser.add_argument('--polyorder', type=int, default=3, help='Savitzky-Golay polynomial order')
    args = parser.parse_args()
    raise SystemExit(main(args.csv_dir, args.out_dir, window_length=args.window_length, polyorder=args.polyorder))

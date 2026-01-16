#!/usr/bin/env python3
"""This version applies a small local smoother (median or gaussian) before differencing, (original docstring trimmed)"""

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d


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


def quadratic_vertex_from_three(x, y):
    try:
        coeffs = np.polyfit(x, y, 2)
    except Exception:
        return None
    a, b, c = coeffs
    if abs(a) < 1e-12:
        return None
    return -b / (2.0 * a)


def find_peak_fd_smoothed(p, y, method='median', window=3, sigma=1.0):
    if method == 'median':
        wl = int(window)
        if wl % 2 == 0:
            wl += 1
        wl = max(3, wl)
        y_s = medfilt(y, kernel_size=wl)
    elif method == 'gaussian':
        y_s = gaussian_filter1d(y, sigma=sigma)
    else:
        raise ValueError('unknown smoothing method')

    pmid = 0.5 * (p[:-1] + p[1:])
    d = (y_s[1:] - y_s[:-1]) / (p[1:] - p[:-1])

    i_max = int(np.nanargmax(d))

    idxs = np.array([i_max - 1, i_max, i_max + 1])
    idxs = idxs[(idxs >= 0) & (idxs < len(d))]

    if len(idxs) >= 3:
        x = pmid[idxs]
        yy = d[idxs]
        xv = quadratic_vertex_from_three(x, yy)
        if xv is not None and (pmid.min() <= xv <= pmid.max()):
            p_peak = float(xv)
        else:
            p_peak = float(pmid[i_max])
    elif len(idxs) == 2:
        x = pmid[idxs]
        yy = d[idxs]
        if abs(yy[1] - yy[0]) < 1e-12:
            p_peak = float(0.5 * (x[0] + x[1]))
        else:
            p_peak = float(x[np.argmax(yy)])
    else:
        p_peak = float(pmid[i_max])

    d_peak = float(np.interp(p_peak, pmid, d))
    fval = float(np.interp(p_peak, p, y_s))
    return p_peak, d_peak, fval, y_s, d


def process_file(path: Path, out_base: Path, method='median', window=3, sigma=1.0):
    theta_list = parse_theta_list_from_filename(path)
    p, rows = read_gt_csv(path)
    n_theta = len(theta_list)
    n_rows = rows.shape[0]

    if n_rows < n_theta:
        print(f"Warning: file {path} has fewer rows ({n_rows}) than parsed theta count ({n_theta}). Using min rows.")
    use_n = min(n_theta, n_rows)

    Lname, zname = extract_L_zeta_from_parent(path.parent.name)
    out_dir = out_base / 'fd_smoothed' / Lname / zname
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    fig, axes = plt.subplots(use_n, 1, figsize=(6, 2.8 * use_n), constrained_layout=True)
    if use_n == 1:
        axes = [axes]

    for i in range(use_n):
        theta = theta_list[i]
        y = rows[i]
        try:
            p_peak, d_peak, fval, y_s, d = find_peak_fd_smoothed(p, y, method=method, window=window, sigma=sigma)
        except Exception as e:
            print(f"Failed for {path.name} theta {theta}: {e}")
            p_peak, d_peak, fval = np.nan, np.nan, np.nan
            y_s = y
            d = np.full_like(y[:-1], np.nan)

        results.append({'file': path.name, 'parent': path.parent.name, 'L': Lname, 'zeta': zname, 'theta': theta, 'p_peak': p_peak, 'd_peak': d_peak, 'f_at_peak': fval})

        ax = axes[i]
        ax.plot(p, y, 'o-', label=f'theta={theta} raw')
        ax.plot(p, y_s, '-', alpha=0.8, label=f'{method} smoothed')
        pmid = 0.5 * (p[:-1] + p[1:])
        ax2 = ax.twinx()
        ax2.plot(pmid, d, color='0.4', linestyle=':', label='d/dp (smoothed)')
        ax.axvline(p_peak, color='C3', linestyle='--', label=f'p*={p_peak:.5f}')
        ax.set_ylabel('Gt Component')
        ax2.set_ylabel('derivative', color='0.4')
        ax.grid(True)
        ax.legend(loc='upper left')

    txtfile = out_dir / f"{sanitize(path.stem)}_critical_points.txt"
    with open(txtfile, 'w') as fh:
        fh.write('# theta,p_peak,d_peak,f_at_peak\n')
        for r in results:
            fh.write(f"{r['theta']},{r['p_peak']},{r['d_peak']},{r['f_at_peak']}\n")

    plotfile = out_dir / f"{sanitize(path.stem)}_critical_points.png"
    fig.suptitle(f"{path.parent.name}  -- {path.name}")
    fig.savefig(plotfile, dpi=200)
    plt.close(fig)

    return results, str(txtfile), str(plotfile)


def main(csv_dir: Path, out_base: Path, method='median', window=3, sigma=1.0):
    csv_dir = Path(csv_dir)
    out_base = Path(out_base)
    files = list(csv_dir.rglob('Gt Component*.csv'))
    if not files:
        print('No Gt Component files found')
        return 1

    all_results = []
    for f in sorted(files):
        res, txt, plot = process_file(f, out_base, method=method, window=window, sigma=sigma)
        all_results.extend(res)
        print(f"Processed {f} -> {txt}, {plot}")

    df = pd.DataFrame(all_results)
    summary = out_base / 'fd_smoothed_critical_points_summary.csv'
    df.to_csv(summary, index=False)
    print(f"Wrote summary to {summary}")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-dir', default='CSV', help='Directory containing CSV files')
    parser.add_argument('--out-dir', default='plots/critical_points', help='Directory to save plots and summaries')
    parser.add_argument('--method', choices=['median', 'gaussian'], default='median')
    parser.add_argument('--window', type=int, default=3, help='window size for median filter')
    parser.add_argument('--sigma', type=float, default=1.0, help='sigma for gaussian filter')
    args = parser.parse_args()
    raise SystemExit(main(args.csv_dir, args.out_dir, method=args.method, window=args.window, sigma=args.sigma))

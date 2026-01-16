#!/usr/bin/env python3
"""For each "Cascade Time" CSV: (original docstring trimmed)"""

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sanitize(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_.=-]", "_", name)


def parse_theta_list_from_filename(path: Path):
    name = path.name
    m = re.search(r'Cascade\s*Time(.*)\.csv', name, flags=re.IGNORECASE)
    fragment = m.group(1) if m else name
    nums = re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', fragment)
    return [float(x) for x in nums]


def extract_L_zeta_from_parent(parent_name: str):
    mL = re.search(r'L\s*=\s*([0-9]+)', parent_name)
    mz = re.search(r'ζ\s*=\s*([0-9\.]+)|zeta\s*=\s*([0-9\.]+)', parent_name)
    L = mL.group(1) if mL else parent_name
    z = (mz.group(1) if mz and mz.group(1) is not None else (mz.group(2) if mz and mz.group(2) is not None else 'unknown'))
    return f"L={L}", f"zeta={z}"


def read_cascade_csv(path: Path):
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
    xv = -b / (2.0 * a)
    # second derivative is 2a
    return xv, 2.0 * a, np.polyval(coeffs, xv)


def find_peak_by_quadratic(p, y):
    i0 = int(np.nanargmax(y))
    idxs = np.array([i0 - 1, i0, i0 + 1])
    idxs = idxs[(idxs >= 0) & (idxs < len(p))]

    if len(idxs) >= 3:
        x = p[idxs]
        yy = y[idxs]
        result = quadratic_vertex_from_three(x, yy)
        if result is not None:
            xv, twoa, fval = result
            # accept only if vertex within neighborhood window
            if (x.min() <= xv <= x.max()):
                return float(xv), float(twoa), float(fval)
    # fallback: discrete maximum
    i = i0
    return float(p[i]), float(np.nan), float(y[i])


def process_file(path: Path, out_base: Path):
    theta_list = parse_theta_list_from_filename(path)
    p, rows = read_cascade_csv(path)
    n_theta = len(theta_list)
    n_rows = rows.shape[0]

    if n_rows < n_theta:
        print(f"Warning: file {path} has fewer rows ({n_rows}) than parsed theta count ({n_theta}). Using min rows.")
    use_n = min(n_theta, n_rows)

    Lname, zname = extract_L_zeta_from_parent(path.parent.name)
    out_dir = out_base / 'fd_cascadetime' / Lname / zname
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    fig, axes = plt.subplots(use_n, 1, figsize=(6, 2.8 * use_n), constrained_layout=True)
    if use_n == 1:
        axes = [axes]

    for i in range(use_n):
        theta = theta_list[i]
        y = rows[i]
        try:
            p_peak, d_peak, fval = find_peak_by_quadratic(p, y)
        except Exception as e:
            print(f"Failed for {path.name} theta {theta}: {e}")
            p_peak, d_peak, fval = np.nan, np.nan, np.nan

        results.append({'file': path.name, 'parent': path.parent.name, 'L': Lname, 'zeta': zname, 'theta': theta, 'p_peak': p_peak, 'd_peak': d_peak, 'f_at_peak': fval})

        ax = axes[i]
        ax.plot(p, y, 'o-', label=f'theta={theta}')
        ax.axvline(p_peak, color='C3', linestyle='--', label=f'p*={p_peak:.5f}')
        ax.set_ylabel('Cascade time')
        ax.grid(True)
        ax.legend()

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


def main(csv_dir: Path, out_base: Path):
    csv_dir = Path(csv_dir)
    out_base = Path(out_base)
    files = list(csv_dir.rglob('Cascade Time*.csv'))
    if not files:
        print('No Cascade Time files found')
        return 1

    all_results = []
    for f in sorted(files):
        res, txt, plot = process_file(f, out_base)
        all_results.extend(res)
        print(f"Processed {f} -> {txt}, {plot}")

    df = pd.DataFrame(all_results)
    summary = out_base / 'fd_cascadetime_critical_points_summary.csv'
    df.to_csv(summary, index=False)
    print(f"Wrote summary to {summary}")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-dir', default='CSV', help='Directory containing CSV files')
    parser.add_argument('--out-dir', default='plots/critical_points', help='Directory to save plots and summaries')
    args = parser.parse_args()
    raise SystemExit(main(args.csv_dir, args.out_dir))

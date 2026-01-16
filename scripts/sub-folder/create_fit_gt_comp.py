#!/usr/bin/env python3
"""Create fit_Gt Component CSV files that contain only p > p_c values per-theta.

For each "Gt Component*.csv" in the CSV directory tree, this script:
 - reads the p grid and theta rows
 - looks up p_c for each theta from the supplied critical summary CSV (expects columns 'parent','theta','p_peak')
 - produces a new CSV named "fit_Gt Component ... .csv" in the same folder with the same header row of p-values,
   but with entries where p <= p_c replaced by empty strings (so the file contains only values for p > p_c).

Usage:
  python3 scripts/create_fit_gt_comp.py --csv-dir CSV --critical-summary plots/critical_points/fd_cascadetime_critical_points_summary.csv
"""

from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd


def parse_theta_list_from_filename(path: Path):
    name = path.name
    m = re.search(r'Gt Component(.*)\.csv', name, flags=re.IGNORECASE)
    fragment = m.group(1) if m else name
    nums = re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', fragment)
    return [float(x) for x in nums]


def read_gt_csv(path: Path):
    df = pd.read_csv(path, header=None)
    p = df.iloc[0].astype(float).values
    rows = df.iloc[1:].astype(float).values
    return p, rows, df


def find_pc_for_theta(crit_df: pd.DataFrame, parent_name: str, filename: str, theta: float):
    # match by parent and theta
    match = crit_df[(crit_df['parent'] == parent_name) & (np.isclose(crit_df['theta'], theta))]
    if len(match) == 0:
        # fallback by file name
        match = crit_df[(crit_df['file'] == filename) & (np.isclose(crit_df['theta'], theta))]
    if len(match) == 0:
        return None
    return float(match['p_peak'].iloc[0])


def main(csv_dir: Path, critical_summary_csv: Path):
    csv_dir = Path(csv_dir)
    if not critical_summary_csv.exists():
        raise FileNotFoundError(f"Critical summary CSV not found: {critical_summary_csv}")

    crit_df = pd.read_csv(critical_summary_csv)

    files = list(csv_dir.rglob('Gt Component*.csv'))
    if not files:
        print('No Gt Component files found')
        return 1

    for f in sorted(files):
        p, rows, df_orig = read_gt_csv(f)
        theta_list = parse_theta_list_from_filename(f)
        n_theta = min(len(theta_list), rows.shape[0])

        # Build output array as strings (so we can write blanks)
        out_rows = []
        for i in range(n_theta):
            theta = theta_list[i]
            y = rows[i]
            p_c = find_pc_for_theta(crit_df, f.parent.name, f.name, theta)
            if p_c is None:
                print(f"Warning: p_c not found for {f} theta {theta}; copying row unchanged")
                out_row = [str(x) for x in y]
            else:
                # Mask values where p <= p_c
                mask = p > p_c
                out_row = [str(v) if m else '' for v, m in zip(y, mask)]
            out_rows.append(out_row)
            # out_rows.append(p_c)  # for debugging
            

        # If there are more rows in the file than theta_list length, copy remaining rows unchanged
        if rows.shape[0] > n_theta:
            for j in range(n_theta, rows.shape[0]):
                out_rows.append([str(x) for x in rows[j]])

        # Prepare output DataFrame: first row p values, then rows
        out_df = pd.DataFrame(out_rows)
        # Prepend header row of p-values (as strings)
        header = [str(x) for x in p]
        out_df_full = pd.concat([pd.DataFrame([header]), out_df], ignore_index=True)

        out_path = f.parent / ('fit_' + f.name)
        out_df_full.to_csv(out_path, header=False, index=False)
        print(f"Wrote {out_path}")

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-dir', default='CSV', help='Directory containing CSV files')
    parser.add_argument('--critical-summary', default='plots/critical_points/fd_cascadetime_critical_points_summary.csv', help='CSV of p_c values to use')
    args = parser.parse_args()
    raise SystemExit(main(Path(args.csv_dir), Path(args.critical_summary)))

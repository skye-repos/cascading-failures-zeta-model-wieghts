#!/usr/bin/env python3

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Notes on fitting choices:
# - scipy.optimize.curve_fit: flexible nonlinear fits, supports bounds and returns parameter covariances. working for discrete data.


def sanitize(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_.=-]", "_", name)


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
    return p, rows


def extract_L_zeta_from_parent(parent_name: str):
    mL = re.search(r'L\s*=\s*([0-9]+)', parent_name)
    mz = re.search(r'ζ\s*=\s*([0-9\.]+)|zeta\s*=\s*([0-9\.]+)', parent_name)
    L = mL.group(1) if mL else parent_name
    z = (mz.group(1) if mz and mz.group(1) is not None else (mz.group(2) if mz and mz.group(2) is not None else 'unknown'))
    return f"L={L}", f"zeta={z}"


def power_model(p, S_c, A, beta, p_c):
    # ensure p>p_c where called
    return S_c + A * np.power(p - p_c, beta)


def fit_beta(p_use, y_use, p_c, S_c_fixed=True, log_fit=False):
    # Fit S(p) = S_c + A*(p - p_c)**beta
    # If S_c_fixed True, keep S_c as known; otherwise fit it.
    # If log_fit is True, perform linear fit in log-space on (S - S_c) = A*(p-p_c)^beta

    if S_c_fixed:
        S_c0 = float(S_c_fixed)
    else:
        S_c0 = float(y_use.min())

    if log_fit:
        # Prepare data for log-space linear fit: ln(y-S_c) = ln A + beta * ln(p-p_c)
        x = p_use - p_c
        yshift = y_use - S_c0
        mask = (x > 0) & (yshift > 0)
        if mask.sum() < 3:
            raise RuntimeError('Not enough positive (p-p_c) and (S-S_c) values for log-fit')
        logx = np.log(x[mask])
        logy = np.log(yshift[mask])
        # linear fit in log-space
        coeffs, cov = np.polyfit(logx, logy, 1, cov=True)
        beta = float(coeffs[0])
        lnA = float(coeffs[1])
        A = float(np.exp(lnA))
        stderr_beta = float(np.sqrt(cov[0, 0]))
        stderr_lnA = float(np.sqrt(cov[1, 1]))
        stderr_A = A * stderr_lnA
        return {'S_c': S_c0, 'A': A, 'beta': beta, 'stderr_A': stderr_A, 'stderr_beta': stderr_beta, 'cov': cov}

    # fallback to nonlinear fit in original space
    A0 = max(1e-6, float(y_use[0] - S_c0))
    beta0 = 1.0

    if S_c_fixed:
        # fit A and beta with S_c fixed
        def model_ab(x, A, beta):
            return S_c0 + A * np.power(x - p_c, beta)
        popt, pcov = curve_fit(model_ab, p_use, y_use, p0=[A0, beta0], bounds=([0.0, 0.0],[np.inf, 10.0]), maxfev=10000)
        A, beta = popt
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else (np.nan, np.nan)
        return {'S_c': S_c0, 'A': float(A), 'beta': float(beta), 'stderr_A': float(perr[0]), 'stderr_beta': float(perr[1]), 'cov': pcov}
    else:
        # fit S_c, A, beta
        def model_sab(x, S_c, A, beta):
            return S_c + A * np.power(x - p_c, beta)
        popt, pcov = curve_fit(model_sab, p_use, y_use, p0=[S_c0, A0, beta0], bounds=([0.0, 0.0, 0.0],[np.inf, np.inf, 10.0]), maxfev=10000)
        S_c, A, beta = popt
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else (np.nan, np.nan, np.nan)
        return {'S_c': float(S_c), 'A': float(A), 'beta': float(beta), 'stderr_Sc': float(perr[0]), 'stderr_A': float(perr[1]), 'stderr_beta': float(perr[2]), 'cov': pcov}


def main(csv_dir: Path, critical_summary_csv: Path, out_base: Path, p_window: float = 0.02, n_points: int = 6, use_Sc_fixed: bool = True, method_name: str = 'fd_cascadetime', log_fit: bool = False, write_to_csv_dir: bool = False):
    csv_dir = Path(csv_dir)
    out_base = Path(out_base)
    out_base.mkdir(parents=True, exist_ok=True)

    if not critical_summary_csv.exists():
        raise FileNotFoundError(f"Critical points summary not found: {critical_summary_csv}")

    crit_df = pd.read_csv(critical_summary_csv)

    files = list(csv_dir.rglob('Gt Component*.csv'))
    if not files:
        print('No Gt Component files found')
        return 1

    all_results = []

    for f in sorted(files):
        p, rows = read_gt_csv(f)
        theta_list = parse_theta_list_from_filename(f)
        use_n = min(len(theta_list), rows.shape[0])
        Lname, zname = extract_L_zeta_from_parent(f.parent.name)
        out_dir = out_base / method_name / Lname / zname
        out_dir.mkdir(parents=True, exist_ok=True)

        results = []
        fig, axes = plt.subplots(use_n, 1, figsize=(6, 2.8 * use_n), constrained_layout=True)
        if use_n == 1:
            axes = [axes]

        for i in range(use_n):
            theta = theta_list[i]
            y = rows[i]

            # find p_c from crit_df matching parent folder and theta (supports Cascade Time summary)
            match = crit_df[(crit_df['parent']==f.parent.name) & (np.isclose(crit_df['theta'], theta))]
            if len(match) == 0:
                # fallback: try matching by file name (legacy behaviour)
                match = crit_df[(crit_df['file']==f.name) & (np.isclose(crit_df['theta'], theta))]
            if len(match) == 0:
                print(f"No p_c found for {f.name} theta {theta} in {critical_summary_csv}")
                continue
            p_c = float(match['p_peak'].iloc[0])

            # interpolate S_c
            S_c = float(np.interp(p_c, p, y))

            # select data p > p_c and within window
            if n_points is not None:
                # pick n_points starting from first p > p_c
                idx = np.where(p > p_c)[0]
                if len(idx) == 0:
                    print(f"No p > p_c points for {f.name} theta {theta}")
                    continue
                sel_idx = idx[:n_points]
            else:
                sel_idx = np.where((p > p_c) & (p <= p_c + p_window))[0]

            if len(sel_idx) < 3:
                print(f"Insufficient points for fit for {f.name} theta {theta} (got {len(sel_idx)}). Consider increasing p_window or n_points.")
                continue

            p_use = p[sel_idx]
            y_use = y[sel_idx]

            try:
                fit = fit_beta(p_use, y_use, p_c, S_c_fixed=S_c if use_Sc_fixed else False, log_fit=log_fit)
            except Exception as e:
                print(f"Fit failed for {f.name} theta {theta}: {e}")
                continue

            # compute diagnostics
            if use_Sc_fixed:
                y_pred = fit['S_c'] + fit['A'] * np.power(p_use - p_c, fit['beta'])
            else:
                y_pred = fit['S_c'] + fit['A'] * np.power(p_use - p_c, fit['beta'])
            resid = y_use - y_pred
            ss_res = np.sum(resid**2)
            ss_tot = np.sum((y_use - np.mean(y_use))**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            rmse = np.sqrt(np.mean(resid**2))

            # pre-failure slopes: linear least-squares slope over the selected window,
            # a simple end-to-end slope, and slope from the fitted power-law at mean p
            try:
                slope_lin = float(np.polyfit(p_use, y_use, 1)[0])
            except Exception:
                slope_lin = np.nan
            slope_end = float((y_use[-1] - y_use[0]) / (p_use[-1] - p_use[0])) if (p_use[-1] - p_use[0]) != 0 else np.nan
            try:
                mean_p = float(np.mean(p_use))
                slope_from_fit = float(fit['A'] * fit['beta'] * np.power(max(mean_p - p_c, 0), max(fit['beta'] - 1, 0)))
            except Exception:
                slope_from_fit = np.nan

            entry = {'file': f.name, 'L': Lname, 'zeta': zname, 'theta': theta, 'p_c': p_c, 'S_c': fit['S_c'], 'A': fit['A'], 'beta': fit['beta'], 'stderr_beta': fit.get('stderr_beta', np.nan), 'r2': r2, 'rmse': rmse, 'n_points': len(p_use), 'slope_lin': slope_lin, 'slope_end': slope_end, 'slope_from_fit': slope_from_fit} 
            results.append(entry)
            all_results.append(entry)

            ax = axes[i]
            ax.plot(p, y, 'o-', label='data')
            # plot fitted curve on fine grid
            xgrid = np.linspace(p_use.min(), p_use.max(), 200)
            yfit = fit['S_c'] + fit['A'] * np.power(np.clip(xgrid - p_c, 0, None), fit['beta'])
            ax.plot(xgrid, yfit, '-', label=f'fit β={fit["beta"]:.3f}')
            ax.axvline(p_c, color='C3', linestyle='--', label=f'p_c={p_c:.5f}')
            ax.set_ylabel('Gt Component')
            ax.grid(True)
            ax.legend()

        # save per-zeta text and plot
        txtfile = out_dir / f"{sanitize(f.stem)}_beta_fit.txt"
        df_out = pd.DataFrame(results)
        df_out.to_csv(txtfile, index=False)

        plotfile = out_dir / f"{sanitize(f.stem)}_beta_fit.png"
        fig.suptitle(f"{f.parent.name}  -- {f.name}  (fit window: p_c -> p_c+{p_window})")
        fig.savefig(plotfile, dpi=200)

        # optionally save outputs next to the CSV in the zeta folder
        if write_to_csv_dir:
            txtfile2 = f.parent / f"{sanitize(f.stem)}_beta_fit.txt"
            df_out.to_csv(txtfile2, index=False)
            plotfile2 = f.parent / f"{sanitize(f.stem)}_beta_fit.png"
            fig.savefig(plotfile2, dpi=200)
            print(f"Also saved to {txtfile2}, {plotfile2}")

        plt.close(fig)

        print(f"Saved fits for {f} -> {txtfile}, {plotfile}")

    summary = out_base / f"{method_name}_beta_fits_summary.csv"
    pd.DataFrame(all_results).to_csv(summary, index=False)
    print(f"Wrote summary to {summary}")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-dir', default='CSV', help='Directory containing CSV files')
    parser.add_argument('--critical-summary', default='plots/critical_points/fd_critical_points_summary.csv', help='CSV of p_c values to use')
    parser.add_argument('--out-dir', default='plots/critical_points/fits', help='Directory to save fit outputs')
    parser.add_argument('--p-window', type=float, default=0.02, help='p window above p_c to include (default 0.02)')
    parser.add_argument('--n-points', type=int, default=6, help='Alternatively, specify number of points above p_c to use (default 6)')
    parser.add_argument('--method-name', default='fd_cascadetime', help='Name tag for output subfolder')
    parser.add_argument('--no-Sc-fit', dest='use_Sc_fixed', action='store_false', help='Do not fix S_c (fit it)')
    parser.add_argument('--log-fit', dest='log_fit', action='store_true', help='Use log-linear fit ln(S-S_c)=lnA + beta ln(p-p_c)')
    parser.add_argument('--save-to-csv-dir', dest='write_to_csv_dir', action='store_true', help='Also save fit outputs (txt, png) into each CSV zeta subdirectory alongside the original CSVs')
    args = parser.parse_args()

    method_name = args.method_name
    if args.log_fit and 'logfit' not in method_name:
        method_name = f"{method_name}_logfit"

    raise SystemExit(main(args.csv_dir, Path(args.critical_summary), Path(args.out_dir), p_window=args.p_window, n_points=args.n_points, use_Sc_fixed=args.use_Sc_fixed, method_name=method_name, log_fit=args.log_fit, write_to_csv_dir=args.write_to_csv_dir))

#!/usr/bin/env python3
"""
temporal_cc_all_pairs.py

Generate cross-correlation plots for ALL EBC pairs.
Outputs a multi-page PDF or individual PNG files for browsing.

ADDED:
- Population statistics for peak-lag stability across tasks
- Saves per-pair metrics to CSV
- Wilcoxon test on delta peak lag vs 0
- % within tolerance (bins and seconds)
"""

import argparse
import numpy as np
import scipy.io as sio
import scipy.signal as signal
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from itertools import combinations
from typing import Tuple, Optional
import os
import csv


def load_spikemat(mat_path: str, var_name: Optional[str] = None) -> np.ndarray:
    mat = sio.loadmat(mat_path)
    if var_name is not None:
        return np.asarray(mat[var_name])
    if 'spike' in mat:
        return np.asarray(mat['spike'])
    if 'spikemat' in mat:
        return np.asarray(mat['spikemat'])
    keys = [k for k in mat.keys() if not k.startswith('__')]
    if len(keys) == 1:
        return np.asarray(mat[keys[0]])
    raise ValueError(f"Cannot auto-detect variable. Available: {keys}")


def bin_spikes(spike_train: np.ndarray, frames_per_bin: int) -> np.ndarray:
    n = spike_train.size
    n_full = (n // frames_per_bin) * frames_per_bin
    truncated = spike_train[:n_full]
    reshaped = truncated.reshape(-1, frames_per_bin)
    return reshaped.sum(axis=1).astype(float)


def detrend_highpass(x: np.ndarray, cutoff_bins: int = 50) -> np.ndarray:
    if len(x) < cutoff_bins * 3:
        return x - x.mean()

    window = min(cutoff_bins * 2 + 1, len(x) // 2)
    if window % 2 == 0:
        window += 1
    window = max(5, window)

    try:
        smoothed = signal.savgol_filter(x, window, polyorder=2)
    except Exception:
        kernel = np.ones(window) / window
        smoothed = np.convolve(x, kernel, mode='same')

    return x - smoothed


def xcorr_coeff(x: np.ndarray, y: np.ndarray, max_lag: int, detrend: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if detrend:
        x = detrend_highpass(x)
        y = detrend_highpass(y)

    x = x - x.mean()
    y = y - y.mean()

    sx = x.std()
    sy = y.std()

    if sx < 1e-12 or sy < 1e-12:
        return np.arange(-max_lag, max_lag + 1), np.full(2 * max_lag + 1, np.nan)

    x = x / sx
    y = y / sy

    n = len(x)
    lags = np.arange(-max_lag, max_lag + 1)
    out = np.zeros(len(lags), dtype=float)

    for i, lag in enumerate(lags):
        if lag < 0:
            out[i] = np.dot(x[-lag:], y[:n + lag]) / (n - abs(lag))
        elif lag > 0:
            out[i] = np.dot(x[:n - lag], y[lag:]) / (n - lag)
        else:
            out[i] = np.dot(x, y) / n

    return lags, out


def compute_peak_metrics(lags: np.ndarray, cc: np.ndarray) -> dict:
    """Compute metrics to help identify good pairs."""
    if np.all(np.isnan(cc)):
        return {'peak_lag': np.nan, 'peak_val': np.nan, 'snr': np.nan}

    # peak in absolute value (captures pos/neg coupling)
    peak_idx = np.nanargmax(np.abs(cc))
    peak_lag = float(lags[peak_idx])
    peak_val = float(cc[peak_idx])

    # SNR: peak value vs std of flanks
    center = len(cc) // 2
    flank_left = cc[:center-10] if center > 10 else cc[:center]
    flank_right = cc[center+10:] if center+10 < len(cc) else cc[center:]
    flanks = np.concatenate([flank_left, flank_right]) if (len(flank_left) + len(flank_right)) > 0 else cc
    noise_std = np.nanstd(flanks) if len(flanks) > 0 else 1e-10
    snr = np.abs(peak_val) / (noise_std + 1e-10)

    return {'peak_lag': peak_lag, 'peak_val': peak_val, 'snr': float(snr)}


def nan_iqr(x: np.ndarray) -> Tuple[float, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan)
    q1 = float(np.percentile(x, 25))
    q3 = float(np.percentile(x, 75))
    return q1, q3


def main():
    ap = argparse.ArgumentParser(description="Generate cross-correlations for all EBC pairs + population stats.")

    ap.add_argument("--of-mat", required=True, help="Path to OF .mat file.")
    ap.add_argument("--chase-mat", required=True, help="Path to Chase .mat file.")
    ap.add_argument("--of-var", default=None, help="Variable name for OF spike data.")
    ap.add_argument("--chase-var", default=None, help="Variable name for Chase spike data.")

    ap.add_argument("--dt", type=float, default=0.008333333333, help="Seconds per frame.")
    ap.add_argument("--bin-width", type=float, default=0.15, help="Bin width in seconds.")
    ap.add_argument("--max-lag", type=int, default=60, help="Max lag in bins.")

    ap.add_argument("--cells", required=True,
                    help='Comma-separated list of EBC cell indices (0-based).')

    ap.add_argument("--output-dir", default="xcorr_pairs", help="Output directory for plots.")
    ap.add_argument("--pdf", action="store_true", help="Also save as single PDF.")

    # ADDED: tolerance thresholds for "% within"
    ap.add_argument("--tol-bins", default="1,2,3",
                    help="Comma-separated tolerances in BINS for Δpeak_lag (e.g. '1,2,3').")

    args = ap.parse_args()

    # Parse cells
    cells = [int(x.strip()) for x in args.cells.split(",") if x.strip()]
    print(f"EBC cells: {cells}")
    print(f"Number of cells: {len(cells)}")

    # All pairs
    pairs = list(combinations(cells, 2))
    print(f"Number of pairs: {len(pairs)}")

    # Load data
    of_spikemat = load_spikemat(args.of_mat, args.of_var)
    chase_spikemat = load_spikemat(args.chase_mat, args.chase_var)

    print(f"OF shape: {of_spikemat.shape}")
    print(f"Chase shape: {chase_spikemat.shape}")

    # Bin all cells
    frames_per_bin = max(1, int(round(args.bin_width / args.dt)))
    bin_sec = frames_per_bin * args.dt
    print(f"Frames/bin: {frames_per_bin}  => bin_sec={bin_sec:.6f} s")

    of_binned = {c: bin_spikes(of_spikemat[c], frames_per_bin) for c in cells}
    ch_binned = {c: bin_spikes(chase_spikemat[c], frames_per_bin) for c in cells}

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Store metrics for ranking + stats
    all_metrics = []

    # Generate plots
    print(f"\nGenerating {len(pairs)} pair plots...")

    pdf = None
    if args.pdf:
        pdf = PdfPages(os.path.join(args.output_dir, "all_pairs.pdf"))

    for idx, (c1, c2) in enumerate(pairs):
        # Compute cross-correlations
        lags, cc_of = xcorr_coeff(of_binned[c1], of_binned[c2], args.max_lag, detrend=True)
        _, cc_ch = xcorr_coeff(ch_binned[c1], ch_binned[c2], args.max_lag, detrend=True)

        # Compute metrics
        metrics_of = compute_peak_metrics(lags, cc_of)
        metrics_ch = compute_peak_metrics(lags, cc_ch)

        # Similarity of curves (shape match)
        similarity = np.nan
        if not (np.all(np.isnan(cc_of)) or np.all(np.isnan(cc_ch))):
            # safe corrcoef: remove nans
            m = np.isfinite(cc_of) & np.isfinite(cc_ch)
            if m.sum() >= 5:
                similarity = float(np.corrcoef(cc_of[m], cc_ch[m])[0, 1])

        avg_snr = (metrics_of['snr'] + metrics_ch['snr']) / 2 if np.isfinite(metrics_of['snr']) and np.isfinite(metrics_ch['snr']) else np.nan

        # Peak-lag shift across tasks
        peak_lag_of = metrics_of['peak_lag']
        peak_lag_ch = metrics_ch['peak_lag']
        d_lag_bins = np.nan
        d_lag_sec = np.nan
        if np.isfinite(peak_lag_of) and np.isfinite(peak_lag_ch):
            d_lag_bins = float(peak_lag_ch - peak_lag_of)
            d_lag_sec = d_lag_bins * bin_sec

        all_metrics.append({
            'c1': c1,
            'c2': c2,
            'of_peak_lag_bins': peak_lag_of,
            'ch_peak_lag_bins': peak_lag_ch,
            'delta_peak_lag_bins': d_lag_bins,
            'delta_peak_lag_sec': d_lag_sec,
            'of_peak_val': metrics_of['peak_val'],
            'ch_peak_val': metrics_ch['peak_val'],
            'of_snr': metrics_of['snr'],
            'ch_snr': metrics_ch['snr'],
            'avg_snr': avg_snr,
            'similarity': similarity,
            'score': (avg_snr * (1 + similarity)) if (np.isfinite(avg_snr) and np.isfinite(similarity)) else np.nan,
        })

        # Create plot
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(lags, cc_of, color='steelblue', linewidth=1.5, label='Open Field')
        ax.plot(lags, cc_ch, color='darkorange', linewidth=1.5, label='Chasing')
        ax.axhline(0, color='black', linestyle='--', linewidth=0.7, alpha=0.6)

        ax.set_xlabel('lag (bins)', fontsize=10)
        ax.set_ylabel('corr.', fontsize=10)

        title = (f'Cells {c1} vs {c2}  |  '
                 f'SNR: OF={metrics_of["snr"]:.1f}, Ch={metrics_ch["snr"]:.1f}  |  '
                 f'Sim={similarity:.2f}  |  Δlag={d_lag_bins:.1f} bins')
        ax.set_title(title, fontsize=10)

        ax.legend(loc='upper right', fontsize=9, frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(-args.max_lag, args.max_lag)

        plt.tight_layout()

        # Save individual PNG
        png_path = os.path.join(args.output_dir, f"pair_{c1:02d}_{c2:02d}.png")
        plt.savefig(png_path, dpi=150, bbox_inches='tight')

        if pdf is not None:
            pdf.savefig(fig, bbox_inches='tight')

        plt.close(fig)

        if (idx + 1) % 20 == 0:
            print(f"  {idx + 1}/{len(pairs)} done...")

    if pdf is not None:
        pdf.close()
        print(f"\nSaved PDF: {os.path.join(args.output_dir, 'all_pairs.pdf')}")

    # ----------------------------
    # ADDED: Save per-pair metrics
    # ----------------------------
    csv_path = os.path.join(args.output_dir, "pair_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"Saved per-pair metrics CSV: {csv_path}")

    # -----------------------------------------
    # ADDED: Population stats on Δ peak lag
    # -----------------------------------------
    d = np.array([m['delta_peak_lag_bins'] for m in all_metrics], dtype=float)
    d_sec = np.array([m['delta_peak_lag_sec'] for m in all_metrics], dtype=float)
    valid = np.isfinite(d)
    d = d[valid]
    d_sec = d_sec[valid]

    print("\n" + "=" * 70)
    print("POPULATION STATS: Peak-lag stability (Chase − OF)")
    print("=" * 70)
    print(f"Valid pairs (finite peak lags in both tasks): {d.size} / {len(all_metrics)}")
    if d.size == 0:
        print("No valid pairs to compute stats. (Likely NaNs from zero-variance binned trains.)")
        return

    med = float(np.median(d))
    q1, q3 = nan_iqr(d)
    med_s = float(np.median(d_sec))
    q1_s, q3_s = nan_iqr(d_sec)

    # Wilcoxon signed-rank vs 0
    # (Requires at least one non-zero difference; handle edge-case)
    w_stat = np.nan
    w_p = np.nan
    try:
        # scipy will error if all differences are 0
        if np.any(np.abs(d) > 0):
            w_stat, w_p = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    except Exception:
        pass

    print(f"Δpeak_lag (bins): median={med:.3f}, IQR=[{q1:.3f}, {q3:.3f}]")
    print(f"Δpeak_lag (sec):  median={med_s:.6f}, IQR=[{q1_s:.6f}, {q3_s:.6f}]")
    print(f"Wilcoxon vs 0: statistic={w_stat}, p={w_p}")

    # % within tolerance
    tol_bins = [int(x.strip()) for x in args.tol_bins.split(",") if x.strip()]
    for tb in tol_bins:
        frac = float(np.mean(np.abs(d) <= tb))
        print(f"% pairs with |Δlag| ≤ {tb} bin(s) (≤ {tb * bin_sec:.3f} s): {100 * frac:.1f}%")

    # Extra: OF vs CH peak-lag correlation
    of_lags = np.array([m['of_peak_lag_bins'] for m in all_metrics], dtype=float)[valid]
    ch_lags = np.array([m['ch_peak_lag_bins'] for m in all_metrics], dtype=float)[valid]
    rho = np.nan
    rho_p = np.nan
    try:
        if of_lags.size >= 3:
            rho, rho_p = stats.spearmanr(of_lags, ch_lags, nan_policy="omit")
    except Exception:
        pass
    print(f"Spearman corr (peak_lag OF vs Chase): rho={rho}, p={rho_p}")

    # Write a text summary too (nice for Overleaf)
    summary_path = os.path.join(args.output_dir, "stats_summary.txt")
    with open(summary_path, "w") as f:
        f.write("POPULATION STATS: Peak-lag stability (Chase − OF)\n")
        f.write("=" * 70 + "\n")
        f.write(f"Valid pairs: {d.size} / {len(all_metrics)}\n")
        f.write(f"bin_sec: {bin_sec:.6f}\n")
        f.write(f"Δpeak_lag (bins): median={med:.6f}, IQR=[{q1:.6f}, {q3:.6f}]\n")
        f.write(f"Δpeak_lag (sec):  median={med_s:.6f}, IQR=[{q1_s:.6f}, {q3_s:.6f}]\n")
        f.write(f"Wilcoxon vs 0: statistic={w_stat}, p={w_p}\n")
        for tb in tol_bins:
            frac = float(np.mean(np.abs(d) <= tb))
            f.write(f"|Δlag| ≤ {tb} bin(s) (≤ {tb * bin_sec:.3f} s): {100 * frac:.2f}%\n")
        f.write(f"Spearman corr (peak_lag OF vs Chase): rho={rho}, p={rho_p}\n")
    print(f"Saved stats summary: {summary_path}")

    # Rank pairs by score (unchanged)
    all_metrics_sorted = [m for m in all_metrics if np.isfinite(m.get('score', np.nan))]
    all_metrics_sorted.sort(key=lambda x: x['score'], reverse=True)

    print("\n" + "=" * 70)
    print("TOP 20 PAIRS (by combined SNR + similarity score)")
    print("=" * 70)
    print(f"{'Rank':<5} {'Pair':<12} {'OF lag':<8} {'Ch lag':<8} {'Δlag':<8} {'Avg SNR':<10} {'Similarity':<10}")
    print("-" * 70)

    for i, m in enumerate(all_metrics_sorted[:20]):
        print(f"{i+1:<5} ({m['c1']:2d},{m['c2']:2d})    "
              f"{m['of_peak_lag_bins']:>6.1f}  {m['ch_peak_lag_bins']:>6.1f}  {m['delta_peak_lag_bins']:>6.1f}  "
              f"{m['avg_snr']:>8.2f}  {m['similarity']:>8.2f}")

    print("\nDone!")


if __name__ == "__main__":
    main()

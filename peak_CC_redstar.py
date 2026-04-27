#!/usr/bin/env python3
"""
Analyze peak CC values for EBCs from any animal.
Flexible version - specify animal and sessions via command line.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import argparse

import COMPLETE_Classification as base


def compute_cross_correlation_improved(rmA, rmB, occ_min=50, min_points=3, use_union_mask=True):
    """Improved cross-correlation that handles sparse occupancy better."""
    A = rmA['rm_ns'].copy().astype(float)
    B = rmB['rm_ns'].copy().astype(float)
    OA = rmA['occ_ns'].copy().astype(float)
    OB = rmB['occ_ns'].copy().astype(float)

    if use_union_mask:
        mask_A = (OA < occ_min) & (OB < occ_min)
        mask_B = mask_A.copy()
    else:
        mask_A = OA < occ_min
        mask_B = OB < occ_min

    A[mask_A] = np.nan
    B[mask_B] = np.nan

    offs_dist = np.arange(-A.shape[0] + 2, A.shape[0] - 1)
    offs_angle = np.arange(-A.shape[1] // 2, A.shape[1] // 2 + (A.shape[1] % 2))

    cc_plot = np.full((len(offs_dist), len(offs_angle)), np.nan)

    for i, oa in enumerate(offs_angle):
        rB = np.roll(B, oa, axis=1)
        for j, od in enumerate(offs_dist):
            if od < 0:
                Bp = rB[-od:, :]
                Ap = A[:A.shape[0] + od, :]
            elif od > 0:
                Bp = rB[:B.shape[0] - od, :]
                Ap = A[od:, :]
            else:
                Ap, Bp = A, rB

            a, b = Ap.ravel(), Bp.ravel()
            m = ~np.isnan(a) & ~np.isnan(b)

            if m.sum() >= min_points:
                cc_plot[j, i] = np.corrcoef(a[m], b[m])[0, 1]

    return cc_plot


def get_peak_cc(cc_plot, exclude_edge_fraction=0.25):
    """Extract peak CC value, excluding edges to avoid artifacts."""
    if cc_plot is None or np.all(np.isnan(cc_plot)):
        return np.nan

    nr, nc = cc_plot.shape
    edge_r = max(3, int(nr * exclude_edge_fraction))
    edge_c = max(2, int(nc * exclude_edge_fraction))

    # Focus on central region
    if nr > 2 * edge_r and nc > 2 * edge_c:
        cc_center = cc_plot[edge_r:nr - edge_r, edge_c:nc - edge_c]
    else:
        cc_center = cc_plot

    # Get peak value
    valid_vals = cc_center[np.isfinite(cc_center)]
    if len(valid_vals) == 0:
        return np.nan

    return float(np.nanmax(valid_vals))


def load_ebc_data(folder_loc, animal, channels, neuron_idx, binsize, occ_min,
                  of_sessions, chase_sessions):
    """Load and compute rate maps for a single EBC from any animal."""
    dt = binsize
    box_edges = base.BOX_EDGES[animal]

    print(f"\n    Loading neuron {neuron_idx}...")

    # Load OF sessions
    x_of_all, y_of_all, hd_of_all, spk_of_all = [], [], [], []

    for sess in of_sessions:
        try:
            print(f"      Loading {sess}...")
            d_of = base.load_session_data(folder_loc, animal, sess, channels, binsize)
            x, y, hd, spk, _, _, _ = base.prepare_data(d_of, neuron_idx, None, True)
            print(f"        ✓ {sess}: {len(x)} samples, {spk.sum()} spikes")
            x_of_all.append(x)
            y_of_all.append(y)
            hd_of_all.append(hd)
            spk_of_all.append(spk)
        except Exception as e:
            print(f"        {sess} failed: {e}")
            continue

    if not x_of_all:
        print(f"         No OF data loaded")
        return None

    x_of = np.concatenate(x_of_all)
    y_of = np.concatenate(y_of_all)
    hd_of = np.concatenate(hd_of_all)
    spk_of = np.concatenate(spk_of_all)

    print(f"      Total OF: {len(x_of)} samples, {spk_of.sum()} spikes")

    # Load chase sessions
    x_c_all, y_c_all, hd_c_all, spk_c_all = [], [], [], []

    for sess in chase_sessions:
        try:
            print(f"      Loading {sess}...")
            d = base.load_session_data(folder_loc, animal, sess, channels, binsize)
            sel = base.extract_chase_intervals(d, sess, animal, "chase")
            if sel is None:
                print(f"        ✗ {sess}: no chase intervals")
                continue

            x, y, hd, spk, _, _, _ = base.prepare_data(d, neuron_idx, sel, True)
            print(f"        ✓ {sess}: {len(x)} samples, {spk.sum()} spikes")

            x_c_all.append(x)
            y_c_all.append(y)
            hd_c_all.append(hd)
            spk_c_all.append(spk)
        except Exception as e:
            print(f"        ✗ {sess} failed: {e}")
            continue

    if not x_c_all:
        print(f"      ✗ No chase data loaded")
        return None

    x_c = np.concatenate(x_c_all)
    y_c = np.concatenate(y_c_all)
    hd_c = np.concatenate(hd_c_all)
    spk_c = np.concatenate(spk_c_all)

    print(f"      Total chase: {len(x_c)} samples, {spk_c.sum()} spikes")

    # Compute rate maps
    print(f"      Computing rate maps...")
    of_full = base.compute_ebc_ratemap(
        x_of, y_of, hd_of, spk_of, box_edges,
        dt_sec=dt, occ_min=occ_min,
        compute_distributions=False, n_shuffles=0
    )
    of_odd, of_even = base.compute_odd_even_splits(
        x_of, y_of, hd_of, spk_of, box_edges,
        dt_sec=dt, is_eboc=False, occ_min=occ_min
    )

    c_full = base.compute_ebc_ratemap(
        x_c, y_c, hd_c, spk_c, box_edges,
        dt_sec=dt, occ_min=occ_min,
        compute_distributions=False, n_shuffles=0
    )
    c_odd, c_even = base.compute_odd_even_splits(
        x_c, y_c, hd_c, spk_c, box_edges,
        dt_sec=dt, is_eboc=False, occ_min=occ_min
    )

    print(f"      ✓ Rate maps computed")

    return of_full, of_odd, of_even, c_full, c_odd, c_even


def analyze_animal_ebcs(classification_csv, animal, folder_loc, channels,
                        of_sessions, chase_sessions, binsize=0.008333, occ_min=50):
    """Analyze peak CC values for all classified EBCs from specified animal."""
    # Load classifications
    df = pd.read_csv(classification_csv)

    # Find neuron index column
    col_idx = None
    for c in ['neuron_idx', 'neuron_index', 'neuron']:
        if c in df.columns:
            col_idx = c
            break

    if col_idx is None:
        raise ValueError(f"Could not find neuron index column. Available: {list(df.columns)}")

    # Filter for EBCs
    col_class = None
    for c in ['classification', 'class', 'label']:
        if c in df.columns:
            col_class = c
            break

    if col_class is not None:
        ebc_mask = df[col_class].astype(str).str.contains('EBC', case=False, na=False)
        df_ebc = df[ebc_mask].copy()
    else:
        df_ebc = df.copy()

    # Filter for tuned+stable if columns exist
    col_sig = None
    for c in ['is_significant', 'tuned', 'is_tuned']:
        if c in df_ebc.columns:
            col_sig = c
            break

    col_stb = None
    for c in ['is_stable', 'stable']:
        if c in df_ebc.columns:
            col_stb = c
            break

    keep = pd.Series(True, index=df_ebc.index)
    if col_sig is not None:
        keep &= df_ebc[col_sig].astype(bool)
    if col_stb is not None:
        keep &= df_ebc[col_stb].astype(bool)

    df_ebc = df_ebc[keep].copy()

    print(f"\n{'=' * 60}")
    print(f"Found {len(df_ebc)} tuned+stable EBCs for {animal}")
    print(f"OF sessions: {', '.join(of_sessions)}")
    print(f"Chase sessions: {', '.join(chase_sessions)}")
    print(f"{'=' * 60}")

    results = []

    for idx, row in df_ebc.iterrows():
        neuron_idx_1based = int(row[col_idx])
        neuron_idx = neuron_idx_1based - 1  # Convert to 0-based

        print(f"\n[{len(results) + 1}/{len(df_ebc)}] Processing neuron {neuron_idx_1based}...")

        try:
            # Load data
            maps = load_ebc_data(
                folder_loc, animal, channels, neuron_idx, binsize, occ_min,
                of_sessions, chase_sessions
            )

            if maps is None:
                print(f"  ✗ SKIPPED (no data)")
                continue

            of_full, of_odd, of_even, c_full, c_odd, c_even = maps

            # Compute CCs
            print(f"    Computing cross-correlations...")
            cc_of_split = compute_cross_correlation_improved(of_odd, of_even, occ_min=occ_min)
            cc_of_chase = compute_cross_correlation_improved(of_full, c_full, occ_min=occ_min)

            # Extract peak values
            peak_of_split = get_peak_cc(cc_of_split)
            peak_of_chase = get_peak_cc(cc_of_chase)

            results.append({
                'animal': animal,
                'neuron_idx': neuron_idx_1based,
                'peak_cc_of_split': peak_of_split,
                'peak_cc_of_chase': peak_of_chase,
            })

            print(f"  ✓ SUCCESS: within={peak_of_split:.3f}, cross={peak_of_chase:.3f}")

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    return pd.DataFrame(results)


def plot_peak_cc_comparison(df_results, animal, out_path=None):
    """Plot comparison of peak CC values."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Remove NaN values
    valid = df_results['peak_cc_of_split'].notna() & df_results['peak_cc_of_chase'].notna()
    df_valid = df_results[valid].copy()

    within = df_valid['peak_cc_of_split'].values
    cross = df_valid['peak_cc_of_chase'].values

    # 1. Distributions
    ax = axes[0]
    ax.hist(within, bins=20, alpha=0.6, label='OF-even vs OF-odd', color='blue', edgecolor='black')
    ax.hist(cross, bins=20, alpha=0.6, label='OF vs Chase', color='red', edgecolor='black')
    ax.axvline(np.median(within), color='blue', ls='--', lw=2)
    ax.axvline(np.median(cross), color='red', ls='--', lw=2)
    ax.set_xlabel('Peak Cross-Correlation')
    ax.set_ylabel('Count')
    ax.set_title(f'{animal}: Distribution of Peak CC Values')
    ax.legend()
    ax.text(0.05, 0.95,
            f'Within: {np.median(within):.3f}\nCross: {np.median(cross):.3f}',
            transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 2. Scatter plot
    ax = axes[1]
    ax.scatter(within, cross, alpha=0.5, s=50)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('Peak CC: OF-even vs OF-odd')
    ax.set_ylabel('Peak CC: OF vs Chase')
    ax.set_title('Within-context vs Cross-context Stability')
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Add statistics
    r, p = stats.pearsonr(within, cross)
    ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.2e}',
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 3. Paired comparison
    ax = axes[2]
    diff = within - cross
    ax.hist(diff, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='k', ls='--', lw=2)
    ax.axvline(np.median(diff), color='red', ls='--', lw=2)
    ax.set_xlabel('Peak CC difference (within - cross)')
    ax.set_ylabel('Count')
    ax.set_title('Stability Difference')

    # Statistical test
    t_stat, p_val = stats.ttest_rel(within, cross)
    w_stat, p_wilcox = stats.wilcoxon(within, cross)

    ax.text(0.05, 0.95,
            f'Paired t-test:\nt = {t_stat:.3f}, p = {p_val:.2e}\n\n'
            f'Wilcoxon:\nW = {w_stat:.1f}, p = {p_wilcox:.2e}',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved figure to {out_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Analyze peak CC values for EBCs from any animal',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ToothMuch (OF1, c1, c2)
  python script.py --animal ToothMuch --of-sessions OF1 --chase-sessions c1,c2 \\
    --classification-csv toothmuch.csv --folder-loc /path/to/data \\
    --channels RSC --output toothmuch_results.csv

  # Arwen (OF1, OF2, c1-c5)
  python script.py --animal Arwen --of-sessions OF1,OF2 --chase-sessions c1,c2,c3,c4,c5 \\
    --classification-csv arwen.csv --folder-loc /path/to/data \\
    --channels RSC --output arwen_results.csv
        """
    )
    parser.add_argument('--animal', required=True,
                        help='Animal name (e.g., Arwen, ToothMuch, PreciousGrape, MimosaPudica)')
    parser.add_argument('--classification-csv', required=True,
                        help='Path to classification CSV')
    parser.add_argument('--folder-loc', required=True,
                        help='Data folder location')
    parser.add_argument('--channels', required=True,
                        help='Channel specification (e.g., RSC)')
    parser.add_argument('--of-sessions', required=True,
                        help='Comma-separated list of OF sessions (e.g., OF1,OF2)')
    parser.add_argument('--chase-sessions', required=True,
                        help='Comma-separated list of chase sessions (e.g., c1,c2,c3)')
    parser.add_argument('--binsize', type=float, default=0.008333,
                        help='Bin size in seconds (default: 0.008333)')
    parser.add_argument('--occ-min', type=float, default=50,
                        help='Minimum occupancy threshold (default: 50)')
    parser.add_argument('--output', required=True,
                        help='Output CSV path for results')
    parser.add_argument('--plot', help='Optional output path for figure')

    args = parser.parse_args()

    # Parse session lists
    of_sessions = [s.strip() for s in args.of_sessions.split(',')]
    chase_sessions = [s.strip() for s in args.chase_sessions.split(',')]

    # Run analysis
    df_results = analyze_animal_ebcs(
        args.classification_csv,
        args.animal,
        args.folder_loc,
        args.channels,
        of_sessions,
        chase_sessions,
        args.binsize,
        args.occ_min
    )

    # Save results
    df_results.to_csv(args.output, index=False)
    print(f"\n{'=' * 60}")
    print(f"Saved results to {args.output}")
    print(f"{'=' * 60}")

    # Print summary statistics
    if len(df_results) == 0:
        print("\n  No neurons successfully analyzed. Check error messages above.")
        return

    print("\n" + "=" * 60)
    print(f"SUMMARY STATISTICS - {args.animal}")
    print("=" * 60)
    print(f"Total EBCs analyzed: {len(df_results)}")
    print(f"\nPeak CC - OF-even vs OF-odd (within-context):")
    print(f"  Mean: {df_results['peak_cc_of_split'].mean():.3f}")
    print(f"  Median: {df_results['peak_cc_of_split'].median():.3f}")
    print(f"  Std: {df_results['peak_cc_of_split'].std():.3f}")
    print(f"\nPeak CC - OF vs Chase (cross-context):")
    print(f"  Mean: {df_results['peak_cc_of_chase'].mean():.3f}")
    print(f"  Median: {df_results['peak_cc_of_chase'].median():.3f}")
    print(f"  Std: {df_results['peak_cc_of_chase'].std():.3f}")

    valid = df_results['peak_cc_of_split'].notna() & df_results['peak_cc_of_chase'].notna()
    if valid.sum() > 0:
        within = df_results.loc[valid, 'peak_cc_of_split'].values
        cross = df_results.loc[valid, 'peak_cc_of_chase'].values
        t_stat, p_val = stats.ttest_rel(within, cross)
        print(f"\nPaired t-test (within vs cross):")
        print(f"  t = {t_stat:.3f}, p = {p_val:.2e}")

    # Generate plot if requested
    if args.plot and len(df_results) > 0:
        plot_peak_cc_comparison(df_results, args.animal, args.plot)
        plt.show()


if __name__ == "__main__":
    main()

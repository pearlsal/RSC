"""
Chase Cell Analysis Pipeline
============================
Identifies neurons significantly excited or suppressed during chasing intervals
by comparing firing rates inside vs outside intervals against a null distribution.

Original R code translated to Python with documentation.

Author: Pearl 
"""

import numpy as np
import pandas as pd
from scipy import io as sio
from scipy.ndimage import gaussian_filter1d
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from pathlib import Path
import warnings


def t_statistic(spikemat: np.ndarray, interval_bins: np.ndarray) -> np.ndarray:
    """
    Calculate unpaired t-statistic of firing rates inside vs outside intervals.

    Parameters
    ----------
    spikemat : np.ndarray
        Spike count matrix, shape (n_timebins, n_cells).
        Each column is a neuron, each row is a time bin.
    interval_bins : np.ndarray
        1D array of time bin indices considered "inside" the interval (e.g., chasing).

    Returns
    -------
    test_stats : np.ndarray
        T-statistic for each cell, shape (n_cells,).
        Positive = higher firing inside intervals.
    """
    n_timebins, n_cells = spikemat.shape

    # Get bins inside and outside intervals
    all_bins = np.arange(n_timebins)
    outside_bins = np.setdiff1d(all_bins, interval_bins)

    spikes_in = spikemat[interval_bins, :]  # (n_in, n_cells)
    spikes_out = spikemat[outside_bins, :]  # (n_out, n_cells)

    n_in = len(interval_bins)
    n_out = len(outside_bins)

    # Calculate t-statistic for each cell
    # t = (mean_in - mean_out) / sqrt(var_in/n_in + var_out/n_out)
    mean_in = np.mean(spikes_in, axis=0)
    mean_out = np.mean(spikes_out, axis=0)

    # Use ddof=1 for sample standard deviation (matches R's sd())
    std_in = np.std(spikes_in, axis=0, ddof=1)
    std_out = np.std(spikes_out, axis=0, ddof=1)

    # Welch's t-test denominator
    se = np.sqrt(std_in ** 2 / n_in + std_out ** 2 / n_out)

    # Avoid division by zero
    se[se == 0] = np.nan
    test_stats = (mean_in - mean_out) / se

    return test_stats


def sample_nonoverlapping_starts(n_timebins: int, lengths: np.ndarray) -> np.ndarray:
    """
    Construct non-overlapping start bins for the given interval lengths.
    Returns 0-indexed starts, sorted.

    Works by distributing the remaining free time into random gaps between intervals.
    """
    lengths = np.asarray(lengths, dtype=int)
    k = lengths.size
    total_len = int(lengths.sum())
    remaining = n_timebins - total_len
    if remaining < 0:
        raise ValueError("Intervals don't fit in session (sum(lengths) > n_timebins).")

    # Randomly split remaining time into (k+1) gaps (>=0)
    # Generate k cut points in [0..remaining], then convert to gaps
    cuts = np.sort(np.random.randint(0, remaining + 1, size=k))
    gaps = np.empty(k + 1, dtype=int)
    prev = 0
    for i, c in enumerate(cuts):
        gaps[i] = c - prev
        prev = c
    gaps[k] = remaining - prev

    starts = np.zeros(k, dtype=int)
    pos = gaps[0]
    starts[0] = pos
    pos += lengths[0]
    for i in range(1, k):
        pos += gaps[i]
        starts[i] = pos
        pos += lengths[i]

    return np.sort(starts)


def build_interval_bins_from_starts(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Build 0-indexed interval bins from starts and lengths."""
    bins = []
    for s, L in zip(starts, lengths):
        bins.extend(range(int(s), int(s) + int(L)))
    return np.asarray(bins, dtype=int)


def generate_null_distribution(
        spikemat: np.ndarray,
        interval_lengths: np.ndarray,
        num_intervals: int | list,
        num_shuffles: int = 1000,
        seed: int = 42,
        verbose: bool = True,
        max_coverage: float = 0.8
) -> np.ndarray:
    """
    Generate null distribution by randomly placing intervals.
    INTENT: match the provided R code as literally as possible.

    Key R-compat details:
    - R uses 1-indexed bins; we emulate that internally.
    - R samples starts from 1 .. (n_timebins - lengths[last]) inclusive.
    - R enforces non-overlap by re-drawing until unique bins == sum(lengths).

    If interval lengths are too long to fit in the null session, they will be
    scaled down proportionally.
    """
    np.random.seed(seed)

    n_timebins, n_cells = spikemat.shape

    # --- Normalize firing for each neuron: spikemat[,cell] /= mean(spikemat[,cell])
    spikemat_norm = spikemat.astype(float).copy()
    cell_means = np.mean(spikemat_norm, axis=0)
    # avoid divide-by-zero
    cell_means[cell_means == 0] = np.nan
    spikemat_norm = spikemat_norm / cell_means  # broadcasts over rows

    # num_intervals can be a scalar or a list/vector
    if isinstance(num_intervals, int):
        num_intervals_options = np.array([num_intervals], dtype=int)
    else:
        num_intervals_options = np.array(num_intervals, dtype=int)

    interval_lengths = np.array(interval_lengths, dtype=int)

    # Check if intervals can fit - if not, scale them down
    total_interval_length = interval_lengths.sum()
    max_allowed = int(n_timebins * max_coverage)

    if total_interval_length > max_allowed:
        scale_factor = max_allowed / total_interval_length
        interval_lengths_scaled = np.maximum(1, (interval_lengths * scale_factor).astype(int))
        if verbose:
            print(f"  ⚠ Scaling interval lengths by {scale_factor:.2f}x to fit in null session")
            print(f"    Original total: {total_interval_length} bins")
            print(f"    Scaled total: {interval_lengths_scaled.sum()} bins (max allowed: {max_allowed})")
        interval_lengths = interval_lengths_scaled

    nulldist = np.zeros((n_cells, num_shuffles), dtype=float)

    for i in range(num_shuffles):
        if verbose and (i + 1) % 100 == 0:
            print(f"Shuffle {i + 1}/{num_shuffles}")

        n_intervals = int(np.random.choice(num_intervals_options))

        # resample lengths until they fit
        for _ in range(10000):
            lengths = np.random.choice(interval_lengths, size=n_intervals, replace=True).astype(int)
            if lengths.sum() <= n_timebins:
                break
        else:
            raise RuntimeError("Could not sample interval lengths that fit in the session.")

        starts = sample_nonoverlapping_starts(n_timebins, lengths)
        interval_bins = build_interval_bins_from_starts(starts, lengths)

        nulldist[:, i] = t_statistic(spikemat_norm, interval_bins)

    return nulldist


def test_cells(
        spikemat: np.ndarray,
        interval_bins: np.ndarray,
        null_distribution: np.ndarray,
        suppressed: bool = False
) -> np.ndarray:
    """
    Test whether cells have significantly elevated/suppressed firing during intervals.

    Matches R's test_cells function exactly.
    """
    n_timebins, n_cells = spikemat.shape
    num_shuffles = null_distribution.shape[1]

    # Normalize firing for each neuron (same as R)
    spikemat_norm = spikemat.copy().astype(float)
    for cell in range(n_cells):
        cell_mean = np.mean(spikemat_norm[:, cell])
        if cell_mean > 0:
            spikemat_norm[:, cell] = spikemat_norm[:, cell] / cell_mean

    # Get actual t-statistics
    test_stats = t_statistic(spikemat_norm, interval_bins)

    # Calculate p-values (one-sided test)
    p_values = np.zeros(n_cells)
    for cell in range(n_cells):
        null_cell = null_distribution[cell, :]
        n_greater_or_equal = np.sum(null_cell >= test_stats[cell])
        p_values[cell] = (n_greater_or_equal + 1) / (num_shuffles + 1)

    # For suppression test, flip the p-values (R way)
    if suppressed:
        p_values = 1 - p_values + 1 / (num_shuffles + 1)

    return p_values


def _group_mean_trace(spikemat_plot: np.ndarray, idx: np.ndarray, normalized: bool) -> np.ndarray:
    n_timebins = spikemat_plot.shape[0]
    if len(idx) == 0:
        return np.zeros(n_timebins)

    X = spikemat_plot[:, idx].astype(float)

    if normalized:
        col_sums = np.sum(X, axis=0)
        col_sums[col_sums == 0] = 1.0
        X = X / col_sums

    if X.shape[1] == 1:
        return X[:, 0]

    return np.mean(X, axis=1)


def smooth_signal(signal: np.ndarray, window: float = 0.02) -> np.ndarray:
    """
    Smooth a 1D signal using a Gaussian-like filter.
    """
    window_size = max(1, int(len(signal) * window))
    sigma = window_size / 4
    return gaussian_filter1d(signal.astype(float), sigma=sigma, mode='nearest')


def plot_chase_activity(
        spikemat: np.ndarray,
        excited_idx: np.ndarray,
        suppressed_idx: np.ndarray,
        interval_starts: np.ndarray,
        interval_lengths: np.ndarray,
        smooth_window: float = 0.02,
        normalized: bool = False,
        title: str = "Chase Cell Activity",
        figsize: tuple = (14, 6)
):
    """
    Plot mean firing rates for excited, suppressed, and indifferent neurons.
    """
    n_timebins, n_cells = spikemat.shape
    all_idx = np.arange(n_cells)

    classified = np.union1d(excited_idx, suppressed_idx)
    indifferent_idx = np.setdiff1d(all_idx, classified)

    spikemat_plot = spikemat.astype(float).copy()

    if normalized:
        cell_sums = np.sum(spikemat_plot, axis=0)
        cell_sums[cell_sums == 0] = 1
        spikemat_plot = spikemat_plot / cell_sums

    excited_mean = _group_mean_trace(spikemat_plot, excited_idx, normalized)
    suppressed_mean = _group_mean_trace(spikemat_plot, suppressed_idx, normalized)
    indifferent_mean = _group_mean_trace(spikemat_plot, indifferent_idx, normalized)

    excited_smooth = smooth_signal(excited_mean, window=smooth_window)
    suppressed_smooth = smooth_signal(suppressed_mean, window=smooth_window)
    indifferent_smooth = smooth_signal(indifferent_mean, window=smooth_window)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n_timebins)

    if len(excited_idx) > 0:
        ax.plot(x, excited_smooth, color='royalblue', linewidth=1,
                label=f'Excited (n={len(excited_idx)})')

    if len(indifferent_idx) > 0:
        ax.plot(x, indifferent_smooth, color='black', linewidth=1,
                label=f'Indifferent (n={len(indifferent_idx)})')

    if len(suppressed_idx) > 0:
        ax.plot(x, suppressed_smooth, color='firebrick', linewidth=1,
                label=f'Suppressed (n={len(suppressed_idx)})')

    for start, length in zip(interval_starts, interval_lengths):
        ax.axvspan(start, start + length - 1, alpha=0.2, color='grey')
        ax.axvline(start, color='black', linestyle='--', linewidth=0.5)
        ax.axvline(start + length - 1, color='black', linestyle='--', linewidth=0.5)

    y_max = max(excited_smooth.max(), indifferent_smooth.max())
    if len(suppressed_idx) > 0:
        y_max = max(y_max, suppressed_smooth.max())
    ax.set_ylim(0, y_max * 1.1)

    ax.set_xlabel('Bin')
    ax.set_ylabel('Normalized mean spikes/bin over all chasing neurons')
    ax.set_title(title)
    ax.legend(loc='upper right')

    plt.tight_layout()
    return fig, ax


def run_analysis(
        spikemat_chase: np.ndarray,
        spikemat_null: np.ndarray,
        chase_intervals: np.ndarray,
        cell_ids: np.ndarray | list = None,
        num_shuffles: int = 1000,
        alpha: float = 0.025,
        seed: int = 42,
        verbose: bool = True
) -> dict:
    """
    Run the complete chase cell analysis.
    """
    n_cells = spikemat_chase.shape[1]
    n_cells_null = spikemat_null.shape[1]
    n_bins_chase = spikemat_chase.shape[0]
    n_bins_null = spikemat_null.shape[0]

    if n_cells != n_cells_null:
        raise ValueError(
            f"Number of cells must match! Chase has {n_cells}, null has {n_cells_null}."
        )

    if cell_ids is None:
        cell_ids = np.arange(1, n_cells + 1)
    else:
        cell_ids = np.array(cell_ids)
        if len(cell_ids) != n_cells:
            raise ValueError(f"cell_ids length ({len(cell_ids)}) must match n_cells ({n_cells})")

    if verbose:
        print(f"Chase session: {n_bins_chase} bins, {n_cells} cells")
        print(f"Null session:  {n_bins_null} bins, {n_cells_null} cells")

        if n_bins_null < n_bins_chase:
            print(f"  ⚠ Note: Null session is shorter ({n_bins_null} vs {n_bins_chase} bins)")
            # Check if chase intervals would even fit
            total_chase_time = sum(int(chase_intervals[i + 1]) - int(chase_intervals[i]) + 1
                                   for i in range(0, len(chase_intervals), 2))
            if total_chase_time > n_bins_null * 0.8:
                print(f"  ⚠ WARNING: Chase intervals ({total_chase_time} bins) exceed 80% of null session!")
                print(f"             Interval lengths will be scaled to fit.")
        elif n_bins_null > n_bins_chase:
            print(f"  ✓ Null session is longer - good for generating diverse null intervals")

    if len(chase_intervals) % 2 != 0:
        raise ValueError("chase_intervals must have even length (start, end pairs)")

    interval_bins = []
    interval_starts = []
    interval_lengths_list = []

    for i in range(0, len(chase_intervals), 2):
        start = int(chase_intervals[i])
        end = int(chase_intervals[i + 1])
        interval_starts.append(start)
        interval_lengths_list.append(end - start + 1)
        interval_bins.extend(range(start, end + 1))

    interval_bins = np.array(interval_bins)
    interval_starts = np.array(interval_starts)
    interval_lengths = np.array(interval_lengths_list)

    interval_bins = interval_bins[interval_bins < n_bins_chase]

    if verbose:
        print(f"\nChase intervals: {len(interval_starts)} bouts")
        print(f"  Total chase bins: {len(interval_bins)} ({100 * len(interval_bins) / n_bins_chase:.1f}% of session)")
        print(f"  Interval lengths: {interval_lengths.min()}-{interval_lengths.max()} bins")

    if verbose:
        print(f"\nGenerating null distribution ({num_shuffles} shuffles)...")

    nulldist = generate_null_distribution(
        spikemat=spikemat_null,
        interval_lengths=interval_lengths,
        num_intervals=len(interval_lengths),
        num_shuffles=num_shuffles,
        seed=seed,
        verbose=verbose
    )

    if verbose:
        print("\nTesting for excitation...")
    p_vals_excitation = test_cells(
        spikemat=spikemat_chase,
        interval_bins=interval_bins,
        null_distribution=nulldist,
        suppressed=False
    )

    if verbose:
        print("Testing for suppression...")
    p_vals_suppression = test_cells(
        spikemat=spikemat_chase,
        interval_bins=interval_bins,
        null_distribution=nulldist,
        suppressed=True
    )

    _, p_vals_exc_adj, _, _ = multipletests(p_vals_excitation, method='fdr_bh')
    _, p_vals_sup_adj, _, _ = multipletests(p_vals_suppression, method='fdr_bh')

    excited_idx = np.where(p_vals_exc_adj <= alpha)[0]
    suppressed_idx = np.where(p_vals_suppression <= alpha)[0]
    all_classified = np.union1d(excited_idx, suppressed_idx)
    indifferent_idx = np.setdiff1d(np.arange(n_cells), all_classified)

    classification = np.array(['indifferent'] * n_cells)
    classification[excited_idx] = 'excited'
    classification[suppressed_idx] = 'suppressed'

    cell_table = pd.DataFrame({
        'cell_id': cell_ids,
        'cell_index': np.arange(n_cells),
        'classification': classification,
        'p_value_excitation': p_vals_excitation,
        'p_value_excitation_adj': p_vals_exc_adj,
        'p_value_suppression': p_vals_suppression,
        'p_value_suppression_adj': p_vals_sup_adj,
    })

    sort_order = {'excited': 0, 'suppressed': 1, 'indifferent': 2}
    cell_table['_sort'] = cell_table['classification'].map(sort_order)
    cell_table = cell_table.sort_values(['_sort', 'p_value_excitation_adj']).drop('_sort', axis=1)
    cell_table = cell_table.reset_index(drop=True)

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"RESULTS SUMMARY")
        print(f"{'=' * 50}")
        print(f"  Total cells:   {n_cells}")
        print(f"  Excited:       {len(excited_idx):3d} ({100 * len(excited_idx) / n_cells:5.1f}%)")
        print(f"  Suppressed:    {len(suppressed_idx):3d} ({100 * len(suppressed_idx) / n_cells:5.1f}%)")
        print(f"  Indifferent:   {len(indifferent_idx):3d} ({100 * len(indifferent_idx) / n_cells:5.1f}%)")
        print(f"{'=' * 50}")

        if len(excited_idx) > 0:
            print(f"\nExcited cells: {list(cell_ids[excited_idx])}")
        if len(suppressed_idx) > 0:
            print(f"Suppressed cells: {list(cell_ids[suppressed_idx])}")

    results = {
        'cell_table': cell_table,
        'excited_idx': excited_idx,
        'suppressed_idx': suppressed_idx,
        'indifferent_idx': indifferent_idx,
        'excited_cell_ids': cell_ids[excited_idx],
        'suppressed_cell_ids': cell_ids[suppressed_idx],
        'indifferent_cell_ids': cell_ids[indifferent_idx],
        'p_values_excited': p_vals_excitation,
        'p_values_suppressed': p_vals_suppression,
        'p_values_excited_adj': p_vals_exc_adj,
        'p_values_suppressed_adj': p_vals_sup_adj,
        'null_distribution': nulldist,
        'interval_bins': interval_bins,
        'interval_starts': interval_starts,
        'interval_lengths': interval_lengths,
    }

    return results


def save_results(
        results: dict,
        output_dir: str = '.',
        prefix: str = 'chase_analysis',
        save_csv: bool = True,
        save_plot: bool = True,
        spikemat: np.ndarray = None,
        smooth_window: float = 0.02,
        title: str = None
):
    """
    Save analysis results to files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    if save_csv:
        csv_path = output_dir / f"{prefix}_cell_classification.csv"
        results['cell_table'].to_csv(csv_path, index=False)
        saved_files['csv'] = str(csv_path)
        print(f"Saved: {csv_path}")

    if save_plot and spikemat is not None:
        plot_path = output_dir / f"{prefix}_activity_plot.pdf"
        fig, ax = plot_chase_activity(
            spikemat=spikemat,
            excited_idx=results['excited_idx'],
            suppressed_idx=results['suppressed_idx'],
            interval_starts=results['interval_starts'],
            interval_lengths=results['interval_lengths'],
            smooth_window=smooth_window,
            normalized=True,
            title=title or prefix
        )

        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_files['plot'] = str(plot_path)
        print(f"Saved: {plot_path}")

    return saved_files


def convert_intervals_to_bins(
        intervals: np.ndarray,
        from_binsize_ms: float,
        to_binsize_ms: float
) -> np.ndarray:
    """
    Convert interval indices from one bin size to another.
    """
    conversion_factor = from_binsize_ms / to_binsize_ms
    converted = np.round(intervals * conversion_factor).astype(int)
    return converted


def prepare_chase_intervals(
        intervals: np.ndarray,
        from_binsize_ms: float = None,
        to_binsize_ms: float = None,
        max_bin: int = None,
        verbose: bool = True
) -> np.ndarray:
    """
    Prepare chase intervals for analysis, with optional bin size conversion.
    """
    intervals = np.array(intervals).flatten()

    if len(intervals) % 2 != 0:
        raise ValueError(f"Intervals must have even length (start, end pairs). Got {len(intervals)}")

    if from_binsize_ms is not None:
        if to_binsize_ms is None:
            raise ValueError("Must specify to_binsize_ms when from_binsize_ms is set")

        if verbose:
            print(f"Converting intervals from {from_binsize_ms}ms to {to_binsize_ms}ms bins")
            print(f"  Conversion factor: {from_binsize_ms / to_binsize_ms:.4f}")

        intervals_original = intervals.copy()
        intervals = convert_intervals_to_bins(intervals, from_binsize_ms, to_binsize_ms)

        if verbose:
            print(f"  Original (in {from_binsize_ms}ms): {intervals_original}")
            print(f"  Converted (in {to_binsize_ms}ms): {intervals}")

    if max_bin is not None:
        clipped = False
        for i in range(len(intervals)):
            if intervals[i] > max_bin:
                intervals[i] = max_bin
                clipped = True
        if clipped and verbose:
            print(f"  Warning: Some intervals clipped to max_bin={max_bin}")

    n_bouts = len(intervals) // 2
    binsize_ms = to_binsize_ms if to_binsize_ms else 50

    if verbose:
        print(f"\nPrepared {n_bouts} chase bouts:")
        total_bins = 0
        for i in range(0, len(intervals), 2):
            start, end = intervals[i], intervals[i + 1]
            duration = end - start + 1
            total_bins += duration
            duration_sec = duration * binsize_ms / 1000
            print(f"  Bout {i // 2 + 1}: bins {start:5d} - {end:5d} ({duration:4d} bins, {duration_sec:5.1f}s)")
        print(f"  Total chase: {total_bins} bins ({total_bins * binsize_ms / 1000:.1f}s)")

    return intervals


# =============================================================================
# DATA LOADING HELPERS
# =============================================================================

def load_spikemat_from_mat(filepath: str, variable_name: str = 'spikemat') -> tuple:
    """
    Load spike matrix and cell names from MATLAB .mat file.
    """
    mat_data = sio.loadmat(filepath)

    if variable_name not in mat_data:
        available = [k for k in mat_data.keys() if not k.startswith('__')]
        raise KeyError(f"Variable '{variable_name}' not found. Available: {available}")

    spikemat = mat_data[variable_name]

    if spikemat.shape[0] < spikemat.shape[1]:
        spikemat = spikemat.T

    cell_ids = None
    if 'cell_names' in mat_data:
        cell_names_raw = mat_data['cell_names']
        if cell_names_raw.dtype.kind == 'U':
            cell_ids = cell_names_raw.flatten()
        elif cell_names_raw.dtype == object:
            cell_ids = np.array([str(x[0]) if hasattr(x, '__len__') else str(x)
                                 for x in cell_names_raw.flatten()])
        else:
            cell_ids = cell_names_raw.flatten().astype(str)

    metadata = {}
    skip_keys = {'__header__', '__version__', '__globals__', variable_name, 'cell_names'}
    for key in mat_data.keys():
        if key not in skip_keys:
            metadata[key] = mat_data[key]

    return spikemat, cell_ids, metadata


def load_chase_session(filepath: str) -> dict:
    """
    Convenience function to load a complete chase session file.
    """
    spikemat, cell_ids, metadata = load_spikemat_from_mat(filepath)

    session = {
        'spikemat': spikemat,
        'cell_ids': cell_ids,
        'n_cells': spikemat.shape[1],
        'n_timebins': spikemat.shape[0],
    }

    for key, val in metadata.items():
        if hasattr(val, 'shape') and len(val.shape) == 2 and val.shape[0] == 1:
            session[key] = val.flatten()
        elif hasattr(val, 'shape') and len(val.shape) == 2 and val.shape[1] == 1:
            session[key] = val.flatten()
        else:
            session[key] = val

    if 'binsize' in session:
        if hasattr(session['binsize'], 'flatten'):
            session['binsize'] = str(session['binsize'].flatten()[0])

    return session


def merge_chase_sessions(
        session_files: list,
        chase_intervals_per_session: dict = None,
        verbose: bool = True
) -> dict:
    """
    Merge multiple chase session files into a single combined session.

    Parameters
    ----------
    session_files : list of str
        List of paths to session .mat files to merge.
    chase_intervals_per_session : dict, optional
        Dictionary mapping session index (0-based) or filename pattern to chase intervals.
        E.g., {0: np.array([...]), 1: np.array([...])}
        Or: {'c1': np.array([...]), 'c2': np.array([...])}
    verbose : bool
        Print progress information.

    Returns
    -------
    merged : dict
        Merged session dictionary containing:
        - 'spikemat': concatenated (n_total_timebins, n_cells)
        - 'cell_ids': cell identifiers
        - 'n_cells': number of cells
        - 'n_timebins': total number of time bins
        - 'session_boundaries': list of (start_bin, end_bin) for each session
        - 'session_files': list of source files
        - 'merged_chase_intervals': combined intervals with offsets applied (if provided)
    """
    if len(session_files) == 0:
        raise ValueError("No session files provided")

    if len(session_files) == 1:
        if verbose:
            print("Only one session provided - loading without merging")
        session = load_chase_session(session_files[0])
        session['session_boundaries'] = [(0, session['n_timebins'] - 1)]
        session['session_files'] = session_files

        # ✅ IMPORTANT: attach chase intervals even for single session
        if chase_intervals_per_session is not None:
            intervals = None

            # Try by index (0)
            if 0 in chase_intervals_per_session:
                intervals = chase_intervals_per_session[0]
            else:
                # Try by filename pattern (e.g. 'c1', 'c2')
                filename = Path(session_files[0]).stem
                for key in chase_intervals_per_session.keys():
                    if isinstance(key, str) and key in filename:
                        intervals = chase_intervals_per_session[key]
                        break

            if intervals is not None and len(intervals) > 0:
                session['merged_chase_intervals'] = np.array(intervals, dtype=int)
                if verbose:
                    print(f"  Single session: {len(intervals) // 2} chase intervals attached")

        return session

    # Load all sessions
    sessions = []
    for i, filepath in enumerate(session_files):
        if verbose:
            print(f"Loading session {i + 1}/{len(session_files)}: {Path(filepath).name}")
        sessions.append(load_chase_session(filepath))

    # Verify cell consistency
    reference_cells = sessions[0]['cell_ids']
    n_cells = sessions[0]['n_cells']

    for i, sess in enumerate(sessions[1:], start=1):
        if sess['n_cells'] != n_cells:
            raise ValueError(
                f"Cell count mismatch: Session 0 has {n_cells} cells, "
                f"session {i} has {sess['n_cells']} cells"
            )
        if not np.array_equal(sess['cell_ids'], reference_cells):
            warnings.warn(
                f"Cell IDs differ between session 0 and session {i}. "
                f"Assuming same neurons in same order."
            )

    # Calculate session boundaries and total length
    session_boundaries = []
    current_bin = 0
    for sess in sessions:
        start = current_bin
        end = current_bin + sess['n_timebins'] - 1
        session_boundaries.append((start, end))
        current_bin = end + 1

    total_timebins = current_bin

    if verbose:
        print(f"\nMerging {len(sessions)} sessions:")
        for i, (sess, (start, end)) in enumerate(zip(sessions, session_boundaries)):
            print(f"  Session {i}: {sess['n_timebins']} bins -> bins {start}-{end}")
        print(f"  Total: {total_timebins} bins ({n_cells} cells)")

    # Concatenate spike matrices
    spikemat_list = [sess['spikemat'] for sess in sessions]
    merged_spikemat = np.vstack(spikemat_list)

    # Build merged session dict
    merged = {
        'spikemat': merged_spikemat,
        'cell_ids': reference_cells,
        'n_cells': n_cells,
        'n_timebins': total_timebins,
        'session_boundaries': session_boundaries,
        'session_files': session_files,
    }

    # Merge behavioral variables if present
    behavioral_vars = ['binned_pos', 'binned_hd', 'binned_speed', 'binned_time']
    for var in behavioral_vars:
        if all(var in sess for sess in sessions):
            arrays = [sess[var] for sess in sessions]
            if arrays[0].ndim == 1:
                merged[var] = np.concatenate(arrays)
            else:
                merged[var] = np.vstack(arrays)
            if verbose:
                print(f"  Merged {var}: shape {merged[var].shape}")

    # Merge chase intervals if provided
    if chase_intervals_per_session is not None:
        merged_intervals = []

        for i, (filepath, (offset, _)) in enumerate(zip(session_files, session_boundaries)):
            intervals = None

            # Try by index first
            if i in chase_intervals_per_session:
                intervals = chase_intervals_per_session[i]
            else:
                # Try by filename patterns (c1, c2, etc.)
                filename = Path(filepath).stem
                for key in chase_intervals_per_session.keys():
                    if isinstance(key, str) and key in filename:
                        intervals = chase_intervals_per_session[key]
                        break

            if intervals is not None:
                intervals = np.array(intervals)
                if len(intervals) > 0:  # Only add if there are intervals
                    adjusted_intervals = intervals + offset
                    merged_intervals.extend(adjusted_intervals.tolist())
                    if verbose:
                        n_intervals = len(intervals) // 2
                        print(f"  Session {i}: {n_intervals} chase intervals (offset +{offset})")
            else:
                if verbose:
                    print(f"  Session {i}: No chase intervals provided")

        if merged_intervals:
            merged['merged_chase_intervals'] = np.array(merged_intervals, dtype=int)
            if verbose:
                total_intervals = len(merged['merged_chase_intervals']) // 2
                print(f"  Total merged intervals: {total_intervals}")

    return merged


def rebin_session(
        session: dict,
        from_binsize_ms: float,
        to_binsize_ms: float,
        verbose: bool = True
) -> dict:
    """
    Rebin a session from one bin size to another (e.g., 8ms to 50ms).

    Parameters
    ----------
    session : dict
        Session dictionary with 'spikemat' and optionally behavioral variables.
    from_binsize_ms : float
        Original bin size in milliseconds (e.g., 8.33).
    to_binsize_ms : float
        Target bin size in milliseconds (e.g., 50).
    verbose : bool
        Print progress information.

    Returns
    -------
    rebinned : dict
        New session dictionary with rebinned data.

    Notes
    -----
    - Spike counts are SUMMED within each new bin
    - Behavioral variables (pos, hd, speed) are AVERAGED within each new bin
    """
    bin_factor = to_binsize_ms / from_binsize_ms

    if bin_factor < 1:
        raise ValueError(f"Cannot rebin to smaller bins: {from_binsize_ms}ms -> {to_binsize_ms}ms")

    if abs(bin_factor - round(bin_factor)) > 0.01:
        bin_factor_int = int(round(bin_factor))
        if verbose:
            print(f"Note: Bin factor {bin_factor:.2f} rounded to {bin_factor_int}")
    else:
        bin_factor_int = int(round(bin_factor))

    spikemat = session['spikemat']
    n_timebins_orig, n_cells = spikemat.shape

    n_timebins_new = n_timebins_orig // bin_factor_int
    n_timebins_used = n_timebins_new * bin_factor_int

    if verbose:
        print(f"Rebinning: {from_binsize_ms}ms -> {to_binsize_ms}ms (factor {bin_factor_int}x)")
        print(f"  Original: {n_timebins_orig} bins")
        print(f"  New: {n_timebins_new} bins")
        if n_timebins_used < n_timebins_orig:
            print(f"  (Truncating {n_timebins_orig - n_timebins_used} bins at end)")

    # Rebin spike matrix by SUMMING
    spikemat_trimmed = spikemat[:n_timebins_used, :]
    spikemat_reshaped = spikemat_trimmed.reshape(n_timebins_new, bin_factor_int, n_cells)
    spikemat_rebinned = np.sum(spikemat_reshaped, axis=1)

    rebinned = {
        'spikemat': spikemat_rebinned,
        'cell_ids': session.get('cell_ids'),
        'n_cells': n_cells,
        'n_timebins': n_timebins_new,
        'binsize': f'{to_binsize_ms}ms',
        'original_binsize': f'{from_binsize_ms}ms',
        'bin_factor': bin_factor_int,
    }

    if 'session_boundaries' in session:
        new_boundaries = []
        for start, end in session['session_boundaries']:
            new_start = start // bin_factor_int
            new_end = min(end // bin_factor_int, n_timebins_new - 1)
            new_boundaries.append((new_start, new_end))
        rebinned['session_boundaries'] = new_boundaries
        if verbose:
            print(f"  Adjusted session boundaries: {new_boundaries}")

    if 'session_files' in session:
        rebinned['session_files'] = session['session_files']

    # Rebin behavioral variables by AVERAGING
    behavioral_vars_1d = ['binned_hd', 'binned_speed', 'binned_time']
    for var in behavioral_vars_1d:
        if var in session and session[var] is not None:
            arr = session[var]
            if len(arr) >= n_timebins_used:
                arr_trimmed = arr[:n_timebins_used]
                arr_reshaped = arr_trimmed.reshape(n_timebins_new, bin_factor_int)
                rebinned[var] = np.mean(arr_reshaped, axis=1)

    if 'binned_pos' in session and session['binned_pos'] is not None:
        pos = session['binned_pos']
        if len(pos) >= n_timebins_used:
            pos_trimmed = pos[:n_timebins_used]
            if pos_trimmed.ndim == 2:
                pos_reshaped = pos_trimmed.reshape(n_timebins_new, bin_factor_int, -1)
                rebinned['binned_pos'] = np.mean(pos_reshaped, axis=1)
            else:
                pos_reshaped = pos_trimmed.reshape(n_timebins_new, bin_factor_int)
                rebinned['binned_pos'] = np.mean(pos_reshaped, axis=1)

    # Handle chase intervals if present
    if 'merged_chase_intervals' in session:
        old_intervals = session['merged_chase_intervals']
        new_intervals = (old_intervals / bin_factor_int).astype(int)
        new_intervals = np.clip(new_intervals, 0, n_timebins_new - 1)
        rebinned['merged_chase_intervals'] = new_intervals
        if verbose:
            print(f"  Adjusted chase intervals to new bin size")

    return rebinned


def build_session_paths(
        folder: str,
        animal: str,
        sessions: list,
        binsize: str = '8ms',
        prefix: str = 'RSC'
) -> list:
    """
    Build file paths for multiple sessions.
    """
    paths = []
    for sess in sessions:
        filename = f"{prefix}_{sess}_binnedshareddata{binsize}.mat"
        path = Path(folder) / animal / filename
        paths.append(str(path))
    return paths
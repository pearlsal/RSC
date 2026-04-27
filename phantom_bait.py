#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
PHANTOM BAIT CONTROL: "Egocentric Cage" Analysis for ETCs
==============================================================================

PURPOSE:
    Test whether egocentric target cell (ETC) firing fields are bait-dependent
    by placing an "egocentric cage" around the animal's head during the open
    field — using the same egocentric bins the bait occupied during pursuit —
    and asking: do the cells fire in a structured way in those bins when no
    bait is present?

LOGIC:
    1. From PURSUIT: get the real bait trajectory in egocentric coords
       (bait_angle, bait_dist). This defines which (angle, dist) bins were
       visited → the "egocentric cage".
    2. From OPEN FIELD: the animal moves freely with no bait. We randomly
       assign "phantom bait" positions by sampling from the pursuit bait
       distribution (preserving the sampling density of egocentric space).
    3. Compute the EBOC ratemap for OF spikes using phantom bait positions.
    4. If ETCs encode the bait specifically, the OF ratemap should show NO
       structured tuning — just a disorganized mess.
    5. Repeat N times (different random assignments) and compare:
       - MRL / MI of real pursuit vs phantom-bait OF
       - Visual comparison of ratemaps

EXPECTED RESULT:
    Pursuit EBOC maps: structured, with clear preferred direction/distance
    Phantom-bait OF maps: flat, disorganized, low MRL/MI

USAGE:
    Modify the LoaderConfig at the bottom of this script, then run:
        python phantom_bait_control.py

    Or import and call:
        from phantom_bait_control import run_phantom_bait_control
        results = run_phantom_bait_control(cfg, neuron_indices=[30, 59, 110])

==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict
import warnings

# Import everything from the main pipeline
from FINAL_COMPLETE_SCRIPT_matlab_rank_threshold import (
    LoaderConfig, BOX_EDGES, GLOBAL_RNG, OCCUPANCY_THRESHOLD, BLOCK_SIZE_BINS,
    load_session_data, extract_chase_intervals, prepare_data,
    compute_eboc_ratemap, compute_ebc_ratemap,
    smooth_mat_wrapped, circ_r, circ_mean, wrap_to_pi,
    _polar_mesh_wrapped, _compute_vmin_vmax,
    parse_neuron_selection,
)

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════
# CORE: Phantom Bait Assignment
# ═══════════════════════════════════════════════════════════════════

def assign_phantom_bait(n_of_bins: int,
                        pursuit_bait_angle: np.ndarray,
                        pursuit_bait_dist: np.ndarray,
                        method: str = "resample",
                        rng: np.random.Generator = None) -> tuple:
    """
    Assign phantom bait positions to OF time bins by sampling from the
    pursuit bait distribution.

    Parameters
    ----------
    n_of_bins : int
        Number of time bins in the OF session.
    pursuit_bait_angle : ndarray
        Egocentric bait angles from pursuit (radians, [-π, π]).
    pursuit_bait_dist : ndarray
        Egocentric bait distances from pursuit (cm).
    method : str
        "resample"  — draw with replacement from pursuit (angle, dist) pairs.
                      Preserves joint distribution of angle & distance.
        "shuffle"   — randomly permute pursuit pairs, tile to OF length.
                      Same marginal stats, different temporal order.
        "independent" — resample angle and distance independently.
                        Breaks joint structure (stricter control).
    rng : Generator
        Random number generator (for reproducibility).

    Returns
    -------
    phantom_angle : ndarray, shape (n_of_bins,)
    phantom_dist  : ndarray, shape (n_of_bins,)
    """
    if rng is None:
        rng = np.random.default_rng(0)

    n_pursuit = len(pursuit_bait_angle)

    if method == "resample":
        # Draw with replacement from (angle, dist) pairs
        idx = rng.integers(0, n_pursuit, size=n_of_bins)
        return pursuit_bait_angle[idx], pursuit_bait_dist[idx]

    elif method == "shuffle":
        # Permute pursuit pairs, tile if OF is longer
        idx = np.arange(n_pursuit)
        rng.shuffle(idx)
        # Tile to cover OF length
        reps = int(np.ceil(n_of_bins / n_pursuit))
        idx_tiled = np.tile(idx, reps)[:n_of_bins]
        return pursuit_bait_angle[idx_tiled], pursuit_bait_dist[idx_tiled]

    elif method == "independent":
        # Resample angle and distance independently (breaks joint structure)
        idx_a = rng.integers(0, n_pursuit, size=n_of_bins)
        idx_d = rng.integers(0, n_pursuit, size=n_of_bins)
        return pursuit_bait_angle[idx_a], pursuit_bait_dist[idx_d]

    else:
        raise ValueError(f"Unknown method: {method}")


# ═══════════════════════════════════════════════════════════════════
# CORE: Compute Phantom-Bait EBOC Ratemap for OF
# ═══════════════════════════════════════════════════════════════════

def compute_phantom_bait_ratemap(
        x_of, y_of, hd_of, spk_of,
        phantom_angle, phantom_dist,
        dt_sec=0.008333,
        occ_min=OCCUPANCY_THRESHOLD,
):
    """
    Compute EBOC-style ratemap for OF data using phantom bait positions.
    Uses the exact same binning as compute_eboc_ratemap.
    """
    return compute_eboc_ratemap(
        x_of, y_of, hd_of, spk_of,
        phantom_angle, phantom_dist,
        dt_sec=dt_sec,
        occ_min=occ_min,
        compute_distributions=False,
        n_shuffles=0,
    )


# ═══════════════════════════════════════════════════════════════════
# CORE: Run the Full Control for One Neuron
# ═══════════════════════════════════════════════════════════════════

def run_single_neuron_control(
        cfg: LoaderConfig,
        neuron_idx: int,
        n_repeats: int = 50,
        method: str = "resample",
) -> Dict:
    """
    For one neuron:
    1. Load pursuit data → real EBOC ratemap (with real bait)
    2. Load OF data → spikes with no bait
    3. Assign phantom bait N times → compute EBOC ratemaps
    4. Compare MRL, MI between real and phantom

    Returns dict with all results.
    """
    dt_sec = cfg.binsize

    # ── 1. Load PURSUIT data (real bait) ──
    print(f"  Loading pursuit data...")
    x_c_all, y_c_all, hd_c_all, spk_c_all = [], [], [], []
    bait_a_all, bait_d_all = [], []

    for sess in cfg.chase_sessions:
        d = load_session_data(cfg.folder_loc, cfg.which_animal, sess, cfg.which_channels, cfg.binsize)
        sel = extract_chase_intervals(d, sess, cfg.which_animal, cfg.chase_or_chill)
        if sel is not None:
            x, y, hd, spk, _, ba, bd = prepare_data(d, neuron_idx, sel, True)
            if ba is None or bd is None:
                print(f"    ⚠ No bait data in {sess}, skipping")
                continue
            x_c_all.append(x); y_c_all.append(y)
            hd_c_all.append(hd); spk_c_all.append(spk)
            bait_a_all.append(ba); bait_d_all.append(bd)

    if len(x_c_all) == 0:
        print(f"    ERROR: No pursuit data with bait for neuron {neuron_idx + 1}")
        return None

    x_c = np.concatenate(x_c_all)
    y_c = np.concatenate(y_c_all)
    hd_c = np.concatenate(hd_c_all)
    spk_c = np.concatenate(spk_c_all)
    bait_a = np.concatenate(bait_a_all)
    bait_d = np.concatenate(bait_d_all)

    # Real pursuit EBOC ratemap
    print(f"  Computing real pursuit EBOC ratemap...")
    pursuit_map = compute_eboc_ratemap(
        x_c, y_c, hd_c, spk_c, bait_a, bait_d,
        dt_sec=dt_sec, occ_min=cfg.occ_min,
        compute_distributions=True, n_shuffles=cfg.n_shuffles,
    )
    pursuit_map['bait_angle'] = bait_a
    pursuit_map['bait_dist'] = bait_d

    print(f"    Pursuit: MRL={pursuit_map['MRL']:.4f}, MI={pursuit_map['MI']:.4f}, "
          f"FR={pursuit_map['firing_rate']:.2f} Hz")

    # ── 2. Load OF data (no bait) ──
    print(f"  Loading open field data...")
    x_of_all, y_of_all, hd_of_all, spk_of_all = [], [], [], []

    for sess in cfg.of_sessions:
        d = load_session_data(cfg.folder_loc, cfg.which_animal, sess, cfg.which_channels, cfg.binsize)
        x, y, hd, spk, _, _, _ = prepare_data(d, neuron_idx, None, True)
        x_of_all.append(x); y_of_all.append(y)
        hd_of_all.append(hd); spk_of_all.append(spk)

    # Optionally add chill periods to OF (same as main pipeline)
    if cfg.add_chill_to_of:
        for sess in cfg.chase_sessions:
            d = load_session_data(cfg.folder_loc, cfg.which_animal, sess, cfg.which_channels, cfg.binsize)
            chill_bins = extract_chase_intervals(d, sess, cfg.which_animal, 'chill')
            if chill_bins is not None:
                x, y, hd, spk, _, _, _ = prepare_data(d, neuron_idx, chill_bins, True)
                x_of_all.append(x); y_of_all.append(y)
                hd_of_all.append(hd); spk_of_all.append(spk)

    x_of = np.concatenate(x_of_all)
    y_of = np.concatenate(y_of_all)
    hd_of = np.concatenate(hd_of_all)
    spk_of = np.concatenate(spk_of_all)

    n_of = len(x_of)
    of_fr = float(np.sum(spk_of)) / (n_of * dt_sec) if n_of > 0 else 0.0
    print(f"    OF: {n_of} bins, FR={of_fr:.2f} Hz")

    # ── 3. Phantom bait: repeated assignments ──
    print(f"  Running {n_repeats} phantom-bait iterations (method={method})...")
    phantom_mrls = np.zeros(n_repeats)
    phantom_mis = np.zeros(n_repeats)
    phantom_maps = []

    for rep in range(n_repeats):
        rng_rep = np.random.default_rng(rep * 1000 + neuron_idx)

        # Assign phantom bait
        ph_angle, ph_dist = assign_phantom_bait(
            n_of, bait_a, bait_d, method=method, rng=rng_rep,
        )

        # Compute EBOC ratemap with phantom bait
        ph_map = compute_phantom_bait_ratemap(
            x_of, y_of, hd_of, spk_of,
            ph_angle, ph_dist,
            dt_sec=dt_sec, occ_min=cfg.occ_min,
        )

        phantom_mrls[rep] = ph_map['MRL']
        phantom_mis[rep] = ph_map['MI']

        # Store first few maps for visualization
        if rep < 5:
            phantom_maps.append(ph_map)

        if (rep + 1) % 10 == 0:
            print(f"    ... {rep + 1}/{n_repeats} done")

    # ── 4. Summary statistics ──
    print(f"\n  ── Results for Neuron {neuron_idx + 1} ──")
    print(f"  Real pursuit:  MRL={pursuit_map['MRL']:.4f}  MI={pursuit_map['MI']:.4f}")
    print(f"  Phantom OF:    MRL={np.mean(phantom_mrls):.4f} ± {np.std(phantom_mrls):.4f}  "
          f"MI={np.mean(phantom_mis):.4f} ± {np.std(phantom_mis):.4f}")
    print(f"  MRL ratio (real/phantom): {pursuit_map['MRL'] / max(np.mean(phantom_mrls), 1e-9):.2f}x")
    print(f"  MI ratio  (real/phantom): {pursuit_map['MI'] / max(np.mean(phantom_mis), 1e-9):.2f}x")

    # p-value: fraction of phantom iterations with MRL >= real
    p_mrl = np.mean(phantom_mrls >= pursuit_map['MRL'])
    p_mi = np.mean(phantom_mis >= pursuit_map['MI'])
    print(f"  p(phantom MRL ≥ real): {p_mrl:.4f}")
    print(f"  p(phantom MI  ≥ real): {p_mi:.4f}")

    return {
        'neuron_idx': neuron_idx,
        'pursuit_map': pursuit_map,
        'phantom_maps': phantom_maps,
        'phantom_mrls': phantom_mrls,
        'phantom_mis': phantom_mis,
        'real_mrl': pursuit_map['MRL'],
        'real_mi': pursuit_map['MI'],
        'x_of': x_of, 'y_of': y_of, 'hd_of': hd_of, 'spk_of': spk_of,
        'x_c': x_c, 'y_c': y_c, 'hd_c': hd_c, 'spk_c': spk_c,
        'bait_a': bait_a, 'bait_d': bait_d,
        'of_fr': of_fr,
        'pursuit_fr': pursuit_map['firing_rate'],
        'p_mrl': p_mrl,
        'p_mi': p_mi,
        'method': method,
        'n_repeats': n_repeats,
    }


# ═══════════════════════════════════════════════════════════════════
# FIGURE: Comparison Panel
# ═══════════════════════════════════════════════════════════════════

def plot_phantom_bait_comparison(result: Dict, cfg: LoaderConfig, save_path: Optional[Path] = None):
    """
    Create comparison figure:
      Row 1: [Pursuit traj (ego)] [Real EBOC map] | [OF traj (ego phantom)] [Phantom EBOC map ×3]
      Row 2: [MRL distribution] [MI distribution]
    """
    ni = result['neuron_idx']
    pursuit_map = result['pursuit_map']
    phantom_maps = result['phantom_maps']

    n_phantom_show = min(3, len(phantom_maps))

    fig = plt.figure(figsize=(5 + 4 * n_phantom_show + 8, 9))

    # ── Top row: ratemaps ──
    n_top = 2 + n_phantom_show  # pursuit ego traj + pursuit map + N phantom maps
    gs_top = fig.add_gridspec(1, n_top, left=0.03, right=0.97, top=0.95, bottom=0.52, wspace=0.5)

    # Pursuit egocentric bait trajectory
    ax_traj = fig.add_subplot(gs_top[0, 0])
    ba = result['bait_a']
    bd = result['bait_d']
    bx = bd * np.cos(ba)
    by = bd * np.sin(ba)
    ax_traj.plot(bx, by, color='0.82', lw=0.5, alpha=0.7)
    # Color by time
    if len(bx) > 1:
        pts = np.column_stack([bx, by])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        t = np.linspace(0, 1, len(segs))
        lc = LineCollection(segs, cmap=cm.get_cmap('turbo'), array=t, linewidths=1.0, alpha=0.8)
        ax_traj.add_collection(lc)
    ax_traj.axhline(0, color='k', ls=':', lw=0.5)
    ax_traj.axvline(0, color='k', ls=':', lw=0.5)
    ax_traj.set_aspect('equal')
    ax_traj.set_title('Pursuit\n(real bait)', fontsize=10, fontweight='bold')
    ax_traj.axis('off')

    # Determine shared color scale across all maps
    all_maps = [pursuit_map] + phantom_maps[:n_phantom_show]
    vmin, vmax, _ = _compute_vmin_vmax(all_maps, cfg.occ_min)

    # Real pursuit EBOC map
    ax_real = fig.add_subplot(gs_top[0, 1])
    pc_real = _polar_mesh_wrapped(ax_real, pursuit_map, 'REAL pursuit', cfg.occ_min, vmin, vmax,
                                  annotate_fr=result['pursuit_fr'])

    # Add MRL/MI annotation
    ax_real.text(0.02, 0.98,
                 f"MRL={pursuit_map['MRL']:.3f}\nMI={pursuit_map['MI']:.3f}",
                 transform=ax_real.transAxes, fontsize=7.5, ha='left', va='top',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.9, lw=0.5))

    # Phantom bait EBOC maps
    for i in range(n_phantom_show):
        ax_ph = fig.add_subplot(gs_top[0, 2 + i])
        pc_ph = _polar_mesh_wrapped(ax_ph, phantom_maps[i],
                                    f'PHANTOM OF #{i+1}', cfg.occ_min, vmin, vmax,
                                    annotate_fr=result['of_fr'])
        ax_ph.text(0.02, 0.98,
                   f"MRL={phantom_maps[i]['MRL']:.3f}\nMI={phantom_maps[i]['MI']:.3f}",
                   transform=ax_ph.transAxes, fontsize=7.5, ha='left', va='top',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='lightcyan', alpha=0.9, lw=0.5))

    # ── Bottom row: distributions ──
    gs_bot = fig.add_gridspec(1, 2, left=0.08, right=0.92, top=0.42, bottom=0.06, wspace=0.35)

    # MRL distribution
    ax_mrl = fig.add_subplot(gs_bot[0, 0])
    ax_mrl.hist(result['phantom_mrls'], bins=25, color='steelblue', alpha=0.75,
                edgecolor='white', lw=0.5, label=f'Phantom OF (n={result["n_repeats"]})')
    ax_mrl.axvline(result['real_mrl'], color='red', lw=2.5, ls='-',
                   label=f'Real pursuit (MRL={result["real_mrl"]:.4f})')
    ax_mrl.set_xlabel('MRL (mean resultant length)', fontsize=11)
    ax_mrl.set_ylabel('Count', fontsize=11)
    ax_mrl.set_title('MRL: Real Pursuit vs Phantom-Bait OF', fontsize=11)
    ax_mrl.legend(fontsize=9)
    ax_mrl.spines['top'].set_visible(False)
    ax_mrl.spines['right'].set_visible(False)

    # Add p-value
    ax_mrl.text(0.98, 0.95, f"p = {result['p_mrl']:.4f}",
                transform=ax_mrl.transAxes, ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

    # MI distribution
    ax_mi = fig.add_subplot(gs_bot[0, 1])
    ax_mi.hist(result['phantom_mis'], bins=25, color='darkorange', alpha=0.75,
               edgecolor='white', lw=0.5, label=f'Phantom OF (n={result["n_repeats"]})')
    ax_mi.axvline(result['real_mi'], color='red', lw=2.5, ls='-',
                  label=f'Real pursuit (MI={result["real_mi"]:.4f})')
    ax_mi.set_xlabel('Skaggs MI (bits/spike)', fontsize=11)
    ax_mi.set_ylabel('Count', fontsize=11)
    ax_mi.set_title('MI: Real Pursuit vs Phantom-Bait OF', fontsize=11)
    ax_mi.legend(fontsize=9)
    ax_mi.spines['top'].set_visible(False)
    ax_mi.spines['right'].set_visible(False)

    ax_mi.text(0.98, 0.95, f"p = {result['p_mi']:.4f}",
               transform=ax_mi.transAxes, ha='right', va='top', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

    fig.suptitle(
        f"Phantom Bait Control — Neuron {ni + 1} | {cfg.which_animal} | "
        f"Method: {result['method']}\n"
        f"If ETCs encode the bait, phantom-bait OF maps should be disorganized "
        f"(low MRL/MI vs real pursuit)",
        fontsize=12, fontweight='bold', y=0.995,
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved figure: {save_path}")
    plt.close(fig)

    return fig


# ═══════════════════════════════════════════════════════════════════
# FIGURE: Population Summary
# ═══════════════════════════════════════════════════════════════════

def plot_population_summary(all_results: List[Dict], cfg: LoaderConfig, save_path: Optional[Path] = None):
    """
    Scatter: real MRL vs mean phantom MRL for each neuron.
    Points should cluster far above the diagonal if ETCs are bait-specific.
    """
    if len(all_results) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    real_mrls = [r['real_mrl'] for r in all_results]
    phantom_mrls_mean = [np.mean(r['phantom_mrls']) for r in all_results]
    phantom_mrls_std = [np.std(r['phantom_mrls']) for r in all_results]

    real_mis = [r['real_mi'] for r in all_results]
    phantom_mis_mean = [np.mean(r['phantom_mis']) for r in all_results]
    phantom_mis_std = [np.std(r['phantom_mis']) for r in all_results]

    neuron_labels = [r['neuron_idx'] + 1 for r in all_results]

    # MRL scatter
    ax = axes[0]
    ax.errorbar(phantom_mrls_mean, real_mrls,
                xerr=phantom_mrls_std, fmt='o', ms=8,
                color='steelblue', ecolor='lightsteelblue', capsize=3, alpha=0.8)
    for i, lbl in enumerate(neuron_labels):
        ax.annotate(str(lbl), (phantom_mrls_mean[i], real_mrls[i]),
                    fontsize=7, ha='left', va='bottom', xytext=(3, 3),
                    textcoords='offset points')

    lim = max(max(real_mrls), max(phantom_mrls_mean)) * 1.15
    ax.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.5, label='unity line')
    ax.set_xlabel('Mean Phantom-Bait OF MRL', fontsize=11)
    ax.set_ylabel('Real Pursuit MRL', fontsize=11)
    ax.set_title('MRL: Real vs Phantom', fontsize=12)
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # MI scatter
    ax = axes[1]
    ax.errorbar(phantom_mis_mean, real_mis,
                xerr=phantom_mis_std, fmt='o', ms=8,
                color='darkorange', ecolor='bisque', capsize=3, alpha=0.8)
    for i, lbl in enumerate(neuron_labels):
        ax.annotate(str(lbl), (phantom_mis_mean[i], real_mis[i]),
                    fontsize=7, ha='left', va='bottom', xytext=(3, 3),
                    textcoords='offset points')

    lim = max(max(real_mis), max(phantom_mis_mean)) * 1.15
    ax.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.5, label='unity line')
    ax.set_xlabel('Mean Phantom-Bait OF MI', fontsize=11)
    ax.set_ylabel('Real Pursuit MI', fontsize=11)
    ax.set_title('MI: Real vs Phantom', fontsize=12)
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle(
        f"Phantom Bait Control — Population ({len(all_results)} neurons) | {cfg.which_animal}\n"
        f"Points above diagonal = bait-dependent tuning",
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.91])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved population summary: {save_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════

def run_phantom_bait_control(
        cfg: LoaderConfig,
        neuron_indices: Optional[List[int]] = None,
        n_repeats: int = 50,
        method: str = "resample",
        output_dir: Optional[str] = None,
) -> List[Dict]:
    """
    Run the phantom bait control for specified neurons.

    Parameters
    ----------
    cfg : LoaderConfig
        Configuration (same as main pipeline).
    neuron_indices : list of int, optional
        0-indexed neuron indices. If None, parses from cfg.which_neurons.
    n_repeats : int
        Number of phantom-bait iterations per neuron (default 50).
    method : str
        Phantom bait assignment method: "resample", "shuffle", "independent".
    output_dir : str, optional
        Where to save figures. Defaults to cfg.output_dir.
    """
    print("=" * 70)
    print("PHANTOM BAIT CONTROL: Egocentric Cage Analysis")
    print("=" * 70)

    if output_dir is None:
        output_dir = cfg.output_dir
    out_dir = Path(output_dir) / "phantom_bait_control" / cfg.which_animal
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine neurons
    if neuron_indices is None:
        sample_sess = cfg.chase_sessions[0]
        sample = load_session_data(cfg.folder_loc, cfg.which_animal,
                                   sample_sess, cfg.which_channels, cfg.binsize)
        n_neurons = sample['spikemat'].shape[0]
        neuron_indices = parse_neuron_selection(cfg.which_neurons, n_neurons)

    print(f"Animals: {cfg.which_animal}")
    print(f"Neurons: {[ni + 1 for ni in neuron_indices]}")
    print(f"Repeats: {n_repeats} | Method: {method}")
    print(f"Output: {out_dir}")

    all_results = []

    for idx, ni in enumerate(neuron_indices):
        print(f"\n{'─' * 60}")
        print(f"Neuron {ni + 1} ({idx + 1}/{len(neuron_indices)})")
        print(f"{'─' * 60}")

        try:
            result = run_single_neuron_control(cfg, ni, n_repeats=n_repeats, method=method)
            if result is None:
                continue

            all_results.append(result)

            # Individual figure
            fig_path = out_dir / f"phantom_bait_neuron{ni + 1}.pdf"
            plot_phantom_bait_comparison(result, cfg, save_path=fig_path)

        except Exception as e:
            print(f"  ✗ ERROR on neuron {ni + 1}: {e}")
            import traceback
            traceback.print_exc()

    # Population summary
    if len(all_results) > 1:
        pop_path = out_dir / "phantom_bait_population_summary.pdf"
        plot_population_summary(all_results, cfg, save_path=pop_path)

    # Save summary CSV
    if len(all_results) > 0:
        import pandas as pd
        rows = []
        for r in all_results:
            rows.append({
                'neuron_idx': r['neuron_idx'] + 1,
                'pursuit_MRL': r['real_mrl'],
                'pursuit_MI': r['real_mi'],
                'pursuit_FR_Hz': r['pursuit_fr'],
                'of_FR_Hz': r['of_fr'],
                'phantom_MRL_mean': np.mean(r['phantom_mrls']),
                'phantom_MRL_std': np.std(r['phantom_mrls']),
                'phantom_MI_mean': np.mean(r['phantom_mis']),
                'phantom_MI_std': np.std(r['phantom_mis']),
                'MRL_ratio': r['real_mrl'] / max(np.mean(r['phantom_mrls']), 1e-9),
                'MI_ratio': r['real_mi'] / max(np.mean(r['phantom_mis']), 1e-9),
                'p_MRL': r['p_mrl'],
                'p_MI': r['p_mi'],
                'method': r['method'],
                'n_repeats': r['n_repeats'],
            })

        df = pd.DataFrame(rows)
        csv_path = out_dir / "phantom_bait_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Saved summary CSV: {csv_path}")
        print(f"\n{'=' * 70}")
        print("SUMMARY")
        print(f"{'=' * 70}")
        print(df.to_string(index=False))

    return all_results


# ═══════════════════════════════════════════════════════════════════
# __main__
# ═══════════════════════════════════════════════════════════════════

def load_etc_indices_from_csv(csv_path: str) -> List[int]:
    """
    Read an EBOC classification CSV and return 0-indexed neuron indices
    for all significant ETCs (is_significant == True).
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    sig = df[df['is_significant'] == True]
    # neuron_idx in CSV is 1-indexed; convert to 0-indexed
    indices = [int(x) - 1 for x in sig['neuron_idx']]
    print(f"  Loaded {len(indices)} significant ETCs from {Path(csv_path).name}")
    print(f"  Neuron IDs (1-indexed): {[i+1 for i in indices]}")
    return indices


# ═══════════════════════════════════════════════════════════════════
# ANIMAL CONFIGS — edit paths/sessions per animal
# ═══════════════════════════════════════════════════════════════════

ANIMAL_CONFIGS = {
    'Arwen': {
        'csv': "/Users/pearls/Library/Mobile Documents/com~apple~CloudDocs/work/EBOC/Arwen/EBOC_ARWEN_classification_summary.csv",
        'of_sessions': ["OF2"],
        'chase_sessions': ["c2"],
    },
    'ToothMuch': {
        'csv': "/Users/pearls/Library/Mobile Documents/com~apple~CloudDocs/work/EBOC/ToothMuch/EBOC_TOOTHMUCH_classification_summary.csv",
        'of_sessions': ["OF1"],
        'chase_sessions': ["c1", "c2"],
    },
    'PreciousGrape': {
        'csv': "/Users/pearls/Library/Mobile Documents/com~apple~CloudDocs/work/EBOC/PreciousGrape/EBOC_PRECIOUSGRAPE_classification_summary.csv",
        'of_sessions': ["OF1", "OF2"],
        'chase_sessions': ["c1", "c2"],
    },
    'MimosaPudica': {
        'csv': "/Users/pearls/Library/Mobile Documents/com~apple~CloudDocs/work/EBOC/MimosaPudica/EBOC_MIMOSAPUDICA_classification_summary.csv",
        'of_sessions': ["OF1", "OF2"],
        'chase_sessions': ["c1", "c2"],
    },
}


if __name__ == "__main__":

    # ── Which animals to run ──
    animals_to_run = ["Arwen"]  # Add more: ["Arwen", "ToothMuch", "PreciousGrape", "MimosaPudica"]

    for animal in animals_to_run:
        print(f"\n{'═' * 70}")
        print(f"  ANIMAL: {animal}")
        print(f"{'═' * 70}")

        acfg = ANIMAL_CONFIGS[animal]

        # Load significant ETC indices from CSV
        etc_indices = load_etc_indices_from_csv(acfg['csv'])
        if len(etc_indices) == 0:
            print(f"  No significant ETCs for {animal}, skipping.")
            continue

        cfg = LoaderConfig(
            folder_loc="/Users/pearls/Work/RSC_project/",
            which_animal=animal,
            which_channels="RSC",
            binsize=0.00833,

            of_sessions=acfg['of_sessions'],
            chase_sessions=acfg['chase_sessions'],

            ebc_or_eboc="EBOC",
            chase_or_chill="chase",
            add_chill_to_of=True,

            n_shuffles=100,
            mi_percentile=99.0,

            which_neurons="all",  # not used — we pass indices directly
            do_plot=True,
            save_results=True,
            output_dir=".",
            occ_min=OCCUPANCY_THRESHOLD,
        )

        results = run_phantom_bait_control(
            cfg,
            neuron_indices=etc_indices,  # ← only significant ETCs
            n_repeats=50,
            method="resample",
        )

    print("\n Phantom bait control complete for all animals!")

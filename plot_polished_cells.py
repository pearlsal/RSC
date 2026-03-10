#!/usr/bin/env python3
"""
Plot-only driver 
===============================================================



USAGE EXAMPLES:
    # Plot specific neurons (1-based indices):
    python plot_polished_cells_modified.py --mode EBC --neuron-list "5,12,47" \
        --folder-loc /path/to/data --animal Arwen --channels RSC --out-dir ./plots

    # Plot from CSV (original behavior):
    python plot_polished_cells_modified.py --mode EBC --classification-csv results.csv \
        --folder-loc /path/to/data --animal Arwen --channels RSC --out-dir ./plots

    # Plot a single cell:
    python plot_polished_cells_modified.py --mode EBC --neuron-list "42" \
        --folder-loc /path/to/data --animal Arwen --channels RSC --out-dir ./plots

FONT HIERARCHY (all Helvetica):
    - Panel/figure titles: 10pt bold
    - Subplot titles: 8pt
    - Axis labels: 8pt
    - Tick labels: 7pt
    - Annotations (FR, etc.): 7pt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from matplotlib.collections import LineCollection
from matplotlib import cm

# =============================================================================
# FONT CONFIGURATION - Helvetica with 12/10/8/7 hierarchy
# =============================================================================
plt.rcParams["pdf.fonttype"] = 42  # TrueType fonts in PDF
plt.rcParams["ps.fonttype"] = 42

# Set Helvetica as default (fallback to Arial, then sans-serif)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

# Font hierarchy: 12 (figure title) / 10 (subplot titles) / 8 (labels) / 7 (ticks)
plt.rcParams["font.size"] = 10  # Base size
plt.rcParams["figure.titlesize"] = 14  # Main figure title (suptitle)
plt.rcParams["axes.titlesize"] = 12  # Subplot titles
plt.rcParams["axes.labelsize"] = 10  # Axis labels (X, Y labels)
plt.rcParams["xtick.labelsize"] = 8  # Tick labels
plt.rcParams["ytick.labelsize"] = 8  # Tick labels
plt.rcParams["legend.fontsize"] = 10  # Legend text

# Optional smoothing (display only)
try:
    from scipy.ndimage import gaussian_filter
except Exception:
    gaussian_filter = None

import COMPLETE_Classification as base


# -----------------------------
# Config
# -----------------------------
@dataclass
class PlotCfg:
    folder_loc: str
    which_animal: str
    which_channels: str
    binsize: float = 0.008333
    occ_min: float = base.OCCUPANCY_THRESHOLD

    of_sessions: Tuple[str, ...] = ("OF1", "OF2")
    chase_sessions: Tuple[str, ...] = ("ob1", "ob2")
    chase_or_chill: str = "chase"
    add_chill_to_of: bool = False

    cmap: str = "jet"

    ring_radii_cm: Tuple[int, ...] = (20, 40, 60, 90)

    # Place ring labels away from 45° tick
    ring_label_angle_deg: float = 22.5

    # Font sizes - Helvetica 12/10/8/7 hierarchy
    theta_tick_fontsize: int = 7  # Tick labels
    title_fontsize: int = 10  # Subplot titles
    fr_fontsize: int = 8  # Firing rate annotation
    label_fontsize: int = 8  # Axis labels
    suptitle_fontsize: int = 12  # Main figure title
    colorbar_label_fontsize: int = 8  # Colorbar label
    colorbar_tick_fontsize: int = 7  # Colorbar ticks
    ring_label_fontsize: int = 7  # Distance ring labels (cm)

    theta_tick_step_deg: int = 45

    display_smooth_sigma: float = 0.0  # 0 = off (display-only smoothing)
    add_colorbar_to_each_polar: bool = False


# -----------------------------
# CSV selection
# -----------------------------
def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def select_neuron_indices_from_csv(csv_path: Path, mode: str) -> List[int]:
    df = pd.read_csv(csv_path)

    col_idx = _find_col(df, ["neuron_idx", "neuron_index", "neuron", "cell", "cell_id"])
    if col_idx is None:
        raise ValueError(f"Could not find neuron index column in {csv_path}. Columns: {list(df.columns)}")

    col_sig = _find_col(df, ["is_significant", "tuned", "is_tuned"])
    col_stb = _find_col(df, ["is_stable", "stable"])
    col_cls = _find_col(df, ["classification", "class", "label"])

    keep = pd.Series(True, index=df.index)
    if col_sig is not None:
        keep &= df[col_sig].astype(bool)
    if col_stb is not None:
        keep &= df[col_stb].astype(bool)
    if col_cls is not None:
        keep &= df[col_cls].astype(str).str.contains(mode, case=False, na=False)

    sub = df.loc[keep].copy()
    if sub.empty:
        raise RuntimeError(f"No neurons selected from {csv_path}. Check CSV columns/filters.")

    # CSV is 1-based; convert to 0-based
    idx1 = sub[col_idx].astype(int).tolist()
    idx0 = [i - 1 for i in idx1 if int(i) > 0]
    return sorted(idx0)


# -----------------------------
# Helpers
# -----------------------------
def nan_aware_gaussian(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or gaussian_filter is None:
        return x
    x = np.array(x, float)
    m = np.isfinite(x).astype(float)
    x0 = np.nan_to_num(x, nan=0.0)
    num = gaussian_filter(x0 * m, sigma=sigma, mode="nearest")
    den = gaussian_filter(m, sigma=sigma, mode="nearest")
    out = num / np.maximum(den, 1e-9)
    out[den < 1e-6] = np.nan
    return out


def get_fr_hz(rm_data: Dict[str, Any], dt_sec: float) -> float:
    if "firing_rate" in rm_data:
        try:
            return float(rm_data["firing_rate"])
        except Exception:
            pass
    if "spike" in rm_data:
        spk = np.asarray(rm_data["spike"]).ravel()
        return float(np.nansum(spk)) / (len(spk) * dt_sec)
    return float("nan")


def compute_vmin_vmax_hz(maps: List[Dict], occ_min: float, dt_sec: float, perc: float = 98.0):
    vals = []
    for m in maps:
        rm = np.array(m["rm"], float).copy()
        occ = np.array(m["occ_ns"], float).copy()
        rm[occ < occ_min] = np.nan
        rm_hz = rm / dt_sec
        vals.append(rm_hz[np.isfinite(rm_hz)])
    if not vals:
        return 0.0, 1.0
    allv = np.concatenate(vals) if len(vals) else np.array([])
    if allv.size == 0:
        return 0.0, 1.0
    vmax = float(np.nanpercentile(allv, perc))
    vmax = max(vmax, 1e-6)
    return 0.0, vmax


# -----------------------------
# Polar plotting (true polar axes)
# -----------------------------
def polar_pcolormesh(ax, rm_data: Dict, title: str, cfg: PlotCfg, vmin: float, vmax: float, annotate_fr: bool = True):
    """
    True polar plot with consistent fonts.
    """
    dt = cfg.binsize

    rm = np.array(rm_data["rm"], float).copy()
    occ = np.array(rm_data["occ_ns"], float).copy()
    rm[occ < cfg.occ_min] = np.nan
    rm_hz = rm / dt

    # display-only smoothing (optional)
    rm_hz = nan_aware_gaussian(rm_hz, cfg.display_smooth_sigma)

    theta_centers = np.asarray(rm_data["params"]["thetaBins"])
    dist_edges = np.asarray(rm_data["params"]["distanceBins"])  # edges, in cm
    ntheta = len(theta_centers)

    # theta edges + wrap for continuity
    theta_edges = np.linspace(-np.pi, np.pi, ntheta + 1)
    dtheta = theta_edges[1] - theta_edges[0]
    theta_edges_wrapped = np.r_[theta_edges, theta_edges[-1] + dtheta]

    # wrap data column
    C = np.c_[rm_hz, rm_hz[:, 0]]

    Θ, R = np.meshgrid(theta_edges_wrapped, dist_edges)

    cmap = plt.get_cmap(cfg.cmap).copy()
    cmap.set_bad("white")  # white background for NaN

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    pc = ax.pcolormesh(
        Θ, R, C,
        cmap=cmap,
        norm=Normalize(vmin=vmin, vmax=vmax),
        shading="flat",
    )

    # Grid + ticks in black
    ax.grid(True, color="black", alpha=0.35, lw=0.8, ls="--")

    # theta ticks (hide 180° label to avoid colorbar overlap)
    step = int(cfg.theta_tick_step_deg)
    ticks = np.arange(0, 360, step)
    labels = [("" if t == 180 else f"{t}°") for t in ticks]
    ax.set_thetagrids(ticks, labels=labels, fontsize=cfg.theta_tick_fontsize, color="black")

    # radial rings
    max_r = float(dist_edges[-1])
    rings = [r for r in cfg.ring_radii_cm if r <= max_r]
    ax.set_yticks(rings)
    ax.set_yticklabels([""] * len(rings))

    # ring labels (moved away from 45° and slightly inward)
    ang = np.deg2rad(cfg.ring_label_angle_deg)
    for r in rings:
        rr = max(r - 2.0, 0.0)
        ax.text(
            ang, rr,
            f"{r} cm",
            color="black", alpha=0.65, fontsize=cfg.ring_label_fontsize,
            ha="left", va="center",
        )

    ax.set_title(title, fontsize=cfg.title_fontsize, pad=10)

    if annotate_fr:
        fr = get_fr_hz(rm_data, dt)
        if np.isfinite(fr):
            ax.text(
                0.02, -0.12,
                f"FR {fr:.2f} Hz",
                transform=ax.transAxes,
                fontsize=cfg.fr_fontsize,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.85),
            )

    return pc


def add_horizontal_cbar(fig, ax, mappable, label="Firing Rate (Hz)", y_offset=-0.22, height="7%", cfg: PlotCfg = None):
    # Get axis position in figure coordinates (more stable for PDF export)
    fig.canvas.draw()  # Force layout calculation
    bbox = ax.get_position()

    # Parse height percentage to fraction
    if isinstance(height, str) and height.endswith("%"):
        h_frac = float(height.rstrip("%")) / 100 * bbox.height
    else:
        h_frac = 0.02  # fallback

    # Create colorbar axes using explicit figure coordinates
    # Center it under the polar plot with 92% of axis width
    cbar_width = bbox.width * 0.92
    cbar_left = bbox.x0 + (bbox.width - cbar_width) / 2
    cbar_bottom = bbox.y0 + y_offset * bbox.height

    cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, h_frac])
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")

    # Use cfg font sizes if provided, else defaults
    label_fs = cfg.colorbar_label_fontsize if cfg else 7
    tick_fs = cfg.colorbar_tick_fontsize if cfg else 6

    cb.set_label(label, fontsize=label_fs)
    cb.ax.tick_params(labelsize=tick_fs)
    return cax


# -----------------------------
# Figure builders
# -----------------------------
def create_ebc_figure(
        of_full, of_odd, of_even,
        c_full, c_odd, c_even,
        root_of, root_c,
        cfg: PlotCfg,
        neuron_idx: int,
):
    """
    New layout:
    1. OF traj | 2. Chase traj | 3. OF full | 4. Chase full | 5. OF vs Chase CC
    6. OF odd  | 7. OF even    | 8. OF odd/even CC
    9. Chase odd | 10. Chase even | 11. Chase odd/even CC
    """
    dt = cfg.binsize
    vmin_of, vmax_of = compute_vmin_vmax_hz([of_odd, of_even, of_full], cfg.occ_min, dt, perc=98.0)
    vmin_c, vmax_c = compute_vmin_vmax_hz([c_odd, c_even, c_full], cfg.occ_min, dt, perc=98.0)

    fig = plt.figure(figsize=(26, 5.4))
    gs = fig.add_gridspec(1, 11, left=0.03, right=0.995, top=0.86, bottom=0.28, wspace=0.65)

    # Panel 1: OF trajectory
    ax1 = fig.add_subplot(gs[0, 0])
    # Panel 2: Chase trajectory
    ax2 = fig.add_subplot(gs[0, 1])
    # Panel 3: OF full (polar)
    ax3 = fig.add_subplot(gs[0, 2], projection="polar")
    # Panel 4: Chase full (polar)
    ax4 = fig.add_subplot(gs[0, 3], projection="polar")
    # Panel 5: OF vs Chase CC
    ax5 = fig.add_subplot(gs[0, 4])
    # Panel 6: OF odd (polar)
    ax6 = fig.add_subplot(gs[0, 5], projection="polar")
    # Panel 7: OF even (polar)
    ax7 = fig.add_subplot(gs[0, 6], projection="polar")
    # Panel 8: OF Odd vs Even CC
    ax8 = fig.add_subplot(gs[0, 7])
    # Panel 9: Chase odd (polar)
    ax9 = fig.add_subplot(gs[0, 8], projection="polar")
    # Panel 10: Chase even (polar)
    ax10 = fig.add_subplot(gs[0, 9], projection="polar")
    # Panel 11: Chase Odd vs Even CC
    ax11 = fig.add_subplot(gs[0, 10])

    # --- Panel 1: OF trajectory ---
    base.plot_trajectory(ax1, root_of, of_full)
    ax1.set_title("Openfiedl Traj", fontsize=cfg.title_fontsize)
    ax1.set_xlabel("X (cm)", fontsize=cfg.label_fontsize)
    ax1.set_ylabel("Y (cm)", fontsize=cfg.label_fontsize)
    ax1.tick_params(labelsize=cfg.theta_tick_fontsize)

    # --- Panel 2: Chase trajectory ---
    base.plot_trajectory(ax2, root_c, c_full)
    ax2.set_title("Chase Traj", fontsize=cfg.title_fontsize)
    ax2.set_xlabel("X (cm)", fontsize=cfg.label_fontsize)
    ax2.set_ylabel("Y (cm)", fontsize=cfg.label_fontsize)
    ax2.tick_params(labelsize=cfg.theta_tick_fontsize)

    # --- Panel 3: OF full polar ---
    pc3 = polar_pcolormesh(ax3, of_full, "Openfield", cfg, vmin_of, vmax_of, annotate_fr=True)

    # --- Panel 4: Chase full polar ---
    pc4 = polar_pcolormesh(ax4, c_full, "Chasing", cfg, vmin_c, vmax_c, annotate_fr=True)

    # --- Panel 5: OF vs Chase CC ---
    cc_cross, aa_cross, dd_cross = base.compute_cross_correlation(of_full, c_full, cfg.occ_min)
    base.plot_cc(ax5, cc_cross, aa_cross, dd_cross, title="OF vs Chase")
    ax5.set_title("OF vs Chase", fontsize=cfg.title_fontsize)
    ax5.set_xlabel("Angle shift (bins)", fontsize=cfg.label_fontsize)
    ax5.set_ylabel("Dist shift (bins)", fontsize=cfg.label_fontsize)
    ax5.tick_params(labelsize=cfg.theta_tick_fontsize)

    # --- Panel 6: OF odd polar ---
    pc6 = polar_pcolormesh(ax6, of_odd, "OF odd", cfg, vmin_of, vmax_of, annotate_fr=True)

    # --- Panel 7: OF even polar ---
    pc7 = polar_pcolormesh(ax7, of_even, "OF even", cfg, vmin_of, vmax_of, annotate_fr=True)

    # --- Panel 8: OF Odd vs Even CC ---
    cc_of, aa_of, dd_of = base.compute_cross_correlation(of_odd, of_even, cfg.occ_min)
    base.plot_cc(ax8, cc_of, aa_of, dd_of, title="OF: Odd vs Even")
    ax8.set_title("OF: Odd vs Even", fontsize=cfg.title_fontsize)
    ax8.set_xlabel("Angle shift (bins)", fontsize=cfg.label_fontsize)
    ax8.set_ylabel("Dist shift (bins)", fontsize=cfg.label_fontsize)
    ax8.tick_params(labelsize=cfg.theta_tick_fontsize)

    # --- Panel 9: Chase odd polar ---
    pc9 = polar_pcolormesh(ax9, c_odd, "Chase odd", cfg, vmin_c, vmax_c, annotate_fr=True)

    # --- Panel 10: Chase even polar ---
    pc10 = polar_pcolormesh(ax10, c_even, "Chase even", cfg, vmin_c, vmax_c, annotate_fr=True)

    # --- Panel 11: Chase Odd vs Even CC ---
    cc_c, aa_c, dd_c = base.compute_cross_correlation(c_odd, c_even, cfg.occ_min)
    base.plot_cc(ax11, cc_c, aa_c, dd_c, title="Chase: Odd vs Even")
    ax11.set_title("Chase: Odd vs Even", fontsize=cfg.title_fontsize)
    ax11.set_xlabel("Angle shift (bins)", fontsize=cfg.label_fontsize)
    ax11.set_ylabel("Dist shift (bins)", fontsize=cfg.label_fontsize)
    ax11.tick_params(labelsize=cfg.theta_tick_fontsize)

    # --- Colorbars ---
    caxes = []
    if cfg.add_colorbar_to_each_polar:
        caxes += [add_horizontal_cbar(fig, ax3, pc3, cfg=cfg), add_horizontal_cbar(fig, ax4, pc4, cfg=cfg)]
        caxes += [add_horizontal_cbar(fig, ax6, pc6, cfg=cfg), add_horizontal_cbar(fig, ax7, pc7, cfg=cfg)]
        caxes += [add_horizontal_cbar(fig, ax9, pc9, cfg=cfg), add_horizontal_cbar(fig, ax10, pc10, cfg=cfg)]
    else:
        # Default: colorbar under OF full and Chase full
        caxes += [add_horizontal_cbar(fig, ax3, pc3, cfg=cfg), add_horizontal_cbar(fig, ax4, pc4, cfg=cfg)]

    D, T = of_full["rm_ns"].shape
    fr_of = get_fr_hz(of_full, cfg.binsize)
    fr_c = get_fr_hz(c_full, cfg.binsize)
    fig.suptitle(
        f"EBC | {cfg.which_animal} | Neuron {neuron_idx + 1} | bins {D}×{T} | FR_OF {fr_of:.2f} Hz | FR_Chase {fr_c:.2f} Hz",
        fontsize=cfg.suptitle_fontsize, fontweight="bold",
    )

    return fig


def create_eboc_figure(c_full, c_odd, c_even, root_c, cfg: PlotCfg, neuron_idx: int):
    dt = cfg.binsize
    vmin_c, vmax_c = compute_vmin_vmax_hz([c_odd, c_even, c_full], cfg.occ_min, dt, perc=98.0)

    fig = plt.figure(figsize=(24, 5.0))
    gs = fig.add_gridspec(1, 7, left=0.03, right=0.995, top=0.86, bottom=0.30, wspace=0.7)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2], projection="polar")
    ax4 = fig.add_subplot(gs[0, 3], projection="polar")
    ax5 = fig.add_subplot(gs[0, 4], projection="polar")
    ax6 = fig.add_subplot(gs[0, 5])
    # ax7 = fig.add_subplot(gs[0, 6])

    base.plot_trajectory(ax1, root_c, c_full)
    ax1.set_title("Animal trajectory (allocentric) with spike overlay", fontsize=cfg.title_fontsize)
    ax1.set_xlabel("X (cm)", fontsize=cfg.label_fontsize)
    ax1.set_ylabel("Y (cm)", fontsize=cfg.label_fontsize)
    ax1.tick_params(labelsize=cfg.theta_tick_fontsize)

    # Rotate egocentric frame so 0 rad = up (north)
    theta = c_full["bait_angle"] + np.pi / 2

    bait_x = c_full["bait_dist"] * np.cos(theta)
    bait_y = c_full["bait_dist"] * np.sin(theta)

    ax2.plot(bait_x, bait_y, color="0.85", lw=0.8)
    if len(bait_x) > 1:
        pts = np.column_stack([bait_x, bait_y])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        t = np.linspace(0, 1, len(segs))
        lc = LineCollection(segs, cmap=cm.get_cmap("Grays"), array=t, linewidths=1.8, alpha=0.95)
        ax2.add_collection(lc)
    ax2.axhline(0, color="k", ls=":", lw=0.6)
    ax2.axvline(0, color="k", ls=":", lw=0.6)
    ax2.set_aspect("equal")
    ax2.set_title("Object trajectory around animal (egocentric)", fontsize=cfg.title_fontsize)
    ax2.axis("off")

    pc3 = polar_pcolormesh(ax3, c_full, "Object", cfg, vmin_c, vmax_c, annotate_fr=True)
    pc4 = polar_pcolormesh(ax4, c_odd, "Object odd", cfg, vmin_c, vmax_c, annotate_fr=True)
    pc5 = polar_pcolormesh(ax5, c_even, "Object even", cfg, vmin_c, vmax_c, annotate_fr=True)

    cc_c, aa_c, dd_c = base.compute_cross_correlation(c_odd, c_even, cfg.occ_min)
    base.plot_cc(ax6, cc_c, aa_c, dd_c, title="Object: Odd vs Even")
    ax6.set_title("Object: Odd vs Even", fontsize=cfg.title_fontsize)
    ax6.set_xlabel("Angle shift (bins)", fontsize=cfg.label_fontsize)
    ax6.set_ylabel("Dist shift (bins)", fontsize=cfg.label_fontsize)
    ax6.tick_params(labelsize=cfg.theta_tick_fontsize)

    # Polar plots
    pc3 = polar_pcolormesh(ax3, c_full, "Object", cfg, vmin_c, vmax_c, annotate_fr=True)
    pc4 = polar_pcolormesh(ax4, c_odd, "Object odd", cfg, vmin_c, vmax_c, annotate_fr=True)
    pc5 = polar_pcolormesh(ax5, c_even, "Object even", cfg, vmin_c, vmax_c, annotate_fr=True)

    # ----------------------------------------------------------
    # THREE separate horizontal colorbars (one under each polar)
    # ----------------------------------------------------------
    add_horizontal_cbar(fig, ax3, pc3, label="Firing rate (Hz)", y_offset=-0.22, height="7%", cfg=cfg)
    add_horizontal_cbar(fig, ax4, pc4, label="Firing rate (Hz)", y_offset=-0.22, height="7%", cfg=cfg)
    add_horizontal_cbar(fig, ax5, pc5, label="Firing rate (Hz)", y_offset=-0.22, height="7%", cfg=cfg)

    # ax7.axis("off")
    # ==========================================================
    # Shared colorbar for Chase / Chase odd / Chase even (EBOC)
    # ==========================================================

    # cax = inset_axes(
    #    ax4,  # anchor to middle polar axis (Chase odd)
    #    width="165%",  # spans all three polar plots

    #    height="8%",
    #    loc="lower center",
    #    bbox_to_anchor=(-0.825, -0.35, 1.65, 1.0),
    #    bbox_transform=ax4.transAxes,
    #    borderpad=0,
    # )

    # cb = fig.colorbar(pc3, cax=cax, orientation="horizontal")
    # cb.set_label("Firing rate (Hz)", fontsize=cfg.colorbar_label_fontsize)
    # cb.ax.tick_params(labelsize=cfg.colorbar_tick_fontsize, width=0.6, length=3)
    # cb.outline.set_visible(False)

    D, T = c_full["rm_ns"].shape
    fr = get_fr_hz(c_full, cfg.binsize)
    mi = float(c_full["MI"]) if "MI" in c_full else float("nan")
    fig.suptitle(
        f"EBOC | {cfg.which_animal} | Neuron {neuron_idx + 1} | bins {D}×{T} | FR {fr:.2f} Hz | MI={mi:.3f}",
        fontsize=cfg.suptitle_fontsize, fontweight="bold",
    )

    return fig


# -----------------------------
# Loaders (skip missing sessions safely)
# -----------------------------
def load_ebc_for_plot(cfg: PlotCfg, neuron_idx: int):
    dt = cfg.binsize
    box_edges = base.BOX_EDGES[cfg.which_animal]

    x_of_all, y_of_all, hd_of_all, spk_of_all = [], [], [], []
    loaded_of = False
    for sess in cfg.of_sessions:
        try:
            d = base.load_session_data(cfg.folder_loc, cfg.which_animal, sess, cfg.which_channels, cfg.binsize)
        except FileNotFoundError:
            print(f"  ⚠️ missing OF {sess} for {cfg.which_animal}, skipping")
            continue
        loaded_of = True
        x, y, hd, spk, _, _, _ = base.prepare_data(d, neuron_idx, None, True)
        x_of_all.append(x)
        y_of_all.append(y)
        hd_of_all.append(hd)
        spk_of_all.append(spk)
    if not loaded_of:
        raise RuntimeError("No OF sessions loaded. Check --of-sessions and folder paths.")

    x_of = np.concatenate(x_of_all)
    y_of = np.concatenate(y_of_all)
    hd_of = np.concatenate(hd_of_all)
    spk_of = np.concatenate(spk_of_all)

    x_c_all, y_c_all, hd_c_all, spk_c_all = [], [], [], []
    loaded_c = False
    for sess in cfg.chase_sessions:
        try:
            d = base.load_session_data(cfg.folder_loc, cfg.which_animal, sess, cfg.which_channels, cfg.binsize)
        except FileNotFoundError:
            print(f"  ⚠️ missing chase {sess} for {cfg.which_animal}, skipping")
            continue
        sel = base.extract_chase_intervals(d, sess, cfg.which_animal, cfg.chase_or_chill)
        if sel is None:
            continue
        loaded_c = True
        x, y, hd, spk, _, _, _ = base.prepare_data(d, neuron_idx, sel, True)
        x_c_all.append(x)
        y_c_all.append(y)
        hd_c_all.append(hd)
        spk_c_all.append(spk)
    if not loaded_c:
        raise RuntimeError("No chase sessions loaded/selected. Check --chase-sessions and intervals.")

    x_c = np.concatenate(x_c_all)
    y_c = np.concatenate(y_c_all)
    hd_c = np.concatenate(hd_c_all)
    spk_c = np.concatenate(spk_c_all)

    of_full = base.compute_ebc_ratemap(x_of, y_of, hd_of, spk_of, box_edges, dt_sec=dt, occ_min=cfg.occ_min,
                                       compute_distributions=False, n_shuffles=0)
    of_odd, of_even = base.compute_odd_even_splits(x_of, y_of, hd_of, spk_of, box_edges, dt_sec=dt,
                                                   is_eboc=False, occ_min=cfg.occ_min)

    c_full = base.compute_ebc_ratemap(x_c, y_c, hd_c, spk_c, box_edges, dt_sec=dt, occ_min=cfg.occ_min,
                                      compute_distributions=False, n_shuffles=0)
    c_odd, c_even = base.compute_odd_even_splits(x_c, y_c, hd_c, spk_c, box_edges, dt_sec=dt,
                                                 is_eboc=False, occ_min=cfg.occ_min)

    root_of = {"x": x_of, "y": y_of, "md": hd_of, "spike": spk_of, "firing_rate": get_fr_hz(of_full, dt)}
    root_c = {"x": x_c, "y": y_c, "md": hd_c, "spike": spk_c, "firing_rate": get_fr_hz(c_full, dt)}
    return of_full, of_odd, of_even, c_full, c_odd, c_even, root_of, root_c


def load_eboc_for_plot(cfg: PlotCfg, neuron_idx: int):
    dt = cfg.binsize

    x_c_all, y_c_all, hd_c_all, spk_c_all = [], [], [], []
    bait_a_all, bait_d_all = [], []
    loaded = False

    for sess in cfg.chase_sessions:
        try:
            d = base.load_session_data(cfg.folder_loc, cfg.which_animal, sess, cfg.which_channels, cfg.binsize)
        except FileNotFoundError:
            print(f"  ⚠️ missing chase {sess} for {cfg.which_animal}, skipping")
            continue

        sel = base.extract_chase_intervals(d, sess, cfg.which_animal, cfg.chase_or_chill)
        if sel is None:
            continue
        loaded = True
        # prepare_data returns: x, y, hd, spk, t, bait_a, bait_d
        # We discard t (time), keep bait_a and bait_d
        x, y, hd, spk, _, bait_a, bait_d = base.prepare_data(d, neuron_idx, sel, True)
        x_c_all.append(x)
        y_c_all.append(y)
        hd_c_all.append(hd)
        spk_c_all.append(spk)
        if bait_a is not None and bait_d is not None:
            bait_a_all.append(bait_a)
            bait_d_all.append(bait_d)

    if not loaded:
        raise RuntimeError("No chase sessions loaded/selected for EBOC.")

    if not bait_a_all:
        raise RuntimeError("No bait (relative angle/distance) data found in sessions. "
                           "EBOC requires binned_rel_ha and binned_rel_dist in .mat files.")

    x_c = np.concatenate(x_c_all)
    y_c = np.concatenate(y_c_all)
    hd_c = np.concatenate(hd_c_all)
    spk_c = np.concatenate(spk_c_all)
    bait_a = np.concatenate(bait_a_all)
    bait_d = np.concatenate(bait_d_all)

    # compute_eboc_ratemap returns the dict directly (not wrapped)
    c_full = base.compute_eboc_ratemap(x_c, y_c, hd_c, spk_c, bait_a, bait_d, dt_sec=dt, occ_min=cfg.occ_min)

    c_odd, c_even = base.compute_odd_even_splits(x_c, y_c, hd_c, spk_c, box_edges=None, dt_sec=dt,
                                                 bait_angle=bait_a, bait_dist=bait_d, is_eboc=True, occ_min=cfg.occ_min)

    root_c = {"x": x_c, "y": y_c, "md": hd_c, "spike": spk_c, "firing_rate": get_fr_hz(c_full, dt)}
    return c_full, c_odd, c_even, root_c


# -----------------------------
# CLI / main
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Plot EBC/EBOC figures with consistent fonts. Can select specific neurons.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot specific neurons (1-based indices):
  python plot_polished_cells_modified.py --mode EBC --neuron-list "5,12,47" \\
      --folder-loc /path/to/data --animal Arwen --channels RSC --out-dir ./plots

  # Plot from CSV:
  python plot_polished_cells_modified.py --mode EBC --classification-csv results.csv \\
      --folder-loc /path/to/data --animal Arwen --channels RSC --out-dir ./plots

  # Plot single cell with PDF output:
  python plot_polished_cells_modified.py --mode EBC --neuron-list "42" --save-pdf \\
      --folder-loc /path/to/data --animal Arwen --channels RSC --out-dir ./plots
        """
    )
    ap.add_argument("--mode", choices=["EBC", "EBOC"], required=True,
                    help="Analysis mode: EBC (boundary) or EBOC (object)")

    # Cell selection - either CSV or direct list
    ap.add_argument("--classification-csv", default="",
                    help="CSV file with neuron classification (optional if --neuron-list provided)")
    ap.add_argument("--neuron-list", type=str, default="",
                    help="Comma-separated list of neuron indices (1-based), e.g., '5,12,47'. "
                         "Overrides --classification-csv if both provided.")

    # Data paths
    ap.add_argument("--folder-loc", required=True, help="Base folder with data")
    ap.add_argument("--animal", required=True, help="Animal name (e.g., Arwen, ToothMuch)")
    ap.add_argument("--channels", required=True, help="Channel set (e.g., RSC)")
    ap.add_argument("--out-dir", required=True, help="Output directory for figures")

    # Data parameters
    ap.add_argument("--binsize", type=float, default=0.008333, help="Time bin size in seconds")
    ap.add_argument("--occ-min", type=float, default=base.OCCUPANCY_THRESHOLD,
                    help="Minimum occupancy threshold (frames)")

    # Session selection
    ap.add_argument("--of-sessions", type=str, default="OF1,OF2",
                    help="Comma-separated open field sessions")
    ap.add_argument("--chase-sessions", type=str, default="c1,c2,c3,c4,c5",
                    help="Comma-separated chase sessions")
    ap.add_argument("--chase-or-chill", choices=["chase", "chill"], default="chase",
                    help="Use 'chase' or 'chill' intervals")

    # Plot options
    ap.add_argument("--cmap", type=str, default="jet", help="Colormap for polar plots")
    ap.add_argument("--colorbar-each", action="store_true",
                    help="Add colorbar to each polar plot (default: only last)")

    # Export options
    ap.add_argument("--dpi", type=int, default=300, help="DPI for PNG output")
    ap.add_argument("--save-pdf", action="store_true", help="Also save vector PDF")

    # Limits
    ap.add_argument("--max-neurons", type=int, default=0,
                    help="Max neurons to plot (0 = all selected)")
    ap.add_argument("--display-smooth-sigma", type=float, default=0.0,
                    help="Display-only gaussian smoothing sigma in bins (0 = off)")

    return ap.parse_args()


def main():
    args = parse_args()

    # Validate that we have some way to select neurons
    if not args.neuron_list and not args.classification_csv:
        raise ValueError("Must provide either --neuron-list or --classification-csv")

    cfg = PlotCfg(
        folder_loc=args.folder_loc,
        which_animal=args.animal,
        which_channels=args.channels,
        binsize=args.binsize,
        occ_min=args.occ_min,
        of_sessions=tuple([s.strip() for s in args.of_sessions.split(",") if s.strip()]),
        chase_sessions=tuple([s.strip() for s in args.chase_sessions.split(",") if s.strip()]),
        chase_or_chill=args.chase_or_chill,
        cmap=args.cmap,
        add_colorbar_to_each_polar=bool(args.colorbar_each),
        display_smooth_sigma=float(args.display_smooth_sigma),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # =========================================
    # CELL SELECTION - neuron-list takes priority
    # =========================================
    if args.neuron_list:
        # User specified neurons directly (1-based input → 0-based internal)
        neuron_idxs = []
        for x in args.neuron_list.split(","):
            x = x.strip()
            if x:
                try:
                    idx = int(x) - 1  # Convert to 0-based
                    if idx >= 0:
                        neuron_idxs.append(idx)
                except ValueError:
                    print(f"  ⚠️ Invalid neuron index: '{x}', skipping")

        if not neuron_idxs:
            raise ValueError("No valid neuron indices in --neuron-list")

        print(f"[plot] User-specified neurons (1-based): {[n + 1 for n in neuron_idxs]}")
    else:
        # Use CSV-based selection
        neuron_idxs = select_neuron_indices_from_csv(Path(args.classification_csv), args.mode)
        print(f"[plot] Selected {len(neuron_idxs)} neurons from CSV for mode={args.mode}")

    # Apply max limit if specified
    if args.max_neurons and args.max_neurons > 0:
        neuron_idxs = neuron_idxs[:args.max_neurons]
        print(f"[plot] Limited to first {args.max_neurons} neurons")

    print(f"[plot] Will plot {len(neuron_idxs)} neuron(s): {[n + 1 for n in neuron_idxs]}")

    # =========================================
    # PLOTTING LOOP
    # =========================================
    success_count = 0
    for n in neuron_idxs:
        try:
            if args.mode == "EBC":
                of_full, of_odd, of_even, c_full, c_odd, c_even, root_of, root_c = load_ebc_for_plot(cfg, n)
                fig = create_ebc_figure(of_full, of_odd, of_even, c_full, c_odd, c_even, root_of, root_c, cfg, n)
                out_png = out_dir / f"EBC_{cfg.which_animal}_neuron{n + 1:04d}.png"
            else:
                c_full, c_odd, c_even, root_c = load_eboc_for_plot(cfg, n)
                fig = create_eboc_figure(c_full, c_odd, c_even, root_c, cfg, n)
                out_png = out_dir / f"EBOC_{cfg.which_animal}_neuron{n + 1:04d}.png"

            fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight", pad_inches=0.02)
            if args.save_pdf:
                out_pdf = out_png.with_suffix(".pdf")
                fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)

            plt.close(fig)
            print(f"  ✓ saved {out_png.name}" + (" (+pdf)" if args.save_pdf else ""))
            success_count += 1

        except Exception as e:
            print(f"  ✗ neuron {n + 1}: {e}")

    print(f"[done] Successfully plotted {success_count}/{len(neuron_idxs)} neurons")


if __name__ == "__main__":
    main()

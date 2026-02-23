#!/usr/bin/env python3
"""
Winner-colored EBC vs ETC GLM plots (ONLY EBC+ETC cells across all animals)

This script:
  1) Loads per-cell GLM .npy files (one file per cell)
  2) Infers animal from path, assigns region (RSC vs SC) via channel ranges
  3) Extracts mean CV deviance + mean CV pseudoR2 per model
  4) Computes (vs null):
       dDev_*  = Dev(null) - Dev(model)   (positive = better)
       dR2_*   = R2(model)  - R2(null)    (positive = better)
     Includes HD-controlled deltas:
       dDev_EBC_HD, dDev_EBOC_HD
       dR2_EBC_HD,  dR2_EBOC_HD
     Unique nested contributions (HD controlled):
       unique_EBC (dev): Dev(EBOC_HD) - Dev(EBC_EBOC_HD)
       unique_ETC (dev): Dev(EBC_HD)  - Dev(EBC_EBOC_HD)
       unique_EBC (R2):  R2(EBC_EBOC_HD) - R2(EBOC_HD)
       unique_ETC (R2):  R2(EBC_EBOC_HD) - R2(EBC_HD)
  5) Merges STRICT-long classification CSV (animal, cell_id/cell_name, mode, classification)
  6) Filters to ONLY cells labeled EBC or ETC (and Both)
  7) Makes winner-colored scatter plots + classification-colored plots
  8) Saves CSVs + prints extracted counts

Run example:
  python3 glm_winner_plots.py \
    --base-dir "/path/to/glm_outputs/" \
    --out-dir "/path/to/out/" \
    --glob "*chaseOnly.npy" \
    --class-csv "/mnt/data/EBC_ETC_EOC_all_animals_long_STRICT.csv"
"""

from __future__ import annotations
import argparse
import re
import sys
import types
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not available, statistical tests will be skipped")

# -----------------------------
# Channel mapping (your provided ranges)
# -----------------------------
channel_ranges = {
    "Arwen": {"RSC": range(124, 385), "SC": range(0, 124)},
    "PreciousGrape": {"RSC": range(84, 250), "SC": range(252, 385)},
    "ToothMuch": {"RSC": set(range(0, 156)).union(set(range(266, 385))), "SC": range(190, 266)},
    "MimosaPudica": {"RSC": set(range(0, 176)).union(set(range(285, 385))), "SC": range(179, 284)},
}
ANIMALS = set(channel_ranges.keys())

# Models expected inside each cell .npy (confirmed from your pipeline)
MODELS_NEEDED = [
    "null", "EBC", "EBOC", "Allo_HD",
    "EBC_HD", "EBOC_HD", "EBC_EBOC", "EBC_EBOC_HD",
]

# -----------------------------
# Regex helpers for filenames like: imec0_cl0015_ch199_chaseOnly.npy
# -----------------------------
CELL_RE = re.compile(r"(imec\d+_cl\d+_ch\d+)", re.IGNORECASE)
CH_RE = re.compile(r"_ch(\d+)", re.IGNORECASE)


# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def apply_numpy_pickle_compat() -> None:
    """Compatibility for numpy pickles referencing numpy._core.*

    Only needed when loading pickles saved with numpy 2.x on numpy 1.x.
    Skip if numpy._core already exists (numpy 2.x) to avoid breaking scipy.
    """
    try:
        import numpy._core  # noqa: F401
        return
    except ImportError:
        pass

    if "numpy._core" not in sys.modules:
        sys.modules["numpy._core"] = types.ModuleType("numpy._core")
    if "numpy._core.multiarray" not in sys.modules:
        sys.modules["numpy._core.multiarray"] = np.core.multiarray


def infer_animal_from_path(p: Path) -> Optional[str]:
    """Infer animal name from folder parts or substring."""
    for part in p.parts:
        if part in ANIMALS:
            return part
    s = str(p).lower()
    for a in ANIMALS:
        if a.lower() in s:
            return a
    return None


def parse_cell_id_and_channel(p: Path) -> Tuple[Optional[str], Optional[int]]:
    m = CELL_RE.search(p.name)
    cell_id = m.group(1) if m else None
    m2 = CH_RE.search(p.name)
    ch = int(m2.group(1)) if m2 else None
    return cell_id, ch


def region_from_channel(animal: str, ch: int) -> Optional[str]:
    if animal not in channel_ranges:
        return None
    if ch in channel_ranges[animal]["RSC"]:
        return "RSC"
    if ch in channel_ranges[animal]["SC"]:
        return "SC"
    return None


def load_cell_npy(path: Path) -> List[Dict[str, Any]]:
    """Load a single cell npy (object array of dicts)."""
    apply_numpy_pickle_compat()
    arr = np.load(path, allow_pickle=True)

    if isinstance(arr, np.ndarray) and arr.dtype == object:
        items = list(arr)
    else:
        raise ValueError(f"Unexpected structure in {path} (expected object array of dicts)")

    if not items or not all(isinstance(x, dict) for x in items):
        raise ValueError(f"{path} does not look like a list of dicts per model")
    return items


def mean_deviance(md: Dict[str, Any]) -> float:
    scores = np.asarray(md["scores"])
    if scores.ndim != 2 or scores.shape[1] < 1:
        raise ValueError(f"Bad scores shape: {scores.shape}")
    return float(np.nanmean(scores[:, 0]))  # deviance is col 0


def mean_pseudoR2(md: Dict[str, Any]) -> float:
    """Extract mean pseudo-R² across CV folds (column 1)."""
    scores = np.asarray(md["scores"])
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError(f"Bad scores shape: {scores.shape}")
    return float(np.nanmean(scores[:, 1]))  # pseudoR2 is col 1


def extract_deviances(path: Path) -> Dict[str, Dict[str, float]]:
    """Extract mean deviance and pseudoR2 for all MODELS_NEEDED that exist in this file."""
    models = load_cell_npy(path)
    results: Dict[str, Dict[str, float]] = {}
    for md in models:
        model_name = md.get("model", None)
        if model_name in MODELS_NEEDED and "scores" in md:
            results[model_name] = {
                "dev": mean_deviance(md),
                "r2": mean_pseudoR2(md),
            }
    return results


# -----------------------------
# Classification: STRICT long format
# -----------------------------
def load_classification_strict_long(csv_path: Path) -> pd.DataFrame:
    """
    Loads STRICT-long classification CSV:
      must contain columns:
        animal, mode, classification
      and one of:
        cell_name / cell_id / cell

    Returns one row per (animal, cell_id) with:
      cell_class in {EBC, ETC, Both, Neither}
    """
    cdf = pd.read_csv(csv_path)

    cell_col = None
    for c in ["cell_name", "cell_id", "cell"]:
        if c in cdf.columns:
            cell_col = c
            break

    required = {"animal", "mode", "classification"}
    if not required.issubset(set(cdf.columns)) or cell_col is None:
        raise ValueError(
            f"Classification CSV must contain {sorted(required)} plus one of "
            f"['cell_name','cell_id','cell'].\nFound columns: {list(cdf.columns)}"
        )

    cdf = cdf.copy()
    cdf["animal"] = cdf["animal"].astype(str)
    cdf["cell_id"] = cdf[cell_col].astype(str)
    cdf["mode"] = cdf["mode"].astype(str)
    cdf["classification"] = cdf["classification"].astype(str)

    # boolean flags
    cdf["is_EBC"] = (cdf["mode"] == "EBC") & (cdf["classification"] == "EBC")
    cdf["is_ETC"] = (cdf["mode"].isin(["ETC", "EBOC"])) & (cdf["classification"].isin(["ETC", "EBOC"]))

    wide = (
        cdf.groupby(["animal", "cell_id"], as_index=False)[["is_EBC", "is_ETC"]]
        .any()
    )

    def mk_label(r) -> str:
        if r["is_EBC"] and r["is_ETC"]:
            return "Both"
        if r["is_EBC"]:
            return "EBC"
        if r["is_ETC"]:
            return "ETC"
        return "Neither"

    wide["cell_class"] = wide.apply(mk_label, axis=1)
    return wide[["animal", "cell_id", "cell_class"]]


# -----------------------------
# Plotting (winner colored, like your example)
# -----------------------------
def classification_scatter(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    title: str,
    out_path: Path,
    clip_percentile: float = 98,
) -> None:
    """
    Scatter colored by CLASSIFICATION (EBC vs ETC vs Both).
    """
    x = df[xcol].values.astype(float)
    y = df[ycol].values.astype(float)
    cell_class = df["cell_class"].values

    plt.figure(figsize=(6.6, 6.2))

    colors = {"EBC": "tab:blue", "ETC": "tab:orange", "Both": "tab:purple"}

    for cls in ["EBC", "ETC", "Both"]:
        m = cell_class == cls
        if np.any(m):
            if cls == "EBC":
                n_correct = np.sum((x[m] - y[m]) > 0)
            elif cls == "ETC":
                n_correct = np.sum((y[m] - x[m]) > 0)
            else:
                n_correct = m.sum()

            plt.scatter(
                x[m], y[m],
                s=70, alpha=0.7, c=colors[cls],
                label=f"{cls} (n={m.sum()}, {n_correct} correct)"
            )

    finite_x = x[np.isfinite(x)]
    finite_y = y[np.isfinite(y)]
    all_vals = np.concatenate([finite_x, finite_y])

    lo = float(np.percentile(all_vals, 100 - clip_percentile))
    hi = float(np.percentile(all_vals, clip_percentile))
    pad = (hi - lo) * 0.05
    lo -= pad
    hi += pad

    plt.xlim(lo, hi)
    plt.ylim(lo, hi)

    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5, color="gray")

    n_outliers = np.sum((x < lo) | (x > hi) | (y < lo) | (y > hi))
    if n_outliers > 0:
        plt.text(
            0.98, 0.02, f"{n_outliers} outliers clipped",
            transform=plt.gca().transAxes,
            ha="right", va="bottom", fontsize=8, color="gray"
        )

    plt.text(
        0.95, 0.15, "EBC better →",
        transform=plt.gca().transAxes,
        ha="right", va="bottom", fontsize=9,
        color="tab:blue", alpha=0.7
    )
    plt.text(
        0.15, 0.95, "↑ ETC better",
        transform=plt.gca().transAxes,
        ha="left", va="top", fontsize=9,
        color="tab:orange", alpha=0.7
    )

    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(title)
    plt.legend(frameon=False, fontsize=9, loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def winner_scatter(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    title: str,
    out_path: Path,
    eps: float = 1e-6,
    clip_percentile: float = 98,
) -> None:
    """
    Scatter colored by which model wins per cell:
      Blue   = EBC > ETC
      Orange = ETC > EBC
      Gray   = equal (within eps)
    """
    x = df[xcol].values.astype(float)
    y = df[ycol].values.astype(float)

    winner = np.full(len(df), "equal", dtype=object)
    winner[(x - y) > eps] = "EBC > ETC"
    winner[(y - x) > eps] = "ETC > EBC"

    plt.figure(figsize=(6.6, 6.2))

    for lab, col in [("EBC > ETC", "tab:blue"), ("ETC > EBC", "tab:orange"), ("equal", "tab:gray")]:
        m = winner == lab
        if np.any(m):
            plt.scatter(x[m], y[m], s=60, alpha=0.75, c=col, label=f"{lab} (n={m.sum()})")

    finite_x = x[np.isfinite(x)]
    finite_y = y[np.isfinite(y)]
    all_vals = np.concatenate([finite_x, finite_y])

    lo = float(np.percentile(all_vals, 100 - clip_percentile))
    hi = float(np.percentile(all_vals, clip_percentile))
    pad = (hi - lo) * 0.05
    lo -= pad
    hi += pad

    plt.xlim(lo, hi)
    plt.ylim(lo, hi)

    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="gray")

    n_outliers = np.sum((x < lo) | (x > hi) | (y < lo) | (y > hi))
    if n_outliers > 0:
        plt.text(
            0.98, 0.02, f"{n_outliers} outliers clipped",
            transform=plt.gca().transAxes,
            ha="right", va="bottom", fontsize=8, color="gray"
        )

    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(title)
    plt.legend(frameon=False, fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def make_comparison_boxplot(
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    """
    Boxplot comparing a metric between EBC, ETC, and Both cell types.
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    groups = ["EBC", "ETC", "Both"]
    colors = {"EBC": "tab:blue", "ETC": "tab:orange", "Both": "tab:purple"}

    data_to_plot = []
    positions = []
    labels = []

    for i, grp in enumerate(groups):
        grp_data = df[df["cell_class"] == grp][metric].dropna().values
        if len(grp_data) > 0:
            data_to_plot.append(grp_data)
            positions.append(i)
            labels.append(f"{grp}\n(n={len(grp_data)})")

    if not data_to_plot:
        plt.close()
        return

    bp = ax.boxplot(data_to_plot, positions=positions, patch_artist=True, widths=0.6)

    for patch, grp in zip(bp["boxes"], [groups[p] for p in positions]):
        patch.set_facecolor(colors[grp])
        patch.set_alpha(0.7)

    for pos, grp, y in zip(positions, [groups[p] for p in positions], data_to_plot):
        x = np.random.normal(pos, 0.08, size=len(y))
        ax.scatter(x, y, alpha=0.4, s=15, c=colors[grp], edgecolors="none")

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def make_all_region_plots(df: pd.DataFrame, out_dir: Path, tag: str) -> None:
    """
    Creates classification-colored and winner-colored scatter plots for both deviance and R2.
    """
    if df.empty:
        return

    # ============ STATISTICAL COMPARISON ============
    ebc_cells = df[df["cell_class"] == "EBC"]
    etc_cells = df[df["cell_class"] == "ETC"]

    if len(ebc_cells) > 0 and len(etc_cells) > 0 and HAS_SCIPY:
        print(f"\n=== {tag}: STATISTICAL COMPARISON ===")

        eboc_hd_ebc = ebc_cells["dR2_EBOC_HD"].values
        eboc_hd_etc = etc_cells["dR2_EBOC_HD"].values
        _, pval_hd = stats.mannwhitneyu(eboc_hd_etc, eboc_hd_ebc, alternative="greater")
        print(f"dR2_EBOC_HD (ETC vs EBC cells): Mann-Whitney U p={pval_hd:.4g}")
        print(f"  ETC cells median: {np.median(eboc_hd_etc):.4f}")
        print(f"  EBC cells median: {np.median(eboc_hd_ebc):.4f}")

        ebc_hd_ebc = ebc_cells["dR2_EBC_HD"].values
        ebc_hd_etc = etc_cells["dR2_EBC_HD"].values
        _, pval_hd2 = stats.mannwhitneyu(ebc_hd_ebc, ebc_hd_etc, alternative="greater")
        print(f"dR2_EBC_HD (EBC vs ETC cells): Mann-Whitney U p={pval_hd2:.4g}")
        print(f"  EBC cells median: {np.median(ebc_hd_ebc):.4f}")
        print(f"  ETC cells median: {np.median(ebc_hd_etc):.4f}")

        u_eboc_ebc = ebc_cells["unique_EBOC_r2"].values
        u_eboc_etc = etc_cells["unique_EBOC_r2"].values
        _, pval = stats.mannwhitneyu(u_eboc_etc, u_eboc_ebc, alternative="greater")
        print(f"Unique EBOC R² (ETC vs EBC cells): Mann-Whitney U p={pval:.4g}")
        print(f"  ETC cells median: {np.median(u_eboc_etc):.4f}")
        print(f"  EBC cells median: {np.median(u_eboc_ebc):.4f}")

        u_ebc_ebc = ebc_cells["unique_EBC_r2"].values
        u_ebc_etc = etc_cells["unique_EBC_r2"].values
        _, pval2 = stats.mannwhitneyu(u_ebc_ebc, u_ebc_etc, alternative="greater")
        print(f"Unique EBC R² (EBC vs ETC cells): Mann-Whitney U p={pval2:.4g}")
        print(f"  EBC cells median: {np.median(u_ebc_ebc):.4f}")
        print(f"  ETC cells median: {np.median(u_ebc_etc):.4f}")

        df_temp = df.copy()
        df_temp["eboc_ratio"] = df_temp["unique_EBOC_r2"] / (
            df_temp["unique_EBC_r2"].abs() + df_temp["unique_EBOC_r2"].abs() + 1e-9
        )
        ratio_ebc = df_temp[df_temp["cell_class"] == "EBC"]["eboc_ratio"].values
        ratio_etc = df_temp[df_temp["cell_class"] == "ETC"]["eboc_ratio"].values
        _, pval3 = stats.mannwhitneyu(ratio_etc, ratio_ebc, alternative="greater")
        print(f"EBOC ratio (ETC vs EBC cells): Mann-Whitney U p={pval3:.4g}")
        print(f"  ETC cells median ratio: {np.median(ratio_etc):.4f}")
        print(f"  EBC cells median ratio: {np.median(ratio_ebc):.4f}")

    # ============ R2 PLOTS (PRIMARY) ============
    classification_scatter(
        df,
        xcol="dR2_EBC_HD",
        ycol="dR2_EBOC_HD",
        title=f"{tag}: EBC+HD vs ETC+HD model fit by classification\n(ΔpseudoR² vs null)",
        out_path=out_dir / f"{tag}_CLASS_dR2_EBC_HD_vs_EBOC_HD.png",
    )

    classification_scatter(
        df,
        xcol="dR2_EBC",
        ycol="dR2_EBOC_HD",
        title=f"{tag}: EBC vs ETC+HD model fit by classification\n(ΔpseudoR² vs null)",
        out_path=out_dir / f"{tag}_CLASS_dR2_EBC_vs_EBOC_HD.png",
    )

    classification_scatter(
        df,
        xcol="dR2_EBC",
        ycol="dR2_EBOC",
        title=f"{tag}: EBC vs ETC model fit by cell classification\n(ΔpseudoR² vs null)",
        out_path=out_dir / f"{tag}_CLASS_dR2_EBC_vs_ETC.png",
    )

    classification_scatter(
        df,
        xcol="unique_EBC_r2",
        ycol="unique_EBOC_r2",
        title=f"{tag}: UNIQUE R² contributions by cell classification\n(HD-controlled)",
        out_path=out_dir / f"{tag}_CLASS_UNIQUE_R2_EBC_vs_ETC.png",
    )

    winner_scatter(
        df,
        xcol="dR2_EBC_HD",
        ycol="dR2_EBOC_HD",
        title=f"{tag}: EBC+HD vs ETC+HD per cell (ΔpseudoR² vs null)\nEBC + ETC + Both cells",
        out_path=out_dir / f"{tag}_WINNER_dR2_EBC_HD_vs_EBOC_HD.png",
    )

    winner_scatter(
        df,
        xcol="dR2_EBC",
        ycol="dR2_EBOC_HD",
        title=f"{tag}: EBC vs ETC+HD per cell (ΔpseudoR² vs null)\nEBC + ETC + Both cells",
        out_path=out_dir / f"{tag}_WINNER_dR2_EBC_vs_EBOC_HD.png",
    )

    winner_scatter(
        df,
        xcol="dR2_EBC",
        ycol="dR2_EBOC",
        title=f"{tag}: EBC vs ETC per cell (ΔpseudoR² vs null)\nEBC + ETC + Both cells",
        out_path=out_dir / f"{tag}_WINNER_dR2_EBC_vs_ETC.png",
    )

    winner_scatter(
        df,
        xcol="unique_EBC_r2",
        ycol="unique_EBOC_r2",
        title=f"{tag}: UNIQUE R² EBC vs ETC (HD-controlled)\nEBC + ETC + Both cells",
        out_path=out_dir / f"{tag}_WINNER_UNIQUE_R2_EBC_vs_ETC.png",
    )

    # ============ DEVIANCE PLOTS (NOW ALSO HD-CONTROLLED) ============

    # Fair comparison: EBC+HD vs EBOC+HD
    classification_scatter(
        df,
        xcol="dDev_EBC_HD",
        ycol="dDev_EBOC_HD",
        title=f"{tag}: EBC+HD vs ETC+HD model fit by cell classification\n(dDeviance vs null)",
        out_path=out_dir / f"{tag}_CLASS_dDev_EBC_HD_vs_EBOC_HD.pdf",
    )

    winner_scatter(
        df,
        xcol="dDev_EBC_HD",
        ycol="dDev_EBOC_HD",
        title=f"{tag}: EBC+HD vs ETC+HD per cell (dDeviance vs null)\nEBC + ETC + Both cells",
        out_path=out_dir / f"{tag}_WINNER_dDev_EBC_HD_vs_EBOC_HD.pdf",
    )

    # EBC vs EBOC+HD (the deviance version of your key R2 plot)
    classification_scatter(
        df,
        xcol="dDev_EBC",
        ycol="dDev_EBOC_HD",
        title=f"{tag}: EBC vs ETC+HD model fit by cell classification\n(dDeviance vs null)",
        out_path=out_dir / f"{tag}_CLASS_dDev_EBC_vs_EBOC_HD.pdf",
    )

    winner_scatter(
        df,
        xcol="dDev_EBC",
        ycol="dDev_EBOC_HD",
        title=f"{tag}: EBC vs ETC+HD per cell (dDeviance vs null)\nEBC + ETC + Both cells",
        out_path=out_dir / f"{tag}_WINNER_dDev_EBC_vs_EBOC_HD.pdf",
    )

    # Legacy: EBC vs ETC (no HD)
    classification_scatter(
        df,
        xcol="dDev_EBC",
        ycol="dDev_EBOC",
        title=f"{tag}: EBC vs ETC model fit by cell classification\n(dDeviance vs null)",
        out_path=out_dir / f"{tag}_CLASS_dDev_EBC_vs_ETC.pdf",
    )

    classification_scatter(
        df,
        xcol="unique_EBC_given_EBOC_HD",
        ycol="unique_EBOC_given_EBC_HD",
        title=f"{tag}: UNIQUE contributions by cell classification\n(HD-controlled, deviance)",
        out_path=out_dir / f"{tag}_CLASS_UNIQUE_dDev_EBC_vs_ETC_HD.pdf",
    )

    winner_scatter(
        df,
        xcol="dDev_EBC",
        ycol="dDev_EBOC",
        title=f"{tag}: EBC vs ETC per cell (dDeviance vs null)\nEBC + ETC + Both cells",
        out_path=out_dir / f"{tag}_WINNER_dDev_EBC_vs_ETC.pdf",
    )

    winner_scatter(
        df,
        xcol="unique_EBC_given_EBOC_HD",
        ycol="unique_EBOC_given_EBC_HD",
        title=f"{tag}: UNIQUE EBC vs UNIQUE ETC (HD-controlled, deviance)\nEBC + ETC + Both cells",
        out_path=out_dir / f"{tag}_WINNER_UNIQUE_dDev_EBC_vs_ETC_HD.pdf",
    )

    # ============ COMPARISON BOXPLOTS ============
    make_comparison_boxplot(
        df,
        metric="unique_EBOC_r2",
        title=f"{tag}: Unique EBOC R² by cell classification",
        ylabel="Unique EBOC R² contribution",
        out_path=out_dir / f"{tag}_BOXPLOT_unique_EBOC_r2.png",
    )

    make_comparison_boxplot(
        df,
        metric="unique_EBC_r2",
        title=f"{tag}: Unique EBC R² by cell classification",
        ylabel="Unique EBC R² contribution",
        out_path=out_dir / f"{tag}_BOXPLOT_unique_EBC_r2.png",
    )

    make_comparison_boxplot(
        df,
        metric="dR2_EBOC_HD",
        title=f"{tag}: EBOC+HD model fit by cell classification",
        ylabel="ΔR² (EBOC+HD vs null)",
        out_path=out_dir / f"{tag}_BOXPLOT_dR2_EBOC_HD.png",
    )

    make_comparison_boxplot(
        df,
        metric="dR2_EBC_HD",
        title=f"{tag}: EBC+HD model fit by cell classification",
        ylabel="ΔR² (EBC+HD vs null)",
        out_path=out_dir / f"{tag}_BOXPLOT_dR2_EBC_HD.png",
    )


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-dir",
        required=True,
        help="Root folder containing animal subfolders with GLM .npy files (one file per cell).",
    )
    ap.add_argument("--out-dir", required=True, help="Output folder for CSV + plots.")
    ap.add_argument(
        "--glob",
        default="*.npy",
        help="Glob pattern to match cell files (default: *.npy). Example: '*chaseOnly.npy'",
    )
    ap.add_argument(
        "--animals",
        default=None,
        help="Optional comma-separated list of animals to include (e.g. 'ToothMuch,Arwen').",
    )
    ap.add_argument(
        "--class-csv",
        default=None,
        help="STRICT-long classification CSV (recommended). Required if you want ONLY EBC+ETC cells.",
    )

    args = ap.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    ensure_dir(out_dir)

    selected_animals = None
    if args.animals:
        selected_animals = {a.strip() for a in args.animals.split(",") if a.strip()}
        print(f"Filtering to animals: {sorted(selected_animals)}")

    files = sorted(base_dir.rglob(args.glob))
    if not files:
        raise SystemExit(f"No files found under {base_dir} matching pattern: {args.glob}")

    rows = []
    skipped_parse = 0
    skipped_load = 0
    skipped_missing_models = 0

    for f in files:
        animal = infer_animal_from_path(f)
        if animal is None:
            skipped_parse += 1
            continue

        if selected_animals is not None and animal not in selected_animals:
            continue

        cell_id, ch = parse_cell_id_and_channel(f)
        if cell_id is None or ch is None:
            skipped_parse += 1
            continue

        region = region_from_channel(animal, ch)
        if region is None:
            skipped_parse += 1
            continue

        try:
            devs = extract_deviances(f)
        except Exception:
            skipped_load += 1
            continue

        if not all(k in devs for k in MODELS_NEEDED):
            skipped_missing_models += 1
            continue

        # === DEVIANCE (lower is better, so null - model = positive means better) ===
        dev_null = devs["null"]["dev"]
        dDev_EBC = dev_null - devs["EBC"]["dev"]
        dDev_EBOC = dev_null - devs["EBOC"]["dev"]
        dDev_HD = dev_null - devs["Allo_HD"]["dev"]

        # ✅ NEW: HD-controlled deviance deltas
        dDev_EBC_HD = dev_null - devs["EBC_HD"]["dev"]
        dDev_EBOC_HD = dev_null - devs["EBOC_HD"]["dev"]

        # Unique nested contributions (deviance)
        unique_EBC_given_EBOC_HD_dev = devs["EBOC_HD"]["dev"] - devs["EBC_EBOC_HD"]["dev"]
        unique_EBOC_given_EBC_HD_dev = devs["EBC_HD"]["dev"] - devs["EBC_EBOC_HD"]["dev"]

        # === PSEUDO-R2 (higher is better, so model - null = positive means better) ===
        r2_null = devs["null"]["r2"]
        dR2_EBC = devs["EBC"]["r2"] - r2_null
        dR2_EBOC = devs["EBOC"]["r2"] - r2_null
        dR2_HD = devs["Allo_HD"]["r2"] - r2_null
        dR2_EBC_HD = devs["EBC_HD"]["r2"] - r2_null
        dR2_EBOC_HD = devs["EBOC_HD"]["r2"] - r2_null

        # Unique nested contributions (R2)
        unique_EBC_given_EBOC_HD_r2 = devs["EBC_EBOC_HD"]["r2"] - devs["EBOC_HD"]["r2"]
        unique_EBOC_given_EBC_HD_r2 = devs["EBC_EBOC_HD"]["r2"] - devs["EBC_HD"]["r2"]

        rows.append(dict(
            animal=animal,
            region=region,
            cell_id=cell_id,
            channel=ch,
            file=str(f),

            # Raw deviances
            dev_null=dev_null,
            dev_EBC=devs["EBC"]["dev"],
            dev_EBOC=devs["EBOC"]["dev"],
            dev_Allo_HD=devs["Allo_HD"]["dev"],
            dev_EBC_HD=devs["EBC_HD"]["dev"],
            dev_EBOC_HD=devs["EBOC_HD"]["dev"],
            dev_EBC_EBOC=devs["EBC_EBOC"]["dev"],
            dev_EBC_EBOC_HD=devs["EBC_EBOC_HD"]["dev"],

            # Delta deviance (positive = better than null)
            dDev_EBC=dDev_EBC,
            dDev_EBOC=dDev_EBOC,
            dDev_Allo_HD=dDev_HD,
            dDev_EBC_HD=dDev_EBC_HD,          # ✅ NEW
            dDev_EBOC_HD=dDev_EBOC_HD,        # ✅ NEW
            unique_EBC_given_EBOC_HD=unique_EBC_given_EBOC_HD_dev,
            unique_EBOC_given_EBC_HD=unique_EBOC_given_EBC_HD_dev,

            # Raw R2
            r2_null=r2_null,
            r2_EBC=devs["EBC"]["r2"],
            r2_EBOC=devs["EBOC"]["r2"],
            r2_Allo_HD=devs["Allo_HD"]["r2"],
            r2_EBC_HD=devs["EBC_HD"]["r2"],
            r2_EBOC_HD=devs["EBOC_HD"]["r2"],
            r2_EBC_EBOC=devs["EBC_EBOC"]["r2"],
            r2_EBC_EBOC_HD=devs["EBC_EBOC_HD"]["r2"],

            # Delta R2 (positive = better than null)
            dR2_EBC=dR2_EBC,
            dR2_EBOC=dR2_EBOC,
            dR2_Allo_HD=dR2_HD,
            dR2_EBC_HD=dR2_EBC_HD,
            dR2_EBOC_HD=dR2_EBOC_HD,
            unique_EBC_r2=unique_EBC_given_EBOC_HD_r2,
            unique_EBOC_r2=unique_EBOC_given_EBC_HD_r2,
        ))

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(
            "No usable cells found. Common causes:\n"
            "- filenames missing '_ch###' so channel can't be parsed\n"
            "- animal name not present in the path\n"
            "- some cells missing one of the required models\n"
        )

    # Merge classification (STRICT long)
    if args.class_csv:
        class_path = Path(args.class_csv).expanduser().resolve()
        class_df = load_classification_strict_long(class_path)

        print("\n=== DIAGNOSTIC: Classification CSV contents ===")
        raw_csv = pd.read_csv(class_path)
        print(f"Columns: {list(raw_csv.columns)}")
        if "mode" in raw_csv.columns:
            print(f"Unique 'mode' values: {sorted(raw_csv['mode'].dropna().unique())}")
        if "classification" in raw_csv.columns:
            print(f"Unique 'classification' values: {sorted(raw_csv['classification'].dropna().unique())}")

        df = df.merge(class_df, on=["animal", "cell_id"], how="left")
        df["cell_class"] = df["cell_class"].fillna("Unknown")

        print("\n=== CLASS LABEL COUNTS (after merge) ===")
        print(df["cell_class"].value_counts(dropna=False).to_string())

        print("\n=== CLASS COUNTS by ANIMAL ===")
        print(df.groupby(["animal", "cell_class"]).size().unstack(fill_value=0).to_string())

        print("\n=== CLASS COUNTS by REGION ===")
        print(df.groupby(["region", "cell_class"]).size().unstack(fill_value=0).to_string())

        # Filter to EBC + ETC + Both cells
        df = df[df["cell_class"].isin(["EBC", "ETC", "Both"])].copy()

        print("\n=== USING EBC + ETC + Both CELLS ===")
        print(df["cell_class"].value_counts().to_string())

    else:
        df["cell_class"] = "Unknown"
        print("\nWARNING: --class-csv not provided, so I cannot filter to ONLY EBC+ETC-labeled cells.")
        print("The plots will include ALL cells.\n")

    # Save CSVs
    df.to_csv(out_dir / "glm_EBC_ETC_ONLY_ALL.csv", index=False)
    df_rsc = df[df["region"] == "RSC"].copy()
    df_sc = df[df["region"] == "SC"].copy()
    df_rsc.to_csv(out_dir / "glm_EBC_ETC_ONLY_RSC.csv", index=False)
    df_sc.to_csv(out_dir / "glm_EBC_ETC_ONLY_SC.csv", index=False)

    # Make plots
    make_all_region_plots(df, out_dir, tag="ALL")
    make_all_region_plots(df_rsc, out_dir, tag="RSC")
    make_all_region_plots(df_sc, out_dir, tag="SC")

    # Final summary
    print("\nDONE ✅")
    print(f"Base dir: {base_dir}")
    print(f"Out dir:  {out_dir}")
    print(f"Usable EBC+ETC cells: {len(df)}  (RSC={len(df_rsc)}, SC={len(df_sc)})")
    print(f"Skipped (parse/region): {skipped_parse}")
    print(f"Skipped (load errors):  {skipped_load}")
    print(f"Skipped (missing required models): {skipped_missing_models}")


if __name__ == "__main__":
    main()

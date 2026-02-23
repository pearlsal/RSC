#!/usr/bin/env python3
"""
Wall vs Bait decoding + spike-time circular-shift null distributions
===================================================================

What this script does
---------------------
1) Runs your REAL decoders (wall + bait) on a test window
2) Runs NULL decoders by circularly shifting spike times (np.roll-in-time equivalent)
3) Computes p-values + z-scores (real vs null)
4) Saves a single publication-ready 3×4 panel figure:
   Row 1: True vs Decoded scatter
   Row 2: Time series (True + Decoded)
   Row 3: Null distributions (directly below corresponding panel)

How to run (recommended)
------------------------
Option A (recommended): provide a decoder module that exposes:
  - decode_wall(spike_times_train: dict, behavior_test: dict) -> (dec_dist, dec_bear)
  - decode_bait(spike_times_train: dict, behavior_test: dict) -> (dec_dist, dec_bear)

And provide a .npz with:
  - time_test_s
  - true_wall_dist, true_wall_bear
  - true_bait_dist, true_bait_bear
  - train_T_seconds
  - spike_times_train_pickle  (path to a pickle file storing dict[cell_id] -> np.array spike times (sec) in [0,T))

Example:
  python wall_bait_null_panel.py \
    --decoder-module /path/to/your_decoder_impl.py \
    --input-npz /path/to/test_bundle.npz \
    --out-prefix /path/to/out/wall_vs_bait_C2train_C1test \
    --n-shuffles 600 \
    --min-shift 10

Option B (demo): run with synthetic data (just to verify the figure/layout)
  python wall_bait_null_panel.py --demo --out-prefix ./demo

Notes
-----
- This script DOES NOT implement your Bayesian decoder. It calls YOUR functions.
- Nulls are done by circularly shifting spike times for each neuron independently.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
from dataclasses import dataclass
from typing import Dict, Tuple, Callable, Any

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Paper style
# =============================================================================
def set_paper_style():
    plt.rcParams.update({
        "font.family": "Helvetica",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
    })


# =============================================================================
# Angle helpers
# =============================================================================
def wrap_deg(a: np.ndarray) -> np.ndarray:
    """Wrap degrees to [-180, 180)."""
    return (a + 180.0) % 360.0 - 180.0


# =============================================================================
# Null shuffle (circular time shift)
# =============================================================================
def circular_shift_spikes(spike_times: np.ndarray, T: float, min_shift: float, rng: np.random.Generator) -> np.ndarray:
    """Circularly shift spike times by a random amount in (min_shift, T-min_shift)."""
    if T <= 2 * min_shift:
        shift = T / 2.0
    else:
        shift = rng.uniform(min_shift, T - min_shift)
    shifted = (spike_times + shift) % T
    shifted.sort()
    return shifted


def compute_p_value(real_value: float, null_values: np.ndarray, higher_better: bool) -> float:
    null_values = np.asarray(null_values)
    if higher_better:
        return float(np.mean(null_values >= real_value))
    else:
        return float(np.mean(null_values <= real_value))


def compute_zscore(real_value: float, null_values: np.ndarray) -> float:
    null_values = np.asarray(null_values)
    mu = float(np.mean(null_values))
    sd = float(np.std(null_values, ddof=1))
    if sd == 0.0:
        return float("nan")
    return float((real_value - mu) / sd)


# =============================================================================
# Plot helpers
# =============================================================================
def add_panel_label(ax, label: str, x: float = -0.12, y: float = 1.08):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left")


def scatter_true_vs_dec(ax, true, dec, xlabel, ylabel, title, color):
    ax.scatter(true, dec, s=10, alpha=0.8, color=color, edgecolors="none")
    mn = float(min(np.min(true), np.min(dec)))
    mx = float(max(np.max(true), np.max(dec)))
    ax.plot([mn, mx], [mn, mx], "k--", lw=1.2, label="Perfect")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper left")


def timeseries(ax, t, true, dec, ylabel, title, color):
    ax.plot(t, true, lw=1.2, color="black", label="True")
    ax.plot(t, dec, lw=1.0, color=color, alpha=0.9, label="Decoded")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper right")


def plot_null(ax, null_vals, real_val, xlabel, higher_better, color, title_prefix):
    null_vals = np.asarray(null_vals)
    ax.hist(null_vals, bins=40, alpha=0.65, color=color)
    ax.axvline(real_val, color="k", lw=2, label="Real")
    p = compute_p_value(real_val, null_vals, higher_better=higher_better)
    z = compute_zscore(real_val, null_vals)
    ax.set_title(f"{title_prefix}\n p={p:.3g} | z={z:.2f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.legend(frameon=False, loc="best")


# =============================================================================
# Decoder loading
# =============================================================================
@dataclass
class Decoders:
    decode_wall: Callable[[Dict[Any, np.ndarray], Dict[str, np.ndarray]], Tuple[np.ndarray, np.ndarray]]
    decode_bait: Callable[[Dict[Any, np.ndarray], Dict[str, np.ndarray]], Tuple[np.ndarray, np.ndarray]]


def load_decoder_module(module_path: str) -> Decoders:
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Decoder module not found: {module_path}")

    spec = importlib.util.spec_from_file_location("user_decoder_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from: {module_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore

    if not hasattr(mod, "decode_wall") or not hasattr(mod, "decode_bait"):
        raise AttributeError(
            "Your decoder module must define BOTH functions:\n"
            "  decode_wall(spike_times_train, behavior_test) -> (dec_dist, dec_bear)\n"
            "  decode_bait(spike_times_train, behavior_test) -> (dec_dist, dec_bear)\n"
        )

    return Decoders(decode_wall=mod.decode_wall, decode_bait=mod.decode_bait)


# =============================================================================
# Metrics computation (REAL)
# =============================================================================
def run_metrics(decoder_fn, spike_times_train, behavior_test, prefix: str):
    """
    prefix in {"wall","bait"}
    expects in behavior_test:
      true_{prefix}_dist, true_{prefix}_bear, time_test_s
    """
    dec_dist, dec_bear = decoder_fn(spike_times_train, behavior_test)

    true_dist = np.asarray(behavior_test[f"true_{prefix}_dist"])
    true_bear = np.asarray(behavior_test[f"true_{prefix}_bear"])
    dec_dist = np.asarray(dec_dist)
    dec_bear = np.asarray(dec_bear)

    dist_err = np.abs(dec_dist - true_dist)
    bear_err = np.abs(wrap_deg(dec_bear - true_bear))

    # correlations (handle degenerate cases safely)
    dist_corr = float(np.corrcoef(true_dist, dec_dist)[0, 1]) if np.std(true_dist) > 0 and np.std(dec_dist) > 0 else float("nan")
    bear_corr = float(np.corrcoef(true_bear, dec_bear)[0, 1]) if np.std(true_bear) > 0 and np.std(dec_bear) > 0 else float("nan")

    return {
        "true_dist": true_dist,
        "true_bear": true_bear,
        "dec_dist": dec_dist,
        "dec_bear": dec_bear,
        "dist_err_mean": float(np.mean(dist_err)),
        "bear_err_mean": float(np.mean(bear_err)),
        "dist_corr": dist_corr,
        "bear_corr": bear_corr,
    }


# =============================================================================
# NULL runner
# =============================================================================
def run_nulls(spike_times_train, behavior_test, decoder_fn, prefix: str,
              T_train: float, n_shuffles: int, min_shift: float, seed: int):
    rng = np.random.default_rng(seed)

    # real
    real = run_metrics(decoder_fn, spike_times_train, behavior_test, prefix=prefix)

    null = {
        "dist_err_mean": np.zeros(n_shuffles, dtype=float),
        "bear_err_mean": np.zeros(n_shuffles, dtype=float),
        "dist_corr": np.zeros(n_shuffles, dtype=float),
        "bear_corr": np.zeros(n_shuffles, dtype=float),
    }

    cell_ids = list(spike_times_train.keys())

    for i in range(n_shuffles):
        shuffled = {}
        for cid in cell_ids:
            spikes = np.asarray(spike_times_train[cid], dtype=float)
            shuffled[cid] = circular_shift_spikes(spikes, T=T_train, min_shift=min_shift, rng=rng)

        m = run_metrics(decoder_fn, shuffled, behavior_test, prefix=prefix)

        null["dist_err_mean"][i] = m["dist_err_mean"]
        null["bear_err_mean"][i] = m["bear_err_mean"]
        null["dist_corr"][i] = m["dist_corr"]
        null["bear_corr"][i] = m["bear_corr"]

    return real, null


# =============================================================================
# Figure builder
# =============================================================================
def make_figure(real_bait, real_wall, null_bait, null_wall, t, suptitle, out_png, out_pdf):
    set_paper_style()
    fig, axes = plt.subplots(3, 4, figsize=(18, 12), constrained_layout=True)

    # Row 1: scatter
    scatter_true_vs_dec(
        axes[0, 0], real_bait["true_dist"], real_bait["dec_dist"],
        "True Bait Distance (m)", "Decoded Bait Distance (m)",
        f"Bait Distance Decoding\nr={real_bait['dist_corr']:.3f}", "red"
    )
    scatter_true_vs_dec(
        axes[0, 1], real_bait["true_bear"], real_bait["dec_bear"],
        "True Bait Bearing (°)", "Decoded Bait Bearing (°)",
        f"Bait Bearing Decoding\nr={real_bait['bear_corr']:.3f}", "red"
    )
    scatter_true_vs_dec(
        axes[0, 2], real_wall["true_dist"], real_wall["dec_dist"],
        "True Wall Distance (m)", "Decoded Wall Distance (m)",
        f"Wall Distance Decoding\nr={real_wall['dist_corr']:.3f}", "blue"
    )
    scatter_true_vs_dec(
        axes[0, 3], real_wall["true_bear"], real_wall["dec_bear"],
        "True Wall Bearing (°)", "Decoded Wall Bearing (°)",
        f"Wall Bearing Decoding\nr={real_wall['bear_corr']:.3f}", "blue"
    )

    # Row 2: time series
    timeseries(
        axes[1, 0], t, real_bait["true_dist"], real_bait["dec_dist"],
        "Bait Distance (m)",
        f"Bait Distance Time Series\nError: {real_bait['dist_err_mean']:.3f} m", "red"
    )
    timeseries(
        axes[1, 1], t, real_bait["true_bear"], real_bait["dec_bear"],
        "Bait Bearing (°)",
        f"Bait Bearing Time Series\nError: {real_bait['bear_err_mean']:.1f}°", "red"
    )
    timeseries(
        axes[1, 2], t, real_wall["true_dist"], real_wall["dec_dist"],
        "Wall Distance (m)",
        f"Wall Distance Time Series\nError: {real_wall['dist_err_mean']:.3f} m", "blue"
    )
    timeseries(
        axes[1, 3], t, real_wall["true_bear"], real_wall["dec_bear"],
        "Wall Bearing (°)",
        f"Wall Bearing Time Series\nError: {real_wall['bear_err_mean']:.1f}°", "blue"
    )

    # Row 3: null distributions
    # For errors: lower is better
    plot_null(
        axes[2, 0], null_bait["dist_err_mean"], real_bait["dist_err_mean"],
        "Mean Distance Error (m)", higher_better=False, color="red", title_prefix="Bait Distance Null"
    )
    plot_null(
        axes[2, 1], null_bait["bear_err_mean"], real_bait["bear_err_mean"],
        "Mean Bearing Error (°)", higher_better=False, color="red", title_prefix="Bait Bearing Null"
    )
    # For correlations: higher is better
    plot_null(
        axes[2, 2], null_wall["dist_corr"], real_wall["dist_corr"],
        "Distance Correlation", higher_better=True, color="blue", title_prefix="Wall Distance Null"
    )
    plot_null(
        axes[2, 3], null_wall["bear_corr"], real_wall["bear_corr"],
        "Bearing Correlation", higher_better=True, color="blue", title_prefix="Wall Bearing Null"
    )

    # Panel labels A–L
    labels = list("ABCDEFGHIJKL")
    k = 0
    for r in range(3):
        for c in range(4):
            add_panel_label(axes[r, c], labels[k])
            k += 1

    fig.suptitle(suptitle, fontsize=12, fontweight="bold")

    if out_png:
        fig.savefig(out_png, dpi=600, bbox_inches="tight")
    if out_pdf:
        fig.savefig(out_pdf, dpi=600, bbox_inches="tight")

    return fig


# =============================================================================
# Demo data generator (so the script can run end-to-end)
# =============================================================================
def demo_make_fake_bundle(out_prefix: str):
    """
    Makes fake spikes + behavior + simple decoders so you can see the layout.
    NOT scientifically meaningful—just a runnable demo.
    """
    rng = np.random.default_rng(0)

    T_train = 1200.0
    n_cells = 50
    spike_times = {}
    for i in range(n_cells):
        n_spk = rng.integers(200, 800)
        spike_times[i] = np.sort(rng.uniform(0, T_train, size=n_spk))

    # test
    t = np.linspace(0, 1500, 1500)  # 1 Hz for demo
    true_wall_dist = 0.2 + 1.2 * (0.5 + 0.5 * np.sin(t / 90.0))
    true_wall_bear = wrap_deg(180 * np.sin(t / 70.0))
    true_bait_dist = 0.1 + 1.0 * (0.5 + 0.5 * np.sin(t / 55.0 + 1.0))
    true_bait_bear = wrap_deg(180 * np.sin(t / 40.0 + 0.5))

    behavior_test = dict(
        time_test_s=t,
        true_wall_dist=true_wall_dist,
        true_wall_bear=true_wall_bear,
        true_bait_dist=true_bait_dist,
        true_bait_bear=true_bait_bear,
        train_T_seconds=T_train
    )

    # fake decoders (wall strong, bait weaker)
    def decode_wall(spikes_train, behavior):
        tt = behavior["time_test_s"]
        dec_dist = behavior["true_wall_dist"] + rng.normal(0, 0.18, size=tt.shape)
        dec_bear = wrap_deg(behavior["true_wall_bear"] + rng.normal(0, 25, size=tt.shape))
        return dec_dist, dec_bear

    def decode_bait(spikes_train, behavior):
        tt = behavior["time_test_s"]
        # degrade signal + bias to show weaker performance
        dec_dist = 0.35 + 0.15 * rng.normal(size=tt.shape)
        dec_bear = wrap_deg(rng.normal(0, 25, size=tt.shape))
        return dec_dist, dec_bear

    # save bundle
    spk_pkl = out_prefix + "_spikes_train.pkl"
    with open(spk_pkl, "wb") as f:
        pickle.dump(spike_times, f)

    npz_path = out_prefix + "_bundle.npz"
    np.savez(
        npz_path,
        time_test_s=t,
        true_wall_dist=true_wall_dist,
        true_wall_bear=true_wall_bear,
        true_bait_dist=true_bait_dist,
        true_bait_bear=true_bait_bear,
        train_T_seconds=T_train,
        spike_times_train_pickle=spk_pkl
    )
    return npz_path, Decoders(decode_wall=decode_wall, decode_bait=decode_bait)


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder-module", type=str, default=None,
                    help="Path to python file defining decode_wall() and decode_bait().")
    ap.add_argument("--input-npz", type=str, default=None,
                    help="NPZ containing time_test_s, true_* arrays, train_T_seconds, spike_times_train_pickle.")
    ap.add_argument("--out-prefix", type=str, required=True,
                    help="Output prefix for figure + logs.")
    ap.add_argument("--n-shuffles", type=int, default=600)
    ap.add_argument("--min-shift", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--demo", action="store_true", help="Run a runnable demo with synthetic data.")
    args = ap.parse_args()

    if args.demo:
        npz_path, decoders = demo_make_fake_bundle(args.out_prefix)
        bundle = np.load(npz_path, allow_pickle=True)
        behavior_test = {k: bundle[k] for k in bundle.files}
        # load spikes
        with open(str(behavior_test["spike_times_train_pickle"]), "rb") as f:
            spike_times_train = pickle.load(f)
    else:
        if args.decoder_module is None or args.input_npz is None:
            raise SystemExit("Need --decoder-module and --input-npz (or use --demo).")
        decoders = load_decoder_module(args.decoder_module)
        bundle = np.load(args.input_npz, allow_pickle=True)
        behavior_test = {k: bundle[k] for k in bundle.files}
        with open(str(behavior_test["spike_times_train_pickle"]), "rb") as f:
            spike_times_train = pickle.load(f)

    # pull essentials
    t = np.asarray(behavior_test["time_test_s"], dtype=float)
    T_train = float(np.asarray(behavior_test["train_T_seconds"]).item())

    # run real + nulls
    real_bait, null_bait = run_nulls(
        spike_times_train, behavior_test, decoders.decode_bait, prefix="bait",
        T_train=T_train, n_shuffles=args.n_shuffles, min_shift=args.min_shift, seed=args.seed
    )
    real_wall, null_wall = run_nulls(
        spike_times_train, behavior_test, decoders.decode_wall, prefix="wall",
        T_train=T_train, n_shuffles=args.n_shuffles, min_shift=args.min_shift, seed=args.seed + 1
    )

    # print key stats
    print("\n=== REAL METRICS ===")
    print("BAIT:",
          f"dist_err={real_bait['dist_err_mean']:.3f} m | bear_err={real_bait['bear_err_mean']:.1f}° | "
          f"dist_r={real_bait['dist_corr']:.3f} | bear_r={real_bait['bear_corr']:.3f}")
    print("WALL:",
          f"dist_err={real_wall['dist_err_mean']:.3f} m | bear_err={real_wall['bear_err_mean']:.1f}° | "
          f"dist_r={real_wall['dist_corr']:.3f} | bear_r={real_wall['bear_corr']:.3f}")

    print("\n=== NULL SIGNIFICANCE (p, z) ===")
    checks = [
        ("Bait dist err", real_bait["dist_err_mean"], null_bait["dist_err_mean"], False),
        ("Bait bear err", real_bait["bear_err_mean"], null_bait["bear_err_mean"], False),
        ("Wall dist r",   real_wall["dist_corr"],     null_wall["dist_corr"],     True),
        ("Wall bear r",   real_wall["bear_corr"],     null_wall["bear_corr"],     True),
    ]
    for name, rv, nv, higher in checks:
        p = compute_p_value(rv, nv, higher_better=higher)
        z = compute_zscore(rv, nv)
        print(f"{name:12s}: p={p:.3g} | z={z:.2f}")

    # save outputs
    out_png = args.out_prefix + "_WITH_NULL.png"
    out_pdf = args.out_prefix + "_WITH_NULL.pdf"

    suptitle = (
        "COMPREHENSIVE WALL vs BAIT DECODING SUMMARY\n"
        "(Real + Spike-Shift Null Controls)"
    )

    make_figure(real_bait, real_wall, null_bait, null_wall, t, suptitle, out_png, out_pdf)

    # save metrics too
    np.savez(args.out_prefix + "_metrics_and_nulls.npz",
             real_bait=real_bait,
             real_wall=real_wall,
             null_bait=null_bait,
             null_wall=null_wall)

    print(f"\nSaved:\n  {out_png}\n  {out_pdf}\n  {args.out_prefix}_metrics_and_nulls.npz")


if __name__ == "__main__":
    main()

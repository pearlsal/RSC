# =============================================================================
# MIMOSAPUDICA: CHASE ANALYSIS + SPEED FLIP-FLOP CONTROL
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from chase_cell_analysis_merge_analysis import (
    load_chase_session, merge_chase_sessions, rebin_session,
    build_session_paths, run_analysis, save_results
)

# ── CONFIG ──────────────────────────────────────────────────────────────────
folder = '/Users/pearls/Work/RSC_project/Data/'
animal = 'MimosaPudica'
binsize = '8ms'
target_binsize_ms = 50.0

chase_intervals_by_session = {
    'c1': [6570, 9359, 10630, 11046, 18612, 20167, 20519, 21387, 21564, 23129,
           24357, 25379, 25485, 27196, 62785, 63685, 65580, 68161, 69099, 70314,
           70506, 71418, 72757, 73200, 112741, 113430, 113906, 115142, 123128, 123779,
           124809, 125532, 137937, 139414, 140320, 140733, 144503, 144972, 157185, 157682,
           160558, 161196, 175193, 175875, 176651, 177511, 178329, 181987, 183352, 185336,
           186056, 189012, 223731, 224325, 224532, 225538, 256141, 259516, 295391, 295979,
           296389, 297440],
    'c2': [55, 1607, 2944, 4945, 5550, 6438, 6842, 7425, 11065, 11928,
           13549, 15909, 17554, 20111, 25364, 26707, 27821, 30559, 31524, 32850,
           44364, 44912, 58539, 58914],
}
sessions_to_merge = ['c1', 'c2']

# ── STEP 1: MERGE ──────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Merging chase sessions")
print("=" * 60)

session_files = build_session_paths(folder, animal, sessions_to_merge, binsize=binsize)
for f in session_files:
    print(f"  {f}")

merged = merge_chase_sessions(
    session_files=session_files,
    chase_intervals_per_session=chase_intervals_by_session,
    verbose=True
)
print(f"\nMerged (8ms): {merged['n_cells']} cells, {merged['n_timebins']} bins")

# ── STEP 2: REBIN ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Rebinning to 50ms")
print("=" * 60)

merged_50ms = rebin_session(merged, from_binsize_ms=8.33, to_binsize_ms=target_binsize_ms, verbose=True)
chase_intervals_ready = merged_50ms['merged_chase_intervals']
print(f"\nRebinned: {merged_50ms['n_cells']} cells, {merged_50ms['n_timebins']} bins")
print(f"Chase intervals: {len(chase_intervals_ready) // 2} bouts")

# ── STEP 3: OPEN FIELD ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Loading open field")
print("=" * 60)

of_8ms = load_chase_session(f"{folder}{animal}/RSC_OF1_binnedshareddata{binsize}.mat")
of_50ms = rebin_session(of_8ms, from_binsize_ms=8.33, to_binsize_ms=target_binsize_ms, verbose=True)
print(f"Open field: {of_50ms['n_cells']} cells, {of_50ms['n_timebins']} bins")

assert merged_50ms['n_cells'] == of_50ms['n_cells'], \
    f"Cell mismatch! Chase={merged_50ms['n_cells']}, OF={of_50ms['n_cells']}"

# ── STEP 4: CHASE ANALYSIS ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Running chase cell analysis (1000 shuffles)")
print("=" * 60)

chase_results = run_analysis(
    spikemat_chase=merged_50ms['spikemat'],
    spikemat_null=of_50ms['spikemat'],
    chase_intervals=chase_intervals_ready,
    cell_ids=merged_50ms['cell_ids'],
    num_shuffles=1000,
    alpha=0.025,
    verbose=True
)

save_results(
    results=chase_results,
    output_dir=f"{folder}{animal}/analysis_output",
    prefix=f"MimosaPudica_RSC_c1c2_merged_chase_50ms",
    save_csv=True,
    save_plot=True,
    spikemat=merged_50ms['spikemat'],
    title="MimosaPudica RSC c1+c2 merged (50ms) - Chase Cell Activity"
)

# ── STEP 5: SPEED CONTROL ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Speed flip-flop control (open field)")
print("=" * 60)

exc_idx = chase_results['excited_idx']
sup_idx = chase_results['suppressed_idx']
print(f"Chase-excited:    n={len(exc_idx)}")
print(f"Chase-suppressed: n={len(sup_idx)}")

if len(exc_idx) < 3 or len(sup_idx) < 3:
    print(f"\n⚠ WARNING: Need ≥3 cells in both groups. Exc={len(exc_idx)}, Sup={len(sup_idx)}")

spikemat_of = of_50ms['spikemat']
speed_of = of_50ms['binned_speed']
print(f"Speed shape: {speed_of.shape}, range [{np.nanmin(speed_of):.2f}, {np.nanmax(speed_of):.2f}]")

speed_percentile = 20.0
valid = np.isfinite(speed_of) & (speed_of >= 0)
low_thresh = np.percentile(speed_of[valid], speed_percentile)
high_thresh = np.percentile(speed_of[valid], 100 - speed_percentile)

low_bins = np.where(valid & (speed_of <= low_thresh))[0]
high_bins = np.where(valid & (speed_of >= high_thresh))[0]
print(f"Speed thresholds: low ≤ {low_thresh:.2f}, high ≥ {high_thresh:.2f}")
print(f"  Low-speed bins:  {len(low_bins)}")
print(f"  High-speed bins: {len(high_bins)}")

rates_low = np.mean(spikemat_of[low_bins, :], axis=0)
rates_high = np.mean(spikemat_of[high_bins, :], axis=0)

denom = rates_high + rates_low
denom[denom == 0] = np.nan
smi = (rates_high - rates_low) / denom

print("\n--- Within-group: high vs low speed (Wilcoxon) ---")
p_exc, p_sup, p_between = np.nan, np.nan, np.nan

if len(exc_idx) >= 3:
    w_exc, p_exc = stats.wilcoxon(rates_high[exc_idx], rates_low[exc_idx])
    print(f"  Excited:    W={w_exc:.1f}, p={p_exc:.4f}")
else:
    print(f"  Excited:    too few cells (n={len(exc_idx)})")

if len(sup_idx) >= 3:
    w_sup, p_sup = stats.wilcoxon(rates_high[sup_idx], rates_low[sup_idx])
    print(f"  Suppressed: W={w_sup:.1f}, p={p_sup:.4f}")
else:
    print(f"  Suppressed: too few cells (n={len(sup_idx)})")

print("\n--- Between groups: speed modulation index (Mann-Whitney) ---")
if len(exc_idx) >= 3 and len(sup_idx) >= 3:
    u_stat, p_between = stats.mannwhitneyu(smi[exc_idx], smi[sup_idx], alternative='two-sided')
    print(f"  U={u_stat:.1f}, p={p_between:.4f}")
    print(f"  Excited  SMI: {np.nanmean(smi[exc_idx]):.3f} ± {np.nanstd(smi[exc_idx]):.3f}")
    print(f"  Suppressed SMI: {np.nanmean(smi[sup_idx]):.3f} ± {np.nanstd(smi[sup_idx]):.3f}")
else:
    print("  Cannot compare — need ≥3 cells in both groups")

# ── PLOT ────────────────────────────────────────────────────────────────────
def p_to_stars(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return 'n.s.'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [1.2, 1]})

group_list = []
if len(exc_idx) > 0: group_list.append(('Excited', exc_idx, 'royalblue'))
if len(sup_idx) > 0: group_list.append(('Suppressed', sup_idx, 'firebrick'))

for i, (label, idx, color) in enumerate(group_list):
    pos_lo, pos_hi = i - 0.15, i + 0.15
    ax1.boxplot(rates_low[idx], positions=[pos_lo], widths=0.22, patch_artist=True,
                boxprops=dict(facecolor='white', edgecolor=color),
                medianprops=dict(color=color, linewidth=1.5),
                whiskerprops=dict(color=color), capprops=dict(color=color),
                flierprops=dict(marker='o', markerfacecolor=color, markersize=3, alpha=0.5))
    ax1.boxplot(rates_high[idx], positions=[pos_hi], widths=0.22, patch_artist=True,
                boxprops=dict(facecolor=color, edgecolor=color, alpha=0.4),
                medianprops=dict(color=color, linewidth=1.5),
                whiskerprops=dict(color=color), capprops=dict(color=color),
                flierprops=dict(marker='o', markerfacecolor=color, markersize=3, alpha=0.5))
    for j in range(len(idx)):
        ax1.plot([pos_lo, pos_hi], [rates_low[idx[j]], rates_high[idx[j]]],
                 color=color, alpha=0.25, linewidth=0.7)
        ax1.scatter([pos_lo, pos_hi], [rates_low[idx[j]], rates_high[idx[j]]],
                    color=color, s=12, alpha=0.5, zorder=2)

ax1.set_xticks(range(len(group_list)))
ax1.set_xticklabels([f'Chase-{l}\n(n={len(idx)})' for l, idx, _ in group_list])
ax1.set_ylabel('Mean firing rate (spikes/bin)')
ax1.set_title(f'Bottom {speed_percentile:.0f}% (open) vs Top {100-speed_percentile:.0f}% (filled) speed')
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

rng = np.random.default_rng(42)
for i, (label, idx, color) in enumerate(group_list):
    jitter = rng.uniform(-0.12, 0.12, size=len(idx))
    ax2.scatter(np.full(len(idx), i) + jitter, smi[idx],
                color=color, s=25, alpha=0.6, edgecolors='white', linewidths=0.3)
    mean_val = np.nanmean(smi[idx])
    sem_val = np.nanstd(smi[idx]) / np.sqrt(np.sum(np.isfinite(smi[idx])))
    ax2.errorbar(i, mean_val, yerr=sem_val, fmt='s', color=color, markersize=8,
                 capsize=4, capthick=1.5, markeredgecolor='black', markeredgewidth=0.5, zorder=5)

if np.isfinite(p_between) and len(group_list) == 2:
    y_top = max(np.nanmax(smi[exc_idx]) if len(exc_idx) else 0,
                np.nanmax(smi[sup_idx]) if len(sup_idx) else 0) + 0.08
    ax2.plot([0, 0, 1, 1], [y_top, y_top+0.02, y_top+0.02, y_top], color='black', linewidth=1)
    ax2.text(0.5, y_top+0.03, p_to_stars(p_between), ha='center', va='bottom', fontsize=9)

ax2.axhline(0, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)
ax2.set_xticks(range(len(group_list)))
ax2.set_xticklabels([f'Chase-{l}' for l, _, _ in group_list])
ax2.set_ylabel('Speed modulation index\n(high−low)/(high+low)')
ax2.set_title('Open field speed modulation')
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

fig.suptitle('MimosaPudica: Speed Flip-Flop Control (Open Field)', fontsize=13, fontweight='bold', y=1.02)
fig.tight_layout()

speed_fig_path = f"{folder}{animal}/analysis_output/MimosaPudica_speed_flipflop_control.pdf"
fig.savefig(speed_fig_path, dpi=300, bbox_inches='tight')
print(f"\nSaved: {speed_fig_path}")
plt.show()
print("\n✓ Done!")
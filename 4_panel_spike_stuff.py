#!/usr/bin/env python3
"""
EBOC Summary Figure Generator  (v3 — ETC overlay)
===================================================
Per-animal, 4 polar panels:
  1. Target Occupancy
  2. Spike Occupancy
  3. Rat-to-target Ratemap  (mean of per-cell raw ratemaps)
  4. Cell Peak Locations
       grey  +  = all active cells
       red   +  = ETCs only  (is_significant == True from classification CSV)

CSV path pattern:
    <csv_dir>/<animal>/EBOC_<animal>_classification_summary.csv

Usage: edit the CONFIG block at the bottom, then run.
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path
from scipy.signal import convolve2d
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# ── constants ────────────────────────────────────────────────────────────────
OCCUPANCY_THRESHOLD = 20

N_THETA       = 18
THETA_EDGES   = np.linspace(-np.pi, np.pi, N_THETA + 1)
THETA_CENTERS = (THETA_EDGES[:-1] + THETA_EDGES[1:]) / 2.0
DIST_EDGES_CM = np.arange(0.0, 90.0 + 4.5, 4.5)   # 0:4.5:90 → 20 bins
DIST_CENTERS  = (DIST_EDGES_CM[:-1] + DIST_EDGES_CM[1:]) / 2.0
N_DIST        = len(DIST_EDGES_CM) - 1


# ── smoothing ────────────────────────────────────────────────────────────────
def smooth_mat_wrapped(mat, kernel_size, std):
    if std == 0:
        return mat
    ny, nx = kernel_size
    y  = np.arange(-(ny-1)/2.0, (ny-1)/2.0 + 1)
    x  = np.arange(-(nx-1)/2.0, (nx-1)/2.0 + 1)
    Yg, Xg = np.meshgrid(y, x, indexing='ij')
    k  = np.exp(-(Xg**2 + Yg**2) / (2.0*std**2))
    k /= k.sum()
    return convolve2d(mat, k, mode='same', boundary='wrap')


# ── chase interval lookup ────────────────────────────────────────────────────
def extract_chase_intervals(data, session, animal):
    intervals_map = {
        'Arwen': {
            'c1': [1610,7583, 27805,38704, 53618,57788],
            'c2': [3996,4946, 13087,20357, 26277,35577, 44556,47376, 49806,53276,
                   54831,61091, 62611,65211],
            'c4': [2490,9030, 18678,23108, 28007,29057, 34017,36677, 39773,43193,
                   53881,60621, 73509,77469],
        },
        'Tauriel': {
            'c1': [1308,4932, 17604,23452, 32330,34869, 39026,55359, 65412,70414],
            'c2': [2848,12662, 23006,36415, 54592,56622, 72111,75756],
            'c4': [1045,24906, 41991,50003, 53071,85051],
            'c5': [3266,22481, 29292,32927, 36211,41067, 59829,80588],
        },
        'PreciousGrape': {
            'c1': [1328,5662, 6271,9598, 19173,23362, 37066,50673, 60591,61942,
                   81982,89630, 95864,97545, 106412,109122, 109926,112703,
                   118190,120433, 126228,136068, 140494,146517, 147408,152977,
                   160049,172396],
            'c2': [22958,32685, 37162,44870, 52398,59078, 70657,83937, 88255,89486,
                   102991,108700, 119019,120545, 127238,142105, 145235,152788,
                   158909,172790, 181085,190973, 191988,206881, 212596,227509,
                   229707,231460, 244807,259357, 275659,283610, 283965,289594],
        },
        'ToothMuch': {
            'c1': [2864,3142, 3280,9074, 9434,10727, 16364,18090, 18368,21101,
                   21499,22887, 23162,24313, 24821,25109, 39485,40367, 40608,42503,
                   43410,44623, 44969,46337, 51153,51645, 52704,54166, 61215,61390,
                   61683,62515, 63264,65663, 66483,67939, 68358,69678, 72545,72648,
                   85962,86963, 87208,87606, 90513,91619, 95734,97353, 97795,98850,
                   99512,100233, 103172,104317, 105184,106051, 106795,108280,
                   124814,128850, 129306,131594, 132160,134162, 135324,135824,
                   136506,137616, 140332,142947],
            'c2': [3707,11494, 12030,15606, 29560,32998, 33731,37618, 38104,38726,
                   39185,40827, 47692,50182, 56995,60636, 62313,67521, 77232,77588,
                   78058,78553, 80353,82153, 82581,85436, 85777,86744, 87304,87825,
                   88334,89129, 89668,90033, 100036,100710, 101806,104542,
                   104949,106761, 108295,109014, 124283,126088, 127924,129349,
                   139810,140402, 141132,141981, 142657,143119, 143783,145040],
        },
        'MimosaPudica': {
            'c1': [6570,9359, 10630,11046, 18612,20167, 20519,21387, 21564,23129,
                   24357,25379, 25485,27196, 62785,63685, 65580,68161, 69099,70314,
                   70506,71418, 72757,73200, 112741,113430, 113906,115142,
                   123128,123779, 124809,125532, 137937,139414, 140320,140733,
                   144503,144972, 157185,157682, 160558,161196, 175193,175875,
                   176651,177511, 178329,181987, 183352,185336, 186056,189012,
                   223731,224325, 224532,225538, 256141,259516, 295391,295979,
                   296389,297440],
            'c2': [55,1607, 2944,4945, 5550,6438, 6842,7425, 11065,11928, 13549,15909,
                   17554,20111, 25364,26707, 27821,30559, 31524,32850, 44364,44912,
                   58539,58914],
            'c3': [2081,2493, 10037,10755, 12191,12821],
            'noc1': [3792,13781, 23280,25444],
            'noc2': [1696,10318, 19078,26666, 40984,49613, 58052,65581, 73842,81018],
            'noc3': [7381,10883, 11385,11542, 12912,13752, 14561,14976, 15265,15496,
                     19812,24622, 29601,30261, 32089,38813, 49436,57103, 61119,75584],
        },
    }
    animal_map = intervals_map.get(animal, {})
    if session not in animal_map:
        return None
    borders = animal_map[session]
    idx = []
    for j in range(0, len(borders), 2):
        if j + 1 < len(borders):
            idx.extend(range(borders[j], borders[j+1]))
    return np.array(idx, dtype=int)


# ── EBOC binning ─────────────────────────────────────────────────────────────
def bin_eboc(bait_angle, bait_dist, spk):
    """Bin into [D x T] occ, nspk, and smoothed ratemap (matches compute_eboc_ratemap)."""
    ba  = np.asarray(bait_angle).ravel()
    bd  = np.asarray(bait_dist).ravel()
    spk = np.asarray(spk).ravel()

    occ  = np.zeros((N_THETA, N_DIST), dtype=float)
    nspk = np.zeros((N_THETA, N_DIST), dtype=float)

    for i in range(N_THETA):
        lo, hi   = THETA_EDGES[i], THETA_EDGES[i+1]
        ang_mask = (ba >= lo) & (ba < hi) if i < N_THETA-1 else (ba >= lo) & (ba <= hi)
        if not np.any(ang_mask):
            continue
        for k in range(N_DIST):
            d_lo, d_hi = DIST_EDGES_CM[k], DIST_EDGES_CM[k+1]
            dmask = (bd >= d_lo) & (bd < d_hi) if k < N_DIST-1 else (bd >= d_lo) & (bd <= d_hi)
            m = ang_mask & dmask
            n_occ = int(m.sum())
            occ[i, k]  = float(n_occ)
            nspk[i, k] = float((spk[m] > 0).sum()) if n_occ > 0 else 0.0

    occ_ns  = occ.T    # [D, T]
    nspk_ns = nspk.T

    with np.errstate(divide='ignore', invalid='ignore'):
        rm_ns = nspk_ns / occ_ns
    rm_ns[~np.isfinite(rm_ns)] = np.nan

    rm_s = smooth_mat_wrapped(rm_ns, (3, 3), 1.5)
    rm_s[occ_ns < OCCUPANCY_THRESHOLD] = np.nan

    return occ_ns, nspk_ns, rm_s


# ── peak estimator ────────────────────────────────────────────────────────────
def pref_orient_dist(rm_s):
    """
    Preferred (angle, distance) via circular mean over distance-collapsed rates,
    then peak distance along that preferred angle column.
    """
    if not np.any(np.isfinite(rm_s)):
        return np.nan, np.nan

    fr_by_angle = np.nanmean(rm_s, axis=0)
    fr_by_angle[~np.isfinite(fr_by_angle)] = 0.0
    if fr_by_angle.sum() == 0:
        return np.nan, np.nan

    r          = np.sum(fr_by_angle * np.exp(1j * THETA_CENTERS))
    pref_angle = float(np.angle(r))

    diffs = np.abs(((THETA_CENTERS - pref_angle) + np.pi) % (2*np.pi) - np.pi)
    t_idx = int(np.argmin(diffs))
    col   = rm_s[:, t_idx]
    pref_dist = float(DIST_CENTERS[np.nanargmax(col)]) if np.any(np.isfinite(col)) else np.nan

    return pref_angle, pref_dist


# ── classification CSV loader ─────────────────────────────────────────────────
def load_etc_mask(csv_dir, animal, n_cells):
    """
    Returns boolean array length n_cells; True = significant ETC.
    Returns None if CSV not found.
    """
    csv_path = Path(csv_dir) / animal / f"EBOC_{animal}_classification_summary.csv"
    if not csv_path.exists():
        print(f"  NOTE: CSV not found -> {csv_path}")
        return None

    df    = pd.read_csv(csv_path)
    n_etc = int(df['is_significant'].sum()) if 'is_significant' in df.columns else 0
    print(f"  CSV loaded: {len(df)} cells, {n_etc} ETCs")

    mask = np.zeros(n_cells, dtype=bool)
    for _, row in df.iterrows():
        idx = int(row['neuron_idx']) - 1   # CSV is 1-indexed
        if 0 <= idx < n_cells and bool(row['is_significant']):
            mask[idx] = True
    return mask


# ── polar plotting helpers ────────────────────────────────────────────────────
def _dir_labels(ax, max_dist):
    lr = max_dist * 1.14
    specs = [('F', (0,  lr),  'center', 'bottom'),
             ('R', (lr,  0),  'left',   'center'),
             ('B', (0, -lr),  'center', 'top'),
             ('L', (-lr, 0),  'right',  'center')]
    for txt, xy, ha, va in specs:
        ax.text(*xy, txt, ha=ha, va=va, fontsize=9, fontweight='bold')


def _polar_mesh(ax, data_2d, title, vmin=None, vmax=None, cmap='Blues', max_dist=90.0):
    """Plot [D x T] ratemap.  theta=0 -> F (top), theta=pi/2 -> R (right)."""
    C     = np.c_[data_2d, data_2d[:, 0]]
    t_ext = np.r_[THETA_EDGES, THETA_EDGES[-1] + (THETA_EDGES[1]-THETA_EDGES[0])]
    Theta, R_ = np.meshgrid(t_ext, DIST_EDGES_CM)
    Tp = -Theta + np.pi/2
    pc = ax.pcolormesh(R_*np.cos(Tp), R_*np.sin(Tp), C,
                       cmap=cmap, shading='auto',
                       norm=Normalize(vmin=vmin, vmax=vmax))
    for r in [20, 40, 60, 90]:
        if r <= max_dist:
            ax.add_patch(plt.Circle((0,0), r, fill=False, lw=0.5, color='gray', alpha=0.4))
            ax.text(r*0.72, r*0.72, f'{r}', fontsize=6, color='gray', alpha=0.7)
    _dir_labels(ax, max_dist)
    ax.set_aspect('equal')
    m = max_dist * 1.22
    ax.set_xlim(-m, m); ax.set_ylim(-m, m)
    ax.axis('off')
    ax.set_title(title, fontsize=10, pad=8)
    return pc


def _peak_scatter(ax, peak_angles, peak_dists, n_active, etc_mask=None, max_dist=90.0):
    """
    Scatter of preferred (angle, distance) per cell.
      Light grey + : all active cells
      Crimson    + : ETCs (is_significant == True), drawn on top
    """
    for r in [20, 40, 60, 90]:
        if r <= max_dist:
            ax.add_patch(plt.Circle((0,0), r, fill=False, lw=0.5, color='lightgray'))
            ax.text(r*0.72, r*0.72, f'{r}', fontsize=6, color='gray', alpha=0.7)
    for deg in range(0, 360, 45):
        th = -np.deg2rad(deg) + np.pi/2
        ax.plot([0, max_dist*np.cos(th)], [0, max_dist*np.sin(th)],
                color='lightgray', lw=0.5, zorder=0)

    pa    = np.asarray(peak_angles)
    pd_   = np.asarray(peak_dists)
    valid = np.isfinite(pa) & np.isfinite(pd_)

    # all active cells — faint grey
    if np.any(valid):
        th = -pa[valid] + np.pi/2
        ax.scatter(pd_[valid]*np.cos(th), pd_[valid]*np.sin(th),
                   marker='+', s=50, linewidths=1.0,
                   c='#cccccc', zorder=2, alpha=0.7, label='All cells')

    # ETCs — bold crimson on top
    n_etc = 0
    if etc_mask is not None:
        etc_valid = valid & np.asarray(etc_mask, dtype=bool)
        n_etc = int(np.sum(etc_valid))
        if n_etc > 0:
            th = -pa[etc_valid] + np.pi/2
            ax.scatter(pd_[etc_valid]*np.cos(th), pd_[etc_valid]*np.sin(th),
                       marker='+', s=110, linewidths=2.2,
                       c='crimson', zorder=4, alpha=0.95, label=f'ETC (n={n_etc})')

    _dir_labels(ax, max_dist)

    n_valid = int(np.sum(valid))
    if etc_mask is not None:
        title = f'ETC Peaks (red) vs All (grey)\nETC={n_etc}  /  active={n_valid}'
    else:
        title = f'Cell Peak Locations\n(n={n_valid} / {n_active})'

    ax.set_aspect('equal')
    m = max_dist * 1.22
    ax.set_xlim(-m, m); ax.set_ylim(-m, m)
    ax.axis('off')
    ax.set_title(title, fontsize=10, pad=8)

    if etc_mask is not None and n_etc > 0:
        ax.legend(loc='lower right', fontsize=6, framealpha=0.7,
                  markerscale=0.8, handletextpad=0.3)


# ── main per-animal function ──────────────────────────────────────────────────
def make_animal_summary(folder_loc, animal, channels, chase_sessions,
                        binsize=0.00833, speed_thresh=3.0,
                        occ_min=OCCUPANCY_THRESHOLD, output_dir='.',
                        csv_dir=None):
    """
    Parameters
    ----------
    csv_dir : str or None
        Root folder containing <animal>/EBOC_<animal>_classification_summary.csv
        If None, all cells shown in same grey colour with no ETC distinction.
    """
    print(f"\n{'='*60}\n{animal}\n{'='*60}")

    # ── 1. Load & concatenate chase bouts across sessions ───────────────
    sessions_data = []
    n_cells = None

    for sess in chase_sessions:
        fname = f"{channels}_{sess}_binnedshareddata{int(binsize*1000)}ms.mat"
        fpath = Path(folder_loc) / "Data" / animal / fname
        if not fpath.exists():
            print(f"  SKIP (not found): {fpath.name}")
            continue
        print(f"  Loading {sess}...")
        data = sio.loadmat(str(fpath))

        chase_idx = extract_chase_intervals(data, sess, animal)
        if chase_idx is None or len(chase_idx) == 0:
            print(f"    No chase intervals for {animal}/{sess}")
            continue

        ba = data.get('binned_rel_ha',   None)
        bd = data.get('binned_rel_dist', None)
        if ba is None or bd is None:
            print(f"    No bait data in {sess} — skipping")
            continue

        hd  = data['binned_hd'][0,   chase_idx]
        spd = data['binned_speed'][0, chase_idx]
        ba  = ba[0, chase_idx]
        bd  = bd[0, chase_idx]

        # auto-detect metres vs cm
        bd_med = float(np.nanmedian(bd[bd > 0])) if np.any(bd > 0) else 0.0
        if bd_med < 5.0:
            bd = bd * 100.0
            print(f"    Converted dist m->cm (median now {bd_med*100:.1f} cm)")
        else:
            print(f"    Dist already in cm (median {bd_med:.1f} cm)")

        valid = ~np.isnan(hd) & ~np.isnan(ba) & ~np.isnan(bd)
        if speed_thresh > 0:
            valid = valid & (spd > speed_thresh)
        if not np.any(valid):
            print(f"    No valid frames in {sess}")
            continue

        spk_v = data['spikemat'][:, chase_idx][:, valid]

        if n_cells is None:
            n_cells = spk_v.shape[0]
        elif spk_v.shape[0] != n_cells:
            print(f"    Cell count mismatch ({spk_v.shape[0]} vs {n_cells}), skipping {sess}")
            continue

        sessions_data.append({'ba': ba[valid], 'bd': bd[valid], 'spk': spk_v})

    if not sessions_data or n_cells is None:
        print(f"  ERROR: No usable data for {animal}")
        return

    ba_all  = np.concatenate([s['ba']  for s in sessions_data])
    bd_all  = np.concatenate([s['bd']  for s in sessions_data])
    spk_all = np.hstack(      [s['spk'] for s in sessions_data])
    print(f"  Combined: {spk_all.shape[1]} frames, {n_cells} cells")

    # ── 2. Population occupancy ─────────────────────────────────────────
    occ_pop, _, _ = bin_eboc(ba_all, bd_all, np.zeros(spk_all.shape[1]))

    # ── 3. Per-cell ratemaps + peaks ────────────────────────────────────
    rm_sum      = np.zeros((N_DIST, N_THETA), dtype=float)
    rm_count    = np.zeros((N_DIST, N_THETA), dtype=float)
    nspk_pop    = np.zeros((N_DIST, N_THETA), dtype=float)
    peak_angles = np.full(n_cells, np.nan)
    peak_dists  = np.full(n_cells, np.nan)
    n_active    = 0

    for ni in range(n_cells):
        spk = spk_all[ni, :]
        if spk.sum() == 0:
            continue
        n_active += 1
        occ_cell, nspk_cell, rm_s = bin_eboc(ba_all, bd_all, spk)
        nspk_pop += nspk_cell

        # raw (un-normalised) ratemap accumulation:
        # per-cell [0,1] normalisation makes every cell contribute its peak (1.0)
        # somewhere so the population average becomes uniformly high -> flat map
        valid_bins = np.isfinite(rm_s)
        rm_sum[valid_bins]   += rm_s[valid_bins]
        rm_count[valid_bins] += 1

        peak_angles[ni], peak_dists[ni] = pref_orient_dist(rm_s)

    print(f"  Active cells (>=1 spike): {n_active}")
    print(f"  Cells with valid peak:    {int(np.sum(np.isfinite(peak_angles)))}")

    # ── 4. Population ratemap ───────────────────────────────────────────
    min_cells = max(1, int(n_active * 0.2))
    with np.errstate(divide='ignore', invalid='ignore'):
        rm_pop = rm_sum / rm_count
    rm_pop[rm_count < min_cells] = np.nan
    rm_pop[occ_pop  < occ_min]   = np.nan

    valid_v = rm_pop[np.isfinite(rm_pop)]
    rm_vmin = float(np.percentile(valid_v, 5))  if len(valid_v) else 0.0
    rm_vmax = float(np.percentile(valid_v, 98)) if len(valid_v) else 1.0

    nspk_disp = nspk_pop.copy()
    nspk_disp[occ_pop < occ_min] = np.nan

    # ── 5. Load ETC mask from classification CSV ─────────────────────────
    etc_mask = load_etc_mask(csv_dir, animal, n_cells) if csv_dir else None
    n_etc    = int(np.sum(etc_mask)) if etc_mask is not None else 0

    # ── 6. Figure ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))
    fig.patch.set_facecolor('white')

    def safe_vmax(arr, pct=98):
        v = arr[np.isfinite(arr) & (arr > 0)]
        return float(np.percentile(v, pct)) if len(v) else 1.0

    pc1 = _polar_mesh(axes[0], occ_pop,   'Target Occupancy',
                      vmin=0, vmax=safe_vmax(occ_pop), cmap='Blues')
    pc2 = _polar_mesh(axes[1], nspk_disp, 'Spike Occupancy',
                      vmin=0, vmax=safe_vmax(nspk_disp), cmap='Blues')
    pc3 = _polar_mesh(axes[2], rm_pop,    'Rat-to-target Ratemap',
                      vmin=rm_vmin, vmax=rm_vmax, cmap='turbo')
    _peak_scatter(axes[3], peak_angles, peak_dists,
                  n_active=n_active, etc_mask=etc_mask)

    for ax, pc, lbl in zip(axes[:3], [pc1, pc2, pc3],
                            ['Occupancy\n(frames)', 'Spike frames', 'Rate\n(spk/frame)']):
        cb = fig.colorbar(pc, ax=ax, shrink=0.55, pad=0.03, aspect=15)
        cb.set_label(lbl, fontsize=7)
        cb.ax.tick_params(labelsize=6)

    etc_str = f'ETCs: {n_etc}  |  ' if etc_mask is not None else ''
    fig.suptitle(
        f'{animal}  |  Chasing bouts  |  EBOC  |  '
        f'{etc_str}active: {n_active}  |  sessions: {", ".join(chase_sessions)}',
        fontsize=11, fontweight='bold', y=1.01
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"EBOC_summary_{animal}_chase.png"
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_dir / f"EBOC_summary_{animal}_chase.pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓  Saved: {out_png}")


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG — edit these
# ═══════════════════════════════════════════════════════════════════════════

FOLDER_LOC   = "/Users/pearls/Work/RSC_project/"
CHANNELS     = "RSC"
BINSIZE      = 0.00833
SPEED_THRESH = 3.0
OUTPUT_DIR   = "/Users/pearls/PycharmProjects/egostuff/ETC_prime/EBOC_summary"

# Root folder containing  <animal>/EBOC_<animal>_classification_summary.csv
CSV_DIR = "/Users/pearls/Library/Mobile Documents/com~apple~CloudDocs/work/EBOC"

ANIMALS_AND_SESSIONS = {
    "Arwen":         ["c1", "c2", "c4"],
    "ToothMuch":     ["c1", "c2"],
    "PreciousGrape": ["c1", "c2"],
    "MimosaPudica":  ["c1", "c2", "c3"],
    # "Tauriel":     ["c1", "c2", "c4"],
}

if __name__ == "__main__":
    for animal, sessions in ANIMALS_AND_SESSIONS.items():
        make_animal_summary(
            folder_loc     = FOLDER_LOC,
            animal         = animal,
            channels       = CHANNELS,
            chase_sessions = sessions,
            binsize        = BINSIZE,
            speed_thresh   = SPEED_THRESH,
            output_dir     = OUTPUT_DIR,
            csv_dir        = CSV_DIR,
        )
    print("\nAll animals done!")
#!/usr/bin/env python3
"""
ETC Polar HD Tuning: Open – Chase – Open'
==========================================
For each ETC cell, plot polar head-direction tuning curves across conditions
to show that cells with clear allocentric HD tuning in the open field
lose that tuning during the chasing task.

Layout per cell:  OF1  |  Chase  |  OF2  (or OF1 | Chase if no OF2)
Each panel is a polar plot of firing rate vs. allocentric head direction.

Author: Pearl S. / generated for thesis Figure 3 supplement
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from pathlib import Path
from scipy.ndimage import uniform_filter1d
import warnings

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit these to match your paths
# ═══════════════════════════════════════════════════════════════════════

# Base data folder
BASE_PATH = Path("/Users/pearls/Work/RSC_project/Data")

# Output folder
OUTPUT_DIR = Path("/Users/pearls/PycharmProjects/egostuff/ETC_prime/HD_polar_comparison")

# Classification CSV — wide-format file with is_ETC_strict column
# (or set to None and provide ETC_CELL_LISTS manually below)
CLASSIFICATION_CSV = Path("/Users/pearls/PycharmProjects/egocentric/EBC_ETC_EOC_all_animals_WIDE_with_metrics_STRICT.csv")  # e.g. Path("/.../EBC_ETC_EOC_all_animals_WIDE_with_metrics_STRICT.csv")

# Arwen has a separate classification file (uses 'EBOC' label for ETCs)
ARWEN_CSV = Path("/Users/pearls/Work/data/work/EBOC/Arwen/RSC/EBOC_ARWEN_classification_summary.csv")  # e.g. Path("/.../EBOC_ARWEN_classification_summary.csv")

# If CLASSIFICATION_CSV is None, manually list ETC cell indices per animal
# These are 0-based column indices into the spikemat of the .mat file
# Replace with your actual ETC cell indices
ETC_CELL_LISTS = {
    'Arwen':         [],   # fill in: e.g. [3, 7, 15, 22, ...]
    'ToothMuch':     [],
    'PreciousGrape': [],
    'MimosaPudica':  [],
}

# Animal → session configuration
# of_sessions:    list of open-field session names (OF1, OF2, ...)
# chase_sessions: list of chasing session names (c1, c2, ...)
# binsize:        time bin string used in filenames (e.g. '50ms' or '8ms')
ANIMALS = {
    'Arwen': {
        'of_sessions':    ['RSC_OF1_binnedshareddata8ms'],
        'chase_sessions': ['RSC_c1_binnedshareddata8ms',
                           'RSC_c2_binnedshareddata8ms',
                           'RSC_c4_binnedshareddata8ms'],
        'of2_sessions':   ['RSC_OF2_binnedshareddata8ms'],  # second open field if available
    },
    'ToothMuch': {
        'of_sessions':    ['RSC_OF1_binnedshareddata8ms'],
        'chase_sessions': ['RSC_c1_binnedshareddata8ms',
                           'RSC_c2_binnedshareddata8ms'],
        #'of2_sessions':   ['RSC_OF2_binnedshareddata50ms'],
    },
    'PreciousGrape': {
        'of_sessions':    ['RSC_OF1_binnedshareddata8ms'],
        'chase_sessions': ['RSC_c1_binnedshareddata8ms',
                           'RSC_c2_binnedshareddata8ms'],
        'of2_sessions':   ['RSC_OF2_binnedshareddata8ms'],
    },
    'MimosaPudica': {
        'of_sessions':    ['RSC_OF1_binnedshareddata8ms'],
        'chase_sessions': ['RSC_c1_binnedshareddata8ms',
                           'RSC_c2_binnedshareddata8ms',
                           'RSC_c3_binnedshareddata8ms'],
        'of2_sessions':   ['RSC_OF2_binnedshareddata8ms'],
    },
}

# ── Chase interval definitions (same as your main pipeline) ──────────
CHASE_INTERVALS = {
    'Arwen': {
        'c1': [(1610, 7583, 27805, 38704, 53618, 57788)],
        'c2': [(3996, 4946, 13087, 20357, 26277, 35577, 44556, 47376, 49806, 53276, 54831, 61091, 62611,
                65211)],
        'c4': [(2490, 9030, 18678, 23108, 28007, 29057, 34017, 36677, 39773, 43193, 53881, 60621, 73509,
                77469)],
    },
    'ToothMuch': {
        'c1': [(2864, 3142, 3280, 9074, 9434, 10727, 16364, 18090, 18368, 21101, 21499, 22887, 23162, 24313,
                24821, 25109,
                39485, 40367, 40608, 42503, 43410, 44623, 44969, 46337, 51153, 51645, 52704, 54166, 61215,
                61390, 61683,
                62515, 63264, 65663, 66483, 67939, 68358, 69678, 72545, 72648, 85962, 86963, 87208, 87606,
                90513, 91619,
                95734, 97353, 97795, 98850, 99512, 100233, 103172, 104317, 105184, 106051, 106795, 108280,
                124814, 128850,
                129306, 131594, 132160, 134162, 135324, 135824, 136506, 137616, 140332, 142947)],
        'c2': [(3707, 11494, 12030, 15606, 29560, 32998, 33731, 37618, 38104, 38726, 39185, 40827, 47692,
                50182, 56995, 60636,
                62313, 67521, 77232, 77588, 78058, 78553, 80353, 82153, 82581, 85436, 85777, 86744, 87304,
                87825, 88334, 89129,
                89668, 90033, 100036, 100710, 101806, 104542, 104949, 106761, 108295, 109014, 124283, 126088,
                127924, 129349,
                139810, 140402, 141132, 141981, 142657, 143119, 143783, 145040)],
    },
    'PreciousGrape': {
        'c1': [(1328, 5662, 6271, 9598, 19173, 23362, 37066, 50673, 60591, 61942, 81982, 89630, 95864, 97545,
                106412, 109122, 109926, 112703, 118190, 120433, 126228, 136068, 140494, 146517, 147408,
                152977, 160049, 172396)],
        'c2': [(22958, 32685, 37162, 44870, 52398, 59078, 70657, 83937, 88255, 89486, 102991, 108700, 119019,
                120545, 127238, 142105, 145235, 152788, 158909, 172790, 181085, 190973, 191988, 206881,
                212596, 227509, 229707, 231460, 244807, 259357, 275659, 283610, 283965, 289594)],
    },
    'MimosaPudica': {
        'c1': [(6570, 9359, 10630, 11046, 18612, 20167, 20519, 21387, 21564, 23129, 24357, 25379, 25485,
                27196, 62785, 63685,
                65580, 68161, 69099, 70314, 70506, 71418, 72757, 73200, 112741, 113430, 113906, 115142,
                123128, 123779,
                124809, 125532, 137937, 139414, 140320, 140733, 144503, 144972, 157185, 157682, 160558,
                161196,
                175193, 175875, 176651, 177511, 178329, 181987, 183352, 185336, 186056, 189012, 223731,
                224325,
                224532, 225538, 256141, 259516, 295391, 295979, 296389, 297440)],
        'c2': [(55, 1607, 2944, 4945, 5550, 6438, 6842, 7425, 11065, 11928, 13549, 15909, 17554, 20111, 25364,
                26707, 27821, 30559, 31524, 32850, 44364, 44912, 58539, 58914)],
        'c3': [(2081, 2493, 10037, 10755, 12191, 12821)],
    },
}
# NOTE: Replace the placeholder intervals above with your actual chase
#       bout intervals (in seconds). If you use frame-based masking instead,
#       see the alternative approach in extract_chase_mask().

# Analysis parameters
N_HD_BINS = 36           # number of angular bins (10° each)
SPEED_THR = 5.0          # cm/s minimum speed
SMOOTHING_SIGMA = 3      # bins for Gaussian smoothing of tuning curve
BINSIZE_S = 0.008        # time bin in seconds (50 ms)

# Plotting
MAX_CELLS_PER_PAGE = 6   # rows per PDF page
FIGSIZE_PER_ROW = 2.8    # inches per row

# Font
plt.rcParams['font.family']
plt.rcParams['font.size'] = 9
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


# ═══════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_session(animal, session_name):
    """Load a single .mat session file and return dict of arrays."""
    fpath = BASE_PATH / animal / f"{session_name}.mat"
    if not fpath.exists():
        print(f"  ⚠ File not found: {fpath}")
        return None

    data = sio.loadmat(str(fpath), squeeze_me=True, struct_as_record=False)

    # Extract head direction (try multiple field names)
    hd = None
    for field in ['binned_hd', 'binned_md', 'hd', 'md']:
        if field in data:
            hd = np.array(data[field], dtype=float).flatten()
            break
    if hd is None:
        print(f"  ⚠ No HD field found in {session_name}")
        return None

    # Extract speed
    speed = None
    for field in ['binned_speed', 'speed']:
        if field in data:
            speed = np.array(data[field], dtype=float).flatten()
            break

    # Extract spike matrix — (n_cells, n_timebins) or (n_timebins, n_cells)
    spk = None
    for field in ['spikemat', 'spike', 'spikes', 'binned_spikes']:
        if field in data:
            spk = np.array(data[field], dtype=float)
            break
    if spk is None:
        print(f"  ⚠ No spike field found in {session_name}")
        return None

    # Ensure spk is (n_cells, n_timebins)
    n_time = len(hd)
    if spk.shape[0] == n_time and spk.shape[1] != n_time:
        spk = spk.T  # was (timebins, cells) → (cells, timebins)
    elif spk.shape[1] == n_time:
        pass  # already (cells, timebins)
    else:
        # Ambiguous — try matching
        if spk.shape[1] == n_time:
            pass
        elif spk.shape[0] == n_time:
            spk = spk.T
        else:
            print(f"  ⚠ Spike shape {spk.shape} doesn't match n_time={n_time}")
            return None

    return {
        'hd': hd,
        'speed': speed,
        'spikemat': spk,
        'n_cells': spk.shape[0],
        'n_time': spk.shape[1],
    }


def extract_chase_mask(data_dict, animal, session_name):
    """
    Return boolean mask for chasing time bins.

    CHASE_INTERVALS stores flat tuples of alternating (start, end, start, end, ...)
    frame indices at 120 Hz (8.33 ms per frame).

    We convert these to 50 ms bin indices: frame_idx * (8.33ms / 50ms) = frame_idx / 6
    """
    n = data_dict['n_time']
    mask = np.zeros(n, dtype=bool)

    # Parse session tag (e.g. 'RSC_c1_binnedshareddata50ms' → 'c1')
    sess_tag = session_name.split('_')[1]  # 'c1', 'c2', etc.

    intervals_raw = CHASE_INTERVALS.get(animal, {}).get(sess_tag, [])

    if len(intervals_raw) == 0:
        return mask

    # Flatten if nested in a list: [(v1, v2, v3, ...)] → (v1, v2, v3, ...)
    if isinstance(intervals_raw[0], (list, tuple)):
        flat = list(intervals_raw[0])
    else:
        flat = list(intervals_raw)

    if len(flat) % 2 != 0:
        print(f"  ⚠ Odd number of interval values for {animal}/{sess_tag}, truncating last")
        flat = flat[:-1]

    # Convert 120 Hz frame indices → 50 ms bin indices
    FRAME_RATE = 120.0  # Hz
    frame_dt = 1.0 / FRAME_RATE  # seconds per frame
    for i in range(0, len(flat), 2):
        # Convert frame index to 50ms bin index
        i_start = int(flat[i] * frame_dt / BINSIZE_S)
        i_end = min(int(flat[i + 1] * frame_dt / BINSIZE_S), n)
        if i_start < n:
            mask[i_start:i_end] = True

    pct = 100.0 * mask.sum() / n
    print(f"    {sess_tag}: {mask.sum()}/{n} bins = {pct:.1f}% chasing")
    return mask


# ═══════════════════════════════════════════════════════════════════════
#  HD TUNING CURVE
# ═══════════════════════════════════════════════════════════════════════

def compute_hd_tuning(hd, spikes_1d, speed=None, speed_thr=SPEED_THR,
                      n_bins=N_HD_BINS, smooth_sigma=SMOOTHING_SIGMA):
    """
    Compute occupancy-normalized HD tuning curve.

    Parameters
    ----------
    hd : 1-D array, head direction in radians [-pi, pi]
    spikes_1d : 1-D array, spike counts per time bin for one cell
    speed : 1-D array or None
    speed_thr : float, minimum speed in cm/s
    n_bins : int
    smooth_sigma : float, Gaussian smoothing width in bins

    Returns
    -------
    tuning : 1-D array (n_bins,) firing rate in Hz
    bin_centres : 1-D array (n_bins,) in radians
    mvl : float, mean vector length (Rayleigh)
    pref_dir : float, preferred direction in radians
    peak_fr : float, peak firing rate
    """
    # Valid samples
    valid = ~np.isnan(hd) & ~np.isnan(spikes_1d)
    if speed is not None:
        valid &= ~np.isnan(speed) & (speed >= speed_thr)

    hd_v = hd[valid]
    spk_v = spikes_1d[valid]

    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Spike count and occupancy per bin
    spike_counts, _ = np.histogram(hd_v, bins=bin_edges, weights=spk_v)
    occupancy, _    = np.histogram(hd_v, bins=bin_edges)

    # Firing rate (Hz) = (spike_count / binsize) / occupancy_in_bins
    # occupancy is in number of time bins → convert to seconds
    occ_sec = occupancy * BINSIZE_S
    with np.errstate(divide='ignore', invalid='ignore'):
        tuning = np.where(occ_sec > 0, spike_counts / occ_sec, 0.0)

    # Smooth (circular)
    if smooth_sigma > 0:
        # Pad for circularity
        pad = int(3 * smooth_sigma)
        padded = np.concatenate([tuning[-pad:], tuning, tuning[:pad]])
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(padded, sigma=smooth_sigma)
        tuning = smoothed[pad:-pad]

    # Mean vector length
    R = np.sum(tuning * np.exp(1j * bin_centres))
    mvl = np.abs(R) / np.sum(tuning) if np.sum(tuning) > 0 else 0.0
    pref_dir = np.angle(R)
    peak_fr = np.max(tuning)

    return tuning, bin_centres, mvl, pref_dir, peak_fr


# ═══════════════════════════════════════════════════════════════════════
#  CONCATENATE SESSIONS
# ═══════════════════════════════════════════════════════════════════════

def concat_sessions(animal, session_list, chase_filter=False):
    """
    Load and concatenate multiple sessions. Returns combined arrays.

    Parameters
    ----------
    animal : str
    session_list : list of session name strings
    chase_filter : bool, if True apply chase mask

    Returns
    -------
    dict with 'hd', 'speed', 'spikemat' (n_cells × n_time_total) or None
    """
    all_hd, all_speed, all_spk = [], [], []
    n_cells_ref = None

    for sess in session_list:
        d = load_session(animal, sess)
        if d is None:
            continue

        if n_cells_ref is None:
            n_cells_ref = d['n_cells']
        elif d['n_cells'] != n_cells_ref:
            print(f"  ⚠ Cell count mismatch in {sess}: {d['n_cells']} vs {n_cells_ref}")
            continue

        if chase_filter:
            mask = extract_chase_mask(d, animal, sess)
        else:
            mask = np.ones(d['n_time'], dtype=bool)

        # Speed filter is applied later in compute_hd_tuning
        all_hd.append(d['hd'][mask])
        if d['speed'] is not None:
            all_speed.append(d['speed'][mask])
        all_spk.append(d['spikemat'][:, mask])

    if len(all_hd) == 0:
        return None

    result = {
        'hd': np.concatenate(all_hd),
        'spikemat': np.hstack(all_spk),
        'n_cells': n_cells_ref,
    }
    if len(all_speed) == len(all_hd):
        result['speed'] = np.concatenate(all_speed)
    else:
        result['speed'] = None

    return result


# ═══════════════════════════════════════════════════════════════════════
#  LOAD ETC CELL INDICES
# ═══════════════════════════════════════════════════════════════════════

def get_etc_cells(animal):
    """
    Return list of 0-based cell indices that are classified as ETCs.

    Handles two CSV formats:
    - Main wide CSV: columns 'animal', 'is_ETC_strict', 'neuron_idx_ETC'
    - Arwen separate CSV: columns 'classification' (=='EBOC'), 'neuron_idx'

    All indices are 1-based in the CSVs (MATLAB) → converted to 0-based here.
    """
    import pandas as pd

    # Arwen has a separate classification file
    if animal == 'Arwen':
        arwen_csv = ARWEN_CSV
        if arwen_csv is not None and arwen_csv.exists():
            df = pd.read_csv(arwen_csv)
            sub = df[df['classification'] == 'EBOC']
            indices = sorted((sub['neuron_idx'] - 1).astype(int).tolist())
            return indices
        else:
            print(f"  ⚠ Arwen CSV not found. Set ARWEN_CSV path.")
            return ETC_CELL_LISTS.get(animal, [])

    if CLASSIFICATION_CSV is not None:
        df = pd.read_csv(CLASSIFICATION_CSV)
        sub = df[(df['animal'] == animal) & (df['is_ETC_strict'] == True)]
        if len(sub) == 0:
            return []
        # neuron_idx_ETC is 1-based (MATLAB) → subtract 1 for 0-based Python
        indices = sorted((sub['neuron_idx_ETC'] - 1).astype(int).tolist())
        return indices
    else:
        return ETC_CELL_LISTS.get(animal, [])


# ═══════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════════

def polar_tuning_panel(ax, tuning, bin_centres, mvl, peak_fr, title,
                       color='#3274A1', fill_alpha=0.25):
    """Draw one polar HD tuning curve on a polar axis."""
    # Close the curve
    centres_closed = np.append(bin_centres, bin_centres[0])
    tuning_closed = np.append(tuning, tuning[0])

    ax.fill(centres_closed, tuning_closed, alpha=fill_alpha, color=color)
    ax.plot(centres_closed, tuning_closed, color=color, linewidth=1.5)

    # Preferred direction arrow
    pref = bin_centres[np.argmax(tuning)]
    ax.annotate('', xy=(pref, np.max(tuning)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    ax.set_title(f"{title}\nMVL={mvl:.2f}  peak={peak_fr:.1f} Hz",
                 fontsize=8, pad=12)
    ax.set_theta_zero_location('N')   # 0° at top
    ax.set_theta_direction(-1)        # clockwise
    ax.tick_params(labelsize=6)


def plot_cell_row(fig, gs_row, tuning_list, label):
    """
    Plot one row of polar panels for a single cell.

    tuning_list: list of dicts with keys:
        'tuning', 'bin_centres', 'mvl', 'peak_fr', 'title', 'color'
    """
    n_panels = len(tuning_list)
    for j, t in enumerate(tuning_list):
        ax = fig.add_subplot(gs_row[j], projection='polar')
        polar_tuning_panel(ax, t['tuning'], t['bin_centres'],
                           t['mvl'], t['peak_fr'], t['title'],
                           color=t['color'])

    # Cell label on the left
    # Use the first axis to place text
    ax0 = fig.add_subplot(gs_row[0], projection='polar')
    # (label is placed via fig.text instead)


def create_comparison_figure(animal, cell_indices, of_data, chase_data, of2_data=None):
    """
    Create PDF with polar HD tuning: OF → Chase → OF' for each ETC cell.
    """
    has_of2 = of2_data is not None
    n_cols = 3 if has_of2 else 2
    n_cells = len(cell_indices)

    if n_cells == 0:
        print(f"  No ETC cells for {animal}, skipping.")
        return

    # Colors
    col_of  = '#3274A1'   # blue
    col_ch  = '#E1812C'   # orange
    col_of2 = '#2CA02C'   # green

    output_path = OUTPUT_DIR / f"ETC_HD_polar_{animal}.pdf"

    with PdfPages(str(output_path)) as pdf:
        page_idx = 0
        for start in range(0, n_cells, MAX_CELLS_PER_PAGE):
            batch = cell_indices[start : start + MAX_CELLS_PER_PAGE]
            n_rows = len(batch)

            fig_h = FIGSIZE_PER_ROW * n_rows + 1.2
            fig = plt.figure(figsize=(n_cols * 3.2, fig_h))

            gs = GridSpec(n_rows, n_cols, figure=fig,
                          hspace=0.55, wspace=0.30,
                          top=0.92, bottom=0.04, left=0.08, right=0.95)

            col_labels = ['Open Field', 'Chasing']
            col_colors = [col_of, col_ch]
            if has_of2:
                col_labels.append("Open Field'")
                col_colors.append(col_of2)

            fig.suptitle(f'{animal} — ETC allocentric HD tuning across contexts',
                         fontsize=12, fontweight='bold')

            # Column headers
            for c, lab in enumerate(col_labels):
                fig.text(0.08 + (c + 0.5) * 0.87 / n_cols, 0.95, lab,
                         ha='center', va='bottom', fontsize=10,
                         fontweight='bold', color=col_colors[c])

            for row_i, cell_idx in enumerate(batch):

                # ── OF1 tuning ──
                if of_data is not None and cell_idx < of_data['n_cells']:
                    t_of, bc, mvl_of, _, pk_of = compute_hd_tuning(
                        of_data['hd'],
                        of_data['spikemat'][cell_idx],
                        speed=of_data['speed']
                    )
                else:
                    t_of = np.zeros(N_HD_BINS)
                    bc = np.linspace(-np.pi, np.pi, N_HD_BINS, endpoint=False)
                    mvl_of, pk_of = 0, 0

                # ── Chase tuning ──
                if chase_data is not None and cell_idx < chase_data['n_cells']:
                    t_ch, bc, mvl_ch, _, pk_ch = compute_hd_tuning(
                        chase_data['hd'],
                        chase_data['spikemat'][cell_idx],
                        speed=chase_data['speed']
                    )
                else:
                    t_ch = np.zeros(N_HD_BINS)
                    mvl_ch, pk_ch = 0, 0

                # ── OF2 tuning ──
                if has_of2 and of2_data is not None and cell_idx < of2_data['n_cells']:
                    t_of2, bc, mvl_of2, _, pk_of2 = compute_hd_tuning(
                        of2_data['hd'],
                        of2_data['spikemat'][cell_idx],
                        speed=of2_data['speed']
                    )
                else:
                    t_of2 = np.zeros(N_HD_BINS)
                    mvl_of2, pk_of2 = 0, 0

                # Shared radial scale across all panels for this cell
                rmax = max(pk_of, pk_ch, pk_of2 if has_of2 else 0) * 1.15
                if rmax == 0:
                    rmax = 1.0

                # ── Plot OF1 ──
                ax_of = fig.add_subplot(gs[row_i, 0], projection='polar')
                polar_tuning_panel(ax_of, t_of, bc, mvl_of, pk_of,
                                   f"Cell {cell_idx}", color=col_of)
                ax_of.set_rlim(0, rmax)

                # ── Plot Chase ──
                ax_ch = fig.add_subplot(gs[row_i, 1], projection='polar')
                polar_tuning_panel(ax_ch, t_ch, bc, mvl_ch, pk_ch,
                                   f"Cell {cell_idx}", color=col_ch)
                ax_ch.set_rlim(0, rmax)

                # ── Plot OF2 ──
                if has_of2:
                    ax_of2 = fig.add_subplot(gs[row_i, 2], projection='polar')
                    polar_tuning_panel(ax_of2, t_of2, bc, mvl_of2, pk_of2,
                                       f"Cell {cell_idx}", color=col_of2)
                    ax_of2.set_rlim(0, rmax)

                # Cell ID label on the far left
                fig.text(0.02, 1 - (row_i + 0.5) / n_rows * 0.88 - 0.06,
                         f"Cell\n{cell_idx}",
                         ha='center', va='center', fontsize=7,
                         fontweight='bold', rotation=0)

            pdf.savefig(fig, dpi=200)
            plt.close(fig)
            page_idx += 1

    print(f"  ✓ Saved {page_idx} pages → {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════
#  SUMMARY FIGURE: Best examples only
# ═══════════════════════════════════════════════════════════════════════

def select_best_examples(animal, cell_indices, of_data, chase_data, of2_data=None,
                         n_best=6, min_of_mvl=0.15):
    """
    Select cells with the clearest OF HD tuning that drops during chasing.
    Ranks by: MVL_OF - MVL_chase (largest drop).

    Returns list of (cell_idx, mvl_of, mvl_chase, delta) sorted by delta.
    """
    results = []
    for ci in cell_indices:
        if of_data is None or ci >= of_data['n_cells']:
            continue
        if chase_data is None or ci >= chase_data['n_cells']:
            continue

        _, _, mvl_of, _, pk_of = compute_hd_tuning(
            of_data['hd'], of_data['spikemat'][ci], speed=of_data['speed'])
        _, _, mvl_ch, _, _ = compute_hd_tuning(
            chase_data['hd'], chase_data['spikemat'][ci], speed=chase_data['speed'])

        if mvl_of >= min_of_mvl and pk_of > 0.5:  # has real HD tuning in OF
            delta = mvl_of - mvl_ch
            results.append((ci, mvl_of, mvl_ch, delta))

    # Sort by largest drop
    results.sort(key=lambda x: -x[3])
    return results[:n_best]


def create_best_examples_figure(all_results, output_path):
    """
    Create a single summary figure with the best example cells across animals.

    all_results: list of (animal, cell_idx, of_data, chase_data, of2_data,
                          mvl_of, mvl_chase)
    """
    n_cells = len(all_results)
    if n_cells == 0:
        print("No good examples found.")
        return

    has_of2 = any(r[4] is not None for r in all_results)
    n_cols = 3 if has_of2 else 2

    col_of  = '#3274A1'
    col_ch  = '#E1812C'
    col_of2 = '#2CA02C'

    fig_h = FIGSIZE_PER_ROW * n_cells + 1.5
    fig = plt.figure(figsize=(n_cols * 3.2, fig_h))

    gs = GridSpec(n_cells, n_cols, figure=fig,
                  hspace=0.55, wspace=0.30,
                  top=0.92, bottom=0.04, left=0.12, right=0.95)

    fig.suptitle('ETC allocentric HD tuning: Open Field → Chasing',
                 fontsize=13, fontweight='bold')

    col_labels = ['Open Field', 'Chasing']
    col_colors = [col_of, col_ch]
    if has_of2:
        col_labels.append("Open Field'")
        col_colors.append(col_of2)
    for c, lab in enumerate(col_labels):
        fig.text(0.12 + (c + 0.5) * 0.83 / n_cols, 0.95, lab,
                 ha='center', va='bottom', fontsize=10,
                 fontweight='bold', color=col_colors[c])

    for row_i, (animal, ci, of_d, ch_d, of2_d, mvl_of, mvl_ch) in enumerate(all_results):

        t_of, bc, mvl_of_, _, pk_of = compute_hd_tuning(
            of_d['hd'], of_d['spikemat'][ci], speed=of_d['speed'])
        t_ch, bc, mvl_ch_, _, pk_ch = compute_hd_tuning(
            ch_d['hd'], ch_d['spikemat'][ci], speed=ch_d['speed'])

        rmax = max(pk_of, pk_ch) * 1.15
        if of2_d is not None and ci < of2_d['n_cells']:
            t_of2, bc, mvl_of2_, _, pk_of2 = compute_hd_tuning(
                of2_d['hd'], of2_d['spikemat'][ci], speed=of2_d['speed'])
            rmax = max(rmax, pk_of2 * 1.15)
        else:
            t_of2, mvl_of2_, pk_of2 = None, 0, 0
        if rmax == 0:
            rmax = 1.0

        # OF1
        ax = fig.add_subplot(gs[row_i, 0], projection='polar')
        polar_tuning_panel(ax, t_of, bc, mvl_of_, pk_of, '', color=col_of)
        ax.set_rlim(0, rmax)

        # Chase
        ax = fig.add_subplot(gs[row_i, 1], projection='polar')
        polar_tuning_panel(ax, t_ch, bc, mvl_ch_, pk_ch, '', color=col_ch)
        ax.set_rlim(0, rmax)

        # OF2
        if has_of2:
            ax = fig.add_subplot(gs[row_i, 2], projection='polar')
            if t_of2 is not None:
                polar_tuning_panel(ax, t_of2, bc, mvl_of2_, pk_of2, '', color=col_of2)
                ax.set_rlim(0, rmax)
            else:
                ax.set_visible(False)

        # Row label
        fig.text(0.02, 1 - (row_i + 0.5) / n_cells * 0.88 - 0.06,
                 f"{animal}\nCell {ci}\nΔMVL={mvl_of-mvl_ch:.2f}",
                 ha='center', va='center', fontsize=7,
                 fontweight='bold')

    fig.savefig(str(output_path), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✓ Best examples summary → {output_path}")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_best = []  # for summary figure

    for animal, cfg in ANIMALS.items():
        print(f"\n{'='*60}")
        print(f"  {animal}")
        print(f"{'='*60}")

        # Get ETC cell indices
        etc_cells = get_etc_cells(animal)
        if len(etc_cells) == 0:
            print(f"  ⚠ No ETC cells listed for {animal}. "
                  "Fill in ETC_CELL_LISTS or set CLASSIFICATION_CSV.")
            continue
        print(f"  {len(etc_cells)} ETC cells")

        # Load data for each condition
        print(f"  Loading Open Field sessions: {cfg['of_sessions']}")
        of_data = concat_sessions(animal, cfg['of_sessions'], chase_filter=False)

        print(f"  Loading Chase sessions: {cfg['chase_sessions']}")
        chase_data = concat_sessions(animal, cfg['chase_sessions'], chase_filter=True)

        of2_data = None
        if cfg.get('of2_sessions'):
            print(f"  Loading Open Field' sessions: {cfg['of2_sessions']}")
            of2_data = concat_sessions(animal, cfg['of2_sessions'], chase_filter=False)

        if of_data is None or chase_data is None:
            print(f"  ⚠ Could not load data for {animal}")
            continue

        # Full PDF with all ETCs
        create_comparison_figure(animal, etc_cells, of_data, chase_data, of2_data)

        # Collect best examples for summary
        best = select_best_examples(animal, etc_cells, of_data, chase_data, of2_data)
        for ci, mvl_of, mvl_ch, delta in best:
            all_best.append((animal, ci, of_data, chase_data, of2_data, mvl_of, mvl_ch))

    # Summary figure with best examples across all animals
    if all_best:
        # Sort by delta MVL and take top examples
        all_best.sort(key=lambda x: -(x[5] - x[6]))
        top_n = min(8, len(all_best))
        summary_path = OUTPUT_DIR / "ETC_HD_polar_best_examples.pdf"
        create_best_examples_figure(all_best[:top_n], summary_path)

    print(f"\n{'='*60}")
    print(f"  All done! Output in: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
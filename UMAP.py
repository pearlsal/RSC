#!/usr/bin/env python3
import os
import numpy as np
from scipy.io import loadmat
import h5py
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
import scipy.stats as stats

# --- UMAP shim (ADD right after `import umap`) ---
try:
    _ = umap.UMAP  # does it exist?
except AttributeError:
    try:
        import umap.umap_ as _umap_mod

        umap.UMAP = _umap_mod.UMAP  # monkey-patch so the rest of your code works
    except Exception as e:
        raise ImportError(
            "UMAP not available. Install the correct package with: pip install umap-learn"
        ) from e
# -------------------------------------------------


########################################
# 0) Configuration & chase intervals
########################################

folder_loc = "/Users/pearls/Work/RSC_project/"
animal = "ToothMuch"
sessions = ["RSC_OF1", "RSC_c2"]
import matplotlib as mpl

mpl.rcParams["svg.fonttype"] = "none"  # keep text as real fonts

OUT_DIR = os.path.join(folder_loc, "Figures")
os.makedirs(OUT_DIR, exist_ok=True)

binsize = 0.008  # seconds per bin
time_skip = 12  # for downsampling

###this is for ToothMuch Day_2
N_NEURONS = 80
etc_list = np.array([4, 10, 12, 16, 21, 24, 26, 28, 29, 34, 37, 40, 43, 49, 59, 67, 71, 77, 80]) - 1
ebc_list = np.array([3, 9, 14, 21, 24, 25, 26, 29, 37, 38, 39, 45, 47, 49, 51, 61, 62, 66, 70, 71]) - 1
# old list
# #eboc_list = np.array([3, 4, 5, 8, 10, 12, 14, 15, 16, 19, 21, 24, 26, 28, 29, 32, 34, 35, 37, 40, 41, 49, 50, 56, 62, 67, 71, 77, 80])-1
# ebc_list  = np.array([7, 9, 21, 24, 25, 26, 29, 34, 37, 38, 39, 40, 47, 49, 51, 54, 56, 61, 62, 65, 66, 67, 69, 70, 71, 74, 77])-1
non_etc = np.setdiff1d(np.arange(N_NEURONS), np.array(etc_list, dtype=int))
non_ebc = np.setdiff1d(np.arange(N_NEURONS), np.array(ebc_list, dtype=int))

# chase_intervals_c1 = [
#    2864, 3142, 3280, 9074, 9434, 10727, 16364, 18090, 18368, 21101, 21499, 22887, 23162, 24313,
#    24821, 25109, 39485, 40367, 40608, 42503, 43410, 44623, 44969, 46337, 51153, 51645, 52704,
#    54166,61215, 61390, 61683, 62515, 63264, 65663, 66483, 67939, 68358, 69678, 72545, 72648, 85962,
#    86963,87208, 87606, 90513, 91619, 95734, 97353, 97795, 98850, 99512, 100233, 103172, 104317, 105184,
#    106051, 106795, 108280, 124814, 128850, 129306, 131594, 132160, 134162, 135324, 135824,
#    136506,137616, 140332, 142947
# ]


chase_intervals_c1 = [
    3707, 11494, 12030, 15606, 29560, 32998, 33731, 37618, 38104, 38726, 39185, 40827, 47692,
    50182, 56995, 60636, 62313, 67521, 77232, 77588, 78058, 78553, 80353, 82153, 82581, 85436,
    85777, 86744, 87304, 87825, 88334, 89129, 89668, 90033, 100036, 100710, 101806, 104542,
    104949, 106761, 108295, 109014, 124283, 126088, 127924, 129349, 139810, 140402, 141132,
    141981, 142657, 143119, 143783, 145040
]


########################################
# 1) Helper functions
########################################

def construct_file_path(folder, animal, session, binsize):
    ms_str = f"{int(round(binsize * 1000))}ms"
    return os.path.join(folder, f"Data/{animal}/{session}_binnedshareddata{ms_str}.mat")


def safe_row_subset(mat, idx):
    if not isinstance(idx, (list, tuple, np.ndarray)):
        idx = [idx]
    idx = [int(i) for i in idx if 0 <= int(i) < mat.shape[0]]
    if len(idx) == 0:
        print("WARNING: no valid indices for this subset.")
        return np.zeros((0, mat.shape[1]), dtype=mat.dtype)
    return mat[idx, :]


def load_session_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at: {file_path}")
    try:
        with h5py.File(file_path, 'r') as f:
            if "spikemat" in f:
                return np.array(f["spikemat"])
            for key in f:
                arr = np.array(f[key])
                if arr.ndim == 2:
                    return arr
    except OSError:
        pass
    mat_data = loadmat(file_path)
    if "spikemat" in mat_data:
        return mat_data["spikemat"]
    for key in mat_data:
        arr = mat_data[key]
        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            return arr
    raise ValueError(f"No suitable 2D data found in {file_path}")


def expand_intervals_to_indices(interval_pairs):
    all_indices = []
    for (start, end) in interval_pairs:
        all_indices.extend(range(int(start), int(end) + 1))
    return np.unique(all_indices)


def subset_spikemat_by_intervals(spikemat, interval_pairs):
    expanded_idx = expand_intervals_to_indices(interval_pairs)
    valid_idx = [i for i in expanded_idx if 0 <= i < spikemat.shape[1]]
    valid_idx = np.unique(valid_idx)
    return spikemat[:, valid_idx]


def smoothcols(matrix, sigma=3):
    if matrix.ndim != 2 or matrix.size == 0:
        return matrix
    return np.apply_along_axis(lambda x: gaussian_filter1d(x, sigma=sigma), axis=1, arr=matrix)


def flat_to_pairs(flat_list):
    """Convert [start1, end1, start2, end2, ...] to [(start1, end1), (start2, end2), ...]"""
    return [(flat_list[i], flat_list[i + 1]) for i in range(0, len(flat_list) - 1, 2)]


def process_two_spikemats(spikemat1, spikemat2, timeskip=12, num_pca_components=6):
    s1_smooth = spikemat1
    s2_smooth = spikemat2

    combined_data = np.hstack((s1_smooth, s2_smooth))
    whichsession = np.hstack((np.zeros(s1_smooth.shape[1]), np.ones(s2_smooth.shape[1])))

    print('shape of combined data', np.shape(combined_data))

    normed_neurons = np.zeros_like(combined_data, dtype=float)
    for i in range(combined_data.shape[0]):
        dummy = np.sqrt(combined_data[i, :])
        normed_neurons[i, :] = stats.norm.ppf(stats.rankdata(dummy) / (len(dummy) + 1))

    normed_neurons = normed_neurons[:, ::timeskip]
    whichsession = whichsession[::timeskip]

    print('shape of combined data after skipping', np.shape(normed_neurons))

    pca_all = PCA(n_components=num_pca_components)
    pcadneurons = pca_all.fit_transform(normed_neurons.T)  # T x dim
    return pcadneurons, whichsession


def create_umap_plot(pcadneurons, whichsession, title="UMAP", n_components=2, savename="umap.svg"):
    total_points = pcadneurons.shape[0]
    reducer = umap.UMAP(
        metric='euclidean',
        n_neighbors=min(100, max(2, total_points - 1)),
        min_dist=0.8,
        n_components=n_components
    )
    embedding = reducer.fit_transform(pcadneurons)

    fig = plt.figure(figsize=(8, 6))
    color_map = {0: 'blue', 1: 'orange', 2: 'magenta'}
    colors = [color_map[int(s)] for s in whichsession]

    if n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.5, c=colors, s=6, edgecolors='none')
    else:
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], alpha=0.7, c=colors, s=6)

    ax.set_title(title)
    fig.tight_layout()

    svg_path = os.path.join(OUT_DIR, savename)
    fig.savefig(svg_path, bbox_inches="tight", transparent=True, metadata={"Title": title})
    print(f"Saved SVG: {svg_path}")
    plt.show()


# -------------------- NEW: Chill plot --------------------
def plotchillingandwhatevere(smoothedspikes, chillingtimepoints, binsize, title="Chill timeline"):
    """
    Quick visualization of 'chill' vs 'not-chill' over time + population rate.
    - smoothedspikes: neurons x time (already smoothed, e.g., sigma=18)/binsize
    - chillingtimepoints: 1D array of time-bin indices considered chill
    - binsize: seconds per bin
    """
    T = smoothedspikes.shape[1]
    chill_mask = np.zeros(T, dtype=bool)
    chill_mask[np.clip(chillingtimepoints, 0, T - 1)] = True

    # Simple stripe (magenta for chill)
    stripe = np.ones((1, T)) * 0.9
    stripe[0, chill_mask] = 0.2

    # Mean population rate for context
    pop_rate = smoothedspikes.mean(axis=0)
    pop_rate = gaussian_filter1d(pop_rate, sigma=10)

    fig = plt.figure(figsize=(10, 3.6))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.imshow(stripe, aspect='auto', cmap='gray', interpolation='nearest')
    ax1.set_yticks([])
    ax1.set_xlim(0, T - 1)
    ax1.set_title(title + " (magenta=chill)")
    # overlay magenta boxes
    ax1.scatter(np.where(chill_mask)[0], np.zeros(np.sum(chill_mask)), s=1, c='magenta')
    ax1.set_xticks([])

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(pop_rate, lw=1.0)
    ax2.set_ylabel("Mean rate (Hz)")
    ax2.set_xlabel(f"Time bins (bin={binsize * 1000:.0f} ms)")
    ax2.set_xlim(0, T - 1)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------

########################################
# 2) Main logic
########################################

def main():
    # specify how many umap dimensions to use
    n_components = 2

    # 2a) Load spikemats from .mat files
    file_path_OF1 = construct_file_path(folder_loc, animal, "RSC_OF1", binsize)
    file_path_c1 = construct_file_path(folder_loc, animal, "RSC_c1", binsize)

    spikemat_OF1 = load_session_data(file_path_OF1)  # shape: (neurons, time)
    spikemat_c1 = load_session_data(file_path_c1)

    print("Loaded spikemat_OF1:", spikemat_OF1.shape)
    print("Loaded spikemat_c1 :", spikemat_c1.shape)

    # Smooth to firing-rate-like traces (Hz)
    spikemat_OF1 = smoothcols(spikemat_OF1, sigma=18) / binsize
    spikemat_c1 = smoothcols(spikemat_c1, sigma=18) / binsize

    # ---- Chill timepoints (complement of chase) & plot ----
    # Build chill indices for c1 by complementing chase intervals within the session length
    all_idx = np.arange(spikemat_c1.shape[1])
    # chase_idx = expand_intervals_to_indices(chase_intervals_c1)
    chase_idx = expand_intervals_to_indices(flat_to_pairs(chase_intervals_c1))
    chase_idx = chase_idx[(chase_idx >= 0) & (chase_idx < spikemat_c1.shape[1])]
    chill_idx = np.setdiff1d(all_idx, chase_idx)

    # Plot chill overlay/timeline
    plotchillingandwhatevere(smoothedspikes=spikemat_c1,
                             chillingtimepoints=chill_idx,
                             binsize=binsize,
                             title="RSC_c1: Chill vs Not-Chill")

    # 2b) (Optional) Subset with chase intervals for UMAP OF vs Chase
    # If you want UMAP to be strictly OF vs Chase, keep using process_two_spikemats as-is:
    spikemat1 = spikemat_OF1
    spikemat2 = spikemat_c1
    print("Final spikemat1 (OF):", spikemat1.shape)
    print("Final spikemat2 (Chase session full):", spikemat2.shape)

    # A) All cells
    all_pcadneurons, all_whichsession = process_two_spikemats(spikemat1, spikemat2, timeskip=time_skip)
    create_umap_plot(all_pcadneurons, all_whichsession,
                     title="UMAP: All Cells (OF vs. Chase)", n_components=n_components, savename="umap_all.pdf")

    # B) EBC only
    valid_ebc = [idx for idx in ebc_list if idx < spikemat1.shape[0]]
    spikemat1_ebc = spikemat1[valid_ebc, :]
    spikemat2_ebc = spikemat2[valid_ebc, :]
    ebc_pcadneurons, ebc_whichsession = process_two_spikemats(spikemat1_ebc, spikemat2_ebc, timeskip=time_skip)
    create_umap_plot(ebc_pcadneurons, ebc_whichsession,
                     title="UMAP on only EBCs (OF vs. Chase)", n_components=n_components, savename="umap_ebc.pdf")

    # C) EBOC only
    valid_etc = [idx for idx in etc_list if idx < spikemat1.shape[0]]
    spikemat1_etc = spikemat1[valid_etc, :]
    spikemat2_etc = spikemat2[valid_etc, :]
    etc_pcadneurons, etc_whichsession = process_two_spikemats(spikemat1_etc, spikemat2_etc, timeskip=time_skip)
    create_umap_plot(etc_pcadneurons, etc_whichsession,
                     title="UMAP on only ETCs  (OF vs. Chase)", n_components=n_components, savename="umap_etc.pdf")

    # (D) Non-ETC
    sp1_nonetc = safe_row_subset(spikemat1, non_etc)
    sp2_nonetc = safe_row_subset(spikemat2, non_etc)
    nonetc_pcadneurons, nonetc_whichsession = process_two_spikemats(sp1_nonetc, sp2_nonetc, timeskip=time_skip)
    create_umap_plot(nonetc_pcadneurons, nonetc_whichsession,
                     title="UMAP (D): Non-ETC (OF vs. Chase)", n_components=n_components, savename="umap_nonetc.pdf")

    # (E) Non-EBC
    sp1_nonebc = safe_row_subset(spikemat1, non_ebc)
    sp2_nonebc = safe_row_subset(spikemat2, non_ebc)
    nonebc_pcadneurons, nonebc_whichsession = process_two_spikemats(sp1_nonebc, sp2_nonebc, timeskip=time_skip)
    create_umap_plot(nonetc_pcadneurons, nonebc_whichsession,
                     title="UMAP (D): Non-EBC (OF vs. Chase)", n_components=n_components, savename="umap_nonebc.pdf")


if __name__ == "__main__":
    main()

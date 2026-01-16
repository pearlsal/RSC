"""
Merged Chase Session Analysis
=============================
Run chase cell analysis on merged sessions (e.g., c1 + c2) with rebinning to 50ms.

Usage: Just run this script after setting your animal name.
"""

from chase_cell_analysis_merge_sessions import (
    merge_chase_sessions,
    build_session_paths,
    rebin_session,
    run_analysis,
    save_results,
    load_chase_session
)
import numpy as np

# =============================================================================
# CONFIGURATION - EDIT THIS SECTION
# =============================================================================

folder = '/Users/pearls/Work/RSC_project/Data/'
animal = 'MimosaPudica'  # Change this for different animals
binsize = '8ms'       # Source data bin size
target_binsize_ms = 50  # Rebin to 50ms for analysis

# Define which sessions to merge for each animal
animal_session_config = {
    'Arwen': ['c1'],
    'ToothMuch': ['c1'],
    'PreciousGrape': ['c2'],
    'MimosaPudica': ['c1'],
}

# Define chase intervals for EACH session (in 8.33ms bins)
# Format: [start1, end1, start2, end2, ...]
chase_intervals_by_session = {
    'Arwen': {
        'c1': np.array([1610, 7583, 27805, 38704, 53618, 57788]),
        'c2': np.array([3996, 4946, 13087, 20357, 26277, 35577, 44556, 47376,
                        49806, 53276, 54831, 61091, 62611, 65211]),
        'c4': np.array([2490, 9030, 18678, 23108, 28007, 29057, 34017, 36677, 39773, 43193, 53881, 60621, 73509,
                             77469]),  # Add intervals if needed
    },
    'ToothMuch': {
        'c1': np.array([2864, 3142, 3280, 9074, 9434, 10727, 16364, 18090,
                        18368, 21101, 21499, 22887, 23162, 24313, 24821, 25109,
                        39485, 40367, 40608, 42503, 43410, 44623, 44969, 46337,
                        51153, 51645, 52704, 54166, 61215, 61390, 61683, 62515,
                        63264, 65663, 66483, 67939, 68358, 69678, 72545, 72648,
                        85962, 86963, 87208, 87606, 90513, 91619, 95734, 97353,
                        97795, 98850, 99512, 100233, 103172, 104317, 105184, 106051,
                        106795, 108280, 124814, 128850, 129306, 131594, 132160, 134162,
                        135324, 135824, 136506, 137616, 140332, 142947]),
        'c2': np.array([3707, 11494, 12030, 15606, 29560, 32998, 33731, 37618,
                        38104, 38726, 39185, 40827, 47692, 50182, 56995, 60636,
                        62313, 67521, 77232, 77588, 78058, 78553, 80353, 82153,
                        82581, 85436, 85777, 86744, 87304, 87825, 88334, 89129,
                        89668, 90033, 100036, 100710, 101806, 104542, 104949, 106761,
                        108295, 109014, 124283, 126088, 127924, 129349,
                        139810, 140402, 141132, 141981, 142657, 143119, 143783, 145040]),
    },
    'PreciousGrape': {
        'c1': np.array([1328, 5662, 6271, 9598, 19173, 23362, 37066, 50673, 60591, 61942, 81982, 89630, 95864, 97545,
                          106412, 109122, 109926, 112703, 118190, 120433, 126228, 136068, 140494, 146517, 147408,
                          152977, 160049, 172396]),

        'c2': np.array([22958, 32685, 37162, 44870, 52398, 59078, 70657, 83937, 88255, 89486, 102991, 108700, 119019,
                          120545, 127238, 142105, 145235, 152788, 158909, 172790, 181085, 190973, 191988, 206881,
                          212596, 227509, 229707, 231460, 244807, 259357, 275659, 283610, 283965, 289594]),
    },
    'MimosaPudica': {
        'c1': np.array([6570, 9359, 10630, 11046, 18612, 20167, 20519, 21387, 21564, 23129, 24357, 25379, 25485,
                          27196, 62785, 63685,
                          65580, 68161, 69099, 70314, 70506, 71418, 72757, 73200, 112741, 113430, 113906, 115142,
                          123128, 123779,
                          124809, 125532, 137937, 139414, 140320, 140733, 144503, 144972, 157185, 157682, 160558,
                          161196,
                          175193, 175875, 176651, 177511, 178329, 181987, 183352, 185336, 186056, 189012, 223731,
                          224325,
                          224532, 225538, 256141, 259516, 295391, 295979, 296389, 297440]),
        'c2': np.array([55, 1607, 2944, 4945, 5550, 6438, 6842, 7425, 11065, 11928, 13549, 15909, 17554, 20111, 25364,
                          26707, 27821, 30559, 31524, 32850, 44364, 44912, 58539, 58914]),
        'c3': np.array([2081, 2493, 10037, 10755, 12191, 12821]),  # Add intervals if needed
    },
    }

# =============================================================================
# RUN ANALYSIS - NO NEED TO EDIT BELOW
# =============================================================================

if __name__ == "__main__":
    
    print("=" * 60)
    print(f"MERGED CHASE ANALYSIS: {animal}")
    print("=" * 60)
    
    # Get sessions for this animal
    sessions_to_merge = animal_session_config.get(animal, ['c1'])
    
    # Build file paths
    session_files = build_session_paths(folder, animal, sessions_to_merge, binsize=binsize)
    print(f"\nSessions to merge: {sessions_to_merge}")
    print(f"Files:")
    for f in session_files:
        print(f"  {f}")
    
    # Get chase intervals for this animal
    animal_intervals = chase_intervals_by_session.get(animal, {})
    
    # Step 1: Merge sessions (still in 8ms bins)
    print("\n" + "-" * 40)
    print("STEP 1: Merging sessions")
    print("-" * 40)
    merged = merge_chase_sessions(
        session_files=session_files,
        chase_intervals_per_session=animal_intervals,
        verbose=True
    )
    
    print(f"\nMerged session (8ms): {merged['n_cells']} cells, {merged['n_timebins']} bins")
    print(f"Session boundaries: {merged['session_boundaries']}")
    
    # Step 2: Rebin from 8ms to 50ms
    print("\n" + "-" * 40)
    print("STEP 2: Rebinning to 50ms")
    print("-" * 40)
    merged_50ms = rebin_session(
        merged,
        from_binsize_ms=8.33,
        to_binsize_ms=target_binsize_ms,
        verbose=True
    )
    
    print(f"\nRebinned session (50ms): {merged_50ms['n_cells']} cells, {merged_50ms['n_timebins']} bins")
    print(f"Session boundaries (50ms): {merged_50ms['session_boundaries']}")
    
    # Get chase intervals (automatically adjusted during rebinning)
    if 'merged_chase_intervals' in merged_50ms:
        chase_intervals_ready = merged_50ms['merged_chase_intervals']
        print(f"Chase intervals (50ms bins): {len(chase_intervals_ready) // 2} intervals")
    else:
        raise ValueError("No chase intervals found! Check chase_intervals_by_session config.")
    
    # Step 3: Load and rebin open field for null distribution
    print("\n" + "-" * 40)
    print("STEP 3: Loading open field (null distribution)")
    print("-" * 40)
    openfield_8ms = load_chase_session(f"{folder}{animal}/RSC_OF1_binnedshareddata{binsize}.mat")
    openfield = rebin_session(openfield_8ms, from_binsize_ms=8.33, to_binsize_ms=target_binsize_ms, verbose=True)
    print(f"Open field (50ms): {openfield['n_cells']} cells, {openfield['n_timebins']} bins")
    
    # Step 4: Run analysis
    print("\n" + "-" * 40)
    print("STEP 4: Running analysis")
    print("-" * 40)
    results = run_analysis(
        spikemat_chase=merged_50ms['spikemat'],
        spikemat_null=openfield['spikemat'],
        chase_intervals=chase_intervals_ready,
        cell_ids=merged_50ms['cell_ids'],
        num_shuffles=1000,
        alpha=0.025,
        verbose=True
    )
    
    # Step 5: Save results
    print("\n" + "-" * 40)
    print("STEP 5: Saving results")
    print("-" * 40)
    sessions_str = '_'.join(sessions_to_merge)
    saved_files = save_results(
        results=results,
        output_dir=f"{folder}{animal}/analysis_output",
        prefix=f"{animal}_RSC_{sessions_str}_merged_chase_50ms",
        save_csv=True,
        save_plot=True,
        spikemat=merged_50ms['spikemat'],
        title=f"{animal} RSC {sessions_str} merged (50ms bins) - Chase Cell Activity"
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Animal: {animal}")
    print(f"Sessions merged: {sessions_to_merge}")
    print(f"Total bins (50ms): {merged_50ms['n_timebins']}")
    print(f"Total cells: {merged_50ms['n_cells']}")
    print(f"\nExcited cells: {len(results['excited_cell_ids'])}")
    for cell_id in results['excited_cell_ids']:
        print(f"  {cell_id}")
    print(f"\nSuppressed cells: {len(results['suppressed_cell_ids'])}")
    for cell_id in results['suppressed_cell_ids']:
        print(f"  {cell_id}")
    print("=" * 60)

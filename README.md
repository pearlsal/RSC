# Active pursuit gates egocentric coding in Retrosplenial Cortex

[![bioRxiv](https://img.shields.io/badge/bioRxiv-2026.02.14.705560-blue)](https://www.biorxiv.org/content/10.64898/2026.02.14.705560v1)

Analysis code for characterising egocentric spatial coding in the rat retrosplenial cortex (RSC) during free foraging and naturalistic prey capture. This repository accompanies the manuscript:

> **Saldanha P.**, Bjerke M., Dunn B.A., Whitlock J.R. (2026). *Active pursuit gates egocentric coding in Retrosplenial Cortex.* bioRxiv. doi: [10.64898/2026.02.14.705560](https://www.biorxiv.org/content/10.64898/2026.02.14.705560v1)

---

## Overview

Spatial navigation is commonly studied in static environments, but adaptive behaviour frequently hinges on tracking moving goals in real time. Using Neuropixels recordings in the retrosplenial cortex (RSC) of freely moving rats engaged in naturalistic bait-chasing, we identified three functionally distinct egocentric cell populations:

**Egocentric Boundary Cells (EBCs)** encode the direction and distance of environmental boundaries relative to the animal's body axis. Their tuning remains stable across behavioural contexts.

**Egocentric Target Cells (ETCs)** encode the egocentric location of a moving prey item (cricket). These cells are specific to active pursuit and lose allocentric head-direction tuning during chasing.

**Egocentric Object Cells (EOCs)** encode the egocentric bearing of stationary objects.

Boundary-related tuning remains stable across contexts, whereas target-coding cells are specific to chasing and shift tuning dynamically — reducing allocentric head direction while enhancing egocentric features during pursuit. Together, these results show that stable environmental representations coexist with flexible goal-related egocentric coding in RSC.

---

## Repository Structure

| Script | Description |
|---|---|
| `COMPLETE_Classification.py` | **Core classification pipeline.** EBC/EBOC detection with shuffle-based significance testing, sequential classification following Alexander et al. (2020), stability filtering, and MATLAB-matched rate-map computation. Runs across all animals and serves as the imported backbone for several downstream scripts. |
| `organize_EBC_ETC_EOC.py` | Merges per-animal classification CSVs into a unified wide-format table with strict EBC/ETC/EOC flags and per-animal summary counts. |
| `phantom_bait.py` | **Phantom-bait control.** Validates ETC specificity by placing an "egocentric cage" around the animal during open-field sessions (no prey) and testing whether structured tuning persists — it should not. Imports from the core pipeline. |
| `plot_polished_cells.py` | Publication-quality polar ratemap figures for individual EBC/ETC/EOC cells. Supports CLI arguments for specific neuron lists or CSV-based batch plotting. Helvetica font hierarchy for journal figures. |
| `peak_CC_redstar.py` | Cross-correlation analysis of EBC tuning across session halves and conditions. Computes peak CC values with edge exclusion and star-marker placement. |
| `temporal_cc_withStats.py` | Temporal cross-correlation stability for all EBC pairs. Generates multi-page PDFs, per-pair CSV metrics, Wilcoxon tests on peak-lag stability, and tolerance statistics. |
| `etc_hd_OF.py` | ETC head-direction tuning comparison across Open Field → Chase → Open Field conditions |
| `chase_cell_analysis_merge_sessions.py` | Chase cell identification pipeline. Classifies neurons as chase-excited, chase-suppressed, or indifferent via permutation testing of firing rates inside vs. outside chase intervals, with FDR correction. |
| `run_merged_chase_analysis.py` | Driver script for merged chase-session analysis. Configures per-animal session merging and rebinning from 8 ms to 50 ms bins. |
| `speed_vs_chase.py` | Speed flip-flop control analysis. Confirms that chase modulation is not explained by running speed alone by comparing firing rates during chase vs. speed-matched open-field periods. |
| `make_glmDeviance_plots.py` | GLM deviance and pseudo-R² visualisation. Loads per-cell GLM outputs, computes unique nested contributions of EBC vs. ETC models (HD-controlled), and generates winner-coloured scatter plots. |
| `egocentricbayesiandecoder.py` | Egocentric Bayesian decoder framework. Includes an `EBCSimulator` class for generating synthetic EBC populations and decoding wall distance/bearing from population activity. |
| `wall_bait_decoder_complete.py` | Wall vs. bait decoding with circular-shift null distributions. Produces a publication-ready 3×4 panel figure (scatter, time series, null distributions) with p-values and z-scores. |
| `UMAP.py` | UMAP dimensionality reduction of RSC population activity across open-field and chase sessions. Visualises state-space separation between behavioural epochs. |

---

## Animals

Data were collected from rats recorded with Neuropixels probes in RSC:

| Alias |
|---|
| Arwen |
| ToothMuch |
| PreciousGrape |
| MimosaPudica |



---

## Dependencies

The codebase requires Python ≥ 3.8 and the following packages:

**Core:** NumPy, SciPy, Matplotlib, Pandas

**Analysis:** scikit-learn, [umap-learn](https://umap-learn.readthedocs.io/), statsmodels

**Data I/O:** h5py, scipy.io (for MATLAB `.mat` files)

**Visualisation:** Seaborn, matplotlib (with PDF/SVG backend support)

### Setup

```bash
git clone https://github.com/pearlsal/RSC.git
cd RSC
pip install numpy scipy matplotlib pandas scikit-learn umap-learn statsmodels h5py seaborn
```

---

## Data

This repository contains analysis code only. The underlying Neuropixels recordings and processed spike-sorted data are available at <!-- ADD DATA REPOSITORY LINK (e.g. DANDI / Figshare / Zenodo) -->.

Input data are expected as MATLAB `.mat` files containing binned spike matrices (8 ms bins) and behavioural tracking variables (position, head direction, bait position). Paths are configured at the top of each script — update `folder_loc` / `BASE_PATH` to point to your local data directory.

---

## Usage

### Cell-type classification

The core pipeline classifies neurons as EBC, EBOC (ETC), or EOC using shuffle-based significance testing:

```python
from FINAL_COMPLETE_SCRIPT_matlab_rank_threshold import LoaderConfig, run_full_analysis

cfg = LoaderConfig(
    folder_loc="/path/to/data/",
    which_animal="ToothMuch",
    which_channels="RSC",
    ebc_or_eboc="EBC",       # or "EBOC" for target cells
    which_neurons="all",
    n_shuffles=1000
)
results = run_full_analysis(cfg)
```

### Phantom-bait control

```bash
python phantom_bait.py
```

### Chase cell analysis

```bash
# Edit animal name in run_merged_chase_analysis.py, then:
python run_merged_chase_analysis.py
```

### Publication figures

```bash
# Plot specific cells (1-based indices):
python plot_polished_cells.py --mode EBC --neuron-list "5,12,47" \
    --folder-loc /path/to/data --animal Arwen --channels RSC --out-dir ./plots

# ETC head-direction comparison:
python etc_hd_OF.py

# GLM deviance plots:
python make_glmDeviance_plots.py \
    --base-dir "/path/to/glm_outputs/" \
    --out-dir "/path/to/out/" \
    --glob "*chaseOnly.npy" \
    --class-csv "/path/to/EBC_ETC_EOC_all_animals_long_STRICT.csv"
```

---

## Key Implementation Notes

**MATLAB-to-Python parity.** The core classification script was translated from MATLAB and validated for numerical equivalence. Rate maps are computed in dimensionless units (spikes/frame) to match the original implementation. A critical 1-based vs. 0-based indexing bug was identified and corrected during development.

**RNG reproducibility.** A global RNG (`np.random.default_rng(42)`) ensures each neuron receives unique shuffles across runs. Earlier versions used identical shuffles for all neurons, which inflated significance rates.

**Sequential testing.** Classification follows the procedure in Alexander et al. (2020): tuning significance is tested first, followed by stability, to control the family-wise error rate at α = 0.05.

**Speed filtering.** A 5 cm/s minimum speed threshold is applied throughout to exclude stationary periods.

---

## Example Outputs

The scripts generate publication-quality figures including polar egocentric ratemaps for classified cells, cross-correlation stability matrices across session time bins, UMAP state-space embeddings coloured by behavioural epoch, GLM deviance scatter plots showing ego vs. allo variable contributions, head-direction tuning curves across Open Field / Chase / Open Field conditions, phantom-bait control comparisons (real pursuit vs. shuffled open-field), and wall vs. bait Bayesian decoding panels with null distributions.

All colormaps default to `jet` for consistency with manuscript figures.

---

## Citation

If you use this code, please cite:

```bibtex
@article{saldanha2026active,
  title={Active pursuit gates egocentric coding in Retrosplenial Cortex},
  author={Saldanha, Pearl and Bjerke, Martin and Dunn, Benjamin Adric and Whitlock, Jonathan Robert},
  journal={bioRxiv},
  year={2026},
  doi={10.64898/2026.02.14.705560}
}
```

---

## Acknowledgements

This work was performed at the [Kavli Institute for Systems Neuroscience](https://www.ntnu.edu/kavli), Norwegian University of Science and Technology (NTNU), Trondheim, Norway. Funded by Research Council of Norway Investigator grants 300709 and 759033.

---

## Contact

**Pearl Saldanha** — [GitHub](https://github.com/pearlsal)

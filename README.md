# Reproducible Code for "Inferring the Hidden Structure of Scientific Knowledge Flows from Citation Networks"

This folder contains the Python and MATLAB source code used to produce the results in the companion paper. **No input data files are included.** You must generate or obtain the data yourself before running the pipeline.

## Paper
Spelta, A., Pagnottoni, P., & Pecora, N. (2026). Inferring the Hidden Structure of Scientific Knowledge Flows from Citation Networks. 
---

## Repository Contents

### Python scripts (data preparation, model fitting, validation)

| Script | Purpose |
|--------|---------|
| `DataDownload.py` | Downloads filtered OpenAlex bibliometric records (2000–2023) and builds the concept×concept citation tensor |
| `doMatlabvariables.py` | Converts the NumPy tensor and concept list into a MATLAB `.mat` file |
| `mainBO.py` | **Core model.** Bayesian hyperparameter optimization + final inverse unbalanced optimal transport (UOT) fit. Produces the latent cost decomposition C(t) = C⁰ + E⁽ᵗ⁾ |
| `main.py` | Earlier/alternative UOT optimization script (fixed hyperparameters, no Bayesian search) |
| `provaBootstrap.py` | Temporal bootstrap and permutation significance tests for the baseline vs. deviation share |
| `sensitivity.py` | Empirical identifiability validation (temporal mean diagnostic, bootstrap stability, hyperparameter sensitivity) |
| `leiden_cpm_directed.py` | Leiden-CPM community detection on a directed graph with a resolution-parameter sweep |

### MATLAB scripts (results, figures, tables)

| Script | Purpose |
|--------|---------|
| `Analysis.m` | Main analysis pipeline (topological and community analysis of the baseline) |
| `RQ2.m` | RQ1–RQ2: variance decomposition, PCA, centrality, modularity, dynamic topology |
| `RQ2_Et.m` | Deep analysis of the temporal deviation matrices E(t) |
| `RQ4_5.m` | RQ4–RQ5: predictive power of latent geometry; COVID-19 vs. AI shock detection |
| `Real_C0.m` | Reviewer response: compares latent C⁰ baseline against the raw citation matrix (incremental value) |
| `CommunityRobust.m` / `CommunityRobust2.m` | Community-detection robustness: Louvain vs. Leiden-CPM vs. MCL comparison |
| `PlotLoss.m` | Plots the training loss history |

---

## Prerequisites

### Python
- Python 3.9 or later
- Packages: `torch`, `scipy`, `pandas`, `numpy`, `bayes_opt`, `tqdm`, `matplotlib`, `seaborn`

Install with:
```bash
pip install torch scipy pandas numpy bayes-opt tqdm matplotlib seaborn
```

### MATLAB
- MATLAB R2020b or later
- Required toolboxes: Statistics and Machine Learning Toolbox (for `pca`, `corr`, `kmeans`, etc.)
- The scripts reference the **Brain Connectivity Toolbox (BCT)**. Several MATLAB files call `addpath('E:\2019_03_03_BCT')` — you must either:
  - Download BCT and place it at that path, or
  - Edit the `addpath` line in each MATLAB script to point to your local BCT installation.

---

## Recommended Run Order

The pipeline has two branches: Python (model fitting) and MATLAB (analysis/visualization). Run Python first, then MATLAB.

### Step 1 — Download and prepare data (Python)

Run:
```bash
python DataDownload.py
```

**What it does:** Queries the OpenAlex API for ~10,000 journal articles per year from 2000 to 2023 (filtered by document type, at least one reference, cited-by-count > 50). Builds a global concept vocabulary (top 300 most frequent concepts) and constructs a third-order citation tensor P⁽ᵗ⁾ ∈ ℝ^(300×300×24).

**What it produces:**
- `ot_filtered_tensor.npy` — the citation tensor (300 × 300 × 24)
- `concept_list.csv` — the 300 concept names

> ⚠️ This step requires internet access to the OpenAlex API and can take a long time (24 years × API pagination). The output files are large and are **not** included in this repository.

### Step 2 — Prepare MATLAB data (Python)

Run:
```bash
python doMatlabvariables.py
```

**What it does:** Loads `ot_filtered_tensor.npy` and `concept_list.csv`, then saves them as a MATLAB file.

**What it produces:**
- `openalex_tensor.mat` — contains `data_3d`, `Ts`, `concepts`, `normalization_factor`, `year_labels`

### Step 3 — Fit the inverse UOT model (Python)

Run the Bayesian optimization version:
```bash
python mainBO.py
```

**What it does:** Loads the tensor from `ot_filtered_tensor.npy` and `concept_list.csv`. Runs Bayesian optimization over 6 hyperparameters (ε, λ, learning rate, reg_c0, reg_e, reg_temp) using a Gaussian-process surrogate, then performs a final optimization with the best hyperparameters. Recovers C⁰ (time-invariant baseline) and E⁽ᵗ⁾ (year-specific deviations) for each year 2000–2023.

**What it produces (in `results_bayesopt/`):**
- `C0_baseline.csv`, `E_YEAR.csv`, `C_YEAR.csv` (24 files each) — cost matrices indexed by concept names
- `results.npz` — all arrays (C0, E, C_full, concepts, years)
- `loss_history.csv` / `loss_history.json` — loss at every iteration
- `concept_outflow.csv`, `concept_inflow.csv`, `concept_volatility.csv`, `top_concept_pairs.csv` — concept-level flow analysis
- `best_hyperparameters.txt`, `bo_logs.json` — optimization record

Alternatively, run `python main.py` for a fixed-hyperparameter version (outputs to `results/`).

### Step 4 — Significance testing (Python)

Run:
```bash
python provaBootstrap.py
```

**What it does:** Runs temporal bootstrap (resampling years with replacement) and temporal permutation (shuffling year labels) tests to assess whether the baseline share of cost magnitude is statistically meaningful.

**What it produces:** Bootstrap/permutation distributions, confidence intervals, and a comparison figure (`bootstrap_permutation.png`).

### Step 5 — Identifiability validation (Python)

Run:
```bash
python sensitivity.py
```

**What it does:** Validates Theorems 3 and 5 from the paper: temporal mean diagnostic (||mean(Eₜ)||_F / ||C⁰||_F), bootstrap stability of C⁰, and sensitivity to hyperparameters (REG_C0, REG_E, REG_TEMP).

**What it produces (in `identifiability_validation/`):** JSON diagnostics, `.npz` arrays, heatmaps, and a LaTeX summary table.

### Step 6 — MATLAB analysis and figures

Open and run each MATLAB script in MATLAB. They read from the Python output folder (`results_bayesopt/` by default) and write figures/tables to their own output folders.

| Script | Reads from | Writes to |
|--------|-----------|-----------|
| `Analysis.m` | `results_bayesopt/` | `matlab_analysis/` |
| `RQ2.m` | `results_bayesopt/` | `rq2_analysis_output/` |
| `RQ2_Et.m` | `results_bayesopt/` | `rq2_analysis_output/figures_Et` + `tables_Et` |
| `RQ4_5.m` | `results_bayesopt/` | `rq3_rq4_analysis_output/` |
| `Real_C0.m` | `results_bayesopt/` + `openalex_tensor.mat` | `matlab_analysis/` |
| `CommunityRobust.m` | `results_bayesopt/` | `reviewer_robustness_corrected/` |
| `CommunityRobust2.m` | `results_bayesopt/` | `reviewer_leiden_corrected/` |
| `PlotLoss.m` | `results_bayesopt/` | loss plot |

> ⚠️ Before running MATLAB scripts, edit any `addpath('E:\2019_03_03_BCT')` lines to point to your local Brain Connectivity Toolbox installation. Also verify that relative paths (e.g., `'results_bayesopt'`) match your working directory.

---

## What the Code Produces (Paper Results)

Running the full pipeline reproduces the following paper findings:

1. **Latent cost decomposition** C(t) = C⁰ + E⁽ᵗ⁾ where C⁰ (persistent baseline) accounts for ~93% of total cost magnitude.
2. **Five macro-communities** in the baseline: biomedical (C1), computational-behavioral (C2), physical-engineering (C3), socio-environmental (C4), and astrophysics (C5).
3. **Three dominant modes of structural change** via PCA: computational consolidation, materials-science realignment, and the COVID-19 shock.
4. **Predictive validity:** changes in latent transport costs anticipate subsequent citation growth.
5. **Shock detection:** COVID-19 (2020) and AI/LLM boom (2023) appear as structural anomalies in E⁽ᵗ⁾.

---

## Portability Notes

- **Hardcoded paths:** Several MATLAB scripts contain absolute Windows paths (e.g., `addpath('E:\2019_03_03_BCT')` and output folder names). Edit these before running on a different machine or operating system.
- **GPU vs CPU:** Python scripts auto-detect CUDA (`DEVICE = "cuda" if torch.cuda.is_available() else "cpu"`). They run on CPU if no GPU is present, but more slowly.
- **Working directory:** Python scripts expect input files (`ot_filtered_tensor.npy`, `concept_list.csv`) in the current working directory. MATLAB scripts expect `results_bayesopt/` relative to the MATLAB working directory.
- **Random seeds:** Bayesian optimization and community detection use fixed seeds (`random_state=42`, `rng(42)`) for reproducibility.

---

## Data Availability

The raw OpenAlex data is downloaded programmatically via `DataDownload.py` (requires internet access and an OpenAlex API email). The processed tensor `ot_filtered_tensor.npy` (~17 MB) and the full raw tensor `openalex_tensor.mat` (~155 MB) are **not** included in this repository due to size constraints. Regenerate them by running `DataDownload.py` followed by `doMatlabvariables.py`.

---

## Citation

If you use this code, please cite:

> Spelta, A., Pagnottoni, P., & Pecora, N. (2026). Inferring the Hidden Structure of Scientific Knowledge Flows from Citation Networks.

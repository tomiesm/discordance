# Morphology-Transcriptome Discordance in Spatial Transcriptomics

Code for the paper: **"Morphology-transcriptome discordance identifies biologically distinct cell states in spatial transcriptomics of breast and colorectal cancer"** (Nature Computational Science, 2026).

This repository implements a complete pipeline for quantifying discordance between histological morphology and gene expression in spatial transcriptomics data. Foundation model embeddings (UNI2-h, Virchow2, H-Optimus-0) are used to predict gene expression from H&E image patches via LOPO cross-validation. Spots where predictions systematically fail — morphology-transcriptome discordant spots — are shown to represent biologically meaningful cell states.

> **Paper:** [DOI placeholder](https://doi.org/10.xxxx/xxxxx)

## Data Availability

This study uses **18 IDC Xenium sections** from the [HEST dataset](https://huggingface.co/datasets/MahmoodLab/hest) (v1.3.0):

| Cohort | HEST Sample IDs | Patients | Genes |
|--------|----------------|----------|-------|
| Discovery (Biomarkers) | TENX191–TENX202 (11 sections) | 4 | 280 |
| Validation (10x Public + Janesick) | TENX95, TENX97–TENX99, NCBI783–NCBI785 | 4 | 280 |

COAD generalization uses 4 additional sections: TENX111, TENX147, TENX148, TENX149 (351-gene Xenium Human Colon panel).

Foundation models require gated access:
- [UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) (MahmoodLab)
- [Virchow2](https://huggingface.co/paige-ai/Virchow2) (Paige AI)
- [H-Optimus-0](https://huggingface.co/bioptimus/H-optimus-0) (Bioptimus)

## Installation

### Option 1: Conda (recommended)

```bash
conda env create -f environment.yaml
conda activate discordance
```

### Option 2: pip

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch 2.1+ with CUDA 12.1, 4 GPUs recommended (tested on 4x NVIDIA A6000).

## Repository Structure

```
├── config.yaml              # Pipeline configuration (cohorts, models, thresholds)
├── src/                     # Source modules
│   ├── data.py              # Data loading (HEST h5ad + patches)
│   ├── embeddings.py        # Foundation model encoders
│   ├── regressors.py        # Ridge, XGBoost, MLP regressors
│   ├── discordance.py       # D_raw, D_cond, rank discordance
│   ├── spatial.py           # Spatial weights, Moran's I, boundary rings
│   ├── pathways.py          # GMT parsing, pathway enrichment (Mann-Whitney U)
│   ├── de_analysis.py       # Quartile DE (Wilcoxon), meta-analysis
│   ├── matching.py          # Morphology-matched DE (k-NN in embedding space)
│   ├── deconvolution.py     # Cell type marker scoring, nuclear morphometry
│   ├── gene_annotations.py  # UniProt localization, GO annotations, gene features
│   ├── plotting.py          # Figure styling and data loading helpers
│   └── utils.py             # Logging, seeding, GPU utilities
├── scripts/                 # Pipeline scripts (run in order)
│   ├── 00_download.py       # Download HEST data
│   ├── 01_qc_and_splits.py  # QC, log-transform, LOPO splits
│   ├── 02_extract_embeddings.py  # Extract foundation model embeddings
│   ├── 03_train_predict.py  # Train regressors, LOPO-CV predictions
│   ├── 04_discordance_scores.py  # Compute D_raw, D_cond
│   ├── 05_multimodel_agreement.py  # Gate 1: multi-model Spearman
│   ├── 06_spatial_structure.py     # Gate 2: Moran's I
│   ├── 07_dual_track.py           # Gate 3: split-half gene-set stability
│   ├── 08_de_analysis.py          # Unmatched + matched DE
│   ├── 09_pathway_enrichment.py   # Hallmark pathway enrichment
│   ├── 10_deconvolution.py        # Cell type deconvolution
│   ├── 11_gene_predictability.py  # Per-gene predictability OLS model
│   ├── 12_within_patient.py       # Within-patient reproducibility
│   ├── 13_heldout_validation.py   # Held-out cohort validation
│   ├── 14_encoder_consistency.py  # Cross-encoder consistency
│   ├── 15_bridge_gene_replication.py  # Bridge gene cross-cohort replication
│   ├── 16_summary_report.py       # Generate summary report
│   ├── 17_compute_figure_data.py  # Precompute figure data
│   ├── 17b_compute_reproducibility.py  # Reproducibility analyses
│   ├── 18_main_figures.py         # Main figures (Fig 1-6)
│   ├── 19_supplementary_figures.py  # Supplementary figures
│   ├── 20_coad_generalization.py  # COAD generalization analysis
│   ├── 20_tables.py               # Supplementary tables
│   ├── 21_idc_visium_generalization.py  # IDC Visium generalization
│   ├── 22_interior_only_de.py     # Interior-only sensitivity analysis
│   └── 23_figure1_schematic.py    # Figure 1 schematic generation
└── data/
    └── gene_sets/
        └── h.all.v2024.1.Hs.symbols.gmt  # MSigDB Hallmark gene sets
```

## Reproducing the Analysis

Run scripts sequentially from the repository root:

```bash
# 1. Download data from HEST
python scripts/00_download.py

# 2. QC, log-transform, generate LOPO splits
python scripts/01_qc_and_splits.py

# 3. Extract foundation model embeddings (requires GPU)
python scripts/02_extract_embeddings.py

# 4. Train regressors and generate predictions (LOPO-CV)
python scripts/03_train_predict.py

# 5. Compute discordance scores
python scripts/04_discordance_scores.py

# 6-7. Validation gates
python scripts/05_multimodel_agreement.py
python scripts/06_spatial_structure.py
python scripts/07_dual_track.py

# 8-11. Biological characterization
python scripts/08_de_analysis.py
python scripts/09_pathway_enrichment.py
python scripts/10_deconvolution.py
python scripts/11_gene_predictability.py

# 12-15. Reproducibility and robustness
python scripts/12_within_patient.py
python scripts/13_heldout_validation.py
python scripts/14_encoder_consistency.py
python scripts/15_bridge_gene_replication.py

# 16. Summary report
python scripts/16_summary_report.py

# 17-19. Figures and tables
python scripts/17_compute_figure_data.py
python scripts/17b_compute_reproducibility.py
python scripts/18_main_figures.py
python scripts/19_supplementary_figures.py
python scripts/20_tables.py

# 20-22. Generalization and sensitivity analyses
python scripts/20_coad_generalization.py
python scripts/21_idc_visium_generalization.py
python scripts/22_interior_only_de.py

# 23. Figure 1 schematic (requires outputs from step 18)
python scripts/23_figure1_schematic.py
```

All outputs are written to `./outputs/` (configurable in `config.yaml`).

Most scripts accept `--help` for additional options (e.g., `--cohort`, `--encoder`, `--gpu`).

## Configuration

Edit `config.yaml` to modify:
- **Cohort definitions**: Sample IDs, patient mappings, gene panels
- **Encoder settings**: Model IDs, embedding dimensions, batch sizes
- **Regressor hyperparameters**: Ridge alpha, XGBoost/MLP settings
- **Decision thresholds**: Spearman cutoffs, FDR thresholds, spatial parameters
- **Hardware**: Number of GPUs, workers, memory settings

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@article{discordance2026,
  title={Morphology-transcriptome discordance identifies biologically distinct
         cell states in spatial transcriptomics of breast and colorectal cancer},
  author={[Authors]},
  journal={Nature Computational Science},
  year={2026},
  doi={10.xxxx/xxxxx}
}
```

# Cell-resolved EMT feasibility experiment

User-authorized continuation of `EMT_NEXT_ANALYSIS_PROPOSAL.md`. All new code,
downloads, intermediate matrices, reports and plots stay in this directory.
Original code/manuscripts and the correction pipeline are read-only inputs.

The primary task is to establish whether the measured Xenium panel can support
an EMT-associated malignant-cell interpretation, using verified source cell
labels, measured transcripts and a donor-held-out external RNA reference.
The analysis keeps tumor identity, EMT-associated expression, spatially coherent
neighborhoods and enrichment of prediction residuals as separate questions.

`protocol.json` records choices before EMT outcomes. Annotation schemas and
gene/transcript availability were inspected to establish feasibility. These
cohorts were previously examined for the paper and other experiments; this is
exploratory, not confirmatory. A failed feasibility gate is an informative
outcome; no valid EMT zones are promised.

Primary references: Janesick et al. 2023 (GSE243280), Wu et al. 2021 (GSE176078),
and the unchanged MSigDB Hallmark GMT already used in the paper. Downloads have
source URLs and checksums. Vendor ZIPs are accessed through standard HTTP ranges
to retrieve the needed count matrices, cell tables and boundaries only.

The three sections represent two patients. Candidate TF coexpression maps are
descriptive and are not tumor-cell EMT diagnoses. The published "Transitional
Cells" category remains a separate source label.

## Completed result

Read [RESULTS.md](RESULTS.md). The fixed analysis found 2,753 measured
TF/epithelial coexpression candidates in source-tumor cells and 50 descriptive
100 µm neighborhood components. Neither marker specificity nor reference
transfer establishes validated EMT zones. Corrected aggregate discordance has
near-zero adjusted associations with candidate prevalence. The raw-count,
barcode, coordinate and independent numerical checks passed.

- [Overview](figures/overview.png)
- [All sections: identity, measured markers and H&E context](figures/all_section_evidence.pdf)
- [Every candidate component: transcript and boundary evidence](figures/all_candidate_transcript_evidence.pdf)
- [Independent verification](results/VERIFICATION.json)
- [Exact broader-Visium source match](results/VISIUM_COMPATIBILITY.json)

## Reproduction

The run used `/home/tmk-gpu/anaconda3/envs/torch/bin/python` (Python 3.12).
Exact package versions are recorded in `run_environment.json`. No GPU training
is needed. Run from this directory. Source URLs and SHA256 values are in
`sources/Janesick_downloads.json`, `sources/Wu_download.json` and
`sources/vendor/{sample}/download_manifest.json`. The source files and extracted
Wu matrix are retained locally. `fetch_vendor.py` retrieves only the selected
members of the large public vendor archives; it is not needed for an existing
download. The local original HEST files and corrected prediction outputs remain
read-only inputs; their project-relative paths are explicit in the scripts.

```bash
EMT_PYTHON=/home/tmk-gpu/anaconda3/envs/torch/bin/python
"$EMT_PYTHON" -B prepare_reference.py
"$EMT_PYTHON" -B prepare_cells.py
"$EMT_PYTHON" -B reference_observability.py
"$EMT_PYTHON" -B analyze_cells.py
"$EMT_PYTHON" -B residual_and_visium.py
"$EMT_PYTHON" -B plot_evidence.py
"$EMT_PYTHON" -B -m unittest discover -s tests -v
"$EMT_PYTHON" -B verify_results.py
"$EMT_PYTHON" -B finalize_experiment.py
```

The current results are already complete; these commands rerun analysis and
may overwrite outputs and snapshot timestamps. For changed hypotheses, input
versions or thresholds, create a new experiment directory instead. The
existing-data shortcuts in preparation scripts are not a general cache
invalidation mechanism. `frozen/`, `reference_frozen/` and
`cell_analysis_frozen/` retain staged code provenance; `final_source/` records
the final implementation. The result report identifies post-primary
diagnostics and implementation corrections separately from the fixed choices.

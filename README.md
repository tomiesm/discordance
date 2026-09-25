# Prediction residuals in spatial transcriptomics reveal microenvironmental heterogeneity in cancer

**Author: Tomas Iesmantas**

This repository contains the code for the revised manuscript. Histology embeddings
from UNI2-h, Virchow2 and H-Optimus-0 predict spatial gene expression. Conditional
discordance ranks locations by the magnitude of prediction error relative to
locations with similar total expression. Q1 and Q4 identify relatively well and
poorly predicted locations within each section. Gene, program and cell analyses
then characterize their biological associations.

## Revision status

The ridge model now fits an intercept using training data only. PCA centers image
features, so fitting ridge without an intercept omitted the expression baseline.
The correction changes predictions and residuals even when Pearson correlations
remain similar. Regression tests cover constant expression, translation of the
expression scale and exclusion of validation targets from the training baseline.

The revision also corrects discovery donor identities. The 11 discovery sections
represent **11 distinct donors**, divided among four prediction holdout groups.
Historical P03–P06 configuration keys identify these holdouts (B01–B04), not four
patients with repeated sections. The folds remain unchanged and exclude test
donors from training. Final biological summaries use the corrected donor registry.

The final analyses retain the generic magnitude score and Q1/Q4 comparison.
Residual patterns have heterogeneous biological associations; discordance does
not by itself identify EMT. The revision includes a patient-specific EMT residual
association and a separate statistical test of spatial coexpression.

The executed revision analyses are in [paper_revision/](paper_revision/README.md).
They include donor aggregation, gene and program exclusion, overlap adjustment,
spatial inference, boundary sensitivity, external applications and coexpression
tests. Earlier pipeline scripts remain available for intermediate computations;
**running scripts 00–23 alone does not reproduce the final revised inference**.
In particular, old discovery repeat-section comparisons and spot-level significance
summaries must not be substituted for the final donor and spatial analyses.

## Data

The analysis uses public spatial transcriptomics data from
[HEST](https://huggingface.co/datasets/MahmoodLab/hest), version 1.3.0.

| Cohort | Sections | Biological groups | Genes |
| --- | --- | --- | --- |
| Discovery, 10x Biomarkers | TENX191–TENX193 and TENX195–TENX202 | 11 distinct donors; four prediction holdouts | 280 |
| Validation, 10x Public and Janesick | TENX95, TENX97–TENX99, NCBI783–NCBI785 | Four donor groups; seven sections | 280 |

The discovery source includes IDC and DCIS. See the
[specimen registry](paper_revision/methodology_audit/stage_15_donor_correction/tables/specimen_registry.csv)
and [fold registry](paper_revision/methodology_audit/stage_15_donor_correction/tables/fold_registry.csv)
for the exact section, donor and holdout assignments.
COAD uses TENX111, TENX147, TENX148 and TENX149. The Visium application and its
independence restrictions are documented in the revision analyses.

Raw images, expression data, embeddings, prediction arrays and large intermediate
results are not included. Revision manifests record the input sources and hashes.

## Installation

```bash
conda env create -f environment.yaml
conda activate discordance
```

Alternatively, install `requirements.txt` in a compatible Python environment.
The original environment files describe the base pipeline. Exact versions used
for the revision are recorded in
[environment_versions.json](paper_revision/focused_revision/build/environment_versions.json)
and the experiment environment files. The recorded revision environment uses
Python 3.12; the original environment is not an exact revision environment lock.

Embedding extraction requires suitable GPU hardware and access to the gated
[UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h),
[Virchow2](https://huggingface.co/paige-ai/Virchow2) and
[H-Optimus-0](https://huggingface.co/bioptimus/H-optimus-0) models.

## Code layout and execution

- `src/`: data loading, embeddings, corrected regressors, scores and analysis helpers.
- `scripts/`: the original pipeline with the executed computational corrections.
- `tests/`: ridge regression tests.
- `paper_revision/clean_repo/`: executed revision pipeline snapshot and refit scripts.
- `paper_revision/methodology_audit/`: successive analysis stages; Stage 15 supplies final donor aggregation.
- `paper_revision/experiments/`: cell coexpression and spatial statistical comparisons.
- `paper_revision/focused_revision/build/`: figure generation from corrected results.
- `paper_revision/minimal_text_revision/build/`: final figure presentation adjustments.

For the base pipeline, edit `config.yaml` for local data paths and hardware, then
run from the repository root:

```bash
python scripts/00_download.py
python scripts/01_qc_and_splits.py
python scripts/02_extract_embeddings.py
python scripts/03_train_predict.py
python scripts/04_discordance_scores.py
```

Subsequent numbered scripts provide historical intermediate analyses. Follow the
[revision guide](paper_revision/README.md) for the revised analysis sequence and
required inputs. The archived scripts preserve the executed workspace layout;
they require intermediate data and are not a standalone reproduction from this
checkout. Plotting scripts also require the original figure assets. No complete
fresh-machine rerun is claimed.

## Tests

These tests use synthetic data and do not require the large analysis inputs:

```bash
python -B -m unittest discover -s tests -v
python -B -m unittest discover -s paper_revision/experiments/emt_cells_v1/tests -v
python -B -m unittest discover -s paper_revision/experiments/emt_spatial_enrichment_v1/tests -v
```

The revision directories also contain the executed independent verification
scripts and their recorded results. Full verification requires their recorded
analysis inputs.

## License

MIT License. Copyright (c) 2026 Tomas Iesmantas. See [LICENSE](LICENSE).

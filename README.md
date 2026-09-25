# Prediction residuals in spatial transcriptomics reveal microenvironmental heterogeneity in cancer

**Author: Tomas Iesmantas**

Code for the current manuscript. Histology embeddings predict spatial gene
expression; conditional discordance compares absolute prediction errors among
locations with similar total expression. Q1 and Q4 identify relatively well and
poorly predicted locations within each section. Subsequent analyses examine
gene expression, programs, cell composition and spatial coexpression.

## Layout

- `src/`: prediction models, score definitions and shared analysis functions.
- `src/paper/`: the current paper's statistical analyses and figure code.
- `scripts/`: data download, model fitting and ordered analysis commands.
- `data/`: the Hallmark gene definitions and discovery specimen registry.
- `tests/`: tests of ridge fitting, cell phenotype definitions and spatial statistics.

Generated data, numerical results and figures go into `data/hest/`, `data/v3/`
and `outputs/`. They are excluded from version control.

## Data and grouping

The analysis uses public [HEST](https://huggingface.co/datasets/MahmoodLab/hest)
data. Discovery comprises 11 breast sections from 11 distinct donors, with four
prediction holdout groups. Validation comprises seven sections from four donor
groups. COAD uses four sections; the Visium application includes a paired group
that is excluded together during fitting.

Historical P03–P06 keys in `config.yaml` define discovery prediction holdouts,
not biological patients. `data/discovery_specimens.csv` supplies the corrected
donor assignments. Final summaries give equal weight to biological donors;
discovery sections do not provide within-patient repeat comparisons.

The ridge model includes an intercept fitted using training data only.
Q1/Q4 and the generic absolute-error definition are retained. Biological
associations are context dependent; discordance alone does not label EMT.

## Environments

The original model environment is defined by `environment.yaml`:

```bash
conda env create -f environment.yaml
```

The final statistical analyses used Python 3.12 and the versions listed in
`requirements-analysis.txt`. Install these in a separate analysis environment:

```bash
conda create -n discordance-analysis python=3.12 pip
conda activate discordance-analysis
pip install -r requirements-analysis.txt
```

Use the model environment's Python executable for `--model-python` below.
Keeping it separate preserves scikit-learn 1.4.0 for the fitted model pipeline
and the joint Visium fit. The analysis environment uses scikit-learn 1.6.1.
Embedding extraction requires GPU hardware and access to the gated UNI2-h,
Virchow2 and H-Optimus-0 models.

## Reproduce the current results

Run from the repository root. The default data layout is `data/hest/` and
`data/v3/`; outputs are written under `outputs/`. First inspect the sequence:

```bash
python scripts/24_current_paper.py --list
```

Then run it with the model environment's interpreter:

```bash
python scripts/24_current_paper.py --model-python /path/to/discordance/bin/python
```

The phases can also be run separately, in order: `download`, `models`,
`analysis`, `cells`, `final`. For example, `--phase final` reruns final donor
summaries and table export after the preceding phases have completed.
The existing numerical definitions and random seeds are retained. The cutoff
sensitivity uses 20%, 25% and 30% tails.

Final donor summaries are saved in `outputs/paper/donors/tables/`; numbered
supplementary tables are exported to `outputs/paper/Tables/`. The spatial
coexpression analysis is saved under `outputs/cells/` and remains separate
from the primary Q1/Q4 residual comparison.

Generate the result panels after the numerical analyses:

```bash
python scripts/25_paper_figures.py
```

These are saved under `outputs/paper_figures/`. Public HEST images, expression
matrices and source cell annotations are required for the corresponding
analyses. Full fitting and permutation runs are computationally intensive.

## Tests

```bash
python -B -m unittest discover -s tests -v
```

## License

MIT License. Copyright (c) 2026 Tomas Iesmantas. See [LICENSE](LICENSE).

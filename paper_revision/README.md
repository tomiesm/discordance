# Code for the manuscript revision

This directory contains the executed analysis source and selected supporting
records. Scientific source was copied from the verified revision archives.
[source_manifest.json](source_manifest.json) records each copied file and its
SHA256 hash. Historical reports describe the stage at which they were written;
the final donor correction and manuscript conclusions take precedence.

## Analysis sequence

1. `clean_repo/src/regressors.py` restores the ridge intercept. The
   `clean_repo/scripts/revision_01_*` scripts refit and compare the archived
   predictions, then propagate the initial correction through the original
   pipeline. Their paths assume the original project layout.
2. `methodology_audit/stage_01_*` through `stage_09_external_replication` examine
   provenance, prediction quality, conditional scores, gene partition reliability,
   biological contrasts, matching and spatial inference, individual genes, cell
   composition and external applications.
3. `stage_10_decile_sensitivity/` contains the historical cutoff analysis. Only
   the 20%, 25% and 30% tails enter the revised paper; the directory name does not
   mean that decile results are retained.
4. Stages 11 and 12 examine numerical gene features and physical boundaries.
   Stage 13 supplies preparation code used by later figures. Stage 14 supplies
   boundary biology, source-preserving profile comparisons and the tumor-rich
   EMT residual analysis.
5. `stage_15_donor_correction/correct.py` supplies the final discovery donor map
   and recomputes donor summaries, section correlations, numerical feature fits
   and cohort comparisons. `verify.py` independently checks these calculations.
   Its discovery interpretation supersedes the earlier grouping. The included
   `tables/specimen_registry.csv` and `tables/fold_registry.csv` give final
   identities and confirm the donor separation in prediction folds.
6. `experiments/emt_cells_v1/` contains the cell coexpression analysis.
   `experiments/emt_spatial_enrichment_v1/` contains the conditional spatial
   comparison, sensitivities and statistical tests. The manuscript uses the
   primary spatial pair-count result; other exploratory localization results
   remain documented in the source reports. Three sections represent two
   patients and are not three independent patients.
7. `focused_revision/build/` supplies the main and supplementary figure code.
   `minimal_text_revision/build/refresh_figure_displays.py` makes the final
   presentation adjustments without changing the numerical analyses.

## Inputs and paths

The original workspace placed `data/`, `outputs_v3/`, `clean_repo/` and
`paper_revision/` beside one another. This directory preserves the revision
subtree. Reusing these scripts requires the source files, intermediate outputs
and caches recorded in the stage manifests. The root checkout is the public
pipeline; it is not the entire original workspace. Set `hest_dir` in the
archived `clean_repo/config.yaml` to your local HEST directory before use.
Local paths in historical JSON manifests identify the executed inputs; they
are provenance records, not downloads bundled with the repository.

The revision scripts include deliberate overwrite guards and preserve earlier
analysis stages. Read the stage protocol before rerunning it. The figure scripts
also depend on manuscript figure assets and fixed numerical tables that are
not part of this code publication. Manuscript sources, reviewer correspondence,
large arrays and raw data are not included here.

The final supplementary CSV tables accompany the revised manuscript. This code
publication includes the compact coexpression statistics and the final specimen
and fold registries, but not every intermediate numerical table.

## Checks for this publication

The root README gives three synthetic test commands. The publication check also
parses every Python file, verifies copied source hashes, confirms unchanged
configuration values and validates the donor/fold registries. Results are saved
in `publication_checks.json`. These are checks of the published code and
registries, not a repeat of the full model fitting or biological analyses.

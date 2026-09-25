# EMT-associated coexpression: spatial enrichment comparison

Read [RESULTS.md](RESULTS.md). The requested statistical comparison is complete
once `EXPERIMENT_COMPLETE.json` is present. This directory is isolated from the
original code/manuscript and the completed cell experiment.

- [Observed versus null overview](figures/spatial_enrichment_overview.pdf)
- [Exploratory connected-region extent maps](figures/component_extent_maps.pdf)
- [Region coordinates and evidence](results/component_extent_with_coordinates.csv)
- [All initial statistics](results/statistics.csv)

Run with `/home/tmk-gpu/anaconda3/envs/torch/bin/python -B` from this directory.
The computational order is `spatial_test.py`, `verify_stats.py`,
`component_extent.py`, `verify_extent.py`, `plot_results.py`, `finalize.py`.
Statistical unit tests use `-m unittest discover -s tests -v`.
All results are already saved. The extent script intentionally refuses to
overwrite its frozen specification; changed analyses should use a new directory.

`protocol.json` precedes the permutation outcomes. `extent_addendum.json` was
added after the initial tests and before computing the region-extent null.
These timing distinctions and all negative/sensitivity results are explicit
in the report. Null reference intervals are not biological confidence intervals.
Only two patients are represented by the three sections.

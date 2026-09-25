# Conditional-score reliability and spatial structure

2026-09-20. Attribution: COMPUTATION. Fixed B1 inputs; no model/score replacement
or biological program selection. All 180 reproduction checks pass, with maximum
discrepancy 1.67e-16. The initial edge-sum verification mixed float32 and float64
arithmetic; the preserved run log records that failed numerical assertion. The
completed verification uses float64 values for both sparse-matrix and explicit
edge-sum formulas, and retains the original saved-Moran comparison.

## The actual conditional score is reliable across gene halves

The original seed-42 raw-gene-half results reproduce for all nine model
configurations in all 18 sections. That historical test did not validate the
conditional score directly. The new analysis uses 20 fixed gene partitions,
averages absolute errors across three ridge encoders, and conditions each gene
half on expression from its own genes. Median correlations remain high.

| Score/source | Discovery half-gene rho | Validation half-gene rho | Discovery Q4 overlap | Validation Q4 overlap |
|---|---:|---:|---:|---:|
| Ridge mean, raw | 0.906 | 0.796 | 79.2% | 73.1% |
| Ridge mean, conditional on own half | 0.862 | 0.807 | 75.8% | 74.0% |
| Ridge mean, conditional on shared full panel | 0.868 | 0.862 | 76.3% | 76.0% |
| Constant median baseline, raw | 0.967 | 0.894 | 87.2% | 79.4% |
| Constant median baseline, conditional on own half | 0.657 | 0.764 | 62.2% | 73.3% |

Numbers are medians over partitions within each section, then medians over
sections. Partitions and sections are not counted as independent patients. Q4
overlap is the fraction of one half's Q4 retained by the other's Q4. These new
ridge-only, repeated-partition summaries have a different scope from the
original gate across nine configurations and should not silently replace its
numbers without relabeling.

The own-half conditional result supports a broad score rather than one driven
by the exact gene subset. Shared expression conditioning can increase apparent
agreement, especially in validation, so the own-half result is the appropriate
primary version of this sensitivity. Correlated genes, shared expression amount,
and technical effects remain common to both halves. The substantial reliability
of a constant-prediction baseline demonstrates why reliability alone does not
establish morphology-specific or EMT-specific biology.

## Spatial structure survives a local-graph sensitivity

| Quantity | Discovery median Moran's I, k=6 | Validation median Moran's I, k=6 | Discovery, 150 µm radius | Validation, 150 µm radius |
|---|---:|---:|---:|---:|
| Current conditional score | 0.497 | 0.587 | 0.486 | 0.585 |
| Raw error | 0.575 | 0.600 | 0.558 | 0.597 |
| Constant-baseline conditional error | 0.345 | 0.427 | 0.337 | 0.413 |
| Within-section-centering diagnostic | 0.442 | 0.595 | 0.431 | 0.587 |
| Sum log1p expression | 0.711 | 0.789 | 0.693 | 0.779 |
| Detected gene count | 0.730 | 0.706 | 0.718 | 0.697 |

All 18 current-score Moran values exactly reproduce the existing result. The
radius graph includes cardinal/diagonal neighbors on the 100-micrometre grid
without forcing a fixed number of neighbors across larger gaps. The observed
structure remains under this sensitivity. The original k=6 graphs have median
edge length 100 micrometres, maximum lengths 224–500 micrometres, and 0.52–1.77%
of edges longer than 150 micrometres. TENX201 has two connected components under
both graphs; TENX200 has one under k=6 and two under the radius graph. No radius
graph has isolated points. These differences do not explain away the observed
spatial autocorrelation.

Baseline error, expression amount and detection are spatially structured as
well. The current score's spatial structure is a retained result, while its
cause requires biological characterization and technical/composition controls.
Moran values are not evidence for anatomical correspondence between different
serial sections.

## Limit the interpretation of the original permutation null

The original “geometry-preserving” permutation groups spots by distance to the
**convex hull of spot coordinates**, then shuffles scores within those quantile
rings. It tests exchangeability within those rings. It does not preserve actual
tissue holes, local score autocorrelation, expression depth, or compartments.
Its small p-values should not be described as proving independence from all
tissue geometry or technical structure. No new point-shuffling p-values are
used here to strengthen that claim. The already-completed k=4/6/8 analysis was
not redundantly rerun.

## Checkpoint B interpretation

Retain broad conditional-score reliability and spatial organization, with their
specific scopes. Retain the Stage 3 qualification that pooled centering does not
ensure within-section expression independence. No replacement metric is selected
from these findings. The next biological analyses must distinguish full-score
description from tests excluding each tested program's genes from both grouping
and its conditioning covariate. They must also show unadjusted and expression/
detection-adjusted associations, patient-level heterogeneity, and spatial
uncertainty before assigning a biological interpretation.

Outputs: [half-gene results](gene_half_stability.csv),
[fixed gene partitions](gene_partitions.json),
[spatial comparisons](spatial_structure.csv),
[graph diagnostics](graph_diagnostics.csv), and [checks](checks.json).

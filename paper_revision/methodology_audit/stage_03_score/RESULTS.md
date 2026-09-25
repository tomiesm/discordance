# What the current conditional score measures

2026-09-20. Attribution: COMPUTATION. Current B1 predictions, score and Q1/Q4
groups are unchanged. No biological program was used in these diagnostics.

## Definition and verification

The actual score is the mean absolute prediction error across genes and three
ridge encoders, minus the corresponding mean error in a pooled cohort expression
decile. The conditioning variable is **sum of log1p(count) across modeled genes**,
not total counts, log1p(total counts), or library-normalized transcription.
The bin means use all held-out predictions in a cohort. Thus this is an empirical
error-magnitude deviation from a cohort reference, not a signed biological
residual or an independently trained calibration of a new patient's error.

All six encoder/cohort reconstructions agree with the stored conditional scores.
Fifty-five independent aggregation/rank checks also pass. Pooled bin means are
approximately zero, as intended. This verifies the implementation of the stated
centering operation; it does not make the score independent of expression depth.

## Expression dependence remains within sections

Across discovery sections, Spearman correlation with the conditioning variable
ranges from **−0.748 to +0.517** after correction; in validation it ranges from
**−0.352 to +0.070**. Median correlations are −0.109 and −0.190, respectively.
The corresponding raw-error medians are +0.466 and +0.083. The correction often
reduces positive dependence, but can introduce or strengthen negative dependence
because sections have different error–expression relationships within pooled bins.

Examples illustrate why a pooled mean or one cohort correlation is insufficient:

- TENX195: raw correlation −0.243, conditional −0.748. Q1/Q4 median panel counts
  are 19,250.5/1,369, with a count standardized difference of −2.56.
- TENX191: raw +0.813, conditional +0.517. Q1/Q4 median counts are
  4,208.5/36,176, with standardized difference +2.78.
- NCBI785: raw +0.083, conditional −0.289. Q1/Q4 median counts are
  11,123/5,490. This section belongs to P07, whose prediction calibration was
  already flagged in Stage 2.

Within-section expression-decile Q4 fractions can be very uneven. For TENX191
they range from 2.0% to 91.9%, and for TENX195 from 0.16% to 70.1%, despite Q4
containing approximately 25% of each whole section. These profiles can reflect
mean and variance differences; centering alone does not standardize variance.

Measured count amount can carry cellularity and biological information as well
as technical effects. These observations do **not** establish that every
count-associated signal is an artifact. They do establish that “depth-corrected”
must not imply complete within-section depth independence, and that biological
contrasts need explicit count/detection and composition sensitivities.

## Ranking sensitivity and constant-prediction comparison

The table gives descriptive medians across sections, not independent-patient
effect estimates or significance tests. Overlap is the fraction of current Q4
members retained by the comparison's Q4.

| Comparison with current conditional score | Discovery rank rho | Validation rank rho | Discovery Q4 overlap | Validation Q4 overlap |
|---|---:|---:|---:|---:|
| Raw mean absolute error | 0.818 | 0.883 | 73.6% | 81.0% |
| Same centering performed within each section | 0.898 | 0.870 | 84.6% | 81.1% |
| Conditionally centered training-median baseline error | 0.314 | 0.397 | 44.2% | 45.3% |

Changing the calibration population is consequential for some sections: current
versus within-section-centered Q4 overlap ranges from 58.4% to 91.8% in discovery
and 72.3% to 89.2% in validation. Q1 can be even more sensitive (minimum 37.4%
discovery overlap). This diagnostic estimates section baselines from the section's
own outcomes. It is not independently calibrated validation and is not selected
as the new primary score here.

The constant predictor lacks spatially varying histology input but shares a
substantial fraction of Q4 locations with the histology-based score. It can
reflect transcriptomic heterogeneity and spatial tissue structure as well as
technical variation. Shared structure is not proof of artifact, and differences
are not proof of biological specificity. Keep this comparator alongside the
Stage 2 evidence that histology improves prediction for seven of eight patients.

Exact 20/80, 25/75, and 30/70 thresholds, ties, group sizes and recovery fractions
are recorded in [cutoffs_and_tail_sensitivities.csv](cutoffs_and_tail_sensitivities.csv).
All lower/upper groups are disjoint. No cutoff was chosen using pathway results.

## Error is distributed broadly across genes

The largest single-gene contribution to section-wide mean absolute error is at
most 2.36%. The top ten genes account for 7.2–15.4% across all sections; median
shares are 8.4% in discovery and 10.9% in validation. Effective contributor counts
(inverse sum of squared contribution shares) range from 174.5 to 255.7 of 280
genes. Error magnitude is therefore not dominated by a tiny gene subset.

Between 199 and 278 genes per section have higher mean absolute error in Q4 than
Q1; the top ten positive contributions account for at most 19.2% of their total.
This supports broad error differences. Because those genes also help define the
groups, it is descriptive evidence and does not independently validate enrichment
of any program. All gene contributions are retained without selecting favorites.

## Decision and next dependency

The current score is computationally valid for its pooled-cohort definition.
Its full within-section depth-independence interpretation is unsupported. Retain
B1 for traceability, and keep raw error and within-section centering as explicit
diagnostics. A replacement primary score has not been chosen. Next check the
actual conditional half-gene reliability and spatial structure against the same
constant baseline, then settle the score/technical sensitivity definitions before
broad biological inference. Do not optimize the definition for EMT.

Results and figures: [correlations](score_covariate_correlations.csv),
[ranking comparisons](score_comparisons.csv),
[pooled-bin composition](pooled_bin_composition.csv),
[quartile covariates](quartile_covariate_contrasts.csv),
[gene concentration](gene_concentration.csv),
[overview figure](figures/score_diagnostics.pdf),
[section expression profiles](figures/section_expression_profiles.pdf), and
[independent checks](independent_checks.json).

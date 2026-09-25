# Selection, overlap, uncertainty and matching audit

2026-09-20. Attribution: COMPUTATION. Complements the broad biological atlas;
no original model, score, DE result or manuscript file was changed.

## Direct self-inclusion and technical overlap

Every tested program is excluded from the diagnostic grouping error and its
expression-conditioning covariate. The primary technical comparison then uses
five-by-five bins of outside-program total counts and detection within sections,
retaining strata with at least ten Q1 and ten Q4 spots and weighting their
differences by harmonic overlap. Ten-by-ten strata are a predefined sensitivity.

This reduces direct constituent-gene selection and avoids extrapolating across
nonoverlapping count/detection strata. It does not remove correlated genes,
all composition or morphology, or unmeasured confounding. Count/detection
adjustment changes the estimand and can remove biology as well as technical
variation. The adjusted result therefore accompanies the unadjusted result.

Overlap matters: across primary section–program comparisons, median retained
fractions are 98.1% of Q1 and 83.2% of Q4, but minima are 37.1% and 12.1%.
An adjusted effect with low coverage describes the overlap population, not all
of Q4. Ten-by-ten strata can retain still fewer locations; all fractions are
saved rather than choosing the most favorable balance scheme.

## Spatial intervals are conditional and scale-sensitive

The fixed-group block bootstrap uses 500 draws and the predeclared 800- and
400-micrometre scales. All reported primary/sensitivity comparisons have 500
valid draws. Sections contain 48–344 occupied blocks at 800 micrometres and
164–1,332 at 400 micrometres. Predictions, score calibration, quartile thresholds
and technical strata remain fixed; these intervals do not include full-pipeline
re-estimation uncertainty.

The estimator agrees with an independent explicitly weighted spot/stratum
calculation to 2.08e-16. In 200 simulated datasets with independent resampled
blocks and a known 0.5 effect, nominal 95% intervals cover 94.5%. In an explicitly
adverse simulation with correlation 0.85 extending **between** resampled blocks,
coverage falls to 63.5%. The latter is a demonstrated limitation, not a passing
validation. It prevents interpreting these intervals as universally calibrated
for the tissues' unknown dependence range. The simple simulation does not verify
every aspect of the adaptive multi-stratum estimator either.

Use these intervals as conditional spatial sensitivities with their block-scale
assumptions, alongside independent-patient evidence. A stronger significance claim
would require further dependence-range/full-pipeline assessment and a justified
null; none is supplied by merely shuffling individual spot labels.

## Patients remain the biological replication units

Section effects are averaged equally within patients; four patients are then
weighted equally within each cohort. The outputs show patient values, approximate
t intervals with three degrees of freedom, and leave-one-patient-out means.
The t intervals rely on a small-sample distributional approximation. Exact
two-sided sign tests cannot attain p<0.125 with four nonzero patient effects;
their family-adjusted values therefore do not establish significance at 0.05.
This limitation is transparent, not a reason to replace them with inflated
spot-level significance. It also is not evidence that the measured effects are
zero. Effect size, scope, heterogeneity and independent supporting evidence must
carry the interpretation.

Program exclusion is an independent-support **diagnostic**, not a guarantee of
statistical independence. In particular, the uniformly positive absolute-error
contrasts demonstrate cross-gene error structure, while correlated factors can
make that structure shared among many programs. No program-specific biological
null has been declared rejected on this evidence alone.

## Original morphology matching has limited and uneven coverage

The original UNI matching calipers and retained counts reproduce for all 18
sections. k=1/5/10 sensitivities were computed without replacing the original
matches. At the original k=5:

- Median Q4 retention is 70.2%, ranging from 5.6% to 93.7%.
  TENX191 retains just 69 of 1,238 Q4 locations; TENX195 retains 27.4%.
- The rule accepts a Q4 location when its **nearest** Q1 neighbor meets the
  caliper, then uses all five neighbors. A median 25.8% of retained Q4 locations
  have at least one farther neighbor beyond that caliper (range 6.1–81.2%).
  This reproduces the code; it must not be described as every matched pair meeting
  the distance threshold.
- A Q1 control can be reused up to 467 times. Median maximum reuse per section
  is 149. Kish effective control counts based on reuse weights range 45.8–492.9.
  Neither these effective counts nor matched rows create independent patients.
- Distances are Euclidean on unit-normalized embeddings, with cosine distance
  equal to half the squared reported distance. Ordering is consistent, but the
  numerical distance label must be corrected.

Matching generally improves measured balance but does not guarantee it. Median
absolute standardized count imbalance falls from 0.301 to 0.169; detection
imbalance from 0.573 to 0.325; sum-log-expression imbalance from 0.508 to 0.426.
Results conditional on the retained matching subset cannot be generalized to
all Q4 locations without showing overlap and selection. Reused controls and
spatial dependence also invalidate an automatic assumption of independent
matched-delta rows for the original signed-rank p-values.

A further source discrepancy: the matched meta-DE caller substitutes mean
log-expression difference into the field called `log2fc`, while unmatched DE
uses the log2 ratio of mean log-expression. A shared threshold of 0.25 therefore
does not impose the same biological effect requirement in the two analyses.
Revised bridge/matched claims need explicitly defined effects and patient/spatial
inference, rather than simply retaining their original significance counts.

## Consequence for the revision

Retain the broad descriptive atlas and its patient-specific associations.
Demote claims that pooled centering, morphology matching or original spot-level
p-values alone establish technical independence or population-level pathway
activation. Do not automatically discard every association because its controls
change the estimate. Full-pipeline uncertainty, a calibrated technical/selection
null if required for a stronger claim, and revised external validation remain
open. No manuscript wording has been edited.

Outputs: [section effects and spatial intervals](section_inference.csv),
[patient effects](patient_effects.csv), [cohort summaries](cohort_patient_summary.csv),
[bootstrap blocks](bootstrap_blocks.csv), [estimator verification](estimator_verification.json),
[matching diagnostics](matching_diagnostics.csv),
[matching balance](matching_covariate_balance.csv), and [matching checks](matching_checks.json).

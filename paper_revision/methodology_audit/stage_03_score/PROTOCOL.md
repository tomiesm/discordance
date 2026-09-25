# Current magnitude score: technical and ranking diagnostics

2026-09-20. Written before this stage's computations. Builds on the completed
IDC prediction-quality audit. No biological programs or cell phenotypes are
tested here. No new primary score is selected by this protocol.

1. Reconstruct the current score from the archived raw scores and exact saved
   conditioning covariate. The covariate is sum_g log1p(count_ig), not total
   counts or log1p(total counts). Reconstruct each encoder separately and then
   average, preserving the original float32 raw-error bin-mean arithmetic.
   Independently check bin edges, mean corrections, and saved score identity.
2. Report raw and conditional score associations with panel counts, total
   non-control counts, detected genes, the actual conditioning covariate, and
   tissue-mask fraction, within each section. Include section-specific
   expression-decile mean/SD and Q4 frequency, because a rank correlation alone
   cannot detect nonmonotonic dependence. These are descriptive associations,
   not causal attribution or a declaration that count dependence is artifact.
3. Report the composition of each pooled calibration bin by patient, and its
   within-patient/section mean conditional error. Zero pooled bin means do not
   guarantee equivalent calibration within patients, equal variance, or balanced
   group covariates. Describe Q4-Q1 expression/detection/tissue differences.
4. Compare within-section raw vs current conditional ranks and tail membership.
   Retain fixed 20/80 and 30/70 tail sensitivities alongside 25/75 thresholds.
   Record exact cutoffs, ties, sizes, and overlaps. As a diagnostic of the
   calibration population only, apply the same ten-bin subtraction separately
   within each section. It uses section outcomes and is not an independently
   trained calibration or a proposed primary replacement.
5. Use the existing training-only median-expression baseline error as a negative
   comparator. Apply identical pooled conditional centering to that baseline
   error, then compare score ranks and Q1/Q4 membership. Similarity would show
   structure shared with a predictor that lacks spatially varying histology
   input; it would not prove all shared structure is technical. Keep predictive
   utility from Stage 2 separate from score-specificity comparisons.
6. Quantify per-gene contributions to mean absolute error and to the Q4-Q1
   absolute-error difference. Retain every gene, panel coverage, concentration
   (top 1/10 shares, effective number of contributing genes), and patient/section
   distribution. This checks dominance without choosing a biology-weighted score.
7. Preserve all outputs and independently verify selected quantities. Review
   results before deciding whether a calibration change is justified. Original
   data, models, B1 scores, groups, and manuscript remain unchanged.

Do not interpret outcome-conditioned score centering as library normalization,
nor a negative conditional score as transcriptional downregulation. The score
is signed relative to expected error magnitude, not signed prediction residual.
All comparisons are retrospective; this protocol is not historical preregistration.

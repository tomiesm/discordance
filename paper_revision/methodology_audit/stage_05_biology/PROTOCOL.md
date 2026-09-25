# Broad biological characterization: fixed definitions

2026-09-20. Written after the technical/reliability checkpoint and before this
stage's new program contrasts. This is a retrospective revision, not a claim of
historical preregistration or untouched validation. No EMT-specific grouping is
introduced. Original inputs and B1 outputs remain unchanged.

## Groups and biological quantities

Keep current B1 conditional Q1/Q4 as the primary **description** of the paper's
groups, accurately called error relative to pooled expression-bin means. Its
within-section depth dependence is known. It is not being redefined as a fully
normalized or independently calibrated score. Preserve raw-error and section-
centered sensitivities chosen before these program results.

For independent-support diagnostics, exclude the tested Hallmark's entire
measured gene set from both mean absolute error and the sum-log-expression
conditioning covariate. Recompute pooled ten-bin centering and section Q1/Q4.
Call this “program-excluded” grouping; correlated genes/shared technical effects
mean it is not proof of complete statistical independence. For common-panel
outcomes, still exclude the entire measured Hallmark, not just the common members.

Use all 50 Hallmarks in the existing v2024.1 collection, requiring at least five
measured members. Report every coverage result, including untestable pathways.
For each eligible program report three separate unweighted gene means:

1. Observed log1p expression: what transcription is present.
2. Signed residual, averaged over encoders: under/overprediction of that expression.
3. Absolute residual, averaged over encoders: prediction-error magnitude.

These share log1p(count) units; they do not all measure pathway activation. Report
raw-unit Q4-Q1 differences and pooled-within-group standardized effects. Show all
four quartiles and continuous-score correlations, preserving Q1 biology. Also
describe each program's prediction error versus its training-median baseline in
Q1 and Q4. Do not infer that all Q4 programs are unlearnable by other models.

The original expression-binned SD(abs residual) scaling remains a separately
labeled legacy sensitivity. Audit weights, sparse/zero-variance bins and gene
concentration; it is not a calibrated signed residual standard error. Primary
raw-unit summaries avoid selecting a new weighting scheme to improve biology.

Produce panel-specific and shared-gene outcomes separately; shared-gene programs
also require five genes. Do not pool different member sets into a single
identically measured cross-panel outcome. Save complete gene-level Q1/Q4 tables,
including observed log-expression differences, detection, actual count-scale
log2 ratios with a stated one-count pseudocount, signed and absolute errors.
Keep the original ratio-of-mean-log-expression quantity explicitly labeled.

## Sensitivities fixed before reading the atlas

- Current full-score description; program-excluded conditional grouping;
  program-excluded raw-error grouping; program-excluded section-centered grouping.
- Primary 25/75 tails; 20/80 and 30/70 sensitivities for program-excluded
  conditional grouping. Retain continuous correlations and all quartiles.
- Technical adjustment as specified in the companion Stage 6 protocol, using
  counts and detection from genes outside the tested program.
- Patient-specific effects, equal-section averages within patients, equal-patient
  cohort summaries, leave-one-patient-out influence. Sections do not add donors.
- Common-gene outcomes are a separate replication sensitivity. Do not select the
  best-performing member set, adjustment, tail fraction, or encoder as primary.

Candidate follow-up will use the complete atlas: report strongest stable positive
and negative associations in both groups, not only EMT. “Stable” means consistent
effect direction under program exclusion and technical adjustment, with patient
heterogeneity and interval uncertainty shown; it is not a substitute for those
uncertainties or a significance label. No new claim is promoted until the
selection/spatial/patient checks are evaluated.

## Exploratory follow-up after the first atlas

The atlas shows differing signed directions across patients/panels and only ten
shared measured EMT-Hallmark genes. Describe the contribution of source-annotated
cell groups to all eligible validation Hallmarks (and shared-gene subsets) in
the existing, count-verified NCBI783–785 cell matrices. Retain the existing QC;
report cell numbers, count fractions, per-cell expression, detection and nuclear
counts. This checks what the measured gene sets represent in these three sections
from two patients; it does not infer a transition, create new Q1/Q4 groups, or
claim independent replication. It is explicitly prompted by this atlas.

For the complete set of programs testable in both panels, describe the variance
of the patient-by-program effect matrix attributable to patient mean, program
mean, and their remainder. This is a balanced additive descriptive decomposition,
without ANOVA p-values or causal attribution. It is an overview of heterogeneity,
not selection of a new biological endpoint.

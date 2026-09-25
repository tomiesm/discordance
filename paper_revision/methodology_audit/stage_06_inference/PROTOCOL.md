# Selection, technical overlap, and uncertainty for the broad atlas

2026-09-20. Fixed before reading new Stage 5 program contrasts. This protocol
separates description, dependence sensitivity, and population uncertainty.

## Technical overlap comparison

For each program-excluded conditional Q1/Q4 comparison, form five quantile bins
of total counts outside the tested program and five quantile bins of detected
genes outside that program, within each section. Cross these fixed bins, merging
duplicate edges implicitly. Retain strata with at least ten Q1 and ten Q4 spots.
Estimate the stratum Q4-Q1 difference, then average with harmonic-overlap weights
2*n1*n4/(n1+n4). Report retained fractions, strata and effective weights.

This estimates an association among comparable measured-count/detection strata.
It can remove variation related to cellularity or biology as well as technical
variation; report it alongside the unadjusted association. It does not control
all cell composition, morphology or unmeasured confounding, and is not causal.
Use 10-by-10 strata as a predeclared sensitivity. Do not extrapolate into strata
with no group overlap. If no valid strata exist, report non-estimability.

## Spatial and patient uncertainty

For the primary program-excluded conditional 25/75 comparison, use square
800-micrometre spatial blocks (fixed origin at each section's minimum x/y) and
500 bootstrap draws of occupied blocks, seed 20260920. Report percentile 95%
intervals for unadjusted and overlap-adjusted raw-unit differences. A
400-micrometre block sensitivity is specified in advance. Record block counts
and invalid replicates. Report uncertainty conditional on the fitted predictions,
calibration, thresholds and technical strata; this first bootstrap does not
refit those components and must not be called full-pipeline uncertainty.

Spatial dependence beyond the chosen block sizes and grid alignment remain
limitations. These intervals are within-section evidence, not new patients.
Do not turn them into a fraction-of-significant-sections population test.

Average repeated sections equally within each patient, then average patients
equally within each cohort. Show all four patient values, an explicitly
approximate t interval with three degrees of freedom, and leave-one-patient-out
means. Exact two-sided sign tests can only be coarse with four patients; their
smallest possible p-value is 0.125. Apply BH separately within each
cohort/outcome family across all eligible pathways if presenting these p-values.
Do not select an asymptotic method simply to achieve significance. A wide interval
or nonsignificant result is not evidence that an association is zero.

The program-exclusion diagnostic reduces direct self-inclusion but cannot remove
correlated-gene/shared-factor selection. Full-score absolute-error enrichment is
partly definitional. No spot-label permutation is used as a null of absent biology.
If proposing a stronger technical/selection-null test later, specify its null and
verify it on simulations before applying its p-values to the data.

## Verification and additional audit work

Before trusting the atlas: verify the overlap estimator against an explicit
weighted matched-stratum calculation; check bootstrap coverage on a simple
known-effect independent-block simulation and expose failures under longer-range
dependence rather than treating one simulation as universal validation. Confirm
gene-exclusion membership and covariates contain no tested genes. Independently
recompute selected effects from saved arrays. Preserve definitions and seeds.

Audit the existing morphology-matching execution separately: k=1/5/10 retention,
caliper distances, farther-neighbor violations, repeated control reuse, and
expression/detection balance. Original Euclidean distances on normalized
embeddings are monotone in cosine distance but are not numerically cosine
distances. Matched pairs are not independent when controls repeat or spots are
spatially correlated. Do not reuse their original spot-level Wilcoxon p-values
as proof of an adjusted population effect.

Further full-pipeline re-estimation, registered serial-section inference, and
cell-lineage-specific claims remain explicit tasks; they are not certified by
this first broad atlas. The design follows the general separation of selection
and evidence and respects nested biological sampling; see
[circular analysis](https://www.nature.com/articles/nn.2303) and
[pseudoreplication](https://www.nature.com/articles/s41467-021-21038-1).
These sources do not establish validity of this particular spatial bootstrap.

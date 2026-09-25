# First broad biological reassessment

2026-09-20. Attribution: COMPUTATION. Read alongside the
[inference report](../stage_06_inference/RESULTS.md). This is a retrospective
atlas and first selection/technical sensitivity, not the finished paper revision.
The primary magnitude score and original files remain unchanged.

## What survives

Poor prediction extends across gene programs. After removing each program's
genes from both the grouping score and its expression-conditioning variable,
and comparing locations within count/detection overlap strata, absolute program
error is higher in Q4 for **all 232 patient–program combinations evaluated**
(31 programs × four discovery patients; 27 × four validation patients).
Standardized adjusted differences range from 0.397 to 2.968.

This strengthens the general prediction-quality interpretation: the score
identifies locations where errors extend to genes that did not define the groups.
It is not a result restricted to EMT. Correlated genes/shared technical factors
remain, and broad error propagation is not proof of a specific biological cause
or of intrinsic unpredictability by all possible models.

The atlas separately records observed expression, signed residuals and absolute
errors, including Q1, Q2, Q3, Q4, continuous associations, raw-error and within-
section-centering sensitivities, and prediction utility against the training-
median baseline. No pathway is omitted for giving an inconvenient direction.

## Signed pathway differences predominantly vary by patient

For the 25 Hallmarks testable in both panels, a descriptive additive decomposition
of the 25-by-eight patient effect matrix attributes **92.2%** of the variation in
count/detection-adjusted signed Q4-Q1 contrasts to the patient mean, 1.0% to the
program mean, and 6.9% to the remaining interaction/deviation. Before adjustment,
the patient component is 86.8%. These are summaries of the observed effect matrix,
not causal variance components or an ANOVA significance test. Panel-specific
gene sets differ and are not being treated as identical measurements.

The practical implication is that many programs shift in the same signed
direction within a patient. Treating each such shift as independent evidence of
pathway activation would overinterpret the results. Shared prediction bias,
technical factors and patient biology are possible explanations; this analysis
does not determine their relative contributions. It motivates a separate generic
calibration investigation rather than selecting a pathway-specific score.

See the complete [patient-by-program figure](figures/patient_program_atlas.pdf).

## EMT: error magnitude survives; uniform positive direction does not

The discovery panel measures 25 EMT-Hallmark genes, validation 19, with ten shared.
For program-excluded, count/detection-adjusted comparisons:

| Outcome | Discovery mean standardized contrast | Validation mean standardized contrast |
|---|---:|---:|
| Observed expression | −0.451 | −0.256 |
| Signed residual | −0.242 | −0.006 |
| Absolute error | +1.695 | +1.421 |

These average sections within patients and then four patients per cohort. An
adjusted contrast is divided by the original Q1/Q4 pooled SD; it is not an
unadjusted Cohen's d or a claim of pathway activation. Patient uncertainty is
large for the signed mean. The validation mean near zero combines opposing
effects, rather than demonstrating that every patient has no association:

| Patient | Signed contrast | Absolute-error contrast |
|---|---:|---:|
| P03 | +0.200 | +1.714 |
| P04 | −1.211 | +1.781 |
| P05 | −0.049 | +1.364 |
| P06 | +0.092 | +1.920 |
| P01 | −0.983 | +1.165 |
| P02 | +0.775 | +1.361 |
| P07 | +1.206 | +1.863 |
| P08 | −1.022 | +1.295 |

The ten shared genes also show higher absolute error (+1.194/+1.266), but not a
consistent positive signed contrast (−0.152/−0.001). Their observed-expression
contrasts are −0.723/−0.289. The current evidence supports EMT-associated genes
being among broadly harder-predicted programs in Q4. It does not make generic
Q4 an EMT-activation or EMT-transition-zone classifier. Earlier positive P08
zone-specific signed-residual findings use different groups and remain separate,
case-specific evidence; this atlas neither substitutes for nor erases them.

## What the measured EMT program represents in available cell data

An explicitly exploratory follow-up used the existing count-verified cell
matrices and unchanged source annotations/QC, profiling **all** eligible
validation Hallmarks. In the three Janesick sections, source-labeled stromal
cells contribute 65.0%, 69.5% and 48.3% of measured panel-EMT transcripts
(NCBI785, NCBI784, NCBI783). Source-labeled tumor cells contribute 11.5%, 8.5% and
6.7%. Stromal cells comprise 25.5%, 32.7% and 22.5% of QC cells, respectively.

These measurements directly show substantial stromal contribution to the
measured Hallmark signal in these sections. They do not imply absence of tumor
EMT, validate the annotations as a perfect lineage gold standard, or measure Q4
enrichment: this is whole-section source-cell context from two patients. Nuclear
count fractions and common-gene results are also retained. In particular, a
mixture-level Hallmark mean cannot be identified with epithelial-cell transition
without additional lineage evidence.

The ten shared EMT members are ACTA2, CXCL12, DST, FBLN1, MMP2, MYLK, OXTR,
PDGFRB, SFRP1 and SFRP4. They differ from the epithelial/EMT-TF coexpression
phenotype used in the earlier cell experiments. See
[cell-context figure](figures/emt_cell_context.pdf) and
[all cell/program values](source_cell_program_context.csv).

## Other emerging biology remains modest or cohort-specific

Observed early-estrogen-response expression is higher in adjusted Q4 in both
panel analyses (+0.238 discovery/+0.106 validation). Restricting to its six shared
genes gives +0.129/+0.140, with positive effects in three of four patients per
cohort. Approximate patient intervals include zero in both. This is a small
exploratory candidate, not a replacement headline or established replication.

Several immune/stromal-related observed programs favor Q1 in discovery, while
validation effects are weaker or heterogeneous. Allograft rejection, for example,
has shared-gene adjusted contrasts −0.361/−0.091. Reduced proliferation is not a
general Q4 result in this atlas: observed G2M expression has small positive cohort
mean adjusted contrasts (+0.246/+0.096). These are program-member associations,
not direct measurements of cell fractions or pathway activity.

Only five Hallmarks have at least five shared measured genes: allograft rejection,
apical junction, EMT, early estrogen response and KRAS signaling up. Same-name
agreement for other pathways tests a broader functional interpretation using
different genes, not exact-member replication. No new biological candidate is
promoted solely because one sensitivity is favorable.

## Scaling and gene-effect corrections

The original studentized scores reproduce for all 530 section–program pairs.
Their denominator is the SD of absolute residuals in expression bins, not a
signed-residual standard error. No occupied bin in the inspected gene/section
data has fewer than three locations. Effective scale minima reach 0.0493,
maximum inverse weights 20.3, and maximum absolute scaled residual 39.6.
This is not a numerical explosion or a few genes dominating the aggregate, but
it is a substantive gene/location weighting choice. Using raw signed means
instead changes the cohort-mean direction for three discovery and five validation
programs under the same full-score groups. Preserve both definitions explicitly.

The gene atlas reports actual count-mean ratios with a **one-count pseudocount**,
observed log-expression differences, signed/absolute errors, and detection.
The old ratio of mean log1p expression is retained under a literal label, not
called ordinary log2 fold change. Matched-DE mean log-expression differences are
another estimand and cannot inherit the same fold-change threshold uncritically.
The old bridge-gene “significant/reproducible” list has not been revalidated here.

## Verification and remaining work

532 construction/legacy/count checks and 9,566 independent arithmetic/group
checks pass. The verification corrected an inclusive-quartile endpoint mismatch
in the initial descriptive Q1/Q2 profile table; Q1/Q4 contrasts and score groups
were already correctly inclusive and did not change. Independent calculations
also verify observed/signed outcomes from targets and predictions, outside-program
counts, source-cell gene availability and transcript-count conservation.

This checkpoint establishes an atlas and bounded sensitivities, not completion
of external replication, full-pipeline inference or manuscript integration.
Remaining limitations and matching issues are documented in the Stage 6 report.

Outputs: [coverage](coverage.csv), [section contrasts](section_contrasts.csv),
[all quartiles](quartile_profiles.csv), [gene characterization](gene_characterization.csv),
[program prediction utility](program_prediction_quality.csv),
[patient effects](../stage_06_inference/patient_effects.csv),
[patient uncertainty](../stage_06_inference/cohort_patient_summary.csv),
[scaling diagnostics](legacy_scaling_diagnostics.csv), and
[independent checks](independent_checks.json).

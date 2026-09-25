# Gene-level biological audit protocol

2026-09-20. Retrospectively fixed before new gene-exclusion results. All 280 genes
in each IDC cohort, 18 sections/eight patients, corrected B1 predictions from
three ridge encoders. Preserve original score, targets and analyzed membership.

## Quantities and comparison groups

Primary descriptive groups are archived B1 conditional Q1/Q4 within each section.
For every gene separately, exclude that gene from BOTH the mean absolute-error
score and the conditioning sum of log1p expression, then recompute ten pooled
cohort bins and section quartiles. Use float64 arithmetic on stored float32
targets/predictions for these new scores; verify a direct excluded-gene sum.
This reduces direct self-inclusion; correlated genes and shared factors remain.
No group is chosen for favorable marker/EMT enrichment.

Report observed mean log1p expression, mean actual counts, detection, predicted
expression, signed residual and mean absolute error over encoders. Count-scale
log2 ratios use an explicitly stated one-count pseudocount; they are not the
original ratio of mean logged values. Q1/Q4 prediction utility is compared with
each gene's median-expression predictor fitted only to outer-training patients.
Keep all four original quartiles. Restricted-group correlations are descriptive
and may be affected by range restriction/selection.

For full and gene-excluded grouping, report unadjusted effects and the same
fixed five-by-five count/detection overlap estimator as Stage 6, with the tested
gene excluded from covariates, >=10 spots of each group per stratum and harmonic
weights. Retain support fractions and non-estimability. Store raw and standardized
effects, each gene's rank/group overlap and attenuation on identical scales.

Average sections equally within patients, then patients equally within cohorts.
Show patient effects, approximate t intervals (df=3), leave-one-patient-out means
and numbers of patients in each direction. Optional exact sign-test probabilities
and BH adjustment use all genes within cohort/outcome/grouping/adjustment families;
do not interpret thousands of spots as biological replication. No new spot-wise
significance calls or universal spatial-interval claims.

## Predictability, replication and named claims

Connect each gene's whole-section prediction correlation/MAE/baseline gain to
Q4-minus-Q1 observed-expression and absolute-error contrasts. Summarize these
across genes within each patient, with Spearman correlations and descriptive
partial-rank associations controlling gene mean expression, detection and
expression SD. No gene-independence p-values or causal predictability claims.
Record prediction gains in Q1 and Q4 to characterize well-predicted tissue.

Compare all 90 shared genes across IDC panels without selecting genes by
spot-level significance. Report direction agreement for all measured contrasts,
patient consistency and size; do not recycle the old 41-gene selection.

The headline-marker register is defined from the manuscript/reviewer requests:
epithelial EPCAM,KRT8,KRT18,KRT19,KRT17,CDH1,MUC1,SCUBE2,ESR1,FOXA1,TP63,
COL17A1,DSP,CLDN4,GATA3; canonical EMT VIM,CDH2,FN1,SNAI1,SNAI2,ZEB1,ZEB2;
ECM/stromal HSPG2,FBLN1,COL4A1,LAMB1,MMP2,PDGFRB,TIMP1,COL1A1,COL3A1,
DCN,FAP,ACTA2,LUM; macrophage CD163,CD68,CSF1R,CD14,ITGAM; proliferation
MKI67,CENPF,PCLAF; and all measured members of the existing complement Hallmark.
Absent genes are recorded as absent, not negative biological findings. This
register organizes interpretation after the all-gene analysis; it does not define
the score. Source-cell lineage attribution belongs to the next stage.

## Checks and deliverable

Verify gene inclusion/exclusion, score arithmetic, archived full-score contrasts,
training-only baselines, explicit weighted effects and patient averaging using
separate formulations. Save all-gene tables, patient/cohort summaries, common-gene
comparison, named-marker coverage and evidence table. This stage's descriptive
results do not certify cell identity, temporal transition or full-pipeline
selection-null significance. Review these limits before proceeding to cell
composition and external replication.

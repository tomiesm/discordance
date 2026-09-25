# First prediction-quality audit: existing IDC fits

2026-09-20. Defined before the new baseline comparisons. Read-only analysis of
the 72 existing outer-patient test predictions (three encoders, three regressors,
four patients in each of two IDC cohorts). Ridge uses correction 01; the other
regressors are the preserved original fits. No model selection or refitting here.

The Stage 1 membership and patch-coordinate audit permits these diagnostics on
the current analyzed population. Anatomical registration remains an inherited
dataset limitation, and the Visium grouped-fold correction is handled separately.

1. Reconstruct each fold's training expression from the other patient folds'
   archived targets. All target/ID orders must agree across model configurations.
   Estimate per-gene mean and median from training patients only. Verify the
   reconstructed training mean against the saved corrected-ridge training mean.
2. Compare held-out MAE with a training-median baseline and held-out MSE/RMSE with
   a training-mean baseline. Report signed bias, negative-prediction fraction,
   per-gene correlation, and fractions of genes/spots improving on baselines.
   Keep section and patient-fold results; do not use spot-level p-values.
3. Describe existing Q1/Q4 locations, defined by the unchanged three-ridge mean
   conditional score: actual absolute errors, baseline errors, fraction better
   predicted than baseline, total expression, and detection. Report all quartiles
   to show their relationship to actual prediction quality. Do not change the
   score or use biological endpoints in this diagnostic.
4. Distinguish averaging encoder errors from the error of the averaged prediction.
   Use the former for the main group diagnostics, consistent with the score.
5. Check stored PCA explained variance against the manuscript's >95% statement.
   Record model provenance and code-path discrepancies. Do not pick a regressor
   from biology or silently replace the ridge score if an alternative has lower
   error; any proposed change needs a separate decision and downstream plan.
6. Validate a subset with separate direct calculations and protect input files.
   Cohort summaries equally weight patient-fold summaries, with their full
   distribution retained. Baseline comparisons establish predictive utility of
   the evaluated pipeline, not intrinsic biological predictability.

## Encoder provenance check (before recomputation)

Re-encode the first, middle and last supplied patches in each of the 18 IDC
sections (54 patches per encoder) using the existing published-wrapper code and
locally cached pretrained checkpoints, with the original configuration and
mixed precision disabled. Offline mode prevents downloading/replacing weights.
Fail explicitly if a loader reports random-initialization fallback. Compare
archived and new embeddings by relative L2 error and cosine similarity; a
relative L2 tolerance of 1e-3 is a numerical reproduction check, not a scientific
performance criterion. Save diagnostics separately; do not replace embeddings.

### Numerical reproduction follow-up

The first check loaded all three pretrained models successfully, but some vectors
exceeded the declared 1e-3 relative-L2 tolerance (maximum 0.00618). Preserve that
result. The audit used batches of eight and omitted the extraction script's
deterministic seeding settings. Repeat with those settings explicitly enabled,
save the compared vectors, and reproduce original contiguous extraction batches
for NCBI784 (largest discrepancy) and TENX193 (near-exact control). This is a
numerical diagnostic, not an expanded random validation sample. Quantify changes
in the fixed corrected-ridge predictions before deciding whether the differences
matter. Do not alter the tolerance, archived embeddings, models, or groups.

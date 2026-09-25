# External replication after gene/cell audit

2026-09-20. Fixed after Stages 7–8 and before replacement fits or new external
biological results. All writes are isolated here; preserve B1 outputs and original
manuscripts. This completes the third analysis step of SEQUENCE_04.md.

## Predictions and specimen grouping

Visium: follow the Stage 1 grouped split specification. Jointly hold out TENX13
and TENX14 (same Block A specimen), training on the other eight sections. Refit
only this joint fold for UNI, Virchow2 and H-Optimus-0 with the original sklearn
1.4.0 environment, training-only StandardScaler/PCA256, seed42, corrected ridge
intercept, original lsqr solver and alpha=100/(256*n_genes). Targets, membership,
gene panel and embeddings are unchanged. Reuse the other 24 encoder/test-fold
fits after verifying identical train/test membership and gene order. Nine known
specimen groups do not certify nine unrelated donors; unspecified links remain
unknown. NCBI776 is P07, paired with Xenium, not a new independent patient.

COAD: reuse the 12 corrected ridge fits with their four existing patient/sample
holdouts. Verify IDs, targets, gene order, residuals and training means before use.

## Required propagation and comparisons

Recompute Visium pooled conditional scores over all sections after replacing the
two Block A section predictions. Even unchanged sections can change conditional
scores because pooled bin means change. Record raw/conditional rank and quartile
overlap against B1 for all sections. Keep original B1-like generic full-panel
grouping; do not use an EMT-defined or patient-offset-adjusted score.

Evaluate per-section/per-gene prediction correlation, MAE and training-only
median-baseline gain, with genes and spots separately identified. Average the two
Block A sections within their specimen before external group summaries. For
cross-technology predictability, compare common measured genes to each IDC cohort
separately using equal-patient/equal-section summaries. Preserve the original
expression-only Moran's I features, whose targets/coordinates did not change,
and compare them with revised gene prediction correlations; no gene-independence
p-values or causal assertion that spatial structure governs predictability.

For every Hallmark with >=5 measured genes, keep observed log expression, signed
residual and absolute error distinct. Primary description uses full generic Q1/Q4;
the support analysis excludes the program from score and conditioning, with the
same five-by-five outside-program count/detection overlap estimator as Stage 6.
Compare program-excluded 20/80,25/75,30/70 tails. Report each external specimen's
effects before averaging; do not collapse signs into a percentage called pathway
replication without specifying the outcome, measured genes and comparison panel.

Cross-panel common-member program comparison requires >=5 shared measured genes;
evaluate the same members in both platforms, while excluding all measured members
of the program from each platform's grouping and conditioning. Existing IDC
program arrays suffice only where they have exactly the intended members;
otherwise recompute the needed summaries from B1 arrays without changing groups.
Treat original same-name pathway comparisons separately from exact-member ones.

For COAD, give approximate four-patient intervals and influence summaries.
For Visium, show known-specimen summaries and leave-group-out sensitivity, with
NCBI776 separated in independent-external claims. Because other donor links are
unresolved, group-based intervals are descriptive assumptions, not certified
independent-patient population inference. No new spot-significance counts.

## Verification and deliverable

Verify train/test separation, target/embedding identity and saved-model prediction
reproduction; check independent centered-ridge predictions on a deterministic
gene subset with the same lsqr settings. Record solver/PCA versions, mean
expression and fit times. Check reused-fold metadata and recompute selected
baseline metrics separately. Reconstruct new scores from residual arrays and
independently verify program/overlap summaries and Block A group averaging.

Save replacement fits, a reference registry for reused immutable fits, propagated
scores, all program effects and cross-panel comparisons, summary figures and a
claim-by-claim replication assessment. The final paper response uses only claims
supported at the appropriate replication level; neither a common program name
nor a same-patient cross-platform comparison is an independent patient replicate.

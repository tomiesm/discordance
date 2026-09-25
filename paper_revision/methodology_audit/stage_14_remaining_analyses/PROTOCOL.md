# Remaining claim-specific analyses

2026-09-20. Authorized by “proceed with remaining investigations/analyses.”
Written before examining the new estimates. Existing broad audits are complete;
these are bounded follow-ups for unresolved claims, not another score search or
figure redesign. Original code, manuscript and Stage 1–12 outputs remain fixed.

## A. Biological associations after physical-boundary exclusion

Reuse all 32 sections and every eligible Hallmark from the frozen program arrays.
Also evaluate all already registered, measured IDC marker genes, without selecting
them by the new results. Compare all modeled locations with exclusion of the
union of tissue-mask, image and input-expression-grid boundary bands at 200 and
400 µm, exactly as in Stage 12. These are tissue/acquisition boundaries, not
annotated tumor–stroma interfaces.

Keep predictions, scores, original whole-section Q1/Q4 cutoffs and conditioning
fixed. Show full-score unadjusted descriptions and tested-program/gene-excluded
unadjusted and overlap-adjusted comparisons. Five-by-five outside-member count
and detection strata have edges defined on the original section; require at
least ten retained Q1 and Q4 locations per stratum, and at least 30 per arm for a
reported contrast. Recompute overlap weights on the retained population. Save
raw expression/signed/absolute-error contrasts and divide by the same original
whole-section outcome SD when comparing band widths. Never recompute interior
quartiles or optimize band width.

Report every section's support and non-estimable contrasts. Compare each interior
effect with its full-section result on the same estimable sections, average
sections equally within patient/known group, then groups equally. Record missing
sections/groups explicitly. No independent-spot significance claim or claim of
causal edge correction follows. Retention of one contrast does not prove all
signals boundary-independent. Visium summaries retain its known-group and P07
qualifications; IDC remains the primary biological evidence.

## B. Within-patient pathway-profile similarity

Reuse Stage 10 quartile section effects, with the observed-expression,
program-excluded, overlap-adjusted profile as the primary comparison. Other
observed/signed/absolute and full/excluded unadjusted profiles are sensitivities.
Use all eligible pathways in each cohort, with an identical member set of rows
across sections; no significance-selected pathways. Correlate profiles across
sections. Average within-patient pair correlations within each patient, and
between-patient correlations within each unordered patient pair, before taking
their respective means. A singleton contributes no within-patient comparison.
Save every section-pair value, patient summary and leave-one-patient-out result.

Enumerate whole-section grouping assignments preserving each patient's section
count and cohort. For validation, the primary enumeration also preserves the
10x-public versus Janesick source blocks (P01/P02 versus P07/P08); the broader
cohort-only enumeration is a sensitivity. Every assignment recomputes the whole
statistic, so reused pairwise correlations are not treated as independent
observations. Use a one-sided exact reference tail for greater within-patient
similarity. Two primary cohort comparisons receive Holm adjustment. With only
nine source-preserving validation partitions, evidence resolution is limited.

These reference probabilities require exchangeability of section profiles under
the specified no-grouping-association null. They are conditional diagnostics,
not a full-pipeline randomization test: shared held-out models, preparation and
patient identity are not disentangled. Do not claim patient-specific biological
mechanism from a small reference probability. Exchangeability restrictions are
central to permutation validity; see [Winkler et al., 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4010955/).

## C. EMT residuals in source-tumor-rich locations

Address R5.3 directly using the three mapped Janesick sections. Keep the existing
program-excluded EMT quartile groups and measured program members. Require at
least 20 mapped QC cells and at least 50% source Tumor cells; 75% is a fixed
sensitivity. Retain the unrestricted eligible population as a reproduction check.
Require at least 30 spots per arm. These thresholds define source-tumor-rich
mixtures; they do not establish EMT or pure tumor-cell measurements.

Compare observed expression, signed residual and absolute error. Use the same
unadjusted, technical and composition regressions as Stage 8, on the same
eligible observations within each population. Include stromal and other
non-Tumor fractions in the full model; preserve composition as a continuous
covariate within the restricted subset. Use fixed-origin 800/1600 µm spatial
blocks and 499 resamples; intervals require at least 95% full-rank draws. Record
group counts, rank and non-estimable fits without relaxing thresholds.

Reproduce the unrestricted Stage 8 estimates/intervals, verify regression
coefficients independently, and verify at least one aggregated block draw against
an explicitly repeated-row fit. Within-cell expression is a different endpoint
and is not substituted for this whole-spot residual comparison. No new EMT score,
cell phenotype, model fit or biological threshold selection is authorized here.

## Deliverable and stopping point

Save scripts, complete tables, verification, source hashes and a concise Markdown
report under this directory. Update the remaining-work register: finished
analyses, findings that change local manuscript claims, and limitations needing
no further experiment under the current associative scope. No TeX or publication
figure changes occur during this sequence.

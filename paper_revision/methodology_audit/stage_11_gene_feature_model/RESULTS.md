# Gene-feature model audit: results

2026-09-20. **Retain the association between spatial expression organization and assessed gene predictability. Rebuild the figure with all-section, patient-balanced summaries and emphasize the numerical feature model.** The original OLS and gene-CV calculations reproduce, but their inputs and biological labels require correction. This is an explanatory analysis of measured genes; it does not improve the histology-to-expression predictor or validate RNA-free prediction of unseen genes.

## What was checked and changed

- Original response: per-encoder Pearson correlation pooled across all held-out spots/folds, then averaged across encoders. It was not an average of per-section correlations. The revised response averages the existing encoder-averaged section correlations within patient, then equally across patients.
- Original expression moments include all raw section spots rather than exactly the modeled spots; the variance averages within-section variances and omits between-section mean variation. Revised first/second moments use exactly modeled spots with equal-section/equal-patient weights and include mixture variance.
- Original Moran’s I is calculated only on the first three configured sections. The revised feature covers all 11 discovery and seven validation sections, using the same six-neighbor graph definition and patient-balanced aggregation.
- Five-fold gene splits remain seed 42. Training-fold scaling and categorical encoding are explicit; numerical OLS with an intercept is invariant to this scaling change here. Largest full-model held-out prediction change from global to fold-specific preprocessing: biomarkers 1.3e-15, 10x_janesick 2.4e-15. No harmful scaling leakage was demonstrated.

These changes alter the response and feature definitions. Better correlations between versions are not measured gains on one unchanged prediction task. Within each revised cohort, the incremental comparison of models does use the same response, genes and folds.

## Retained numerical result

| cohort       | model                   |   pearson |   spearman |   mae |   relative_MSE_gain |   predictive_R2 |
|:-------------|:------------------------|----------:|-----------:|------:|--------------------:|----------------:|
| biomarkers   | mean_only               |     0.635 |      0.674 | 0.089 |               0.406 |           0.404 |
| biomarkers   | expression_pathway      |     0.634 |      0.662 | 0.089 |               0.403 |           0.401 |
| biomarkers   | numeric_with_spatial    |     0.801 |      0.829 | 0.065 |               0.642 |           0.641 |
| biomarkers   | full_cached_annotations |     0.836 |      0.854 | 0.058 |               0.700 |           0.699 |
| 10x_janesick | mean_only               |     0.701 |      0.736 | 0.095 |               0.500 |           0.491 |
| 10x_janesick | expression_pathway      |     0.717 |      0.717 | 0.092 |               0.523 |           0.514 |
| 10x_janesick | numeric_with_spatial    |     0.808 |      0.825 | 0.076 |               0.660 |           0.654 |
| 10x_janesick | full_cached_annotations |     0.817 |      0.830 | 0.072 |               0.673 |           0.667 |

Adding Moran’s I to expression mean/CV/pathway count increases revised gene-CV Pearson r from .634 to .801 in discovery and .717 to .808 in validation. Mean-only models give .635 and .701. The numerical model’s positive spatial coefficients are .110 and .116 correlation-response units per one feature SD, conditional on the other numerical features. This supports an association beyond those specified expression features; it does not identify a causal mechanism.

| cohort       | model                   |   coefficient |
|:-------------|:------------------------|--------------:|
| biomarkers   | numeric_with_spatial    |         0.110 |
| biomarkers   | full_cached_annotations |         0.098 |
| 10x_janesick | numeric_with_spatial    |         0.116 |
| 10x_janesick | full_cached_annotations |         0.110 |

[Numerical model cross-validation plot](numeric_gene_cv.png). Raw coefficients, fold predictions, all old/response-only/revised comparisons and rank diagnostics are saved as CSV files. All 24 model fits have full-rank gene-CV folds.

Patient omission recomputes cohort features and responses and repeats the full cached-annotation model. The spatial coefficient remains positive in all eight omissions; CV r ranges .823–.839 in discovery and .775–.840 in validation. This is a sensitivity of cohort summaries, not a fresh held-out-patient test of the gene-feature model.

| cohort       | omitted_patient   |   spatial_coefficient |   pearson |   relative_MSE_gain |
|:-------------|:------------------|----------------------:|----------:|--------------------:|
| biomarkers   | P03               |                 0.095 |     0.823 |               0.679 |
| biomarkers   | P04               |                 0.096 |     0.839 |               0.704 |
| biomarkers   | P05               |                 0.097 |     0.835 |               0.697 |
| biomarkers   | P06               |                 0.103 |     0.837 |               0.701 |
| 10x_janesick | P01               |                 0.104 |     0.775 |               0.609 |
| 10x_janesick | P02               |                 0.133 |     0.840 |               0.710 |
| 10x_janesick | P07               |                 0.105 |     0.827 |               0.689 |
| 10x_janesick | P08               |                 0.110 |     0.805 |               0.653 |

## Annotation issue and figure consequence

The BioMart query in the original code requests `namespace_1003`, and then searches those namespace strings for detailed function keywords. In the discovery cache, 277 genes have namespace strings and become “Other”; three are “Unknown”. Validation instead exactly reproduces the gene-name-prefix fallback for all 280 genes (235 “Other”). These are not equivalent curated GO-Slim functional annotations. The classifier outputs were independently replayed from the caches, without a live annotation update.

The localization cache uses coarse, priority-ordered categories from the first search result, with 96 discovery and 78 validation genes “Unknown”. Its current biological correctness was not independently curated. Keep the full cached-annotation model as a labeled sensitivity, and avoid a main-text conclusion that a consistently curated functional ontology explains the cohort difference. The numerical feature model does not depend on these cached categories.

## Interpretation and verification

RNA-derived mean, variability and Moran features require measured transcription. Random folds of genes test prediction within these panels; genes share programs and are not independent biological replicates. The two cohorts also share 90 genes. Neither ordinary gene-wise OLS p-values nor the same gene CV establishes generalization to unrelated gene families, new patient populations or unmeasured transcripts. Use effect sizes and model comparisons, without causal “governs” or “intrinsically invisible” language.

Original OLS coefficients and archived gene-CV predictions reproduce to floating-point tolerance. A separate solver checks all new OLS/CV predictions. Scalar Moran calculations check five genes in each of 18 sections, and direct spot-weight calculations verify moments and CV. There are 100 recorded construction/reproduction checks plus two annotation replay checks, all passing.

[Protocol](PROTOCOL.md), [checks](checks.json), [annotation replay](annotation_replay_checks.json), [model comparisons](model_metrics.csv), [annotation inventory](annotation_audit.csv), [patient influence](patient_influence.csv).

# Held-out prediction quality and encoder checks

2026-09-20. Attribution: COMPUTATION. Existing corrected ridge and preserved
MLP/XGBoost predictions; no new fits or changes to score definitions.

## Main result

Histology predictions have useful absolute-error performance for seven of eight
IDC patients when averaged across the three ridge encoders and compared with a
training-only, per-gene median-expression baseline. Equal-patient mean relative
MAE improvement is 39.2% in discovery and 23.2% in validation. This establishes
utility for the evaluated pipeline and target, not a universal limit on how
predictable a gene or program is from morphology.

| Patient | Mean ridge MAE | Training-median baseline MAE | Relative improvement | Mean signed error, observed minus predicted |
|---|---:|---:|---:|---:|
| P03 | 0.7103 | 1.2278 | 42.1% | +0.2602 |
| P04 | 0.7527 | 1.1625 | 35.3% | −0.3076 |
| P05 | 0.7145 | 1.2538 | 43.0% | +0.0585 |
| P06 | 0.7272 | 1.1446 | 36.5% | −0.0135 |
| P01 | 0.6885 | 0.9625 | 28.5% | −0.2517 |
| P02 | 0.7065 | 0.9246 | 23.6% | +0.2873 |
| P07 | 1.0551 | 1.0315 | −2.3% | +0.8145 |
| P08 | 0.7016 | 1.2353 | 43.2% | −0.3166 |

Values average encoder metrics within each patient. Discovery and validation
summary values average patients equally; they are not pooled-spot estimates.
Errors are in the log1p(count) target units. Full per-encoder results, including
RMSE against the training-mean baseline, are retained in the output tables.

P07 has substantial underprediction and slightly worse mean MAE than the median
baseline, despite mean gene correlation around 0.614. This is an important
calibration/domain-shift concern, not evidence that the intercept correction
failed. Other patients also have opposing mean biases that can cancel in a pooled
summary. Neither correlation nor a favorable cohort mean establishes calibration
for every patient. The origin of these shifts (technical factors, tissue
composition, patient biology, or model mismatch) remains unresolved. Estimating
an offset from the held-out targets would change the evaluation and could remove
biology; it is not an automatic correction to apply.

Ridge remains competitive with the alternative regressors in this existing
comparison. These diagnostics do not justify selecting a different model to
obtain more favorable biological associations. Training-only model selection
would require a separate protocol if later prediction diagnostics motivate it.

## Do the current quartiles rank actual error?

Yes. Actual mean absolute prediction error increases from Q1 through Q2 and Q3
to Q4 in **all 18 IDC sections**, using the existing conditional score and
section-specific cutoffs. This supports their relative prediction-quality
interpretation. It does not yet show independence from technical variation or
validate a biological enrichment test.

| Cohort | Q1 MAE | Q2 MAE | Q3 MAE | Q4 MAE | Q1 spots better than baseline | Q4 spots better than baseline |
|---|---:|---:|---:|---:|---:|---:|
| Discovery | 0.600 | 0.672 | 0.740 | 0.896 | 99.8% | 83.9% |
| Validation | 0.667 | 0.737 | 0.792 | 0.957 | 97.5% | 66.8% |

Here sections are averaged within patients and patients within cohorts. The
baseline is each outer fold's training-median expression. Many Q4 locations
remain better predicted than this simple reference: Q4 means relatively poorly
predicted within a section, not necessarily useless prediction. Error averaged
over the three encoders is the primary diagnostic, consistent with the score.
The error of an averaged prediction is separately saved and is a different
quantity.

## PCA and execution-description corrections

All 24 corrected IDC ridge models use 256 PCA components. Their retained variance
is 81.7–85.0% for UNI, 85.9–88.4% for H-Optimus-0, and 90.9–93.5% for Virchow2.
The manuscript's “more than 95%” claim is incorrect for these fits. This requires
a reporting correction, but does not on its own establish that 256 components
is a poor prediction choice.

The first inspection used sklearn 1.6.1 and recorded model-version warnings.
All 24 values were then verified in the fitting environment, sklearn 1.4.0,
with zero load warnings and maximum discrepancy 1.11e-16. See
[the verification](pca_original_environment_verification.json).

The source audit also confirms that the XGBoost training caller passes no
validation set, so its conditional early-stopping branch is not executed. MLP's
inner validation uses randomly selected training-patient spots, not a separately
held-out patient. Outer held-out patients remain excluded. These distinctions
must be reported accurately and considered if model selection is revisited.

## Encoder reproduction and numerical limits

All three original wrapper classes successfully loaded existing pretrained
checkpoints in offline mode. No random-initialization fallback occurred. The
first/middle/last patch in every IDC section was re-encoded (54 patches per
encoder). Archived/new cosine similarity was at least 0.99998, supporting
correspondence to the specified pretrained models and preprocessing.

Some vectors nevertheless exceed the protocol's relative-L2 tolerance of 1e-3:
maxima are 0.00618 (UNI), 0.00303 (Virchow2), and 0.00177 (H-Optimus-0). The initial
failed tolerance result remains preserved. Explicit original deterministic
settings, a second GPU type, and original contiguous batches for the largest-
difference section plus a near-exact control did not fully remove the differences.
Their exact historical numerical origin is unresolved. Do not label this check
bitwise/exact reproduction or silently loosen its tolerance.

Propagation through fixed corrected ridge models, in their original sklearn
environment, found mean absolute predicted-expression change 0.000242, maximum
single predicted-value change 0.00758, and maximum spot-MAE change 0.000780 among
180 comparisons. Mean absolute spot-MAE change was 0.0000661. Applying the fixed
models to archived vectors reconstructs saved predictions within 3.34e-6. There
were no model-load warnings. These sampled changes are small relative to the
observed MAEs; they support continuing diagnostics with the archived embeddings.
This is not a full-cohort test of quartile stability under re-extraction. Preserve
the vectors, settings, initial discrepancy, and impact calculations in
[embedding_numerics/](embedding_numerics/).

## Verification and consequence for the plan

All 114 prediction-audit identity/reconstruction checks passed. A separate
calculation with sklearn DummyRegressor and metric functions checked 72 metrics
in two held-out folds, including P07 (maximum discrepancy 7.33e-11), and 54 gene
correlations with scipy (maximum discrepancy 3.31e-14). Reconstructed training
means also match the saved correction means. The shared checkpoint verification
found all 184 monitored original files unchanged.

Proceed to technical/error diagnostics of the current score. Keep the current
ridge model, target, and quartile definitions as B1 while doing so. Do not promote
a replacement score or a biological claim from these performance checks. The
external Visium specimen-group refit remains a dependency before its revised
biological validation. The full figure/table provenance map and anatomical
registration limitation also remain explicitly open.

Outputs: [patient metrics](patient_model_metrics.csv),
[section metrics](section_model_metrics.csv), [gene metrics](gene_model_metrics.csv),
[quartile diagnostics](quartile_prediction_quality.csv), [checks](checks.json),
[independent checks](../checkpoint_02_verification.json), and
[embedding impact](embedding_numerics/prediction_impact_summary.json).

## Follow-up prompted by the broad atlas

The exact identity mean(r²) = mean_g(mean_spot(r_g)²) +
mean_g(var_spot(r_g)) separates gene-specific held-out mean bias from within-fold
variation. Calculated from the already verified gene metrics, the mean-bias
fraction of MSE, averaged over encoders, is 13.2–22.2% in discovery and 33.4–60.5%
in validation (P01 40.9%, P02 51.3%, P07 60.5%, P08 33.4%). See
[the decomposition](patient_bias_mse_decomposition.csv).

This is a descriptive, outcome-aware decomposition. It does not identify the
offsets as technical rather than biological, and is not permission to remove
them using the held-out targets while continuing to claim untouched prediction
evaluation. It strengthens the priority of investigating the calibration/target
question before promoting pathway-specific interpretations of signed residuals.

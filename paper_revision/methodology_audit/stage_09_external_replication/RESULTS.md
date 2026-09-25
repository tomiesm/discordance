# External replication after specimen-group correction

2026-09-20. Three joint Block A Visium fits and the downstream propagation are complete. Original code, B1 outputs and both manuscript copies remain unchanged. See the [frozen protocol](PROTOCOL.md).

## Model correction and prediction quality

TENX13 and TENX14 were excluded together during training for each of three encoders. The other 24 Visium fits retain the same training/test membership and are reused; COAD reuses its 12 corrected fits. This fixes the known shared-specimen leakage in the two Block A test sections. It does not establish that every other Visium section is from an unrelated donor. NCBI776 is P07, already represented in Xenium, and is also excluded in the independent-external sensitivity.

| Dataset | Known specimen groups | Groups beating training-median MAE | Range of relative MAE gain |
|---|---:|---:|---:|
| COAD | 4 | 4/4 | +13.1% to +28.9% |
| Visium | 9 | 3/9 | -22.0% to +2.3% |

MAE gains average encoders and sections within known specimens before the group comparison. A positive gain means smaller absolute error than a training-only per-gene median prediction. Correlation, useful absolute prediction and Q1/Q4 ranking are distinct results.

COAD outperforms its training-median baseline in all four patients. Visium does so in only three of nine known specimen groups overall; global absolute prediction accuracy is therefore limited for this transcriptome-wide application. In its Q1 subsets, Visium improves on the baseline in eight of nine groups (TENX39 is the exception). Report both populations rather than treating relative concordance as uniformly accurate whole-section prediction.

Visium raw MAE rises monotonically from Q1 through Q4 in seven of ten sections and Q4 exceeds Q1 in nine. NCBI682 has Q1 MAE 0.344 versus Q4 0.322. Conditional groups rank error relative to pooled expression-bin means, so they cannot always be called the lowest/highest absolute-error locations across differing expression levels. All four COAD sections are monotone.

The two replacement test sections have the following score changes relative to archived B1:

| Section | Conditional-score rank correlation | Q1 overlap | Q4 overlap |
|---|---:|---:|---:|
| TENX13 | 0.662 | 67.5% | 55.3% |
| TENX14 | 0.489 | 53.9% | 45.9% |

Pooled bin means were recalculated across all 40,350 Visium spots. The other eight sections retain their predictions (maximum raw-score arithmetic difference 1.1e-07), but their conditional scores also change: Q4 overlap ranges from 93.3% to 99.1%. These propagated groups are used throughout this stage.

## Broad error structure versus expression direction

Primary full-score descriptions and program-excluded support analyses are both saved. The following table uses program-excluded grouping and outside-program count/detection overlap adjustment. It counts positive cohort-average contrasts, without converting them into independent discoveries.

| Dataset | Testable programs | Positive observed expression | Positive signed residual | Positive absolute error |
|---|---:|---:|---:|---:|
| COAD | 26 | 6 | 4 | 26 |
| Visium | 50 | 0 | 0 | 50 |

For comparison, the primary full-score, unadjusted descriptions give the following counts:

| Dataset | Positive observed expression | Positive signed residual | Positive absolute error |
|---|---:|---:|---:|
| COAD | 11 | 15 | 26 |
| Visium | 49 | 0 | 50 |

In Visium, 49/50 observed program means are higher in the original generic Q4 comparison and remain so after program exclusion alone; none is higher after count/detection overlap adjustment. That substantial change must accompany the adjusted results. The adjusted analysis compares a different, overlap-weighted population at similar measured abundance/detection; those covariates can also represent biology. It does not prove that the unadjusted associations are artifacts or establish program-specific suppression. The unadjusted EMT mean itself is near zero (−0.024 SD, four of nine groups positive), so the broad raw abundance increase does not supply replicated EMT enrichment.

Across section–program comparisons, median retained Q1/Q4 fractions are 93.8%/87.2% in COAD and 90.0%/89.5% in Visium; minimum Q4 retention is 22.2% and 10.3%, respectively. These overlap limitations constrain the adjusted population, even when the median support is high.

Excluding a program from the score reduces its direct contribution to group selection, but correlated genes and shared tissue/count effects remain. Broad positive absolute-error contrasts support structured prediction difficulty; they do not imply that every program is activated in Q4.

Selected pre-existing biological claims, with standardized Q4−Q1 effects averaged over known groups:

| Dataset / program | Observed expression | Signed residual | Absolute error |
|---|---:|---:|---:|
| COAD / EPITHELIAL_MESENCHYMAL_TRANSITION | -0.692 | -0.527 | +1.718 |
| COAD / COMPLEMENT | -0.446 | -0.198 | +1.447 |
| COAD / E2F_TARGETS | +0.389 | +0.132 | +1.672 |
| COAD / G2M_CHECKPOINT | +0.456 | +0.212 | +1.425 |
| Visium / EPITHELIAL_MESENCHYMAL_TRANSITION | -0.175 | -0.804 | +0.935 |
| Visium / COMPLEMENT | -0.065 | -0.994 | +1.162 |
| Visium / E2F_TARGETS | -0.037 | -1.326 | +1.100 |
| Visium / G2M_CHECKPOINT | -0.032 | -1.345 | +1.158 |

These full-panel programs contain different measured genes across platforms. Their names cannot by themselves establish replication. The exact-member comparisons below address that separately. Per-specimen directions, approximate intervals and leave-group-out ranges are retained in each dataset's tables. For Visium, group-based intervals assume independence of the known groups and are descriptive because remaining donor links are unresolved.

## Exact-member program comparisons

Programs require at least five identical measured members in both panels. Each platform excludes all its measured members of that program from grouping and conditioning. Counts below are cohort-mean direction agreement for overlap-adjusted contrasts; they are not a pathway replication rate. Visium rows exclude P07 from the external mean.

| External / IDC comparator | Eligible programs | Observed direction agreement | Signed direction agreement | Absolute-error direction agreement |
|---|---:|---:|---:|---:|
| COAD / IDC discovery | 3 | 3/3 | 3/3 | 3/3 |
| COAD / IDC validation | 4 | 3/4 | 3/4 | 4/4 |
| Visium / IDC discovery | 30 | 23/30 | 30/30 | 30/30 |
| Visium / IDC validation | 27 | 15/27 | 5/27 | 27/27 |

COAD has only three eligible common-member programs with IDC discovery and four with IDC validation. The measured EMT Hallmark does not reach the common-member threshold for either COAD comparison. Accordingly, COAD cannot provide exact-member EMT replication under this rule. All same-name comparisons, ineligible coverage and common-member effect sizes are also saved.

## Gene predictability and spatial expression

Gene correlations are averaged across encoders and sections within patient/specimen, then equally across patients/specimens. Cross-gene rank correlations below are descriptive; genes are dependent and different technologies measure different count distributions.

| External / IDC comparator | Shared genes | Spearman correlation of predictability |
|---|---:|---:|
| COAD / IDC discovery | 42 | 0.639 |
| COAD / IDC validation | 55 | 0.567 |
| Visium / IDC discovery | 252 | 0.714 |
| Visium / IDC validation | 267 | 0.773 |

COAD gene predictability versus the archived expression-only Moran's I feature has Spearman correlation 0.853. Targets and coordinates are unchanged, so the target-only feature is reused. This is a spatial-expression association, not a causal explanation or certification of the original multivariable gene-CV analysis.

Visium gene predictability versus the archived expression-only Moran's I feature has Spearman correlation 0.956. Targets and coordinates are unchanged, so the target-only feature is reused. This is a spatial-expression association, not a causal explanation or certification of the original multivariable gene-CV analysis.

## Threshold and verification checks

| Dataset / outcome | Same sign across 20/80, 25/75 and 30/70 tails |
|---|---:|
| COAD / absolute | 103/104 specimen–program contrasts |
| COAD / observed | 95/104 specimen–program contrasts |
| COAD / signed | 102/104 specimen–program contrasts |
| Visium / absolute | 445/450 specimen–program contrasts |
| Visium / observed | 414/450 specimen–program contrasts |
| Visium / signed | 438/450 specimen–program contrasts |

Threshold agreement assesses sensitivity of these fixed predictions, not independent biological replication. No new spot-level significance counts or blanket pathway-conservation percentage is used.

Numerical verification includes source IDs/targets/residuals, training-only means and medians, reconstructed full and excluded scores, raw program means, independent stratified weighting, specimen aggregation, exact common members and saved-model reproduction. Refit diagnostics are reported in the [fit record](refit_complete.json); independent family checks are linked below.

The independent centered-ridge check, using the actual deterministic PCA training scores, matches all three models within 1e-6; coefficients and reconstructed intercepts match exactly. The preliminary PCA-design and discrete-quantile checker discrepancies are preserved and explained in [numerical verification notes](NUMERICAL_VERIFICATION.md).

- [COAD independent checks](coad/independent_checks.json), [Visium independent checks](idc_visium/independent_checks.json), [common-member checks](comparison_checks.json).
- [Independent raw IDC common-member effects and patient aggregation](idc_comparator_independent_checks.json).
- [Specimen prediction quality](specimen_prediction_quality.csv), [gene comparison summary](gene_predictability_comparisons.csv), [gene pairs](gene_predictability_pairs.csv).
- [All program comparisons](program_comparisons.csv), [comparison summary](program_comparison_summary.csv), [coverage](common_member_coverage.csv), [threshold sensitivity](tail_sensitivity.csv).
- [COAD specimen effects](coad/specimen_program_effects.csv), [Visium specimen effects](idc_visium/specimen_program_effects.csv), [Visium score changes](idc_visium/score_changes.csv).
- [Overview figure](figures/external_overview.pdf), [program contrasts](figures/external_programs.pdf).

## Consequence for the revision

Retain application of the prediction-error framework to additional datasets and the supported predictability/error-structure associations. Reassess each expression program separately, with coverage and patient/specimen directions visible. Limited Visium overall MAE utility and conditional-versus-raw ordering must remain explicit. Generic Q4 is not established as a uniform EMT-zone class by a large EMT absolute-error contrast. This stage completes the authorized external sequence; figure-level manuscript integration and any stronger inferential claims remain separate tasks.

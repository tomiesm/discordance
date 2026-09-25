# Gene-level Q1/Q4 audit

2026-09-20. Completed first step of the authorized sequence. B1 models, score and manuscript files remain unchanged. Definitions are in [PROTOCOL.md](PROTOCOL.md); this stage follows the whole-program atlas with all-gene comparisons.

## Main findings

- Direct single-gene self-inclusion is small for most genes: median Q1/Q4 overlap after gene exclusion is 99.1%/99.3% in discovery and 99.0%/99.3% in validation. Correlated genes/shared factors still remain.
- After exclusion and count/detection overlap adjustment, average absolute-error contrasts are positive for 552/560 cohort–gene combinations; 492 have positive effects in all four patients of their cohort. This extends the broad poor-prediction signal to individual genes without using each gene in its own score.
- Of all 90 shared genes, adjusted observed-log-expression contrasts agree in direction for 70, with cross-cohort Spearman correlation .720; 49 have the same direction in at least three of four patients in each cohort. The former 41/41 significance-selected replication claim is not retained.
- Signed residual contrasts agree in direction for only 32/90 shared genes. Observed expression, residual direction and prediction error must not be treated as interchangeable biological measurements.

## Named marker evidence

The following values are equal-patient means of section-standardized observed-log-expression Q4−Q1 contrasts after gene exclusion and overlap adjustment. Positive means more expression in Q4; negative means more in Q1. A dash is absence from the measured panel, not evidence of no expression. Patient effects, approximate df=3 intervals and count-scale effects are in the complete tables.

| Gene | Discovery | Validation |
|---|---:|---:|
| EPCAM | +0.548 | +0.135 |
| KRT19 | +0.558 | — |
| KRT17 | -0.078 | — |
| ESR1 | +0.200 | +0.206 |
| FOXA1 | +0.382 | +0.188 |
| TP63 | -0.170 | — |
| HSPG2 | -0.596 | — |
| FBLN1 | -0.574 | -0.219 |
| COL4A1 | -0.596 | — |
| LAMB1 | -0.705 | — |
| MMP2 | -0.750 | -0.087 |
| PDGFRB | -0.837 | -0.196 |
| CD163 | -0.576 | +0.040 |
| CD68 | -0.301 | +0.104 |
| MKI67 | +0.220 | +0.146 |
| CENPF | +0.294 | +0.046 |
| PCLAF | — | +0.126 |
| SNAI1 | — | -0.033 |
| ZEB1 | — | -0.195 |
| ZEB2 | — | -0.087 |

EPCAM/KRT19 enrichment in discovery Q4 and lower ECM/macrophage marker expression are present also in the unadjusted corrected analysis; they are not produced by single-gene exclusion. In discovery, CD163 is negative in all four patients. Validation CD163 is mixed and its small positive adjusted mean combines one positive with three negative patient effects. These findings do not support the submitted broad epithelial-loss/ECM-gain/macrophage-enrichment narrative. KRT17/TP63 remain modest negative effects, so individual epithelial markers cannot be collapsed into a uniform identity-loss assertion.

MKI67 is near zero in unadjusted cohort comparisons, becomes modestly positive with overlap adjustment, and is positive in all four patients in each cohort under that adjustment. A broad claim of suppressed proliferation is unsupported. Validation measures SNAI1/ZEB1/ZEB2, whereas the other canonical markers listed by the reviewer are not present in these modeled panels; the marker coverage file records every requested gene explicitly. Absence in a modeled panel does not imply absence in other source assays.

## What Q1 predicts well and its relationship to differential expression

Strong Q1 prediction gains over training-only medians include EPCAM, FOXA1 and CLDN4 in discovery and KRT8, TACSTD2 and FOXA1 in validation. Full rankings are reported across all genes, including genes for which Q1 does not improve on its baseline. A gene can be accurately predicted in Q1 and still be more highly expressed in Q4; expression level and prediction error are distinct.

There is no universal rule that the most differentially expressed genes are the least predictable. Across discovery patients, correlation of whole-section gene predictability with the absolute adjusted standardized expression contrast ranges −.073 to +.009; validation ranges −.240 to +.395. After descriptive rank adjustment for gene mean expression, detection and SD, discovery values range −.221 to −.103 and validation −.334 to +.224. These are patient-specific across-gene associations, not gene-independence significance tests.

## Uncertainty and limits

All section effects and equal-section patient summaries are retained. Cohort intervals are approximate t intervals with four patients, not definitive population evidence or simultaneous confidence statements. Exact two-sided sign tests with four patients cannot be smaller than .125; no method was switched to obtain smaller probabilities. Overlap adjustment conditions on count and detection, which can remove biologically meaningful abundance differences too; unadjusted results remain alongside it. Marker means are not cell proportions. The next stage evaluates composition and lineage where verified source cells are available.

## Evidence register and next checkpoint

| Submitted claim | Current disposition |
|---|---|
| Q4 indicates poorer prediction | Retained with gene-excluded individual-gene and program support. |
| Epithelial identity is generally depleted in Q4 | Not retained as a general statement; markers and cohorts differ. |
| ECM/stromal and CD163 macrophage expression is generally enriched in Q4 | Not retained; several corrected discovery effects favor Q1. Composition remains to be assessed. |
| Proliferation is generally reduced in Q4 | Not supported by corrected individual-marker and broad-program results. |
| 41/41 selected bridge genes replicate | Replace with all-90 comparisons, effect sizes and patient consistency. |
| Poorly predictable genes explain Q4 differential expression | Mixed associations; provide the explicit predictability–effect analysis. |

Proceed to the cell-composition stage using the existing verified mappings. These results do not identify EMT absence or presence within individual tumor cells. Prior P08 signed-zone evidence retains its separate, patient-specific interpretation.

## Files and verification

- [All section effects](section_effects.csv), [patient effects](patient_effects.csv), [cohort effects](cohort_effects.csv).
- [Q1/Q4 prediction quality](prediction_quality.csv), [four quartiles](quartile_profiles.csv), [Q1 ranking](Q1_predictable_genes.csv).
- [Predictability links](predictability_effect_relationship.csv), [all shared genes](all_bridge_gene_comparisons.csv), [bridge summary](bridge_summary.csv).
- [Named-marker coverage](marker_coverage.csv), [named-marker evidence](named_marker_evidence.csv), [exclusion attenuation](gene_exclusion_attenuation.csv).
- [Patient figure](figures/marker_patient_effects.pdf).
- [Independent verification](independent_checks.json): 21,337 checks passed, including all archived full-score contrasts and direct excluded-gene/bin/weighted-effect reconstruction for fixed sampled genes.
- The initial writer mislabeled an ancillary quartile Spearman column as Pearson. The label and writer were corrected; values and all primary results were unchanged.

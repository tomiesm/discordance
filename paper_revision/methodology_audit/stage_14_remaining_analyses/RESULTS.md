# Results of the remaining claim-specific analyses

2026-09-20. The three investigations in [the protocol](PROTOCOL.md) are complete.
Original code, manuscript, figures and previous numerical analyses are unchanged.
No new prediction models, score definition, cutoff optimization or figure redesign
was introduced. Quartiles remain primary; deciles/calibration experiments remain
internal. These results support local revisions in the existing manuscript.

## 1. EMT-associated residuals in tumor-rich P07 regions

There is a positive, composition-adjusted association in **both P07 sections**:
among locations with at least 20 mapped QC cells and at least 50% source Tumor
cells, Q4 has greater signed EMT-program residual and greater absolute error than
Q1. The tested EMT members were excluded from the score and its conditioning
quantity before defining the original whole-section quartiles. Restricting to
tumor-rich locations did not re-rank them.

| sample   | outcome   |   estimate |   ci_low |   ci_high |   n_Q1 |   n_Q4 |
|:---------|:----------|-----------:|---------:|----------:|-------:|-------:|
| NCBI785  | observed  |      0.112 |   -0.007 |     0.250 |    392 |     44 |
| NCBI785  | signed    |      0.539 |    0.373 |     0.746 |    392 |     44 |
| NCBI785  | absolute  |      0.402 |    0.264 |     0.623 |    392 |     44 |
| NCBI784  | observed  |      0.030 |   -0.121 |     0.159 |    159 |     71 |
| NCBI784  | signed    |      0.418 |    0.280 |     0.507 |    159 |     71 |
| NCBI784  | absolute  |      0.299 |    0.192 |     0.421 |    159 |     71 |

Effects are adjusted Q4−Q1 differences in mean log1p-count expression or residual
units over the **19 measured EMT-Hallmark members**, not standardized effects.
Intervals above use the 1600 µm fixed-origin spatial bootstrap. Signed residual
is observed minus predicted. At 800 µm the signed intervals are +0.391 to +0.688
(NCBI785) and +0.301 to +0.506 (NCBI784); all 499 draws are estimable at both
scales. Positive mean signed residuals occur in both arms, and are larger in Q4:
unadjusted Q1/Q4 means are 0.591/1.231 and 0.608/1.152, respectively.

The corresponding adjusted observed-expression contrasts are only +0.112 and
+0.030, with 1600 µm intervals crossing zero. Subtracting the signed contrast from
the observed contrast gives adjusted predicted-expression contrasts of −0.427
and −0.388. Thus the supported result concerns **EMT-associated transcription
being underpredicted more in the high-error group**, rather than a robust general
increase in measured EMT-program expression.

This directly addresses the reviewer's request about residuals in epithelial/
tumor-rich regions after stromal control. It remains one patient's repeated-section
association. The RNA-derived source Tumor category includes DCIS; the whole-spot
outcome can contain transcripts from other cells even after adjustment for cell
fractions. The 19 measured members include ACTA2, FBLN1, LUM, MMP1/2, PDGFRB,
POSTN and other ECM/remodeling genes; they do not establish a complete canonical
EMT transcription-factor program or a pure tumor-cell endpoint. Larger error in
this program alone does not establish EMT specificity relative to other programs.

**Support matters.** NCBI785 supplies 392 Q1/44 Q4 locations and NCBI784 159/71.
P08 supplies only **6 Q1/190 Q4**, so no tumor-rich contrast is reported there.
At the fixed 75% Tumor sensitivity, NCBI784 remains estimable (83/31): signed
contrast +0.424, 1600 µm interval +0.348 to +0.635. NCBI785 (211/13) and P08
(0/121) are insufficient; their estimates were not rescued by changing thresholds.

This is distinct from the earlier positive P08 signed-residual association with
cell-defined epithelial/TF neighborhoods. Neither result supplies an independent
patient replication of the other, because the comparison populations differ.
These new tumor-rich effects have not been jointly re-estimated after physical
boundary exclusion; the program-level boundary results below are a separate
population. Do not call the tumor-rich finding boundary-independent.

Sources: [effects](tumor_rich_emt_effects.csv), [all population support](tumor_rich_emt_support.csv),
[exported designs](tumor_rich_emt_designs.npz). Main-paper placement: a short,
scoped addition to the existing biology paragraph, with details in S10/S14 and
response R5.3; no new main figure is required.

## 2. Broad prediction difficulty persists away from physical edges

Every tested program retains a positive cohort/group-average absolute-error
contrast after 200 and 400 µm exclusion of the union of tissue-mask, image and
input-grid boundary bands. The program-excluded overlap-adjusted means remain
positive for **31/31 discovery and 27/27 validation programs**, using all 18 IDC
sections and all eight patients. Median interior standardized error differences
at 400 µm are 1.102 and 0.992, respectively. They use the unchanged original
whole-section outcome SD, rather than restandardizing the reduced populations.

| dataset      |   band_um | positive_error_means   | same_observed_direction   | programs_with_all_sections   |
|:-------------|----------:|:-----------------------|:--------------------------|:-----------------------------|
| 10x_janesick |       200 | 27/27                  | 20/27                     | 27/27                        |
| 10x_janesick |       400 | 27/27                  | 16/27                     | 27/27                        |
| biomarkers   |       200 | 31/31                  | 30/31                     | 31/31                        |
| biomarkers   |       400 | 31/31                  | 30/31                     | 31/31                        |
| coad         |       200 | 26/26                  | 26/26                     | 26/26                        |
| coad         |       400 | 26/26                  | 25/26                     | 26/26                        |
| idc_visium   |       200 | 50/50                  | 47/50                     | 43/50                        |
| idc_visium   |       400 | 50/50                  | 46/50                     | 0/50                         |

These counts summarize dependent programs, not independent significant discoveries.
`same_observed_direction` compares full/interior means on the same estimable
sections, averaged first within patient/known group. Visium loses TENX68 for seven
programs at 200 µm and all programs at 400 µm. All remaining external program means
are positive, but the last result covers eight known groups/nine sections; donor
independence is still unresolved. P07-excluded summaries are also saved.

Biological expression directions are more heterogeneous than error magnitude.
At 400 µm, 30/31 discovery program directions persist but only 16/27 validation
directions do. All 32 measured registered discovery marker directions persist;
28/33 validation marker directions persist. Small near-zero changes should not
be interpreted as categorical biological reversals.

The existing corrected findings are not restored to their original directions by
boundary exclusion. At 400 µm, adjusted EPCAM contrasts remain positive
(+0.461/+0.110 SD in discovery/validation); discovery CD163 and FBLN1 remain
negative (−0.571/−0.500 SD). Observed EMT-program means remain negative in all
four datasets: discovery −0.411, validation −0.344, COAD −0.359 and Visium −0.205
SD on the estimable paired populations. This does not establish absence of EMT
within any individual tissue compartment.

Boundary exclusion changes the comparison population. For IDC programs, the
median Q4 retention across sections/programs is about 70% in discovery and 63%
in validation at 400 µm, before additional overlap retention. Tissue masks are
not independently annotated cancer interfaces. These diagnostics show that broad
error differences extend into the interior, while they do not identify whether
boundary associations are technical or biological.

Sources: [all 46,359 section estimates/support rows](boundary_section_effects.csv),
[paired sections](boundary_paired_sections.csv), [patient/group effects](boundary_unit_effects.csv),
[cohort/group summaries](boundary_cohort_summary.csv). Main-paper placement:
brief qualification in the existing spatial/biology sections and supporting
spatial supplement; preserve Figure 3's current role and layout.

## 3. Patient-profile reproducibility is strongly cohort-dependent

For the primary program-excluded, overlap-adjusted observed-expression profiles,
within-patient section similarity is high in both cohorts. Its advantage over
between-patient similarity is **small in discovery and large in validation**.
The statistic averages within-patient pairs within each patient and between-
patient pairs within each patient pair. It does not count reused pairs as
independent observations.

| family       |   within_mean |   between_mean |   difference |   n_partitions |   reference_p_one_sided |   primary_holm_p |
|:-------------|--------------:|---------------:|-------------:|---------------:|------------------------:|-----------------:|
| biomarkers   |        0.7196 |         0.6943 |       0.0253 |          15400 |                  0.2269 |           0.2269 |
| 10x_janesick |        0.9259 |        -0.1090 |       1.0349 |              9 |                  0.1111 |           0.2222 |

The original p≈.004 independent-pair Mann–Whitney result should be removed.
In discovery, the primary within-minus-between difference is only +0.025; it
remains +0.016 to +0.042 under omission of each patient. Validation's difference
is +1.035, and remains +0.927 to +1.134 under patient omission. Signed and
absolute-error profile sensitivities also show larger within-patient similarity
in validation and weak additional similarity in discovery.

Exact reference assignments move whole section profiles while preserving group
sizes. Validation additionally preserves the 10x-public versus Janesick source
blocks: only **nine distinct assignments** exist, giving a minimum possible
one-sided reference probability of 1/9. The broader cohort-only permutation gives
p=1/105 for validation, but that result ignores source restrictions and is not the
primary inference. Holm values above adjust only the two predefined primary
cohort comparisons; other configurations are reported as sensitivities.

These probabilities are conditional on profile exchangeability under the stated
null. They do not separate patient identity from preparation, source or shared
held-out model effects and are not full-pipeline biological tests. Permutation
validity depends on the exchangeability restrictions, not on obtaining a smaller
p-value ([Winkler et al., 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4010955/)).
Keep the large descriptive validation association visible, with its restricted
scope. Replace the blanket significant patient-specific-pathway claim with the
cohort-specific comparison in the existing Figure 6 paragraph and caption.

Sources: [summary](patient_profile_summary.csv), [every pair](patient_profile_pairs.csv),
[patient means](patient_profile_patient_means.csv), [patient omission](patient_profile_leave_one_out.csv),
[enumerated references](patient_profile_reference_distributions.npz).

## Verification and remaining scope

Construction checks: 1858 boundary, 101 profile,
and 25 tumor-rich checks. An additional **198 independent-formula
checks** pass. All 226 recorded input files retain their source hashes.

- Unrestricted boundary effects reproduce the frozen Stage 10 program and Stage 7
  marker estimates. Fixed-score boundary group retention reproduces Stage 12.
- An independent stratum-intercept regression reproduces selected harmonic-
  overlap estimates across all sections and datasets.
- A brute-force enumeration of all 7! validation section permutations independently
  reproduces both unique-partition reference probabilities, including source
  restrictions.
- Unrestricted EMT estimates and both block intervals reproduce Stage 8. Independent
  residualization and statsmodels fits reproduce the restricted estimates, and
  block-sufficient-statistic resamples agree with explicitly repeated-row fits.

The first boundary implementation used an arithmetically different construction
of the same nominal quantile grid. A tied cutoff exposed a maximum 0.00083
contrast discrepancy in the reproduction check. The fixed code uses the exact
original grid convention and now reproduces the reference results. The failed
attempt log is retained; no old results were overwritten.

[Independent verification](verification_summary.json), [protocol](PROTOCOL.md),
[boundary checks](boundary_checks.json), [profile checks](profiles_checks.json),
[tumor checks](tumor_checks.json), [independent checks](independent_checks.json).

These three investigations are complete. Remaining work under the current
associative scope is local manuscript/figure correction and reviewer-response
integration. Anatomical registration, donor independence, full-pipeline biological
uncertainty and EMT-specific causal claims remain qualified limits. They are not
silently marked as completed analyses. No additional broad refit or search for
favorable zones is required by these results.

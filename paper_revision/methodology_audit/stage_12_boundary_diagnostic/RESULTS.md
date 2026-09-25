# Boundary effects and the existing ring control

**Yes: the original analysis includes a relevant boundary-ring permutation control. The maps also show measurable boundary enrichment. Those findings are compatible.** This exploratory follow-up was prompted by the author’s inspection after the decile maps, with widths/geometries frozen before the new comparisons. It leaves predictors, scores and primary groups unchanged.

## What the original code does

`assign_boundary_rings()` measures distance to the convex hull of analyzed spot centers, divides those distances into ten quantile bands, and merges undersized groups. These are edge-distance bands, not literal circles centered on the image. Script 06 shuffles the unchanged conditional scores only within those bands when constructing the Moran permutation reference. That preserves every band’s score distribution and hence a broad bandwise edge gradient. It asks whether arrangement within/between those fixed bands gives more spatial autocorrelation than that restricted shuffling.

It does not subtract an edge trend from individual scores or change Q1/Q4. The corrected archived IDC ring-controlled p-values are .001 in all 18 sections (999 permutations, minimum attainable .001). Their observed Moran values reproduce against the fixed scores. These are conditional spatial tests with within-band exchangeability assumptions, not proof of independence from tissue geometry or of biological cause. The convex hull ignores holes/concavities and does not distinguish tissue, image and expression-coverage boundaries.

The earlier “interior-only DE” script is a different control: it chooses locations by similarity of neighboring histology embeddings and then re-ranks the retained scores. It did not erode physical boundary bands. Neither this distinction nor visible edge enrichment negates the useful original ring test.

## Direct boundary measurements

For the upper 10%, comparison within 200 µm of the actual saved H&E tissue-mask boundary versus farther inside:

| sample   |   n_edge |   edge_fraction |   edge_tail_rate |   interior_tail_rate |   relative_risk |   tail_fraction_in_edge |
|:---------|---------:|----------------:|-----------------:|---------------------:|----------------:|------------------------:|
| NCBI785  |      295 |           0.072 |            0.325 |                0.083 |           3.942 |                   0.235 |
| NCBI784  |      284 |           0.098 |            0.201 |                0.089 |           2.245 |                   0.196 |
| NCBI783  |      620 |           0.177 |            0.194 |                0.080 |           2.414 |                   0.342 |

Rates are fractions, not percentages; relative risk is near-boundary rate divided by interior rate. The high tail is enriched near this boundary in 26/32 sections (8/11 discovery, 7/7 validation, 4/4 COAD, 7/10 Visium). These are descriptive section counts, not independent-patient discoveries. All sections, lower tails, 25% tails and both widths are in the saved tables. [Patient/known-group rate differences](unit_tail_boundary_effects.csv) average repeated sections within unit.

Different boundaries emphasize different tails. At the supplied expression-grid rectangle, P07 lower-decile rates are about 4.40 and 4.30 times higher within 200 µm; the upper-tail ratios are 1.32 and 1.02. P08 upper-tail enrichment at that rectangle is 2.72. Thus “edge effect” cannot be equated with universally increased score, and a straight acquisition edge may cross biological tissue.

The available geometry shows zero partly off-image patches in NCBI785 and NCBI783, and 45 in NCBI784. Literal off-image padding therefore cannot explain the broad P08 boundary pattern. Partial tissue context, RNA coverage, composition and registration remain distinct possible contributors; this diagnostic does not determine their causal shares.

[All three H&E/boundary context pages](cell_linked_boundary_context.pdf), [P08 context](NCBI783_boundary_context.png), [P07 context](NCBI785_boundary_context.png), [ring profiles](ring_tail_profiles.png). The H&E views use saved registration and are not independent anatomical ground truth. The yellow contour is the tissue segmentation; the purple rectangle marks all input expression locations.

## What remains beyond boundaries?

The following diagnostic removes locations within each fixed width of **any** tissue-mask, image or input-grid boundary. It retains the original score and cutoffs; it does not create a fresh guaranteed decile inside the remainder.

| sample   |   band_um |   fraction_retained |   upper_10_retained |   moran_all |   moran_interior |
|:---------|----------:|--------------------:|--------------------:|------------:|-----------------:|
| NCBI785  |       200 |               0.840 |               0.711 |       0.567 |            0.603 |
| NCBI785  |       400 |               0.723 |               0.547 |       0.567 |            0.595 |
| NCBI784  |       200 |               0.791 |               0.732 |       0.547 |            0.575 |
| NCBI784  |       400 |               0.654 |               0.574 |       0.547 |            0.576 |
| NCBI783  |       200 |               0.776 |               0.519 |       0.699 |            0.677 |
| NCBI783  |       400 |               0.628 |               0.319 |       0.699 |            0.571 |

For P08, roughly 48% of the original top decile lies in the combined 200 µm boundary band, and 68% in the 400 µm band. Substantial interior spatial organization remains: Moran’s I is .699 overall, .677 after 200 µm exclusion and .571 after 400 µm exclusion. These values use the fixed 150 µm graph rule, rebuilt on retained points, and are descriptive comparisons rather than a new significance test.

A second diagnostic subtracts only the ten original hull-band means. In NCBI785/784/783 those means account for 3.18%, 1.34% and 5.99% of within-section score variance, respectively. Radius-graph Moran’s I changes .567→.561, .547→.549 and .699→.680. The percentage is variance explained by band means, not the fraction of spatial autocorrelation or technical artifact. This diagnostic is not a replacement score and is not the permutation test itself.

| sample   |   score_variance_fraction_between_ring_means |   moran_radius150_original |   moran_radius150_after_ring_mean_removal |   archived_k6_ring_permutation_p |
|:---------|---------------------------------------------:|---------------------------:|------------------------------------------:|---------------------------------:|
| NCBI785  |                                        0.032 |                      0.567 |                                     0.561 |                            0.001 |
| NCBI784  |                                        0.013 |                      0.547 |                                     0.549 |                            0.001 |
| NCBI783  |                                        0.060 |                      0.699 |                                     0.680 |                            0.001 |

## Consequence for the revision

Retain the conclusion of structured prediction error and explicitly credit the existing boundary-ring control. Add physical boundary context and the fixed-group erosion sensitivity to the spatial figure/supplement. Deciles can display strong boundary-associated patterns as well as interior regions, so visual concentration is insufficient to identify biological transition interfaces. Do not automatically regress out all boundary association: tissue margins can carry biology, and the current diagnostics do not identify a purely technical component.

This follow-up checks spatial score structure, not every gene/program/cell contrast after boundary restriction. A claim specifically about boundary-independent biological enrichment would require that additional population-specific analysis. The present figure plan does not make that stronger claim.

## Verification

All scored locations align to the Stage 1 native coordinates; signed mask/image distances have scalar checks; explicit edge sums independently reproduce graph Moran calculations; the original segment-distance formula agrees with the new hull geometry. Repairs to invalid contours are the previously documented in-memory repairs. Polygonal extraction removes only zero-area line remnants, with exactly zero symmetric-difference area. Two initial assertions caught an undefined geometry-collection boundary and floating-point polygon-area summation order; both failed logs are retained and no failed results are used.

[Protocol](PROTOCOL.md), [geometry/graph checks](checks.json), [original-ring checks](ring_checks.json), [all boundary effects](tail_boundary_enrichment.csv), [interior sensitivity](interior_sensitivity.csv).

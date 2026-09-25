# Statistical comparison of EMT-associated coexpression neighborhoods

2026-09-20. Isolated continuation of `emt_cells_v1`. Existing cell definitions,
coordinates, thresholds, original manuscript and original code are unchanged.

## Answer

**The measured epithelial-marker/EMT-TF coexpression phenotype is spatially
clustered relative to random labeling of the tumor-cell population.** This
association remains detectable in all three sections when labels are shuffled
within source-identity, transcript-count, cell-size and gene-detection strata.
At 100 µm, there are approximately **14%, 19% and 12% more positive-positive
cell pairs** than expected under that stratified null. This is positive
statistical evidence of spatial organization of the EMT-associated phenotype.

The evidence is more specific than a claim that all 50 previously mapped
components are significant. An exploratory connected-region extent test
supports **two regions in NCBI784** under the identity/detection/size-stratified
null after spatial-search and localization-test corrections. Other sections
show overall clustering without individually supported regions under that
correction. Additional conditioning on local cell composition attenuates the
overall result in P08. Nuclear-only clustering is not consistently supported.

The previous near-zero discordance associations answer a different question
and remain unchanged. This comparison supplies the missing spatial-association
analysis; it does not turn those discordance correlations into a positive result.

## Primary test: nearby coexpressing tumor-cell pairs

Cell positions and source tumor populations remain fixed. The mark is the
unchanged binary coexpression phenotype: at least two measured EMT-related TF
genes and at least one epithelial gene in a QC source-labeled tumor cell.
The statistic counts unordered pairs of positive cells <=100 µm apart, with
self-pairs excluded. The null redistributes positive labels within technical
strata while preserving their exact number in every stratum. Each section has
1,999 random realizations. The primary p-values are Holm-adjusted over the
three section tests; dependence between serial sections does not invalidate
Holm correction, but the sections still represent only **two patients**.

| Section | Patient | Observed positive pairs | Exact null expectation | Observed / expected | Primary Holm p |
|---|---|---:|---:|---:|---:|
| NCBI785 | P07 | 1,803 | 1,587.7 | 1.136 | 0.0015 |
| NCBI784 | P07 | 1,128 | 947.0 | 1.191 | 0.0015 |
| NCBI783 | P08 | 1,857 | 1,663.2 | 1.117 | 0.0015 |

Unadjusted Monte Carlo p is 0.0005 in each case, the resolution of these runs;
none of the 1,999 null pair counts reaches its observed value. We do not report
zero or infer a smaller numerical p. Even Holm correction over all 30 pair-count
configurations gives p=0.015 for each primary result. The 50 and 150 µm
technical-stratified sensitivity tests are also positive after that 30-test
correction (maximum adjusted p=0.048).

See [all statistics](results/statistics.csv) and
[observed versus null overview](figures/spatial_enrichment_overview.pdf).
Grey intervals in the overview are central 95% **null reference intervals**,
not confidence intervals for biological effect sizes or population-level effects.

## What the different nulls ask

1. **Within-section random labeling:** positions and total positive count are
   fixed. This tests organization relative to random placement among the actual
   tumor cells, retaining tissue geometry, holes and tumor-cell density.
2. **Identity/detection/size stratification (primary):** exact source label, then
   recursive quantile bins for non-marker total transcripts (five), cell area
   (three) and non-marker detected genes (three). TF and epithelial genes are
   excluded from these transcript/detection covariates. Bins use covariates
   only, not outcomes or coordinates. No child bin has fewer than 30 cells.
3. **Additional local composition:** further splits on stromal and myoepithelial
   fractions among QC cells within 100 µm. This tests organization beyond that
   measured context; it may condition away part of the microenvironmental
   association itself, so it is reported alongside the simpler nulls.

| Section | Within-section pair ratio / p | Identity/detection/size ratio / p | Additional composition ratio / p |
|---|---:|---:|---:|
| NCBI785 | 1.261 / 0.0005 | 1.136 / 0.0005 | 1.155 / 0.0005 |
| NCBI784 | 1.259 / 0.0005 | 1.191 / 0.0005 | 1.226 / 0.0005 |
| NCBI783 | 1.123 / 0.0010 | 1.117 / 0.0005 | 1.044 / 0.0915 |

P-values in this comparison table are raw Monte Carlo p-values; full adjusted
values are in the CSV. The P07 composition-adjusted pair results remain positive
under the 30-configuration correction (p=0.015); P08 does not (p=0.4575).
This is consistent with P08 coexpression clustering being associated with local
cell composition, but does not establish that composition causally explains it.

Stratification controls **bins**, not continuous covariates perfectly. For
example, primary-null positive cells average 76.9, 77.6 and 76.4 detected
non-marker genes, compared with observed 79.1, 79.9 and 78.2. Mean log cell area
also differs within bins. Thus the adjusted results are conditional on a coarse
exchangeability assumption; they do not prove complete removal of technical or
biological differences. Full balance diagnostics, including unfavorable ones,
are saved in [covariate_balance.csv](results/covariate_balance.csv). Bins were
not refined after seeing the results.

## Repeating the original neighborhood search under the null

Every shuffled field is searched using the original rule: >=20 tumor cells,
>=5 positive cells and >=10% positivity in a 100 µm neighborhood. Centers and
their overlapping neighborhoods are evaluated together as a spatial field;
they are not treated as independent observations.

| Section | Observed qualifying centers | Mean under primary stratified null | Raw Monte Carlo p |
|---|---:|---:|---:|
| NCBI785 | 224 | 59.3 | 0.0005 |
| NCBI784 | 431 | 46.0 | 0.0005 |
| NCBI783 | 2,129 | 1,691.1 | 0.0005 |

All three center-count results have Holm p=0.015 across all 30 configurations.
With the additional composition restriction, the corresponding null means are
157.3, 134.0 and 2,126.1, and p-values are 0.131, 0.0005 and 0.489. Only NCBI784
retains clear evidence of excess qualifying centers under that null.

## Which individual regions are supported?

The initial local test compares each center's standardized positive count with
the **maximum across all eligible centers** in each shuffle, accounting for the
spatial search. After correction across sections, none passes under the primary
stratified null. Under unrestricted within-section labeling, 3, 5 and 3 original
components contain a passing center. These initial results are retained in
[component_scan_evidence.csv](results/component_scan_evidence.csv).

Because the user's question concerns connected regions as well as individual
centers, a separate **exploratory region-extent test** was added after those
initial results. The [addendum](extent_addendum.json) was fixed before this
additional calculation. It keeps every original threshold and joins qualifying
centers separated by <=100 µm. Each shuffle repeats the full search and records
the largest connected component. Each observed region is compared against that
whole-section maximum, rather than a conveniently selected local background.

The following primary-stratified regions pass p<=0.05 after Bonferroni correction
for three sections and both localization statistics (peak and extent):

| Section / component | Native centroid (µm) | Qualifying centers | Coexpressing tumor cells in neighborhood union | Corrected extent p |
|---|---|---:|---:|---:|
| NCBI784 / 1 | (679.4, 2184.1) | 158 | 62 | 0.003 |
| NCBI784 / 2 | (1106.5, 2627.9) | 126 | 32 | 0.015 |

Both are in patient P07; their neighborhood unions can overlap. Counts refer
to different things: the connected region consists of qualifying tumor-cell
centers, and each center's neighborhood contains the supporting coexpressing
cells. A region-level p-value does not assign a biological state to every cell.

With added composition stratification, component 1 retains corrected extent
p=0.042 in that null's six-comparison family. This is less robust to a broader
multiple-analysis interpretation: correcting all 39 peak/extent configurations
gives p=0.189 for the NCBI784 composition-conditioned maximum extent, while the
primary-stratified NCBI784 maximum extent remains positive (p=0.0195). These
broader results are reported to make the post-hoc addition explicit, not hidden.

See [region extent maps](figures/component_extent_maps.pdf),
[all region coordinates and evidence](results/component_extent_with_coordinates.csv),
[extent statistics](results/component_extent_statistics.csv), and
[all 39 localization corrections](results/peak_localization_with_extent_multiplicity.csv).
The original [transcript atlas](../emt_cells_v1/figures/all_candidate_transcript_evidence.pdf)
contains every region, including these two. Visual review is available for
context; it is not the basis of their statistical p-values.

## Nuclear-only sensitivity and interpretation

At 100 µm under the primary strata, nuclear-only positive-pair ratios are 1.460,
1.057 and 0.766, with raw one-sided clustering p=0.029, 0.3965 and 0.9735.
None passes the 30-configuration correction. This sensitivity retains fewer
positive cells and removes cytoplasmic transcripts, so it differs in both
measurement and power. It is a limitation of the result's robustness, not a
reason to erase the positive total-cell coexpression association.

A suitable interpretation is: **EMT-associated TF/epithelial coexpression in
source-labeled tumor cells shows nonrandom spatial organization, with localized
enrichment supported in one section under the specified stratified analysis.**
The degree of association with the surrounding cell composition differs by
sample. These statements concern associations in the measured phenotype.
No per-cell visual adjudication or proof of a temporal transition is required
for them. The separate claim that aggregate discordance captures this phenotype
remains unsupported by the previous near-zero correlations.

## Verification and provenance

The original [protocol](protocol.json) was saved before the permutation results,
after the original coexpression maps had already been examined. The extent
addendum is explicitly later and exploratory. Neither is external preregistration.
All 30 initial configurations and nine extent configurations are retained.

Six statistical tests include exhaustive enumeration of a small stratified
null to verify local moments and expected positive-pair counts. Independent
checks verify observed pairs by direct pair distances, exact expectations by
edge probabilities, local moments by direct hypergeometric sums, Holm correction
against statsmodels, and sampled region extents by hierarchical single linkage.
The extent run reproduces the original pair and qualifying-center statistics
for every one of its reused random draws. Original project files and all outputs
of the preceding cell experiment remain byte-identical.

See [initial verification](results/VERIFICATION.json) and
[extent verification](results/EXTENT_VERIFICATION.json). All sources, null draws,
covariates, scripts and hashes are retained. This is a second numerical
calculation path, not a separate analyst's review.

Fixed-location random-labeling tests for spatial gene-expression marks have
precedent in [Identification of spatial expression trends in single-cell gene
expression data](https://pmc.ncbi.nlm.nih.gov/articles/PMC6314435/). The
Monte Carlo p-value uses the observed realization in its denominator and
numerator, following [Phipson and Smyth](https://gksmyth.github.io/pubs/PermPValuesPreprint.pdf).
The biological meaning of the mark and the conditioning variables remains
distinct from the numerical validity of the random-labeling calculation.

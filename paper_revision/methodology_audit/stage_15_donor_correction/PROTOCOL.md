# Discovery donor correction and focused revision

2026-09-20, before recomputing donor-weighted results. Authorized by the author.

The 10x source lists 12 specimens and 12 donors; eleven non-normal specimens are
analyzed. Assign D01–D11 in TENX/source-position order, preserving source IDs.
P03–P06 are retained only as historical split identifiers, renamed B01–B04 in
presentation. Validation P01/P02/P07/P08 relationships retain their documented
scope. Cross-study unrelatedness is not established by different IDs alone.

1. Verify saved train/test section and donor membership in every outer fold.
   Reuse predictions when whole donors were excluded; describe four-group
   cross-validation in discovery and patient-group exclusion in validation.
2. Recompute all submission-facing discovery means from section results using
   equal donor weights. Recompute per-section correlations where older files
   pooled specimens before correlation. Preserve old group-weighted summaries
   as sensitivity evidence, not biological patient results.
3. Propagate weighting into genes, programs, boundary sensitivities, gene-feature
   regression, prediction baselines, bridge and external comparisons. Include
   all measured genes/programs; do not choose results by favorable significance.
4. Report donor distributions and approximate donor t intervals alongside the
   original four-split sensitivity. Shared fitted models and retrospective
   selection limit population inference. Do not count spots, correlated genes,
   or repeated pairs as independent patients.
5. Discovery has no within-donor repeatability comparison. Keep the existing
   Figure 6 role, using documented validation repeat sections for its exemplars.
6. Build clean and color-marked revisions, supplementary material and reviewer
   responses from the existing manuscript architecture. Computation corrections
   and R5/R6/R7 additions receive distinct colors. Preserve original submission,
   copied submission, published repository and completed audit outputs.

The strongest supported story is chosen after checking these corrected results;
no new clinical correlation, metric tuning, decile display or outcome-driven
model refit is introduced. Calibration/adaptation experiments remain internal.

# Boundary diagnostic prompted by the author's map inspection

2026-09-20. This is exploratory: the author noticed apparent edge effects after
viewing the maps. Fix the following comparisons before calculating them. Reuse
all 32 sections, the existing score and whole-section 10%/25% thresholds; do not
refit predictors, choose an edge definition using biology or change the paper.

Separate four geometries, in verified native physical coordinates:

1. Signed distance to the saved H&E tissue-mask boundary (negative outside the
   mask). This is a segmentation boundary, not independently annotated tumor
   or tumor–stroma interface.
   Use the polygonal area after the already documented geometry repair; discard
   zero-area line remnants and verify unchanged area. A geometry collection's
   undefined boundary must not become a missing distance.
2. Signed distance to the H&E image rectangle. Include the previously verified
   partly off-image patch flag; image boundaries differ from tissue margins.
3. Distance to the bounding rectangle of **all input expression locations**,
   before patch filtering. This proxies the supplied acquisition-grid extent;
   it is not a cell or tissue boundary and can be imperfect for irregular grids.
4. Distance to the convex hull of analyzed spot centers. This describes the
   outer sampled envelope and ignores holes/concavities. It is supplementary
   geometry, not a replacement for the actual saved tissue contour.

At fixed 200 and 400 µm widths, report fraction near each boundary, continuous
score/physical-distance association, edge-versus-interior conditional/raw error,
median-baseline error and top/bottom tail proportions and relative risks. Retain
both high and low tails: visual edge alignment need not mean higher error.
Summarize sections within patients/known groups without independent-spot p-values.
Record patch tissue coverage and partial-image status separately. No causal
artifact inference follows from a distance association.

As a diagnostic, remove the union of tissue-mask/image/expression-extent boundary
bands at both widths. Keep original scores and group cutoffs fixed. Report
retention of high/low tails and radius-150 µm spatial structure before/after;
do not rank a new guaranteed 10% inside the remainder. This checks whether all
spatial error structure is confined to boundaries without changing the score.
Do not interpret graph-statistic differences as a formal conditional null.

Create H&E/context views for all three cell-linked sections with tissue contour,
grid/image extent, continuous distances and fixed score groups; include numerical
results for every section. Verify spot/coordinate alignment to Stage 1, geometry
distances with scalar calculations, and graph Moran calculations using an
independent edge-sum expression. Distinguish demonstrated edge association from
unresolved registration, staining, RNA coverage, patch context or actual biology.

Revisit the decile-display/figure recommendation after these results, recording
what changes. This diagnostic does not independently identify the cause; causal
correction/refitting would need its own justified definition.

The author also correctly recalls the existing ten convex-hull boundary bands.
They constrain permutation in the Moran test; they do not adjust the score or
quartile labels. Reuse the already verified ring-controlled results, compare
independently computed hull distances with the original segment formula, and
record ring-wise mean-score variation separately from total spatial structure.
Do not describe visible edge alignment as refuting the existing ring test.

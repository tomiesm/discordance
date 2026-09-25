# Stage 1 continuation: patch coverage and specimen provenance

2026-09-20, before the coverage computations below. Builds on the completed
membership inventory; original inputs and corrected predictions remain read-only.

1. Reconstruct patch top-left coordinates from each expression location using
   the saved patch scale and HEST's documented coordinate convention. Test all
   supplied IDC patches and the COAD/Visium patches without extracting new images.
2. Test the documented HEST tissue-intersection threshold (15% of patch area)
   and slide-bound intersection rule against patch membership. Use saved tissue
   contours where present. Download missing metadata/contours only into this
   audit directory, pinned to a recorded dataset revision. Current remote
   contours are not assumed identical to historical generating contours.
   Report mismatches rather than choosing a threshold to fit observed membership.
3. Describe retained versus excluded locations using total measured non-control
   counts, modeled-panel counts, detected genes, and location/image coverage.
   Report per-section distributions, no biological enrichment tests or spot-level
   inferential claims. Produce coverage plots using the embedded H&E thumbnails.
4. Trace external sample identifiers to primary public metadata and source
   publications. Record confirmed specimen groups separately from unknown donor
   relationships. Reconcile preservation labels. Create a proposed grouped split
   specification if repeated specimens are established; do not refit until the
   affected inputs and exact change have been audited.
5. Inspect count data and spatial units sufficiently to state what the targets
   and footprints measure. Exact coordinate/bookkeeping consistency does not
   certify anatomical image–transcript registration. Carry remaining registration
   uncertainty into the results rather than declaring it solved from barcodes.
6. Verify reconstruction independently on selected locations, record source
   provenance/hashes, protect earlier results, and update the plan and claim
   ledger. This is a data audit, not a new biological discovery experiment.

## Execution amendment: contour availability and geometric validity

The first uncredentialed request for missing HEST files returned 401. Retrying
through the existing authorized Hugging Face credential obtained all 27 missing
metadata/contour files at the same pinned revision; no new access was requested.
The initial incomplete-mask run is retained in initial_missing_mask_run/.

Several existing contours have ring self-intersections. Before computing overlap,
repair their in-memory geometries with Shapely make_valid, record the change in
area, and compare the resulting union with an independently buffer(0)-repaired
union. Original contour files remain untouched. Report any nonzero area
disagreement and any patch-rule mismatches; neither threshold nor coordinates
are optimized to force agreement. This amendment addresses a geometry-format
issue, and makes no change to a biological analysis.

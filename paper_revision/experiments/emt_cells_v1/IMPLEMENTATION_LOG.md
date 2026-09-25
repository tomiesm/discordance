# Implementation and data-reconciliation log

1. NCBI785 transcript-derived counts matched the vendor matrix exactly, but the
   initial global projective coordinate model failed its held-out-coordinate
   check (maximum error 781 pixels). No EMT outcome was computed. The HEST
   transcript coordinates therefore cannot be assumed to follow that global
   transform. The revised reader first tests a projective mapping, then uses
   local thin-plate interpolation of supplied native/H&E transcript pairs where
   necessary. It retains the one-pixel held-out-coordinate acceptance limit,
   increases the fixed control-point density to every 1000th transcript, and
   saves all control points. This changes coordinate reconstruction, not the
   biology, signatures, thresholds or cell assignments. The original source is
   retained in `frozen/` and the failed log is retained.

2. The local interpolation also failed the one-pixel criterion (median 0.205,
   99th percentile 15.3, maximum 140.6 pixels for NCBI785). It is not used.
   Spatial neighborhoods now use the original vendor cell centroids in native
   micrometers directly. H&E displays/secondary spot assignment use the directly
   observed mean H&E location of each cell's assigned Q>=20 biological transcripts.
   Native transcript-centroid displacement from the vendor cell centroid is
   stored to quantify this distinction. No claim is made that the transcript
   centroid equals a geometric cell centroid or that H&E registration was
   independently validated. This fallback retains measured coordinates and
   avoids applying an inaccurate inferred transform. Candidate state definitions,
   genes and native spatial radii remain unchanged. The reference observability
   test is independent of this coordinate issue.

3. The Visium source-match reader encountered duplicate gene symbols in the
   public 10x matrix. It now aligns by the unique Ensembl feature IDs stored in
   both files. This preserves gene identity and does not merge or reorder counts
   heuristically. The already-computed secondary residual associations are
   unchanged; only source matching is rerun.

4. The independent verifier initially read a numeric annotation barcode as the
   literal string `1e+05`. The analysis reader already parsed it correctly as
   cell 100000. The verifier now uses exact Decimal-to-integer conversion for
   numeric source IDs. A separate verifier-only pandas index collision during
   concatenation of streamed transcript batches was fixed with a fresh row
   index. Neither issue changed any primary analysis result; failed check logs
   remain available. The independent checks are implementation checks by a
   second calculation path, not a separate analyst or biological validation.

5. After the primary outputs, descriptive cell-size/depth/local-composition and
   reference-lineage tables were added as diagnostics. They do not change the
   frozen genes, thresholds, cell labels, model fitting or candidate regions.

# First inventory: specification before execution

2026-09-20. Scope: 18 main IDC sections, original preservation, and membership
consistency of the existing intercept-corrected analysis. No new biological test.

1. Rehash all original files in original_snapshot_manifest.json; check copied
   submitted assets against their original hashes. Extra copied assets are not
   certified by that original manifest.
2. Read cohort configuration, gene lists and patient fold files. Check that each
   sample appears in exactly one outer test fold, that train/test patients and
   sections are disjoint, and that mappings agree. This verifies internal metadata
   consistency, not identity against original source-study records.
3. For each raw HEST h5ad, record spot/gene counts, duplicate IDs, in-tissue flags,
   available panel genes and the cohort intersection after configured probe
   removal. Preserve raw and modeled denominators separately.
4. Read supplied patch IDs and image-array dimensions; read each encoder's spot
   IDs and count NaN/Inf embedding rows in chunks. Identify raw spots missing from
   patches/embeddings and match finite embedded spots to saved test membership.
   This is not a visual H&E registration assessment.
5. Verify ridge prediction/target/residual array shapes against spot IDs and gene
   lists, without recomputing values already checked in correction 01. Check that
   all three encoders and the score tables use consistent locations; verify score
   coordinate finiteness. Inspect spatial units later, not inferred here.
6. Write section-level attrition, encoder detail, gene-panel audit, fold checks,
   checks.json and a brief report. All failures are retained and investigated;
   missingness is not silently repaired. Record input paths, sizes and mtimes;
   hash small configuration/ID files, but do not imply full raw-data checksums
   where only filesystem metadata were collected.

Completion means these inventory checks ran. It does not close Stage 1's full
data-quality, source-provenance, or anatomical-registration audit.

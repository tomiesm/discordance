# First data inventory: completed, full Stage 1 remains open

2026-09-20. The specified read-only inventory completed with **879 checks passed,
zero failed**. These are checks of recorded membership, dimensions and preservation;
they are not 879 independent scientific validations.

## Analytic denominators

| Cohort | Raw expression locations | Supplied patches | Corrected scored locations | Raw locations without a patch |
|---|---:|---:|---:|---:|
| IDC discovery | 80,591 | 67,709 | 67,709 | 12,882 |
| IDC validation | 85,049 | 66,624 | 66,624 | 18,425 |
| Total | 165,640 | 134,333 | 134,333 | 31,307 |

The 31,307-location difference is fully accounted for by the supplied patch
inventory. Every supplied patch has an embedding and a corrected prediction/
score at the matching expression location. None is lost during the inspected
alignment or scoring stages. Patch retention ranges from **64.1% to 97.5%**
across sections.

This does not establish why patches were absent. In sections with an `in_tissue`
field, all recorded raw locations have that flag, including locations without
patches. Two sections lack that field. Therefore the loss cannot simply be
described as removal of raw locations flagged outside tissue. Upstream patch
selection, registration/image coverage and tissue masks need inspection before
deciding whether this affects biological representativeness.

The manuscript's larger raw denominators and the smaller analytic denominators
must be explicitly distinguished. This is not evidence that the corrected
pipeline silently lost an additional set of available patches.

## Checks completed

- All 18 sections occur in exactly one outer test fold. Recorded training and
  test patients/sections are disjoint and agree with saved/configured mappings.
- Both modeled 280-gene panels exactly equal their cohort's non-control-gene
  intersection under the configured probe exclusions; all modeled genes exist in
  every contributing expression file.
- Patch and embedding IDs agree in order for all 54 section/encoder combinations.
  Embedding rows contain no NaN or Inf values. Finite aligned membership reproduces
  saved ridge test IDs in order.
- All three encoders use identical test ID order in each fold. Prediction,
  target and residual array dimensions agree with their IDs and gene lists.
  Score-table membership matches all three encoders; saved score coordinates
  are finite.
- All **184 original files** in the existing snapshot manifest retain their
  hashes (80 submission assets, 103 published code files, one review file).
  All 80 copied submission assets covered by that same manifest also match.
  Additional copied assets are outside this particular original-manifest check.
- The sizes and modification timestamps of inspected inputs remain unchanged
  during the run. Large raw/embedding files were not newly fully checksummed;
  their recorded metadata must not be described as full content certification.

## Source-provenance questions found while preparing the next audit

These observations are separate from the 879 checks above. They require source
reconciliation before extending patient-independent generalization claims.

1. **Visium TENX13 and TENX14:** local HEST metadata identify them as Block A
   Section 1 and Block A Section 2. The corresponding official pages confirm
   these titles and matching tissue descriptions. This strongly indicates related
   sections, so the current leave-one-section-out folds should not be assumed
   donor-independent. Audit their shared specimen identity and group them in
   training/test splits if confirmed. Sources:
   [10x section 1](https://www.10xgenomics.com/datasets/human-breast-cancer-block-a-section-1-1-standard-1-1-0),
   [10x section 2](https://www.10xgenomics.com/datasets/human-breast-cancer-block-a-section-2-1-standard-1-1-0).
2. **NCBI776 and the Janesick Xenium sections:** local metadata point to the same
   originating study. The study used serial sections for Visium/Xenium integration
   and an FFPE Visium workflow, whereas local metadata label NCBI776 fresh frozen.
   This flags donor-overlap and preservation metadata for reconciliation; the
   exact accession-to-patient correspondence has not been established in this
   inventory. [Janesick et al. source study](https://www.nature.com/articles/s41467-023-43458-x).
3. **Unknown external donor labels:** local metadata have missing patient fields
   for several Visium samples and TENX111. Missing labels do not establish either
   independence or overlap. Source-study identifiers must determine the mapping.

## Next bounded work

Continue Stage 1 by tracing patch availability and sample/donor identities, then
audit image/expression spatial correspondence and preprocessing. These findings
add priorities to the plan; they do not trigger an indiscriminate rerun of all
models. Existing predictions remain the B1 reference until a concrete upstream
change is specified.

Files: [section inventory](section_inventory.csv),
[encoder details](encoder_inventory.csv), [fold inventory](fold_inventory.csv),
[gene panels](gene_panel_inventory.json), [all checks](checks.json),
[preservation](preservation.json), [input metadata](inputs.json),
[specification](PROTOCOL.md), [executed script](inventory.py).

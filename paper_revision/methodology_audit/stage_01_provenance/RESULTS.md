# Patch coverage and specimen provenance

2026-09-20. Attribution: COMPUTATION; supports later responses on sample counts,
generalization, and spatial interpretation. Original inputs, published code,
manuscript, and earlier correction outputs remain unchanged.

## Result and scope

The archived patch coordinates and membership are exactly reproducible for all
32 IDC, COAD, and Visium sections using the documented HEST geometry rule. There
is no unexplained patch-membership loss. The resulting analyzed population is a
tissue-mask-selected subset of the supplied expression grid. This verification
establishes coordinate and bookkeeping consistency; it does not independently
certify anatomical registration between H&E and transcription.

| Cohort | Raw locations | Supplied patches | Outside H&E bounds | Below tissue threshold |
|---|---:|---:|---:|---:|
| IDC discovery | 80,591 | 67,709 | 2,585 | 10,297 |
| IDC validation | 85,049 | 66,624 | 988 | 17,437 |
| COAD | 19,028 | 15,750 | 157 | 3,121 |
| IDC Visium | 41,857 | 40,350 | 2 | 1,505 |

The two exclusion columns are mutually exclusive. Across IDC, 3,573 locations
have no patch/slide intersection and 27,734 fail the 15% tissue-area threshold,
accounting for all 31,307 missing patches. Retained patches' top-left coordinates
match exactly (maximum discrepancy 0 pixels). Both rule-disagreement counts are
zero in every section. HEST's intersection rule allows partially off-slide
patches; 74 retained IDC patches have that property. Keep this small edge subset
visible in later sensitivity checks rather than silently changing eligibility.

The saved tissue contours and pinned downloaded contours give the same exact
membership result. Invalid polygon rings were repaired only in memory. The
make_valid and independent buffer(0) unions have identical area and zero
symmetric-difference area in all sections. A second scalar implementation
verified 191 selected locations with zero overlap-fraction discrepancy; see
[independent checks](../checkpoint_02_verification.json).

## What is being measured

The physical image-patch widths are 111.918–112.248 micrometres (224 output pixels
at the nominal 0.5 micrometre/pixel scale). IDC/COAD expression-grid spacing is
100 micrometres; the installed HEST Xenium loader pools transcripts into
100-by-100-micrometre bins. Earlier raw-transcript reconstructions independently
support this for NCBI783–785. The 55-micrometre diameter in the Xenium H5AD display
metadata should not be described as the expression-bin size. Visium has its own
spot geometry, with measured nearest-neighbor spacing approximately 100
micrometres; patch, capture spot, and spacing are distinct quantities.

All IDC count matrices inspected here are finite, nonnegative, and integer-valued.
The modeled expression transform remains log1p(count), without library-size
normalization. This audit does not change the target or fit new models.

Retained locations have higher median modeled-panel counts than excluded
locations in all 18 IDC sections. Many exclusions are low-expression background,
but not all are empty. For example, NCBI784 has retained/excluded median counts
7,682/1,991: 988 locations are outside the H&E bounds and 291 fail the tissue
threshold. Visual inspection of its coverage plot confirms the expression grid
extends beyond the image. This is a coverage limitation, not evidence that the
corresponding transcript locations lack biological tissue. Per-section
distributions are in [idc_coverage_distributions.csv](idc_coverage_distributions.csv).

## Specimen relationships and required corrections

**NCBI776 is FFPE Visium from the P07 source specimen.** Its HEST sample title maps
to GEO GSM7782699. The source study's Xenium Sample 1 replicates map to
NCBI785/GSM7780153 and NCBI784/GSM7780154; Sample 2 maps to
NCBI783/GSM7780155 (P08). As a direct identity check, all 90,280,320 entries of
the 4,992-by-18,085 NCBI776 count matrix match the authors' public CytAssist FFPE
matrix exactly, with identical full gene and barcode sets. The local Fresh
Frozen label is therefore corrected in an isolated metadata override. NCBI776
remains useful for cross-platform comparison, but adds no independent patient
to the IDC P07 evidence. Separate platform-specific fitting does not itself
create cross-platform training leakage; the correction concerns provenance and
the independence of replication.

**TENX13 and TENX14 are sections 1 and 2 of the same named Block A.** They must
be held out together for an evaluation that excludes the known specimen from
training. The existing section-wise folds train on the sister section in these
two cases. [The grouped specification](visium_grouped_split_specification.json)
defines nine known-specimen groups and the required correction. With current
targets/preprocessing fixed, only three new ridge fits are needed: jointly hold
out the two sections for each encoder. The other 24 fits have unchanged training
membership and can be reused. These new fits have **not** been run here.

The remaining source relationships and uncertainty are recorded in
[specimen_register.csv](specimen_register.csv). Public descriptions support the
existing P01/P02 replicate groups. The named COAD and other Visium patients must
retain their study context: the same number in different studies is not a donor
match. Distinct dataset IDs without donor linkage are not proof of distinct
donors. No unsupported additional donor merges have been made.

## Consequence for the plan

IDC membership/coordinate provenance is adequate to proceed with prediction and
score diagnostics on the current population, carrying the registration and
coverage limitations. Preserve the current eligibility population. Before
reassessing external Visium biology, apply the grouped Block A fits and distinguish
paired P07 platform evidence from additional-patient replication. Downstream
Visium conditional calibration pools sections, so replacing two sections'
predictions can affect scores in other sections even when their fits are reused.

No new biological association is claimed from these checks. Neither the
expression target nor the original files nor TeX have been changed.

## Reproduction and sources

Run `patch_coverage_audit.py`, `verify_janesick_source.py`, and the parent
`verify_checkpoint_02.py` with the documented local Python environment. Machine
outputs include [coverage_summary.csv](coverage_summary.csv),
[coverage_checks.json](coverage_checks.json),
[spatial_grid_spacing.csv](spatial_grid_spacing.csv),
[source-matrix verification](janesick_source_verification.json), and
[all-section coverage plots](figures/patch_coverage_all_sections.pdf).

Downloaded HEST files are pinned to revision
`7e8d5a0b0aace41d8c8ec0f6ecea80e4ad2a61ec`; exact origins and hashes are in
[the source manifest](sources/hest_source_manifest.json). Installed HEST 1.1.1
and hestcore 1.0.4 source snapshots are retained. Current remote provenance is
not retrospectively asserted to be the original extraction version.

Primary sources: [HEST patch documentation](https://hest.readthedocs.io/en/latest/generated/hest.HESTData.HESTData.html),
[Janesick et al.](https://www.nature.com/articles/s41467-023-43458-x),
[authors' companion repository](https://github.com/10XGenomics/janesick_nature_comms_2023_companion),
[GSE243275](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE243275),
[GSE243168](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE243168),
[Block A section 1](https://www.10xgenomics.com/datasets/human-breast-cancer-block-a-section-1-1-standard-1-1-0),
[Block A section 2](https://www.10xgenomics.com/datasets/human-breast-cancer-block-a-section-2-1-standard-1-1-0).

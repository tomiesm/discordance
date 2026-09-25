# Cell-resolved EMT feasibility: results and interpretation

2026-09-20. Isolated continuation of the cell-identity-first proposal. Original
code, submitted manuscript and copied manuscript are unchanged. This report is
an analysis record for later revision decisions; no manuscript text was edited.

## Decision

**There are directly measured TF/epithelial coexpression candidates in
source-labeled tumor cells, and specific spatial neighborhoods to investigate.**
The fixed rule identifies 2,753 candidate cells and 50 descriptive neighborhood
components across three sections from **two patients**. These are not validated
EMT zones. Restricted marker coverage, nonspecificity across cell types, depth
dependence and cell-assignment uncertainty prevent that stronger conclusion.

**Corrected aggregate discordance does not show a material adjusted association
with candidate prevalence in this analysis.** Candidate areas were defined
without using discordance. Thus this experiment supplies concrete biological
follow-up targets, but does not substantiate the paper's claim that aggregate
discordance identifies EMT transition zones. It also does not establish absence
of EMT: the candidate definition covers only one limited expression phenotype.

Broader expression data are available: local Visium NCBI776 matches the published
source matrix exactly. Tumor-specific mixture/state modeling remains a separate
next stage, not a completed validation result.

Start with the [overview](figures/overview.png),
[all-section maps](figures/all_section_evidence.pdf), and
[transcript/boundary evidence for every candidate component](figures/all_candidate_transcript_evidence.pdf).

## What was measured and checked

The sections are NCBI785 / GSM7780153 and NCBI784 / GSM7780154 (P07, two sections
of the same source sample), and NCBI783 / GSM7780155 (P08). Published annotations,
vendor cell matrices, cell/nuclear boundaries and cell tables were recovered
from [GSE243280](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE243280),
the source of the integrated breast-tumor study by
[Janesick et al.](https://www.nature.com/articles/s41467-023-43458-x).
Annotations support cell identity; they are not independent EMT diagnoses or
genotype-based proof of malignancy. Published transitional, hybrid and uncertain
classes were kept separate and were never relabeled tumor EMT.

All 90,278,364 local transcript records were processed. Assigned Q>=20 counts
match the independently downloaded vendor matrices exactly across all 541
features in each section. Biological matrices contain 313, 313 and 280 genes,
respectively; control probes are excluded from scoring. Cell QC requires at
least 20 biological transcripts and 10 detected biological genes. There are
428,804 total cells, 407,295 passing QC and 111,271 QC source-tumor cells.

The original 184 paper/code/review files and frozen input hashes were checked.
Additional checks independently reconstruct candidate flags, sampled raw
barcode counts and nuclear assignments, native neighborhoods and connected
components, and adjusted correlations. Six scientific unit tests address
same-cell versus mixed-cell evidence, TF multiplicity, identity, normalization
and neighborhood denominators. See [verification](results/VERIFICATION.json),
[count verification](results/count_verification.json), and `logs/`.

## Cell-level finding

The fixed candidate definition is a QC source-tumor cell with at least one
transcript in **two distinct measured EMT-related TF genes**, plus a detected
epithelial gene. The available TFs are SNAI1, ZEB1 and ZEB2. Measured epithelial
genes are CDH1, DSP, EPCAM, KRT7 and KRT8, with CLDN4 additionally available in
NCBI785/784. VIM, CDH2, FN1, SNAI2, TWIST1 and TWIST2 are absent from these Xenium
panels and were not imputed as observed evidence.

| Section | Patient | QC source-tumor cells | Coexpression candidates | Fraction | Also pass using nuclear transcripts only | 100 µm components |
|---|---|---:|---:|---:|---:|---:|
| NCBI785 | P07 | 62,168 | 1,135 | 1.83% | 136 | 12 |
| NCBI784 | P07 | 34,650 | 706 | 2.04% | 115 | 7 |
| NCBI783 | P08 | 14,453 | 912 | 6.31% | 178 | 31 |

Nuclear-only support retains 12.0%, 16.3% and 19.5% of the candidates. This is a
stricter measurement sensitivity, not a validated truth filter: genuine
cytoplasmic RNA is removed along with potential assignment artifacts. The
429 nuclear-supported candidates merit inspection, but are not automatically
confirmed EMT cells.

The rule is not cell-type specific. Among QC ACTA2-positive myoepithelial cells,
TF/epithelial coexpression occurs in 13.1% and 14.0% in the P07 sections, exceeding
the source-tumor fractions. In P08, 3.58% of DST-positive myoepithelial cells and
3.69% of the published transitional category meet the same coexpression rule.
These categories were excluded from tumor candidates. This demonstrates why
tumor identity, myoepithelial context and segmentation must accompany a marker
map. See [all source-label controls](results/source_label_evidence.csv).

Reference-lineage diagnostics also show that the broad Hallmark program is
strong in cancer-associated fibroblasts and other non-tumor lineages. Full-fit
reference projections for those controls are descriptive; they are not
held-out tumor-state validation. The per-donor/type values are retained in
[reference lineage diagnostics](results/posthoc_reference_lineage_diagnostics.csv).

## Spatial definition and its sensitivity

At each QC source-tumor cell centroid, the 100 µm neighborhood must contain at
least 20 QC tumor cells, at least five candidate cells and at least 10% candidate
cells among its tumor cells. Qualifying centers are linked if within 100 µm;
connected components give the reported count. A component can have only one
qualifying center. Its surrounding disk still satisfies the cell-count rule.
No top-quantile selection, discordance selection, spatial null test or claim of
statistically significant hotspots is involved.

| Section | 50 µm: qualifying centers / components | 100 µm: centers / components | 150 µm: centers / components |
|---|---:|---:|---:|
| NCBI785 | 87 / 19 | 224 / 12 | 15 / 7 |
| NCBI784 | 117 / 19 | 431 / 7 | 128 / 7 |
| NCBI783 | 118 / 22 | 2,129 / 31 | 2,502 / 18 |

The extent and fragmentation are scale dependent. In particular, the much
smaller qualifying extent in NCBI785 at 150 µm cautions against presenting a
stable count or area of biological zones. Component neighborhood unions may
overlap even when their qualifying centers are disconnected; summing their
cell counts can double-count cells.

Every 100 µm component has a page of measured transcripts over vendor cell and
nuclear boundaries. The displayed representative is the candidate nearest the
component center, selected without reference to residuals. Crops include
neighboring/unassigned transcripts for context; the title's counts come from
the representative cell's assigned barcode. These figures do not constitute
pathologist review or a full alternative-segmentation validation.

The [component table](results/candidate_neighborhood_components.csv) provides
native coordinates, tumor/candidate counts, nuclear support and stromal/
myoepithelial context. Per-cell assignments and all three spatial scales are
retained in each section's `cell_evidence.parquet`. Post-primary descriptive
size/depth/context comparisons are in
[cell context diagnostics](results/posthoc_cell_context_diagnostics.csv);
they did not alter candidate definitions.

These diagnostics show higher transcript depth and larger segmented cell area
among candidates in every section. Median candidate/other-tumor transcript
counts are 319/227, 329.5/248 and 307.5/240; median areas are 230/135, 221/139 and
318/246 µm². Candidates also have more nearby stromal cells in all sections.
These are unmatched descriptive contrasts, not causal or significance tests.
They reinforce the need to assess detection opportunity and segmentation before
interpreting sparse coexpression as a distinct tumor state. The nuclear-only
sensitivity and secondary composition/depth adjustment do not fully resolve
these alternatives.

## External-reference observability

The [Wu et al. breast-cancer atlas](https://www.nature.com/articles/s41588-021-00911-1),
[GSE176078](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE176078), supplies
100,064 annotated cells from 26 donors. Its 24,489 cancer epithelial cells include
20 donors with at least 100 eligible cancer cells. Each test donor was excluded
from training and preprocessing. Training donor weights were equal, with a
fixed 3,000-cell training cap per donor and no parameter tuning.

Inputs were the exact measured panel genes present in the reference, normalized
within that panel. The target was mean log-normalized expression of Hallmark
EMT genes absent from the entire Xenium panel: 178 genes for the 313-gene panel
and 179 for the 280-gene panel. This is a disjoint transcriptomic proxy, not an
EMT label. Ridge alpha=100 and all other primary choices were frozen in
[protocol.json](protocol.json) before these outcomes.

| Median within-donor correlation (20 held-out donors) | 313-gene panel | 280-gene panel |
|---|---:|---:|
| Learned restricted-panel prediction vs disjoint program | 0.494 | 0.489 |
| Measured panel Hallmark mean vs disjoint program | 0.443 | 0.427 |
| Total RNA count alone vs disjoint program | 0.561 | 0.561 |
| Learned prediction vs program, controlling depth and detected genes | 0.297 | 0.296 |

Both panels pass the fixed operational gate: median rho>=0.30, at least ten
evaluable donors, and at least 70% of donors with rho>=0.30 (observed: 19/20 for
both). **The depth-only comparator is stronger in the median, and adjustment
substantially attenuates performance.** The adjusted comparison is a diagnostic
added after the primary test; it is not a replacement gate. Passing the original
gate establishes limited information about a reference expression program,
not a validated EMT classifier, cross-platform calibration or successful
biological transfer to Xenium.

The projected maps are retained and explicitly labeled predictions. They were
not used to define the measured coexpression candidates or to claim expression
of genes absent from Xenium. See [donor results](results/reference_donor_performance.csv)
and [depth sensitivity](results/reference_depth_sensitivity.csv).

## Does corrected discordance identify these candidates?

This secondary analysis compares candidate fractions within HEST spot disks
with the corrected mean score across the three encoders. It concerns spot-level
candidate prevalence, not a test of binary component membership. Primary
inclusion requires at least five source-tumor cells per spot; at least twenty
is the fixed sensitivity. Controls are stromal, myoepithelial and tumor
fractions, tumor transcript depth, tumor cell count and total spot expression.
Partial Spearman correlations are correlations of rank residuals after linear
projection on the ranked controls. They do not remove every confounder.

| Section | Included spots (>=5 tumor cells) | Raw D_cond rho | Adjusted D_cond rho | Adjusted gene-disjoint D_cond rho |
|---|---:|---:|---:|---:|
| NCBI785 | 1,570 | 0.073 | -0.006 | -0.003 |
| NCBI784 | 936 | 0.090 | 0.029 | 0.026 |
| NCBI783 | 728 | -0.297 | 0.006 | -0.012 |

The disjoint score excludes all predefined epithelial, TF and Hallmark genes,
leaving 253 of the original 280 genes. Mean absolute residuals are reconditioned
within ten pooled validation-cohort expression bins per encoder, then averaged
across encoders. This reduces direct reuse of the candidate-defining genes;
correlated expression still prevents treating it as independent biological
validation. The existing conditional-score implementation and corrected
predictions were used without refitting or selecting favorable genes.

Requiring twenty tumor cells gives adjusted D_cond correlations 0.008, 0.044
and -0.007; the disjoint values are 0.008, 0.060 and -0.021. These sensitivities
do not change the interpretation. No cell/spot-independent p-values or
population-generalization claims are reported for two patients. P07's two
sections are shown separately, not counted as independent patients.

Signed TF residuals have positive adjusted associations (0.171, 0.223, 0.157),
but observed TF expression alone also does (0.185, 0.184, 0.236). These use the
same TFs that define candidates; they are not independent EMT validation and
do not establish incremental predictive value. A nested patient-level
comparison against an independently measured outcome has not been performed.
All scores, including unfavorable comparisons, are saved in
[residual associations](results/residual_associations.csv).

## Coordinates and source matching

Primary cell neighborhoods use **vendor centroids in original micrometers**.
They do not depend on inferred H&E registration. An attempted global projective
mapping failed a held-out-coordinate check, and a local interpolation also
failed the one-pixel acceptance criterion. Neither is used in final results.
The failures and subsequent choices are in [IMPLEMENTATION_LOG.md](IMPLEMENTATION_LOG.md).

H&E displays and secondary spot assignment use the directly observed mean H&E
coordinates of each cell's assigned Q>=20 biological transcripts. This is a
transcript centroid, not the geometric cell centroid. In native coordinates,
median displacement from the vendor centroid is 0.82, 0.92 and 1.03 µm; the
99th percentiles are 4.95, 5.62 and 6.72 µm. Those native discrepancies do not
independently validate the supplied H&E registration. Secondary associations
inherit its uncertainty. Assignment uses the nearest HEST spot within half
the median spot spacing; cells outside the disks are excluded. Assignment
counts and radii are retained in [spot_assignment.json](results/spot_assignment.json).

Local Visium **NCBI776 exactly matches GSM7782699** for all 4,992 local barcodes
and 18,085 local genes, aligned by Ensembl IDs because source gene symbols are
not unique. The matrix contains 198 Hallmark EMT genes. VIM, FN1, CDH2, all six
predefined TFs, and KRT19 have measured counts; KRT18 is absent from this local
matrix. This is the same study's Sample 1, providing a broader assay on serial
tissue, not another independent patient or same-cell validation. No direct
Visium-to-Xenium registration or tumor-specific deconvolution is claimed.
The matched-study scFFPE-seq matrix is downloaded but has not been independently
annotated for this experiment. See [compatibility](results/VISIUM_COMPATIBILITY.json)
and [actual marker coverage](results/NCBI776_marker_coverage.csv).

## Consequences for the paper and next analysis

The strongest defensible current result is **spatially organized, measured
TF/epithelial coexpression within source-labeled tumor populations**, subject to
the stated rule and assignment limitations. It should remain exploratory and
must not be upgraded to temporal transition or validated partial EMT on the
strength of these maps. The need for multiple context-appropriate molecular
and cellular features follows the [EMT consensus](https://pmc.ncbi.nlm.nih.gov/articles/PMC7250738/).

The next useful computational step is tumor-identity and cell-mixture modeling
of the matched broader Visium/scFFPE-seq data, evaluating epithelial and
mesenchymal programs separately, with explicit fibroblast/myoepithelial
alternatives and depth controls. A breast-cancer reference plus spatial
deconvolution and tumor-identity checks is an established analysis pattern,
for example in [Withnell and Secrier](https://link.springer.com/article/10.1186/s13059-024-03428-y);
its published performance does not validate these sections. That next stage
needs its own fixed analysis specification and patient-overlap accounting.
Orthogonal pathology or targeted RNA/protein evidence would strengthen a
same-cell EMT interpretation if such material is available.

Even if stronger EMT evidence emerges, the separate claim that discordance
finds those states still needs a positive, noncircular test. The present result
does not justify changing thresholds or selecting genes to recover the desired
association. There is no basis here to promise a favorable revision outcome or
to declare the broad spatial-transcriptomics premise false.

## Output guide

- [Overview](figures/overview.pdf): reference performance, depth sensitivity,
  measured candidates and adjusted residual associations.
- [Nine-page section atlas](figures/all_section_evidence.pdf): identities,
  measured genes, candidate neighborhoods, predicted program and H&E context
  for all sections. Some continuous maps saturate at their display limits;
  these color limits do not select cells or define neighborhoods.
- [Fifty-page transcript atlas](figures/all_candidate_transcript_evidence.pdf):
  every primary component, with a fixed representative and boundaries.
- `results/{NCBI785,NCBI784,NCBI783}/measured_cells.h5ad`: raw biological counts,
  nuclear counts, source labels and coordinates.
- `results/{sample}/cell_evidence.parquet`: all cell-level scores and fixed
  spatial assignments; `spot_residual_evidence.parquet`: secondary comparisons.
- [Candidate coordinates and counts](results/candidate_neighborhood_components.csv),
  [spatial-scale sensitivity](results/spatial_scale_sensitivity.csv),
  [cell summary](results/CELL_EVIDENCE_SUMMARY.json).
- [Protocol](protocol.json), code snapshots, source download manifests, input
  hashes, verification results and retained logs provide the audit trail.

This is exploratory analysis on previously studied datasets, with choices
fixed before this experiment's outcomes. It is not external preregistration.
Independent numerical checks use a second calculation path within this work;
they are not a separate analyst's review or independent biological evidence.

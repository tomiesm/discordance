# Cell-composition follow-up protocol

2026-09-20. Written after the Stage 7 review and before these new outcomes.
Use only the three already count-verified/mapped Janesick sections NCBI785,
NCBI784 (P07), NCBI783 (P08). No inferred extension of labels to other patients.
Primary B1 score and whole-section quartiles remain fixed. Source labels are
supportive RNA-derived annotations, not an independent lineage gold standard.

## Questions

1. What source-cell mixtures occur in Q1/Q4? Compute QC cell numbers/fractions
   for every existing source_group and source_label, with >=20 assigned QC cells
   per spot for the primary fraction comparison. Retain original whole-section
   cutoffs, not quartiles recomputed to favor a lineage.
2. For all eligible validation Hallmarks and the predefined Stage 7 markers,
   how do observed/signed/absolute spot-level Q4−Q1 associations change after
   adjusting measured cell composition? Use program/gene-excluded conditional
   groups from the fixed definitions, computed over the full validation cohort.
   Single-gene exclusions use the Stage 7 float64 rule; program exclusions reuse
   Stage 5 arrays. This is not a new score or a selection of favorable pathways.
3. Within source Tumor, Stromal and macrophage/myeloid groups where available,
   compare directly measured per-cell expression and detection in assigned Q1/Q4
   spots. Require >=5 QC cells of the lineage per spot; >=20 is a predefined
   sensitivity. Keep cell-weighted and equal-spot summaries distinct. Whole-cell
   assigned measurements do not exactly reproduce transcript-binned targets.

## Adjustment and uncertainty

For whole-spot observed/signed/absolute outcomes, evaluate unadjusted Q4 indicator,
then log1p outside-program/gene counts and detected genes, log1p number of QC cells
and log1p mean cell area; finally add fractions for all observed source groups
except Tumor (reference). Drop constant covariates. Keep a fixed comparison
population for the three fits (>=20 cells, finite covariates, >=30 Q1 and >=30 Q4).
Report design rank/condition, group support and composition contrasts. These are
conditional linear associations, not causal effects or proof of no confounding.

Within-lineage expression comparisons use the same group definitions with >=5
lineage cells (>=20 sensitivity), adjusting mean non-tested-gene cell counts,
non-tested-gene cell detection and cell area. Aggregate cell outcomes to spots
before fitting; never treat cells as independent biological replicates. For
lineage marker registers use only measured genes; absent/insufficient comparisons
remain explicit. For programs use mean log1p of each cell's measured members.

For the primary >=20 total-cell whole-spot and >=5 lineage-cell fits, obtain
conditional 95% percentile intervals from 499 resamples of occupied 800 µm
spatial blocks (seed 20260920); repeat 1600 µm as a dependence-scale sensitivity.
Use native coordinates with fixed section-minimum origin. Fixed predictions,
groups and covariates; no full-pipeline or patient-population uncertainty claim.
Require >=95% estimable bootstrap draws; retain block counts and non-estimable
intervals. Effects from two sections of P07 do not constitute two patients.

Reuse earlier verified EMT phenotype/zone comparisons, including stromal controls
and their spatial-scale sensitivities, rather than rerunning the same experiment.
Do not replace generic Q1/Q4 by the favorable P08 signed EMT grouping. The new
lineage analysis complements the prior spatial phenotype analysis.

## Checks and deliverable

Reconstruct fraction denominators from mapped QC cells; reconcile with previous
spot data and measure mapping eligibility. Verify a subset of aggregated counts
and cell expression directly from the cell matrix, OLS coefficients with an
independent solver, and block-weighted fits against explicitly repeated blocks.
Store all sections/programs/lineages, including negative and missing results.
Review the composition evidence before launching external replication fits.

### Label clarification before calculation

The existing broad `Other non-tumor` group contains macrophages. Define a
disjoint analysis group `Macrophage` using existing source labels containing
"macrophage" (case-insensitive), leaving all other broad labels unchanged.
Use these disjoint groups for composition covariates and the macrophage lineage.
Tumor includes source-annotated DCIS; add its fraction among tumor cells to
within-Tumor adjustment so a DCIS/invasive mixture is not silently described as
a within-state expression change. Retain the original source labels in outputs.

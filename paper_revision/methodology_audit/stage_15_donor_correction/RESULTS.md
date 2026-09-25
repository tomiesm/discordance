# Discovery grouping correction and focused revision

The primary 10x Biomarkers source reports 12 source specimens from 12 donors.
Eleven specimens enter the analysis; the normal S2-Top specimen is excluded.
Historical P03–P06 labels are four prediction holdout groups, now B01–B04, and
must not be interpreted as four patients with repeat sections. Discovery donors
are D01–D11. The corrected pathology description includes IDC and DCIS, with the
source CCH/DCIS designation retained in the registry.

The existing prediction folds are donor-disjoint. No model refitting was needed
for this identity correction: each held-out group excludes its component donors.
Section predictions, scores and primary quartiles were retained. Gene, program,
boundary and prediction summaries were reaggregated with equal discovery donor
weights. Gene correlations were computed within section for all nine model
configurations before donor/cohort averaging. Numerical gene-feature models,
cross-cohort comparisons and figures were updated accordingly.

The correction removes all discovery biological within-patient repeatability
claims. Figure 6 retains P02 and substitutes the genuine P07 repeat pair for the
false discovery pair. The validation whole-profile source-preserving reference
remains p=1/9, with only nine assignments. Validation effects and the P07
tumor-rich residual result are unchanged.

## Findings retained in the focused paper

- Spatially organized conditional errors in all 18 breast sections: original
  ring-reference Moran's I 0.380–0.754, permutation p=.001 each (999 draws).
- After gene exclusion and overlap adjustment, positive mean absolute-error
  contrasts for 276/280 genes in each cohort. All 31/27 program error means
  remain positive, including after 200/400-µm physical boundary exclusion.
- Discovery EPCAM/KRT19 contrasts are positive in 10/11 donors; PDGFRB is
  negative in all 11. This replaces the original general epithelial-loss claim.
- Observed-expression directions agree for 71/90 shared genes (Spearman .723);
  signed-residual directions agree for 42/90. These are different endpoints.
- P07 tumor-rich EMT signed contrasts remain +.539/+.418 log-count units, with
  positive 800/1600-µm conditional block intervals in both sections. Observed
  program-expression contrasts remain small/uncertain. Only one patient is
  estimable for this exact comparison; P08 lacks eligible Q1 support.
- Gene-CV correlations using numerical expression/spatial features are .802 and
  .808, compared with .635/.717 without spatial autocorrelation. Spatial
  coefficients remain positive after omission of each donor.
- COAD remains a four-specimen application, with only 3/4 eligible identical-
  member program comparisons against the two breast panels. Visium remains a
  qualified application with paired P07 excluded from independent comparisons.

The paper preserves its generic magnitude score, Q1/Q4, seven main figure roles,
valid original exemplars, and existing supplementary structure. Deciles,
calibration/adaptation experiments and a grade/stage association study are not
added to the paper. The central biological scope is heterogeneous associations,
with a patient-specific EMT-associated residual example, rather than a universal
transition-zone label.

## Independent verification

`verify.py` does not import `correct.py` or reuse its aggregation helpers.
All 307 checks pass. It verifies source mapping/folds, independently recomputes
all gene cohort means and approximate intervals, reproduces prior split-group
means, confirms unchanged validation means, checks sampled correlations with
SciPy, verifies boundary error directions and donor-omission coefficients, and
checks all Stage 15 input hashes. CSV serialization differences are at numerical
round-trip precision (maximum section-table difference 2.3e-13), not changed
section computations. All 184 original files and all 80 copied submission files
match the original snapshot. See `verification.json` for exact checks.

Manuscript production is in `../../focused_revision/`; source and production
verification are separate from this scientific correction. Prior audit stages
remain immutable historical records and do not override this donor mapping.

# Bounded audit of the original gene-feature regression

2026-09-20. Fixed before new model outcomes, after reading scripts 11/13 and the
submitted figure-5 claim. This analysis is retained for consideration in the
claim/figure architecture, so its computation and interpretation need checking.
No expression-prediction model or primary discordance score is refitted.

1. Reproduce archived B1 OLS and seeded five-fold gene CV on the existing feature
   tables. Check feature ordering, categorical reference levels, rank, missing
   values, standardization and cached annotation provenance. Distinguish the
   actual pooled-across-fold correlation response from the manuscript's wording.
2. Recompute expression mean/variance and Moran's I on the exact modeled spots
   of **all** IDC sections. The original code calculates Moran's I on the first
   three configured sections and computes a within-section variance average
   omitting between-section mean variation. Preserve those as the old definitions;
   do not silently call them all-patient/global estimates.
3. New response: equal-section mean gene prediction correlation within each
   patient, then equal-patient mean, using Stage 7 per-section encoder-averaged
   correlations. New features: the same hierarchical weights for expression
   first/second moments (full mixture variance and CV), and equal-section then
   equal-patient Moran's I on the original six-neighbor graph. Static pathway
   counts and cached annotation labels remain fixed. Record their actual source
   and limits; no current external annotation refresh changes the model.
4. Evaluate the archived model; response-only change; and revised response plus
   all-section features. Use identical shuffled five-fold gene splits, seed42.
   Training-fold mean/SD and dummy levels are fitted only within training genes.
   Record unseen categorical levels and rank deficiencies. With an intercept,
   global versus training-only scaling may be prediction-equivalent for OLS;
   measure that effect rather than asserting harmful leakage automatically.
5. Fixed predictor sets: mean expression only; mean/CV/pathway count; those
   features plus Moran's I; and the full cached-annotation model. Report held-out
   Pearson/Spearman, MAE and prediction R-squared against a training-mean response
   baseline. No tuning by the strongest biology or p-value. Recompute the revised
   full model leaving each patient out of the cohort summaries to measure influence.
6. Verify OLS/CV predictions with an independent solver, Moran's I with selected
   scalar calculations, and weighted moments directly. Original model p-values
   assume independent genes; related genes and overlapping pathways restrict that
   inference. Random gene folds are a within-panel predictive check, not proof of
   transfer to unrelated gene families, new patients or unmeasured transcripts.

The regression's expression/spatial predictors themselves require RNA measurements.
They explain assessed gene predictability; they are not a validated H&E-only
inference tool for unseen RNA. Final claim/figure wording must reflect the measured
association and these validation boundaries. Save outputs separately, preserve
all previous audit checkpoints, and do not edit TeX.

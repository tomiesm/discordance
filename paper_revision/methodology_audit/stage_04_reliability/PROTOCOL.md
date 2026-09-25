# Reliability of the score actually used

2026-09-20, before these computations. Original scores and models remain fixed.
The Stage 3 audit shows within-section expression dependence despite pooled
centering. Reliability can coexist with this dependence; it must not be used to
claim technical independence or a particular biological cause.

1. Reproduce the original seed-42 raw half-gene correlation for all nine model
   configurations against the saved gate file. The original test used raw error.
2. For the primary mean of three ridge errors, use 20 fixed random 50/50 gene
   partitions (RandomState seeds 42–61). Compute half scores independently from
   disjoint genes, and condition each half on its own sum-log-expression using
   the original pooled cohort calibration. Also report raw halves and the shared
   full-panel conditioning covariate as an explicitly shared-covariate sensitivity.
   Report per-section rank correlation and Q4 overlap. These partitions are
   robustness draws, not independent biological replicates. Do not use spot-level
   correlation p-values to support population claims.
3. Repeat these half-panel diagnostics for the training-only median-expression
   baseline, with the same partition/calibration definitions. This comparator
   contains no spatially varying histology prediction. Reliability is therefore
   not sufficient evidence of morphology-specific discordance.
4. Reconstruct the original k=6 row-standardized spatial graph and Moran's I;
   compare with the saved result. Record edge lengths in physical units and
   connected components. Compare observed spatial structure for raw/conditional
   error, constant-baseline conditional error, expression/detection, and the
   within-section-centering diagnostic. Use an undirected 150-micrometre radius
   graph as a predeclared local-graph sensitivity (100-micrometre Xenium lattice,
   including diagonal neighbors). Isolated points have zero outgoing weight.
5. Do not rerun the already-completed k=4/6/8 sensitivity without a changed input.
   Review the ring null's actual interpretation: shuffling within convex-hull
   distance quantiles preserves ring membership, not full tissue geometry,
   local autocorrelation, compartments, or expression depth. Newly reported
   Moran values are descriptive; do not manufacture stronger biological inference
   from another point-shuffling p-value.
6. Verify selected quantities by a separate formula; save partitions, results,
   and scope. Review score/reliability conclusions before defining broad
   biological contrasts and their selection-aware uncertainty protocol.

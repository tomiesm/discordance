"""
Discordance score computation functions.

Stateless, side-effect-free functions for computing:
  - Conditional discordance (depth-corrected)
  - Mean absolute discordance (L1 per gene)
"""

import numpy as np


def compute_conditional_discordance(
    raw_discordance: np.ndarray,
    total_expression: np.ndarray,
    n_bins: int = 10,
    min_bin_size: int = 30,
) -> np.ndarray:
    """Subtract bin-mean discordance conditioned on total expression.

    Bins spots into quantiles by total_expression, computes mean raw discordance
    within each bin, and subtracts. Result has mean ~0 within each bin.

    Small bins are merged with adjacent bins until all bins have >= min_bin_size spots.

    Args:
        raw_discordance: (N_spots,) raw D_i values.
        total_expression: (N_spots,) total expression S_i = sum of target genes.
        n_bins: Number of quantile bins (default 10 = deciles).
        min_bin_size: Minimum spots per bin; small bins merged with neighbors.

    Returns:
        (N_spots,) conditional discordance D_i^{cond}.
    """
    N = len(raw_discordance)
    if N < n_bins * min_bin_size:
        n_bins = max(2, N // min_bin_size)

    # Bin by quantiles of total expression
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(total_expression, percentiles)
    # np.digitize with right=False: bin 0 = below first edge, etc.
    bin_edges_inner = bin_edges[1:-1]
    assignments = np.digitize(total_expression, bin_edges_inner)  # 0..n_bins-1

    # Merge small bins with neighbors
    assignments = _merge_small_bins(assignments, min_bin_size)

    # Subtract bin mean
    D_cond = raw_discordance.copy().astype(np.float64)
    for b in np.unique(assignments):
        mask = assignments == b
        D_cond[mask] -= raw_discordance[mask].mean()

    return D_cond


def _merge_small_bins(assignments: np.ndarray, min_size: int) -> np.ndarray:
    """Merge bins with fewer than min_size spots into adjacent bins."""
    assignments = assignments.copy()
    unique_bins = np.sort(np.unique(assignments))

    changed = True
    while changed:
        changed = False
        unique_bins = np.sort(np.unique(assignments))
        for b in unique_bins:
            count = np.sum(assignments == b)
            if count < min_size and len(unique_bins) > 2:
                # Merge with the adjacent bin that has fewer spots
                idx = np.where(unique_bins == b)[0][0]
                if idx == 0:
                    merge_target = unique_bins[1]
                elif idx == len(unique_bins) - 1:
                    merge_target = unique_bins[-2]
                else:
                    left_count = np.sum(assignments == unique_bins[idx - 1])
                    right_count = np.sum(assignments == unique_bins[idx + 1])
                    merge_target = unique_bins[idx - 1] if left_count <= right_count else unique_bins[idx + 1]
                assignments[assignments == b] = merge_target
                changed = True
                break

    return assignments


def compute_mean_absolute_discordance(residuals: np.ndarray) -> np.ndarray:
    """Mean |residual| across genes (v3 metric).

    Unlike v2's L2 norm, this is scale-invariant to gene count
    and directly interpretable as average per-gene prediction error.

    Args:
        residuals: (N_spots, N_genes) array of signed residuals.

    Returns:
        (N_spots,) array of mean absolute discordance scores.
    """
    return np.mean(np.abs(residuals), axis=1)

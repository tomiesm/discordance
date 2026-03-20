"""
Morphology matching in embedding space for Phase 3.

Provides cosine KNN matching between discordant and concordant spots,
matched delta computation, and paired DE testing.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


def morphology_match(embeddings: np.ndarray,
                     disc_indices: np.ndarray,
                     conc_indices: np.ndarray,
                     k: int = 5,
                     max_distance_percentile: float = 90,
                     subsample_for_threshold: int = 1000,
                     seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Find k nearest concordant neighbors for each discordant spot by cosine distance.

    Uses L2 normalization + euclidean distance (equivalent to cosine distance).

    Args:
        embeddings: Full embedding matrix (N_all_spots, D_embed)
        disc_indices: Indices of discordant spots in embeddings
        conc_indices: Indices of concordant spots in embeddings
        k: Number of nearest neighbors
        max_distance_percentile: Percentile of within-concordant distances for threshold
        subsample_for_threshold: N concordant spots to subsample for threshold estimation
        seed: Random seed

    Returns:
        matched_conc_idx: (N_disc, k) indices into the FULL embedding array
        match_distances: (N_disc, k) cosine distances
        unmatched_mask: (N_disc,) boolean, True if nearest neighbor exceeds threshold
        distance_threshold: The computed tau threshold
    """
    # L2 normalize embeddings (cosine distance = euclidean on unit sphere)
    emb_norm = normalize(embeddings, norm='l2', axis=1)

    emb_disc = emb_norm[disc_indices]
    emb_conc = emb_norm[conc_indices]

    # Estimate distance threshold from within-concordant pairwise distances
    rng = np.random.RandomState(seed)
    n_conc = len(conc_indices)
    if n_conc > subsample_for_threshold:
        sub_idx = rng.choice(n_conc, subsample_for_threshold, replace=False)
        emb_conc_sub = emb_conc[sub_idx]
    else:
        emb_conc_sub = emb_conc

    # Compute pairwise distances within concordant subsample
    # Use k+1 neighbors within concordant set to get typical distances
    nn_within = NearestNeighbors(n_neighbors=min(k + 1, len(emb_conc_sub)),
                                 metric='euclidean', algorithm='auto')
    nn_within.fit(emb_conc_sub)
    within_dists, _ = nn_within.kneighbors(emb_conc_sub)
    # Skip self-distance (column 0), take remaining columns
    within_dists_flat = within_dists[:, 1:].flatten()
    distance_threshold = np.percentile(within_dists_flat, max_distance_percentile)

    # Find k nearest concordant neighbors for each discordant spot
    nn = NearestNeighbors(n_neighbors=min(k, len(emb_conc)),
                          metric='euclidean', algorithm='auto')
    nn.fit(emb_conc)
    distances, local_indices = nn.kneighbors(emb_disc)

    # Map local concordant indices back to full embedding indices
    matched_conc_idx = conc_indices[local_indices]

    # Euclidean distance on unit vectors = sqrt(2 - 2*cos_sim)
    # So cosine distance = distances^2 / 2
    # But for thresholding, we use the euclidean distances directly
    # (threshold was computed in the same space)

    # Flag unmatched spots (nearest neighbor too far)
    unmatched_mask = distances[:, 0] > distance_threshold

    return matched_conc_idx, distances, unmatched_mask, distance_threshold


def compute_matched_deltas(expression: np.ndarray,
                           disc_indices: np.ndarray,
                           matched_conc_idx: np.ndarray,
                           unmatched_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute expression differences between discordant spots and their matched concordant neighbors.

    Delta_ig = y_disc_g - mean(y_matched_conc_g)

    Args:
        expression: Full expression matrix (N_spots, N_genes)
        disc_indices: Indices of discordant spots
        matched_conc_idx: (N_disc, k) indices of matched concordant spots
        unmatched_mask: (N_disc,) boolean mask for unmatched spots

    Returns:
        deltas: (N_matched, N_genes) expression differences
        matched_disc_indices: Indices of the matched discordant spots
    """
    matched_mask = ~unmatched_mask
    matched_disc_idx = disc_indices[matched_mask]
    matched_conc = matched_conc_idx[matched_mask]

    expr_disc = expression[matched_disc_idx]
    # Mean expression across k concordant neighbors
    expr_conc_mean = np.mean(expression[matched_conc], axis=1)

    deltas = expr_disc - expr_conc_mean
    return deltas, matched_disc_idx


def matched_de(deltas: np.ndarray, gene_names: List[str]) -> pd.DataFrame:
    """One-sample Wilcoxon signed-rank test on matched delta values.

    Tests H0: median Delta_g = 0 for each gene.

    Args:
        deltas: (N_matched, N_genes) expression differences
        gene_names: Gene names

    Returns:
        DataFrame with: gene, pval, median_delta, mean_delta, effect_size, fdr
    """
    n_spots = deltas.shape[0]
    records = []

    for g_idx, gene in enumerate(gene_names):
        delta_g = deltas[:, g_idx]

        # Skip if all zeros
        nonzero = np.sum(delta_g != 0)
        if nonzero < 5:
            continue

        median_delta = np.median(delta_g)
        mean_delta = np.mean(delta_g)

        # Effect size: mean / sd
        sd = np.std(delta_g, ddof=1)
        effect_size = mean_delta / sd if sd > 0 else 0

        try:
            stat, pval = stats.wilcoxon(delta_g, alternative='two-sided')
        except ValueError:
            pval = 1.0

        records.append({
            'gene': gene,
            'pval': pval,
            'median_delta': median_delta,
            'mean_delta': mean_delta,
            'effect_size': effect_size,
            'n_matched': n_spots,
        })

    if not records:
        return pd.DataFrame(columns=['gene', 'pval', 'median_delta',
                                     'mean_delta', 'effect_size', 'n_matched', 'fdr'])

    df = pd.DataFrame(records)
    _, fdr, _, _ = multipletests(df['pval'].values, method='fdr_bh')
    df['fdr'] = fdr

    return df.sort_values('pval').reset_index(drop=True)


def matching_quality_report(match_distances: np.ndarray,
                            unmatched_mask: np.ndarray,
                            distance_threshold: float) -> dict:
    """Compute matching quality metrics.

    Args:
        match_distances: (N_disc, k) distances to matched neighbors
        unmatched_mask: (N_disc,) boolean mask
        distance_threshold: Tau threshold

    Returns:
        Dict with quality metrics
    """
    n_total = len(unmatched_mask)
    n_unmatched = int(np.sum(unmatched_mask))

    return {
        'n_discordant': n_total,
        'n_matched': n_total - n_unmatched,
        'n_unmatched': n_unmatched,
        'frac_unmatched': n_unmatched / n_total if n_total > 0 else 0,
        'distance_threshold': float(distance_threshold),
        'median_nn1_distance': float(np.median(match_distances[:, 0])),
        'mean_nn1_distance': float(np.mean(match_distances[:, 0])),
        'p90_nn1_distance': float(np.percentile(match_distances[:, 0], 90)),
    }

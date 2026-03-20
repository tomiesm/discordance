"""
Spatial statistics for discordance analysis.

Provides:
  - Spatial weight construction (KNN)
  - Moran's I with permutation-based inference
  - Level 2 (geometry-preserving) and Level 3 (compartment-conditioned) null models
  - Boundary ring assignment
"""

import numpy as np
import scipy.sparse as sp
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, Optional, Union
import warnings


def build_spatial_weights(
    coords: np.ndarray, n_neighbors: int = 6
) -> sp.csr_matrix:
    """Build row-standardized KNN spatial weight matrix (sparse).

    Args:
        coords: (N, 2) array of spatial coordinates.
        n_neighbors: Number of nearest neighbors (default 6 for Visium hexagonal grid).

    Returns:
        (N, N) sparse CSR weight matrix, row-standardized.
    """
    N = len(coords)
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='euclidean')
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords)

    # Build sparse adjacency (exclude self)
    rows = np.repeat(np.arange(N), n_neighbors)
    cols = indices[:, 1:].ravel()  # skip self
    data = np.ones(len(rows), dtype=np.float64)
    W = sp.csr_matrix((data, (rows, cols)), shape=(N, N))

    # Row-standardize
    row_sums = np.array(W.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    W = sp.diags(1.0 / row_sums) @ W

    return W


def morans_i(values: np.ndarray, W: Union[np.ndarray, sp.spmatrix]) -> float:
    """Compute Moran's I statistic.

    Uses z^T @ W @ z formulation — O(N*k) for sparse W with k neighbors.

    Args:
        values: (N,) array of values.
        W: (N, N) weight matrix (sparse or dense).

    Returns:
        Moran's I statistic.
    """
    N = len(values)
    z = values - values.mean()
    denom = np.sum(z ** 2)
    if denom == 0:
        return 0.0
    numer = z @ (W @ z)
    if sp.issparse(W):
        total_W = W.sum()
    else:
        total_W = np.sum(W)
    I = (N / total_W) * (numer / denom)
    return float(I)


def morans_i_permutation(
    values: np.ndarray,
    W: np.ndarray,
    n_permutations: int = 999,
    permutation_groups: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Tuple[float, float]:
    """Moran's I with permutation-based p-value.

    Args:
        values: (N,) array of values.
        W: (N, N) weight matrix.
        n_permutations: Number of random permutations.
        permutation_groups: Optional (N,) array of group labels.
            If None: Level 1 (random permutation).
            If provided: permute only within groups (Level 2/3).
        seed: Random seed.

    Returns:
        (I_observed, p_value).
    """
    rng = np.random.RandomState(seed)
    I_obs = morans_i(values, W)

    count_ge = 0
    for _ in range(n_permutations):
        if permutation_groups is None:
            perm = rng.permutation(len(values))
            perm_values = values[perm]
        else:
            perm_values = values.copy()
            for grp in np.unique(permutation_groups):
                mask = permutation_groups == grp
                idx = np.where(mask)[0]
                perm_idx = rng.permutation(idx)
                perm_values[idx] = values[perm_idx]

        I_perm = morans_i(perm_values, W)
        if I_perm >= I_obs:
            count_ge += 1

    p = (1 + count_ge) / (1 + n_permutations)
    return I_obs, p


def assign_boundary_rings(
    coords: np.ndarray, n_bins: int = 10, min_bin_size: int = 10
) -> np.ndarray:
    """Assign each spot to a distance-from-boundary ring.

    Uses the convex hull of spot positions as the tissue boundary.
    Spots farther from the boundary get higher ring indices.

    Args:
        coords: (N, 2) spatial coordinates.
        n_bins: Number of annular rings.
        min_bin_size: Merge bins with fewer spots.

    Returns:
        (N,) array of ring assignments (integers).
    """
    N = len(coords)
    if N < 4:
        return np.zeros(N, dtype=int)

    try:
        hull = ConvexHull(coords)
        hull_points = coords[hull.vertices]
    except Exception:
        # Degenerate geometry — all spots get same ring
        return np.zeros(N, dtype=int)

    # Compute distance from each spot to nearest hull edge
    distances = _point_to_hull_distances(coords, hull_points)

    # Bin into quantile rings
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(distances, percentiles[1:-1])
    rings = np.digitize(distances, bin_edges)

    # Merge small rings
    rings = _merge_small_groups(rings, min_bin_size)
    return rings


def _point_to_hull_distances(points: np.ndarray, hull_points: np.ndarray) -> np.ndarray:
    """Compute minimum distance from each point to the convex hull boundary."""
    n_hull = len(hull_points)
    N = len(points)
    distances = np.full(N, np.inf)

    for i in range(n_hull):
        p1 = hull_points[i]
        p2 = hull_points[(i + 1) % n_hull]
        # Distance from each point to line segment p1-p2
        seg_dists = _point_to_segment_distance(points, p1, p2)
        distances = np.minimum(distances, seg_dists)

    return distances


def _point_to_segment_distance(
    points: np.ndarray, p1: np.ndarray, p2: np.ndarray
) -> np.ndarray:
    """Distance from each point to line segment p1-p2."""
    d = p2 - p1
    d_sq = np.dot(d, d)
    if d_sq < 1e-10:
        return np.linalg.norm(points - p1, axis=1)

    t = np.clip(np.dot(points - p1, d) / d_sq, 0.0, 1.0)
    projections = p1 + np.outer(t, d)
    return np.linalg.norm(points - projections, axis=1)


def _merge_small_groups(labels: np.ndarray, min_size: int) -> np.ndarray:
    """Merge groups with fewer than min_size members into adjacent groups."""
    labels = labels.copy()
    changed = True
    while changed:
        changed = False
        unique = np.sort(np.unique(labels))
        if len(unique) <= 2:
            break
        for lab in unique:
            count = np.sum(labels == lab)
            if count < min_size:
                idx = np.where(unique == lab)[0][0]
                if idx == 0:
                    merge_target = unique[1]
                elif idx == len(unique) - 1:
                    merge_target = unique[-2]
                else:
                    merge_target = unique[idx - 1]
                labels[labels == lab] = merge_target
                changed = True
                break
    return labels

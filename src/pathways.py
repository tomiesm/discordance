"""
Pathway-level signed residual computation.

Provides:
  - GMT file parsing
  - Signed pathway residual aggregation
  - Gene-level studentized residuals (bin-based variance estimation)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


def load_gene_sets(gmt_path: str) -> Dict[str, List[str]]:
    """Parse GMT file into {pathway_name: [gene_list]}.

    GMT format: pathway_name<tab>description<tab>gene1<tab>gene2<tab>...

    Args:
        gmt_path: Path to .gmt file.

    Returns:
        Dictionary mapping pathway names to gene lists.
    """
    gene_sets = {}
    with open(gmt_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            name = parts[0]
            genes = [g for g in parts[2:] if g]  # skip description (parts[1])
            gene_sets[name] = genes
    return gene_sets


def compute_pathway_signed_residuals(
    residuals: np.ndarray,
    gene_names: List[str],
    gene_sets: Dict[str, List[str]],
    min_overlap: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute mean signed residual per pathway per spot.

    For each pathway P, computes:
        r_bar_i^(P) = mean of r_i^(g) for g in P ∩ HVGs

    Args:
        residuals: (N_spots, N_genes) signed residuals (target - prediction).
        gene_names: List of N_genes gene names matching residual columns.
        gene_sets: {pathway_name: [gene_list]} from load_gene_sets().
        min_overlap: Minimum genes overlapping with HVGs to include pathway.

    Returns:
        (scores_df, overlap_df):
          - scores_df: DataFrame with columns [pathway, mean_signed_residual]
              and N_spots rows per pathway.
          - overlap_df: DataFrame with columns [pathway, n_overlap, genes_used]
              one row per pathway.
    """
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    overlap_records = []
    pathway_scores = {}

    for pname, pgenes in gene_sets.items():
        overlap = [g for g in pgenes if g in gene_to_idx]
        overlap_records.append({
            'pathway': pname,
            'n_overlap': len(overlap),
            'n_total': len(pgenes),
            'genes_used': ','.join(sorted(overlap)),
        })
        if len(overlap) < min_overlap:
            continue

        gene_idx = [gene_to_idx[g] for g in overlap]
        pathway_resid = residuals[:, gene_idx]
        pathway_scores[pname] = pathway_resid.mean(axis=1)

    # Build scores DataFrame
    records = []
    for pname, scores in pathway_scores.items():
        for i, s in enumerate(scores):
            records.append({'spot_idx': i, 'pathway': pname, 'mean_signed_residual': s})

    scores_df = pd.DataFrame(records) if records else pd.DataFrame(
        columns=['spot_idx', 'pathway', 'mean_signed_residual']
    )
    overlap_df = pd.DataFrame(overlap_records)

    return scores_df, overlap_df


def compute_studentized_gene_residuals(
    residuals: np.ndarray,
    expression: np.ndarray,
    n_bins: int = 20,
    sigma_floor_percentile: float = 1.0,
) -> np.ndarray:
    """Per-gene studentized residuals using bin-based variance estimation.

    For each gene g:
      1. Bin spots by expression level y_i^(g)
      2. Compute SD of |r_i^(g)| within each bin -> sigma_hat(y)
      3. Studentize: t_i^(g) = r_i^(g) / sigma_hat(bin(y_i^(g)))

    Args:
        residuals: (N_spots, N_genes) signed residuals.
        expression: (N_spots, N_genes) expression values (targets).
        n_bins: Number of expression-level bins per gene.
        sigma_floor_percentile: Floor sigma at this percentile to avoid div-by-zero.

    Returns:
        (N_spots, N_genes) studentized residuals.
    """
    N, G = residuals.shape
    studentized = np.zeros_like(residuals, dtype=np.float64)

    actual_bins = min(n_bins, max(2, N // 10))

    for g in range(G):
        expr_g = expression[:, g]
        resid_g = residuals[:, g]

        # Bin by expression level
        percentiles = np.linspace(0, 100, actual_bins + 1)
        bin_edges = np.percentile(expr_g, percentiles[1:-1])
        bins = np.digitize(expr_g, bin_edges)

        # Compute sigma per bin
        sigma_per_bin = {}
        for b in np.unique(bins):
            mask = bins == b
            if mask.sum() < 3:
                sigma_per_bin[b] = np.nan
            else:
                sigma_per_bin[b] = np.std(np.abs(resid_g[mask]))

        # Fill NaN sigmas with global sigma
        global_sigma = np.std(np.abs(resid_g))
        for b in sigma_per_bin:
            if np.isnan(sigma_per_bin[b]) or sigma_per_bin[b] == 0:
                sigma_per_bin[b] = global_sigma

        # Apply sigma floor
        all_sigmas = np.array(list(sigma_per_bin.values()))
        sigma_floor = np.percentile(all_sigmas[all_sigmas > 0], sigma_floor_percentile) if np.any(all_sigmas > 0) else 1.0
        for b in sigma_per_bin:
            sigma_per_bin[b] = max(sigma_per_bin[b], sigma_floor)

        # Studentize
        for b in np.unique(bins):
            mask = bins == b
            studentized[mask, g] = resid_g[mask] / sigma_per_bin[b]

    return studentized


def compute_studentized_pathway_scores(
    studentized_residuals: np.ndarray,
    gene_names: List[str],
    gene_sets: Dict[str, List[str]],
    min_overlap: int = 3,
) -> Dict[str, np.ndarray]:
    """Compute studentized pathway scores from per-gene studentized residuals.

    t_bar_i^(P) = mean of t_i^(g) for g in P ∩ HVGs

    Args:
        studentized_residuals: (N_spots, N_genes) studentized residuals.
        gene_names: Gene names matching columns.
        gene_sets: {pathway: [genes]}.
        min_overlap: Minimum overlapping genes.

    Returns:
        {pathway_name: (N_spots,) scores} for pathways meeting overlap threshold.
    """
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    result = {}

    for pname, pgenes in gene_sets.items():
        overlap_idx = [gene_to_idx[g] for g in pgenes if g in gene_to_idx]
        if len(overlap_idx) < min_overlap:
            continue
        result[pname] = studentized_residuals[:, overlap_idx].mean(axis=1)

    return result

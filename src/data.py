"""
Data loading utilities for v3 IDC Xenium deep-dive with full gene panel.

Loads expression matrices from HEST h5ad files for specified samples
and gene panels.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import scanpy as sc


def load_v3_task(
    sample_ids: List[str],
    hest_dir: str,
    gene_list_path: str,
    normalize: bool = True,
) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    """
    Load expression for v3 samples from hest/st/*.h5ad using full gene panel.

    Args:
        sample_ids: List of sample IDs to load
        hest_dir: Path to HEST data directory
        gene_list_path: Path to gene_list_541.json
        normalize: Whether to apply log1p normalization

    Returns:
        expr_df: Expression DataFrame (n_spots, n_genes), indexed by {sample_id}_{barcode}
        gene_names: List of gene names from the panel
        spot_to_sample: Dict mapping spot_id -> sample_id
    """
    import json

    hest_dir = Path(hest_dir)

    with open(gene_list_path) as f:
        gene_names = json.load(f)

    all_expr = []
    spot_to_sample = {}

    for sample_id in sample_ids:
        adata_path = hest_dir / 'st' / f'{sample_id}.h5ad'
        if not adata_path.exists():
            raise FileNotFoundError(f"Missing h5ad: {adata_path}")

        adata = sc.read_h5ad(adata_path)

        # Subset to panel genes
        available_genes = [g for g in gene_names if g in adata.var_names]
        if len(available_genes) < len(gene_names):
            missing = set(gene_names) - set(available_genes)
            print(f"Warning: {sample_id} missing {len(missing)} genes: {list(missing)[:5]}")

        adata_subset = adata[:, available_genes]

        if normalize:
            sc.pp.log1p(adata_subset)

        expr_matrix = adata_subset.X
        if hasattr(expr_matrix, 'toarray'):
            expr_matrix = expr_matrix.toarray()

        # Prefix spot IDs with sample_id for cross-sample uniqueness
        raw_spot_ids = list(adata_subset.obs_names)
        prefixed_ids = [f"{sample_id}_{sid}" for sid in raw_spot_ids]

        for sid in prefixed_ids:
            spot_to_sample[sid] = sample_id

        sample_df = pd.DataFrame(
            expr_matrix,
            index=prefixed_ids,
            columns=available_genes,
        )
        all_expr.append(sample_df)

    expr_df = pd.concat(all_expr, axis=0)

    # Add any missing genes as zeros
    for gene in gene_names:
        if gene not in expr_df.columns:
            expr_df[gene] = 0.0

    # Ensure column order matches gene_names
    expr_df = expr_df[gene_names]

    return expr_df, gene_names, spot_to_sample

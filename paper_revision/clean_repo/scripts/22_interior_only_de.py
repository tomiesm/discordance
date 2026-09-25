#!/usr/bin/env python3
"""
Script 22: Interior-Only DE Reanalysis (Spatial Bleeding Control)

Addresses the concern that morpho-transcriptomic discordance is driven by
spatial bleeding at tissue compartment boundaries. Restricts analysis to
"morphologically interior" spots — spots whose k=6 spatial neighbors all
have similar UNI embeddings — and repeats the DE analysis. If the signature
persists in interior-only spots, it cannot be explained by spatial bleeding.

Output:
    outputs/phase3/de_interior/{cohort}/
        per_sample/{sample_id}_interior_de.csv
        per_sample/{sample_id}_interior_stats.json
        meta_de_interior.csv
        comparison_with_full.csv
    outputs/figures/supplementary/
        figS22_interior_only_de.pdf
        figS22_interior_only_de.png

Usage:
    python scripts/22_interior_only_de.py
    python scripts/22_interior_only_de.py --cohort discovery
    python scripts/22_interior_only_de.py --skip-figures
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import h5py

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize as l2_normalize

from src.utils import setup_logging, format_time
from src.de_analysis import partition_spots_by_discordance, wilcoxon_de, meta_de
from src.plotting import (
    setup_style, save_figure, annotate_r,
    COLORS, COHORT_LABELS, FULL_WIDTH,
)

# ============================================================
# Config
# ============================================================

K_SPATIAL = 6
THRESHOLD_PERCENTILE = 50.0  # median
ENCODER = "uni"  # Use UNI2-h for morphological homogeneity

COHORT_MAP = {
    "discovery": "biomarkers",
    "validation": "10x_janesick",
}


# ============================================================
# Data loading (from 08_de_analysis.py)
# ============================================================

def load_v3_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_expression_for_sample(sample_id, hest_dir, gene_list):
    import scanpy as sc
    adata = sc.read_h5ad(Path(hest_dir) / "st" / f"{sample_id}.h5ad")
    available = [g for g in gene_list if g in adata.var_names]
    adata = adata[:, available]
    sc.pp.log1p(adata)
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    barcodes = list(adata.obs.index)
    return X.astype(np.float32), available, barcodes


def load_discordance_for_sample(sample_id, scores_dir):
    path = scores_dir / f"{sample_id}_discordance.parquet"
    return pd.read_parquet(path)


def load_embeddings_for_sample(sample_id, encoder_name, embed_dir):
    path = embed_dir / sample_id / f"{encoder_name}_embeddings.h5"
    with h5py.File(path, "r") as f:
        embeddings = f["embeddings"][:]
        spot_ids = [s.decode() if isinstance(s, bytes) else s for s in f["spot_ids"][:]]
    return embeddings, spot_ids


def align_sample_data(disc_df, expression, gene_names, barcodes, embeddings, embed_spot_ids):
    expr_idx = {b: i for i, b in enumerate(barcodes)}
    embed_idx = {b: i for i, b in enumerate(embed_spot_ids)}
    ridge_cols = [c for c in disc_df.columns if c.startswith("D_cond_") and c.endswith("_ridge")]

    aligned_expr = []
    aligned_embed = []
    aligned_dcond = []
    aligned_barcodes = []
    aligned_x = []
    aligned_y = []

    for _, row in disc_df.iterrows():
        barcode = row["barcode"]
        if barcode not in expr_idx or barcode not in embed_idx:
            continue
        emb = embeddings[embed_idx[barcode]]
        if np.any(np.isnan(emb)):
            continue

        aligned_expr.append(expression[expr_idx[barcode]])
        aligned_embed.append(emb)
        aligned_dcond.append(np.mean([row[c] for c in ridge_cols]))
        aligned_barcodes.append(barcode)
        aligned_x.append(row["x"])
        aligned_y.append(row["y"])

    return (
        np.array(aligned_expr, dtype=np.float32),
        np.array(aligned_embed, dtype=np.float32),
        np.array(aligned_dcond, dtype=np.float64),
        aligned_barcodes,
        np.array(aligned_x, dtype=np.float64),
        np.array(aligned_y, dtype=np.float64),
    )


# ============================================================
# Interior spot classification
# ============================================================

def compute_interior_mask(coords, embeddings, k_spatial=6, threshold_percentile=50.0):
    """Identify morphologically interior spots.

    A spot is "interior" if the minimum cosine similarity between its
    embedding and those of its k spatial nearest neighbors exceeds a
    data-driven threshold (percentile of min-neighbor-cosine distribution).

    Args:
        coords: (N, 2) spatial coordinates.
        embeddings: (N, D) morphological embeddings (raw, will be L2-normalized).
        k_spatial: Number of spatial neighbors.
        threshold_percentile: Percentile of min-neighbor-cosine for threshold.

    Returns:
        interior_mask: (N,) boolean.
        min_neighbor_cosine: (N,) minimum cosine sim to spatial neighbors.
        threshold: The computed cosine similarity threshold.
    """
    N = coords.shape[0]

    # Build spatial KNN
    nn = NearestNeighbors(n_neighbors=k_spatial + 1, metric="euclidean")
    nn.fit(coords)
    _, indices = nn.kneighbors(coords)
    neighbor_indices = indices[:, 1:]  # (N, k) — exclude self

    # L2-normalize embeddings for cosine similarity
    emb_norm = l2_normalize(embeddings, norm="l2", axis=1)  # (N, D)

    # Vectorized cosine similarity: dot product of center with each neighbor
    neighbor_embs = emb_norm[neighbor_indices]  # (N, k, D)
    center_expanded = emb_norm[:, np.newaxis, :]  # (N, 1, D)
    cosine_sims = np.sum(neighbor_embs * center_expanded, axis=2)  # (N, k)
    min_neighbor_cosine = cosine_sims.min(axis=1)  # (N,)

    # Data-driven threshold
    threshold = np.percentile(min_neighbor_cosine, threshold_percentile)
    interior_mask = min_neighbor_cosine > threshold

    return interior_mask, min_neighbor_cosine, threshold


# ============================================================
# Per-sample interior DE
# ============================================================

def run_interior_de_for_sample(
    sample_id, config, cohort_name, scores_dir, logger
):
    """Run interior-only DE for a single sample.

    Returns dict with:
        'de_df': DataFrame from wilcoxon_de on interior subset
        'stats': dict of interior classification statistics
    """
    hest_dir = config["hest_dir"]
    embed_dir = Path(config["output_dir"]) / "embeddings"
    gene_list_path = Path(config["v3_data_dir"]) / f"gene_list_{cohort_name}.json"

    with open(gene_list_path) as f:
        gene_list = json.load(f)

    # Load data
    expression, gene_names, barcodes = load_expression_for_sample(
        sample_id, hest_dir, gene_list
    )
    disc_df = load_discordance_for_sample(sample_id, scores_dir)
    embeddings, embed_spot_ids = load_embeddings_for_sample(
        sample_id, ENCODER, embed_dir
    )

    # Align (includes x, y coordinates)
    (expr_aligned, embed_aligned, dcond_aligned,
     barcodes_aligned, x_aligned, y_aligned) = align_sample_data(
        disc_df, expression, gene_names, barcodes, embeddings, embed_spot_ids
    )

    n_total = len(barcodes_aligned)
    logger.info(f"  {sample_id}: {n_total} aligned spots, {len(gene_names)} genes")

    # Compute interior mask
    coords = np.column_stack([x_aligned, y_aligned])
    interior_mask, min_cos, threshold = compute_interior_mask(
        coords, embed_aligned, k_spatial=K_SPATIAL,
        threshold_percentile=THRESHOLD_PERCENTILE,
    )
    n_interior = interior_mask.sum()
    frac_interior = n_interior / n_total
    logger.info(
        f"    Interior: {n_interior}/{n_total} ({frac_interior:.1%}), "
        f"threshold={threshold:.4f}"
    )

    # Filter to interior spots
    expr_interior = expr_aligned[interior_mask]
    dcond_interior = dcond_aligned[interior_mask]

    # Repartition Q4/Q1 within interior subset
    disc_mask, conc_mask = partition_spots_by_discordance(dcond_interior)
    n_disc = disc_mask.sum()
    n_conc = conc_mask.sum()
    logger.info(f"    Interior Q4={n_disc}, Q1={n_conc}")

    # Run DE on interior spots
    de_df = wilcoxon_de(expr_interior, gene_names, disc_mask, conc_mask,
                        min_frac_expressed=0.10)
    n_sig = (de_df["fdr"] < 0.05).sum() if not de_df.empty else 0
    logger.info(f"    Interior DE: {len(de_df)} genes tested, {n_sig} FDR<0.05")

    stats = {
        "sample_id": sample_id,
        "n_total": int(n_total),
        "n_interior": int(n_interior),
        "frac_interior": float(frac_interior),
        "threshold": float(threshold),
        "n_disc_interior": int(n_disc),
        "n_conc_interior": int(n_conc),
        "n_sig_fdr05": int(n_sig),
        "min_cosine_median": float(np.median(min_cos)),
        "min_cosine_mean": float(np.mean(min_cos)),
        "min_cosine_p25": float(np.percentile(min_cos, 25)),
        "min_cosine_p75": float(np.percentile(min_cos, 75)),
    }

    return {"de_df": de_df, "stats": stats}


# ============================================================
# Figure S22
# ============================================================

def generate_figure_s22(comparisons, interior_stats, fig_dir, logger):
    """Generate Supplementary Figure S22: Interior-only DE reanalysis.

    3-panel figure:
      (a) Discovery: full vs interior log2FC scatter
      (b) Validation: full vs interior log2FC scatter
      (c) Per-sample interior fraction bar chart
    """
    import matplotlib.pyplot as plt
    from scipy import stats as sp_stats

    setup_style()

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH * 0.75, FULL_WIDTH * 0.35))

    cohort_keys = ["discovery", "validation"]
    cohort_colors = {
        "discovery": COLORS["discovery"],
        "validation": COLORS["validation"],
    }

    # Panels (a) and (b): log2FC scatter
    for idx, cohort_key in enumerate(cohort_keys):
        ax = axes[idx]
        comp = comparisons[cohort_key]

        # Significance categories (reproducibility >= 0.5)
        both = (comp["full_repro"] >= 0.5) & (comp["interior_repro"] >= 0.5)
        full_only = (comp["full_repro"] >= 0.5) & (comp["interior_repro"] < 0.5)
        int_only = (comp["full_repro"] < 0.5) & (comp["interior_repro"] >= 0.5)
        neither = ~(both | full_only | int_only)

        # Plot "Both" first (background), then minority categories on top
        ax.scatter(comp.loc[both, "full_log2fc"],
                   comp.loc[both, "interior_log2fc"],
                   c=cohort_colors[cohort_key], s=4, alpha=0.4, zorder=1,
                   label=f"Both (n={both.sum()})", rasterized=True)

        ax.scatter(comp.loc[neither, "full_log2fc"],
                   comp.loc[neither, "interior_log2fc"],
                   c=COLORS["neutral"], s=6, alpha=0.6, zorder=2,
                   label=f"Neither (n={neither.sum()})")

        if full_only.sum() > 0:
            ax.scatter(comp.loc[full_only, "full_log2fc"],
                       comp.loc[full_only, "interior_log2fc"],
                       c="#E6AB02", s=8, alpha=0.8, zorder=3,
                       marker="v", label=f"Full only (n={full_only.sum()})")

        if int_only.sum() > 0:
            ax.scatter(comp.loc[int_only, "full_log2fc"],
                       comp.loc[int_only, "interior_log2fc"],
                       c="#7570B3", s=8, alpha=0.8, zorder=3,
                       marker="^", label=f"Interior only (n={int_only.sum()})")

        # Identity line
        lims = [
            min(comp["full_log2fc"].min(), comp["interior_log2fc"].min()) - 0.1,
            max(comp["full_log2fc"].max(), comp["interior_log2fc"].max()) + 0.1,
        ]
        ax.plot(lims, lims, "k--", lw=0.5, alpha=0.5, zorder=0)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        # Correlation
        r, p = sp_stats.pearsonr(comp["full_log2fc"], comp["interior_log2fc"])
        annotate_r(ax, r, p, pos="upper left", fontsize=5)

        ax.set_xlabel("Full DE median log$_2$FC")
        ax.set_ylabel("Interior-only DE median log$_2$FC")
        ax.set_title(COHORT_LABELS.get(cohort_key, cohort_key), fontsize=8)
        ax.legend(fontsize=4.5, loc="lower right", framealpha=0.8,
                  handletextpad=0.3, borderpad=0.3)

    plt.tight_layout()

    out_path = fig_dir / "figS22_interior_only_de"
    save_figure(fig, out_path, formats=("pdf", "png"), dpi=300)
    logger.info(f"  Saved figure: {out_path}.pdf/.png")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Interior-only DE reanalysis")
    parser.add_argument("--cohort", choices=["discovery", "validation"],
                        help="Run only one cohort")
    parser.add_argument("--skip-figures", action="store_true",
                        help="Skip figure generation")
    args = parser.parse_args()

    logger = setup_logging("INFO")
    config = load_v3_config()
    t0 = time.time()

    logger.info("=" * 60)
    logger.info("Interior-Only DE Reanalysis (Spatial Bleeding Control)")
    logger.info("=" * 60)

    output_base = Path(config["output_dir"]) / "phase3" / "de_interior"
    fig_dir = Path(config["output_dir"]) / "figures" / "supplementary"
    fig_dir.mkdir(parents=True, exist_ok=True)

    cohorts_to_run = [args.cohort] if args.cohort else ["discovery", "validation"]

    all_comparisons = {}
    all_interior_stats = {}

    for cohort_key in cohorts_to_run:
        cohort_name = COHORT_MAP[cohort_key]
        samples = config["cohorts"][cohort_key]["samples"]
        scores_dir = Path(config["output_dir"]) / "phase2" / "scores" / cohort_name

        logger.info(f"\n{'='*40}")
        logger.info(f"Cohort: {cohort_key} ({cohort_name}), {len(samples)} samples")
        logger.info(f"{'='*40}")

        out_dir = output_base / cohort_name
        per_sample_dir = out_dir / "per_sample"
        per_sample_dir.mkdir(parents=True, exist_ok=True)

        sample_de_results = []
        sample_stats = []

        for sample_id in samples:
            result = run_interior_de_for_sample(
                sample_id, config, cohort_name, scores_dir, logger
            )
            de_df = result["de_df"]
            stats = result["stats"]

            # Save per-sample results
            de_df.to_csv(per_sample_dir / f"{sample_id}_interior_de.csv", index=False)
            with open(per_sample_dir / f"{sample_id}_interior_stats.json", "w") as f:
                json.dump(stats, f, indent=2)

            sample_de_results.append(de_df)
            sample_stats.append(stats)

        # Meta-DE aggregation
        meta_interior = meta_de(sample_de_results)
        meta_interior.to_csv(out_dir / "meta_de_interior.csv", index=False)
        logger.info(f"\n  Meta-DE interior: {len(meta_interior)} genes")

        n_repro = (meta_interior["reproducibility"] >= 0.5).sum()
        logger.info(f"  Reproducible (>=50% samples): {n_repro}")

        # Compare with full DE
        full_meta_path = (Path(config["output_dir"]) / "phase3" / "de" /
                          cohort_name / "meta_de_unmatched.csv")
        if full_meta_path.exists():
            meta_full = pd.read_csv(full_meta_path)

            comp = meta_full[["gene", "median_log2fc", "reproducibility"]].rename(
                columns={"median_log2fc": "full_log2fc",
                          "reproducibility": "full_repro"}
            ).merge(
                meta_interior[["gene", "median_log2fc", "reproducibility"]].rename(
                    columns={"median_log2fc": "interior_log2fc",
                              "reproducibility": "interior_repro"}
                ),
                on="gene", how="inner"
            )
            comp.to_csv(out_dir / "comparison_with_full.csv", index=False)

            from scipy.stats import pearsonr, spearmanr
            r_p, p_p = pearsonr(comp["full_log2fc"], comp["interior_log2fc"])
            r_s, p_s = spearmanr(comp["full_log2fc"], comp["interior_log2fc"])
            logger.info(f"  Correlation (full vs interior log2FC):")
            logger.info(f"    Pearson  r={r_p:.4f}, p={p_p:.2e}")
            logger.info(f"    Spearman rho={r_s:.4f}, p={p_s:.2e}")

            # Gene-level concordance
            both_repro = ((comp["full_repro"] >= 0.5) &
                          (comp["interior_repro"] >= 0.5)).sum()
            full_only = ((comp["full_repro"] >= 0.5) &
                         (comp["interior_repro"] < 0.5)).sum()
            int_only = ((comp["full_repro"] < 0.5) &
                        (comp["interior_repro"] >= 0.5)).sum()
            logger.info(f"  Gene concordance: both={both_repro}, "
                        f"full_only={full_only}, interior_only={int_only}")

            all_comparisons[cohort_key] = comp
        else:
            logger.warning(f"  Full DE not found: {full_meta_path}")

        all_interior_stats[cohort_key] = sample_stats

        # Summary stats
        mean_frac = np.mean([s["frac_interior"] for s in sample_stats])
        logger.info(f"  Mean interior fraction: {mean_frac:.1%}")

    # Generate figure
    if not args.skip_figures and len(all_comparisons) == 2:
        logger.info("\n=== Generating Figure S22 ===")
        generate_figure_s22(all_comparisons, all_interior_stats, fig_dir, logger)

    elapsed = time.time() - t0
    logger.info(f"\nTotal time: {format_time(elapsed)}")
    logger.info("Done.")


if __name__ == "__main__":
    main()

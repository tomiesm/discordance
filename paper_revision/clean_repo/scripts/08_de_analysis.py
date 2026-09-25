#!/usr/bin/env python3
"""
Script 08: Differential Expression Analysis (Phase 3A)

For each sample and config, partition spots into discordant (top 25%) and
concordant (bottom 25%) by D_cond, then run:
  1. Unmatched Wilcoxon rank-sum DE across all panel genes
  2. Morphology-matched DE (k=5 cosine neighbors in embedding space)
  3. Jaccard overlap between matched and unmatched DE gene sets
  4. Meta-DE aggregation across samples per cohort
  5. Cross-encoder Jaccard comparison

Uses Ridge regressor averaged across 3 encoders as the primary config.

Output:
    outputs/phase3/de/{cohort}/
        per_sample/{sample_id}_unmatched_de.csv
        per_sample/{sample_id}_matched_de.csv
        per_sample/{sample_id}_matching_quality.json
        meta_de_unmatched.csv
        meta_de_matched.csv
        cross_encoder_jaccard.csv
        de_summary.json

Usage:
    python scripts/08_de_analysis.py
    python scripts/08_de_analysis.py --cohort discovery
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import anndata as ad

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.utils import setup_logging, format_time
from src.de_analysis import (
    partition_spots_by_discordance,
    wilcoxon_de,
    meta_de,
    cross_encoder_jaccard,
)
from src.matching import (
    morphology_match,
    compute_matched_deltas,
    matched_de,
    matching_quality_report,
)


def load_v3_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_expression_for_sample(sample_id, hest_dir, gene_list):
    """Load log1p-normalized expression for a single sample, subset to gene_list."""
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
    """Load discordance parquet for a sample."""
    path = scores_dir / f"{sample_id}_discordance.parquet"
    return pd.read_parquet(path)


def load_embeddings_for_sample(sample_id, encoder_name, embed_dir):
    """Load embeddings for a single sample."""
    path = embed_dir / sample_id / f"{encoder_name}_embeddings.h5"
    with h5py.File(path, "r") as f:
        embeddings = f["embeddings"][:]
        spot_ids = [s.decode() if isinstance(s, bytes) else s for s in f["spot_ids"][:]]
    return embeddings, spot_ids


def align_sample_data(disc_df, expression, gene_names, barcodes, embeddings, embed_spot_ids):
    """Align discordance scores, expression, and embeddings by barcode.

    Returns aligned arrays and the D_cond values for the ridge-averaged config.
    """
    # Build lookup for expression and embeddings
    expr_idx = {b: i for i, b in enumerate(barcodes)}
    embed_idx = {b: i for i, b in enumerate(embed_spot_ids)}

    # Use D_cond averaged across 3 encoders with ridge
    ridge_cols = [c for c in disc_df.columns if c.startswith("D_cond_") and c.endswith("_ridge")]

    aligned_expr = []
    aligned_embed = []
    aligned_dcond = []
    aligned_barcodes = []

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

    return (
        np.array(aligned_expr, dtype=np.float32),
        np.array(aligned_embed, dtype=np.float32),
        np.array(aligned_dcond, dtype=np.float64),
        aligned_barcodes,
    )


def run_de_for_sample(
    sample_id, config, cohort_name, encoder_name, scores_dir, logger
):
    """Run unmatched + matched DE for a single sample."""
    hest_dir = config["hest_dir"]
    embed_dir = Path(config["output_dir"]) / "embeddings"
    gene_list_path = Path(config["v3_data_dir"]) / f"gene_list_{cohort_name}.json"
    phase3_config = config.get("phase3", {})
    top_q = phase3_config.get("de_quantile_top", 0.75)
    bot_q = phase3_config.get("de_quantile_bottom", 0.25)
    match_k = 5  # Default k for morphology matching

    with open(gene_list_path) as f:
        gene_list = json.load(f)

    # Load data
    expression, gene_names, barcodes = load_expression_for_sample(
        sample_id, hest_dir, gene_list
    )
    disc_df = load_discordance_for_sample(sample_id, scores_dir)
    embeddings, embed_spot_ids = load_embeddings_for_sample(
        sample_id, encoder_name, embed_dir
    )

    # Align all data sources
    expr_aligned, embed_aligned, dcond_aligned, barcodes_aligned = align_sample_data(
        disc_df, expression, gene_names, barcodes, embeddings, embed_spot_ids
    )

    logger.info(
        f"  {sample_id}: {len(barcodes_aligned)} aligned spots, "
        f"{len(gene_names)} genes"
    )

    # Partition by discordance
    disc_mask, conc_mask = partition_spots_by_discordance(
        dcond_aligned, top_quantile=top_q, bottom_quantile=bot_q
    )
    n_disc = disc_mask.sum()
    n_conc = conc_mask.sum()
    logger.info(f"    Discordant: {n_disc}, Concordant: {n_conc}")

    # 1. Unmatched DE
    unmatched_df = wilcoxon_de(
        expr_aligned, gene_names, disc_mask, conc_mask, min_frac_expressed=0.10
    )
    n_sig_unmatched = (unmatched_df["fdr"] < 0.05).sum() if not unmatched_df.empty else 0
    logger.info(f"    Unmatched DE: {len(unmatched_df)} genes tested, {n_sig_unmatched} FDR<0.05")

    # 2. Matched DE
    disc_indices = np.where(disc_mask)[0]
    conc_indices = np.where(conc_mask)[0]

    matched_conc_idx, match_distances, unmatched_mask, dist_threshold = morphology_match(
        embed_aligned, disc_indices, conc_indices, k=match_k
    )

    quality = matching_quality_report(match_distances, unmatched_mask, dist_threshold)
    logger.info(
        f"    Matching: {quality['n_matched']}/{quality['n_discordant']} matched "
        f"({quality['frac_unmatched']:.1%} unmatched)"
    )

    deltas, matched_disc_idx = compute_matched_deltas(
        expr_aligned, disc_indices, matched_conc_idx, unmatched_mask
    )

    matched_df = matched_de(deltas, gene_names)
    n_sig_matched = (matched_df["fdr"] < 0.05).sum() if not matched_df.empty else 0
    logger.info(f"    Matched DE: {len(matched_df)} genes tested, {n_sig_matched} FDR<0.05")

    # 3. Jaccard between unmatched and matched significant genes
    if not unmatched_df.empty and not matched_df.empty:
        sig_unmatched = set(unmatched_df[unmatched_df["fdr"] < 0.05]["gene"])
        sig_matched = set(matched_df[matched_df["fdr"] < 0.05]["gene"])
        if sig_unmatched or sig_matched:
            union = sig_unmatched | sig_matched
            intersection = sig_unmatched & sig_matched
            jaccard = len(intersection) / len(union) if union else 0
        else:
            jaccard = 0
    else:
        jaccard = 0

    logger.info(f"    Unmatched-Matched Jaccard: {jaccard:.3f}")

    return {
        "unmatched_de": unmatched_df,
        "matched_de": matched_df,
        "matching_quality": quality,
        "n_disc": int(n_disc),
        "n_conc": int(n_conc),
        "n_sig_unmatched": int(n_sig_unmatched),
        "n_sig_matched": int(n_sig_matched),
        "jaccard": float(jaccard),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 3A: DE analysis")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--cohort", type=str, default=None)
    parser.add_argument("--encoder", type=str, default="uni",
                        help="Encoder for embedding-based matching")
    args = parser.parse_args()

    config = load_v3_config(args.config)

    log_dir = Path(config["output_dir"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(
        log_level=config.get("log_level", "INFO"),
        log_file=str(log_dir / "08_de_analysis.log"),
    )

    logger.info("=" * 60)
    logger.info("Phase 3A: Differential Expression Analysis")
    logger.info("=" * 60)

    cohort_keys = [args.cohort] if args.cohort else ["discovery", "validation"]
    overall_start = time.time()

    for cohort_key in cohort_keys:
        cohort_config = config["cohorts"][cohort_key]
        cohort_name = cohort_config["name"]
        samples = cohort_config["samples"]
        scores_dir = Path(config["output_dir"]) / "phase2" / "scores" / cohort_name
        out_dir = Path(config["output_dir"]) / "phase3" / "de" / cohort_name
        sample_dir = out_dir / "per_sample"
        sample_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\nCohort: {cohort_name} ({cohort_key}), {len(samples)} samples")

        # Run per-sample DE
        all_unmatched_de = []
        all_matched_de = []
        summary_records = []

        for sample_id in samples:
            results = run_de_for_sample(
                sample_id, config, cohort_name, args.encoder, scores_dir, logger
            )

            # Save per-sample results
            results["unmatched_de"].to_csv(
                sample_dir / f"{sample_id}_unmatched_de.csv", index=False
            )
            results["matched_de"].to_csv(
                sample_dir / f"{sample_id}_matched_de.csv", index=False
            )
            with open(sample_dir / f"{sample_id}_matching_quality.json", "w") as f:
                json.dump(results["matching_quality"], f, indent=2)

            all_unmatched_de.append(results["unmatched_de"])
            all_matched_de.append(results["matched_de"])
            summary_records.append({
                "sample_id": sample_id,
                "n_disc": results["n_disc"],
                "n_conc": results["n_conc"],
                "n_sig_unmatched": results["n_sig_unmatched"],
                "n_sig_matched": results["n_sig_matched"],
                "jaccard": results["jaccard"],
            })

        # Meta-DE aggregation
        phase3_config = config.get("phase3", {})
        fdr_thresh = phase3_config.get("de_fdr_threshold", 0.05)
        log2fc_thresh = phase3_config.get("de_log2fc_threshold", 0.25)

        meta_unmatched = meta_de(all_unmatched_de, fdr_threshold=fdr_thresh,
                                 log2fc_threshold=log2fc_thresh)
        # Adapt matched DE DataFrames for meta_de (which expects 'log2fc' and 'cohens_d')
        adapted_matched = []
        for mdf in all_matched_de:
            if mdf.empty:
                adapted_matched.append(mdf)
                continue
            m = mdf.copy()
            m["log2fc"] = m["mean_delta"]  # Use mean_delta as log2fc proxy
            m["cohens_d"] = m["effect_size"]
            adapted_matched.append(m)

        meta_matched = meta_de(adapted_matched, fdr_threshold=fdr_thresh,
                               log2fc_threshold=log2fc_thresh)

        meta_unmatched.to_csv(out_dir / "meta_de_unmatched.csv", index=False)
        meta_matched.to_csv(out_dir / "meta_de_matched.csv", index=False)

        n_repro_unmatched = (meta_unmatched["reproducibility"] >= 0.5).sum() if not meta_unmatched.empty else 0
        n_repro_matched = (meta_matched["reproducibility"] >= 0.5).sum() if not meta_matched.empty else 0
        logger.info(f"\n  Meta-DE: {n_repro_unmatched} genes reproducible (>=50% samples) unmatched")
        logger.info(f"  Meta-DE: {n_repro_matched} genes reproducible (>=50% samples) matched")

        # Cross-encoder comparison: run meta-DE per encoder
        logger.info("\n  Cross-encoder comparison...")
        encoder_names = [e["name"] for e in config["encoders"]]
        encoder_meta = {}

        for enc_name in encoder_names:
            # For each encoder, use D_cond from that encoder's ridge config
            enc_unmatched_de = []
            for sample_id in samples:
                disc_df = load_discordance_for_sample(sample_id, scores_dir)
                dcond_col = f"D_cond_{enc_name}_ridge"
                if dcond_col not in disc_df.columns:
                    continue

                gene_list_path = Path(config["v3_data_dir"]) / f"gene_list_{cohort_name}.json"
                with open(gene_list_path) as f:
                    gene_list = json.load(f)

                expression, gene_names, barcodes = load_expression_for_sample(
                    sample_id, config["hest_dir"], gene_list
                )

                # Align barcodes with discordance
                barcode_to_idx = {b: i for i, b in enumerate(barcodes)}
                valid_mask = disc_df["barcode"].isin(barcode_to_idx)
                disc_subset = disc_df[valid_mask]

                expr_indices = [barcode_to_idx[b] for b in disc_subset["barcode"]]
                expr_aligned = expression[expr_indices]
                dcond = disc_subset[dcond_col].values

                disc_mask, conc_mask = partition_spots_by_discordance(dcond)
                de_df = wilcoxon_de(expr_aligned, gene_names, disc_mask, conc_mask)
                enc_unmatched_de.append(de_df)

            encoder_meta[enc_name] = meta_de(enc_unmatched_de, fdr_threshold=fdr_thresh,
                                              log2fc_threshold=log2fc_thresh)

        jaccard_df = cross_encoder_jaccard(encoder_meta, top_n=50)
        jaccard_df.to_csv(out_dir / "cross_encoder_jaccard.csv", index=False)
        logger.info(f"  Cross-encoder Jaccard (top 50):")
        for _, row in jaccard_df.iterrows():
            logger.info(f"    {row['encoder1']} vs {row['encoder2']}: {row['jaccard']:.3f}")

        # Save summary
        summary = {
            "cohort": cohort_name,
            "n_samples": len(samples),
            "encoder_for_matching": args.encoder,
            "per_sample": summary_records,
            "meta_n_reproducible_unmatched": int(n_repro_unmatched),
            "meta_n_reproducible_matched": int(n_repro_matched),
            "cross_encoder_jaccard": jaccard_df.to_dict(orient="records") if not jaccard_df.empty else [],
        }

        # Add top reproducible genes
        if not meta_unmatched.empty:
            top_genes = meta_unmatched.head(20)[["gene", "reproducibility", "median_log2fc", "mean_cohens_d"]].to_dict(orient="records")
            summary["top_20_unmatched_genes"] = top_genes
        if not meta_matched.empty:
            top_genes = meta_matched.head(20)[["gene", "reproducibility", "median_log2fc", "mean_cohens_d"]].to_dict(orient="records")
            summary["top_20_matched_genes"] = top_genes

        with open(out_dir / "de_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    total_time = time.time() - overall_start
    logger.info(f"\nTotal time: {format_time(total_time)}")
    logger.info("Done.")


if __name__ == "__main__":
    main()

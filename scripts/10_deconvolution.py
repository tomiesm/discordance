#!/usr/bin/env python3
"""
Script 10: Cell Type Deconvolution Analysis (Phase 3C)

Two approaches:
  1. Signature-based scoring: mean expression of marker genes per cell type
     (available for all 18 samples)
  2. CellViT nuclear morphometry: per-spot nuclear features from segmentation
     (available for 4 samples: TENX99, TENX95, NCBI785, NCBI783)

For each sample:
  - Compute cell type scores (signature-based)
  - Compare scores between discordant vs concordant spots (morphology-matched, k=5)
  - Cohen's d per cell type per sample
  - If CellViT data available: nuclear morphometry comparison

Tests v2 IDC finding: immune-depleted, epithelial-enriched in discordant spots.

Output:
    outputs/phase3/deconvolution/{cohort}/
        per_sample/{sample_id}_celltype_scores.csv
        per_sample/{sample_id}_celltype_de.csv
        per_sample/{sample_id}_cellvit_morphometry.csv  (if available)
        celltype_summary.csv        (cross-sample Cohen's d per cell type)
        cellvit_summary.csv         (cross-sample nuclear morphometry)
        gene_availability.csv       (which signature genes are in panel)
        deconvolution_summary.json

Usage:
    python scripts/10_deconvolution.py
    python scripts/10_deconvolution.py --cohort validation
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
from scipy import stats
from src.utils import setup_logging, format_time
from src.deconvolution import (
    get_available_signatures,
    compute_celltype_scores,
    gene_availability_report,
    find_cellvit_file,
    extract_nuclear_morphometry,
)
from src.de_analysis import partition_spots_by_discordance
from src.matching import morphology_match, matching_quality_report


def load_v3_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_expression_for_sample(sample_id, hest_dir, gene_list):
    """Load log1p-normalized expression for a single sample."""
    import scanpy as sc
    adata = sc.read_h5ad(Path(hest_dir) / "st" / f"{sample_id}.h5ad")
    available = [g for g in gene_list if g in adata.var_names]
    adata = adata[:, available]
    sc.pp.log1p(adata)
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    barcodes = list(adata.obs.index)
    coords = adata.obsm["spatial"]
    return X.astype(np.float32), available, barcodes, coords


def celltype_de_test(scores_df, disc_mask, conc_mask):
    """Compare cell type scores between discordant and concordant spots."""
    records = []
    for cell_type in scores_df.columns:
        vals_disc = scores_df[cell_type].values[disc_mask]
        vals_conc = scores_df[cell_type].values[conc_mask]

        mean_disc = np.mean(vals_disc)
        mean_conc = np.mean(vals_conc)

        pooled_var = ((np.var(vals_disc, ddof=1) * (len(vals_disc) - 1) +
                       np.var(vals_conc, ddof=1) * (len(vals_conc) - 1)) /
                      (len(vals_disc) + len(vals_conc) - 2))
        pooled_sd = np.sqrt(pooled_var) if pooled_var > 0 else 1e-6
        cohens_d = (mean_disc - mean_conc) / pooled_sd

        try:
            stat, pval = stats.mannwhitneyu(vals_disc, vals_conc, alternative="two-sided")
        except ValueError:
            pval = 1.0

        records.append({
            "cell_type": cell_type,
            "mean_disc": float(mean_disc),
            "mean_conc": float(mean_conc),
            "log2fc": float(np.log2((mean_disc + 1e-6) / (mean_conc + 1e-6))),
            "cohens_d": float(cohens_d),
            "pval": float(pval),
            "n_disc": len(vals_disc),
            "n_conc": len(vals_conc),
        })

    df = pd.DataFrame(records)
    if not df.empty:
        from statsmodels.stats.multitest import multipletests
        _, fdr, _, _ = multipletests(df["pval"], method="fdr_bh")
        df["fdr"] = fdr

    return df


def celltype_matched_de_test(scores_df, disc_indices, matched_conc_idx, unmatched_mask):
    """Compare cell type scores using morphology-matched controls."""
    matched_mask = ~unmatched_mask
    matched_disc = disc_indices[matched_mask]
    matched_conc = matched_conc_idx[matched_mask]  # (N_matched, k)

    records = []
    for cell_type in scores_df.columns:
        vals = scores_df[cell_type].values

        disc_vals = vals[matched_disc]
        # Mean across k matched concordant neighbors
        conc_vals = np.mean(vals[matched_conc], axis=1)

        deltas = disc_vals - conc_vals
        nonzero = np.sum(deltas != 0)
        if nonzero < 5:
            continue

        mean_delta = np.mean(deltas)
        sd_delta = np.std(deltas, ddof=1)
        effect_size = mean_delta / sd_delta if sd_delta > 0 else 0

        try:
            stat, pval = stats.wilcoxon(deltas, alternative="two-sided")
        except ValueError:
            pval = 1.0

        records.append({
            "cell_type": cell_type,
            "mean_delta": float(mean_delta),
            "median_delta": float(np.median(deltas)),
            "effect_size": float(effect_size),
            "pval": float(pval),
            "n_matched": len(deltas),
        })

    df = pd.DataFrame(records)
    if not df.empty:
        from statsmodels.stats.multitest import multipletests
        _, fdr, _, _ = multipletests(df["pval"], method="fdr_bh")
        df["fdr"] = fdr

    return df


def main():
    parser = argparse.ArgumentParser(description="Phase 3C: Deconvolution analysis")
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
        log_file=str(log_dir / "10_deconvolution.log"),
    )

    logger.info("=" * 60)
    logger.info("Phase 3C: Cell Type Deconvolution Analysis")
    logger.info("=" * 60)

    cohort_keys = [args.cohort] if args.cohort else ["discovery", "validation"]
    cellvit_dir = Path(config["hest_dir"]) / "cellvit_seg"
    overall_start = time.time()

    for cohort_key in cohort_keys:
        cohort_config = config["cohorts"][cohort_key]
        cohort_name = cohort_config["name"]
        samples = cohort_config["samples"]
        scores_dir = Path(config["output_dir"]) / "phase2" / "scores" / cohort_name
        embed_dir = Path(config["output_dir"]) / "embeddings"
        out_dir = Path(config["output_dir"]) / "phase3" / "deconvolution" / cohort_name
        sample_out_dir = out_dir / "per_sample"
        sample_out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\nCohort: {cohort_name} ({cohort_key}), {len(samples)} samples")

        # Load gene list for this cohort
        gene_list_path = Path(config["v3_data_dir"]) / f"gene_list_{cohort_name}.json"
        with open(gene_list_path) as f:
            gene_list = json.load(f)

        # Gene availability report
        avail_report = gene_availability_report(gene_list)
        avail_report.to_csv(out_dir / "gene_availability.csv", index=False)
        available_sigs = get_available_signatures(gene_list)
        logger.info(f"  Available cell types: {list(available_sigs.keys())}")
        for ct, genes in available_sigs.items():
            logger.info(f"    {ct}: {genes}")

        all_unmatched_de = []
        all_matched_de = []
        cellvit_records = []

        for sample_id in samples:
            logger.info(f"\n  Sample: {sample_id}")

            # Load expression
            expression, gene_names, barcodes, spatial_coords = load_expression_for_sample(
                sample_id, config["hest_dir"], gene_list
            )

            # Compute cell type scores
            ct_scores, used_genes = compute_celltype_scores(
                expression, gene_names, min_genes=2
            )
            logger.info(f"    {len(barcodes)} spots, {len(ct_scores.columns)} cell types scored")

            # Load discordance
            disc_df = pd.read_parquet(scores_dir / f"{sample_id}_discordance.parquet")
            ridge_cols = [c for c in disc_df.columns if c.startswith("D_cond_") and c.endswith("_ridge")]
            disc_df["D_cond_ridge_mean"] = disc_df[ridge_cols].mean(axis=1)

            # Align by barcode
            barcode_to_idx = {b: i for i, b in enumerate(barcodes)}
            disc_barcode_to_dcond = dict(zip(disc_df["barcode"], disc_df["D_cond_ridge_mean"]))

            # Find common barcodes
            common_barcodes = [b for b in barcodes if b in disc_barcode_to_dcond]
            common_idx = [barcode_to_idx[b] for b in common_barcodes]
            dcond_aligned = np.array([disc_barcode_to_dcond[b] for b in common_barcodes])

            ct_scores_aligned = ct_scores.iloc[common_idx].reset_index(drop=True)
            expr_aligned = expression[common_idx]

            # Save cell type scores
            ct_out = ct_scores_aligned.copy()
            ct_out.insert(0, "barcode", common_barcodes)
            ct_out.insert(1, "D_cond", dcond_aligned)
            ct_out.to_csv(sample_out_dir / f"{sample_id}_celltype_scores.csv", index=False)

            # Partition
            disc_mask, conc_mask = partition_spots_by_discordance(dcond_aligned)
            n_disc = disc_mask.sum()
            n_conc = conc_mask.sum()
            logger.info(f"    Discordant: {n_disc}, Concordant: {n_conc}")

            # Unmatched DE
            unmatched_de = celltype_de_test(ct_scores_aligned, disc_mask, conc_mask)
            unmatched_de["sample_id"] = sample_id
            all_unmatched_de.append(unmatched_de)

            for _, row in unmatched_de.iterrows():
                sig = "*" if row.get("fdr", 1) < 0.05 else ""
                logger.info(
                    f"    {row['cell_type']}: d={row['cohens_d']:.3f} "
                    f"(disc={row['mean_disc']:.3f}, conc={row['mean_conc']:.3f}){sig}"
                )

            # Matched DE using embeddings
            emb_path = embed_dir / sample_id / f"{args.encoder}_embeddings.h5"
            with h5py.File(emb_path, "r") as f:
                embeddings = f["embeddings"][:]
                embed_spot_ids = [s.decode() if isinstance(s, bytes) else s for s in f["spot_ids"][:]]

            # Align embeddings with common_barcodes
            embed_idx_map = {b: i for i, b in enumerate(embed_spot_ids)}
            embed_available = [b for b in common_barcodes if b in embed_idx_map]
            embed_indices = [embed_idx_map[b] for b in embed_available]
            barcode_to_common_idx = {b: i for i, b in enumerate(common_barcodes)}
            common_indices = [barcode_to_common_idx[b] for b in embed_available]

            if len(embed_available) > 100:
                embed_aligned = embeddings[embed_indices]
                dcond_for_match = dcond_aligned[common_indices]

                disc_mask_e, conc_mask_e = partition_spots_by_discordance(dcond_for_match)
                disc_indices = np.where(disc_mask_e)[0]
                conc_indices = np.where(conc_mask_e)[0]

                matched_conc_idx, match_distances, unmatched_mask, dist_threshold = morphology_match(
                    embed_aligned, disc_indices, conc_indices, k=5
                )

                quality = matching_quality_report(match_distances, unmatched_mask, dist_threshold)
                logger.info(f"    Matching: {quality['n_matched']}/{quality['n_discordant']} matched")

                # Matched cell type DE
                ct_scores_for_match = ct_scores_aligned.iloc[common_indices].reset_index(drop=True)
                matched_de = celltype_matched_de_test(
                    ct_scores_for_match, disc_indices, matched_conc_idx, unmatched_mask
                )
                matched_de["sample_id"] = sample_id
                all_matched_de.append(matched_de)

                for _, row in matched_de.iterrows():
                    sig = "*" if row.get("fdr", 1) < 0.05 else ""
                    logger.info(
                        f"    [matched] {row['cell_type']}: "
                        f"delta={row['mean_delta']:.4f}, es={row['effect_size']:.3f}{sig}"
                    )

            # CellViT nuclear morphometry (if available)
            cellvit_path = find_cellvit_file(sample_id, str(cellvit_dir))
            if cellvit_path:
                logger.info(f"    CellViT data found: {cellvit_path.name}")
                morph_df = extract_nuclear_morphometry(
                    cellvit_path,
                    spatial_coords[common_idx],
                    spot_radius_px=56.0,
                )
                if not morph_df.empty and len(morph_df) == len(common_barcodes):
                    morph_df.insert(0, "barcode", common_barcodes)
                    morph_df.to_csv(
                        sample_out_dir / f"{sample_id}_cellvit_morphometry.csv", index=False
                    )

                    # Compare nuclear features between disc/conc
                    for feat in ["nuclear_count", "frac_neoplastic", "frac_inflammatory",
                                 "frac_stromal", "nuclear_density", "mean_nuclear_area"]:
                        if feat in morph_df.columns:
                            vals = morph_df[feat].values
                            valid = ~np.isnan(vals)
                            if valid.sum() > 100:
                                disc_v = vals[disc_mask & valid]
                                conc_v = vals[conc_mask & valid]
                                if len(disc_v) > 10 and len(conc_v) > 10:
                                    mean_d = np.mean(disc_v)
                                    mean_c = np.mean(conc_v)
                                    n_d, n_c = len(disc_v), len(conc_v)
                                    var_d = np.var(disc_v, ddof=1) if n_d > 1 else 0
                                    var_c = np.var(conc_v, ddof=1) if n_c > 1 else 0
                                    pooled_var = ((var_d * (n_d - 1) + var_c * (n_c - 1)) / (n_d + n_c - 2)) if (n_d + n_c) > 2 else 0
                                    pooled_sd = np.sqrt(pooled_var) if pooled_var > 0 else 1e-6
                                    d = (mean_d - mean_c) / pooled_sd
                                    _, pval = stats.mannwhitneyu(disc_v, conc_v, alternative="two-sided")
                                    cellvit_records.append({
                                        "sample_id": sample_id,
                                        "feature": feat,
                                        "mean_disc": float(mean_d),
                                        "mean_conc": float(mean_c),
                                        "cohens_d": float(d),
                                        "pval": float(pval),
                                    })
                                    logger.info(
                                        f"    CellViT {feat}: d={d:.3f} "
                                        f"(disc={mean_d:.3f}, conc={mean_c:.3f}, p={pval:.4f})"
                                    )
            else:
                logger.info(f"    No CellViT data for {sample_id}")

        # Aggregate unmatched cell type DE across samples
        if all_unmatched_de:
            combined_unmatched = pd.concat(all_unmatched_de, ignore_index=True)

            summary_records = []
            for ct in combined_unmatched["cell_type"].unique():
                ct_data = combined_unmatched[combined_unmatched["cell_type"] == ct]
                n_samples = len(ct_data)
                n_sig = (ct_data["fdr"] < 0.05).sum()
                mean_d = ct_data["cohens_d"].mean()
                median_d = ct_data["cohens_d"].median()
                # Direction consistency
                n_positive = (ct_data["cohens_d"] > 0).sum()
                dir_consistency = max(n_positive, n_samples - n_positive) / n_samples

                summary_records.append({
                    "cell_type": ct,
                    "n_samples": n_samples,
                    "n_sig_fdr05": n_sig,
                    "mean_cohens_d": float(mean_d),
                    "median_cohens_d": float(median_d),
                    "direction_consistency": float(dir_consistency),
                    "direction": "up_in_disc" if mean_d > 0 else "down_in_disc",
                })

            summary_df = pd.DataFrame(summary_records).sort_values(
                "mean_cohens_d", key=abs, ascending=False
            )
            summary_df.to_csv(out_dir / "celltype_summary.csv", index=False)

            logger.info(f"\n  Cell type summary ({cohort_name}):")
            for _, row in summary_df.iterrows():
                logger.info(
                    f"    {row['cell_type']}: mean_d={row['mean_cohens_d']:.3f}, "
                    f"sig={row['n_sig_fdr05']}/{row['n_samples']}, "
                    f"dir={row['direction']} ({row['direction_consistency']:.0%})"
                )

        # CellViT summary
        if cellvit_records:
            cellvit_df = pd.DataFrame(cellvit_records)
            cellvit_df.to_csv(out_dir / "cellvit_summary.csv", index=False)

        # Save overall summary
        summary = {
            "cohort": cohort_name,
            "n_samples": len(samples),
            "encoder_for_matching": args.encoder,
            "available_cell_types": {ct: genes for ct, genes in available_sigs.items()},
            "n_cellvit_samples": len(set(r["sample_id"] for r in cellvit_records)) if cellvit_records else 0,
        }
        if all_unmatched_de:
            summary["celltype_effects"] = summary_df.to_dict(orient="records")
        if all_matched_de:
            combined_matched = pd.concat(all_matched_de, ignore_index=True)
            matched_summary = []
            for ct in combined_matched["cell_type"].unique():
                ct_data = combined_matched[combined_matched["cell_type"] == ct]
                matched_summary.append({
                    "cell_type": ct,
                    "n_samples": len(ct_data),
                    "mean_effect_size": float(ct_data["effect_size"].mean()),
                    "n_sig_fdr05": int((ct_data["fdr"] < 0.05).sum()),
                })
            summary["matched_celltype_effects"] = matched_summary

        with open(out_dir / "deconvolution_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    total_time = time.time() - overall_start
    logger.info(f"\nTotal time: {format_time(total_time)}")
    logger.info("Done.")


if __name__ == "__main__":
    main()

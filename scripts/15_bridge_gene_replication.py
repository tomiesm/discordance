#!/usr/bin/env python3
"""
Script 15: Bridge Gene Cross-Cohort Replication (Phase 4C)

Uses the 90 genes shared between the Biomarkers and 10x+Janesick panels to
directly compare prediction quality and discordance patterns across cohorts.

Analyses:
  1. Per-gene Pearson correlation across cohorts (do the same genes predict
     well/poorly in both cohorts?)
  2. Per-gene mean |residual| correlation across cohorts
  3. Discordance-related DE overlap: for the 90 bridge genes, are the same
     genes DE in discordant vs concordant spots in both cohorts?
  4. v2 comparison: for the ~50 HVGs that overlap with bridge genes, compare
     v2 vs v3 prediction quality

This is assumption-free cross-cohort validation: if the same 90 genes show
similar prediction quality in two independent cohorts with different patients,
panels, and processing, the signal is real.

Output:
    outputs/phase4/bridge_genes/
        bridge_gene_pearson.csv         (per-gene Pearson in both cohorts)
        bridge_gene_correlation.json    (cross-cohort agreement metrics)
        bridge_gene_de_overlap.json     (DE replication on bridge genes)

Usage:
    python scripts/15_bridge_gene_replication.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from scipy import stats
from src.utils import setup_logging, format_time


def load_v3_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_per_gene_stats(pred_dir, cohort_name, gene_names, n_folds, encoder_names):
    """Compute per-gene Pearson and mean |residual| for bridge genes only.

    Averages across encoders (Ridge regressor).
    """
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    # Aggregate across folds for each encoder
    encoder_pearsons = {g: [] for g in gene_names}
    encoder_mean_abs_resid = {g: [] for g in gene_names}

    for enc in encoder_names:
        all_targets = []
        all_preds = []
        all_resids = []

        for fold_idx in range(n_folds):
            fold_dir = pred_dir / cohort_name / enc / "ridge" / f"fold{fold_idx}"
            targets = np.load(fold_dir / "test_targets.npy")
            preds = np.load(fold_dir / "test_predictions.npy")
            resids = np.load(fold_dir / "test_residuals.npy")
            all_targets.append(targets)
            all_preds.append(preds)
            all_resids.append(resids)

        targets = np.vstack(all_targets)
        preds = np.vstack(all_preds)
        resids = np.vstack(all_resids)

        for gene in gene_names:
            idx = gene_to_idx[gene]
            yt = targets[:, idx]
            yp = preds[:, idx]
            if np.std(yt) > 1e-10 and np.std(yp) > 1e-10:
                r, _ = stats.pearsonr(yt, yp)
                if np.isnan(r):
                    r = 0.0
            else:
                r = 0.0
            encoder_pearsons[gene].append(r)
            encoder_mean_abs_resid[gene].append(np.mean(np.abs(resids[:, idx])))

    # Average across encoders
    mean_pearsons = {g: np.mean(rs) for g, rs in encoder_pearsons.items()}
    mean_abs_resids = {g: np.mean(rs) for g, rs in encoder_mean_abs_resid.items()}
    return mean_pearsons, mean_abs_resids


def main():
    parser = argparse.ArgumentParser(description="Phase 4C: Bridge gene replication")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_v3_config(args.config)

    log_dir = Path(config["output_dir"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(
        log_level=config.get("log_level", "INFO"),
        log_file=str(log_dir / "15_bridge_gene_replication.log"),
    )

    logger.info("=" * 60)
    logger.info("Phase 4C: Bridge Gene Cross-Cohort Replication")
    logger.info("=" * 60)

    out_dir = Path(config["output_dir"]) / "phase4" / "bridge_genes"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = Path(config["output_dir"]) / "predictions"
    encoder_names = [e["name"] for e in config["encoders"]]
    overall_start = time.time()

    # Load bridge genes
    bridge_path = Path(config["v3_data_dir"]) / "gene_list_bridge.json"
    with open(bridge_path) as f:
        bridge_genes = json.load(f)
    logger.info(f"Bridge genes: {len(bridge_genes)}")

    # Load gene lists for each cohort to get column indices
    cohort_gene_lists = {}
    for cohort_key in ["discovery", "validation"]:
        cohort_name = config["cohorts"][cohort_key]["name"]
        gene_list_path = Path(config["v3_data_dir"]) / f"gene_list_{cohort_name}.json"
        with open(gene_list_path) as f:
            cohort_gene_lists[cohort_key] = json.load(f)

    # Verify bridge genes are in both panels
    disc_set = set(cohort_gene_lists["discovery"])
    val_set = set(cohort_gene_lists["validation"])
    verified_bridge = [g for g in bridge_genes if g in disc_set and g in val_set]
    logger.info(f"Verified bridge genes (in both panels): {len(verified_bridge)}")

    # Compute per-gene stats for bridge genes in each cohort
    results = {}
    for cohort_key in ["discovery", "validation"]:
        cohort_config = config["cohorts"][cohort_key]
        cohort_name = cohort_config["name"]
        n_folds = cohort_config["n_lopo_folds"]
        full_gene_list = cohort_gene_lists[cohort_key]

        logger.info(f"\nComputing stats for {cohort_name}...")
        mean_pearsons, mean_abs_resids = compute_per_gene_stats(
            pred_dir, cohort_name, full_gene_list, n_folds, encoder_names
        )

        # Filter to bridge genes
        results[cohort_key] = {
            "pearsons": {g: mean_pearsons[g] for g in verified_bridge},
            "mean_abs_resids": {g: mean_abs_resids[g] for g in verified_bridge},
        }

    # Build comparison DataFrame
    records = []
    for gene in verified_bridge:
        records.append({
            "gene": gene,
            "pearson_discovery": results["discovery"]["pearsons"][gene],
            "pearson_validation": results["validation"]["pearsons"][gene],
            "mean_abs_resid_discovery": results["discovery"]["mean_abs_resids"][gene],
            "mean_abs_resid_validation": results["validation"]["mean_abs_resids"][gene],
        })

    bridge_df = pd.DataFrame(records)
    bridge_df.to_csv(out_dir / "bridge_gene_pearson.csv", index=False)

    # Cross-cohort correlations
    pearson_disc = bridge_df["pearson_discovery"].values
    pearson_val = bridge_df["pearson_validation"].values
    resid_disc = bridge_df["mean_abs_resid_discovery"].values
    resid_val = bridge_df["mean_abs_resid_validation"].values

    pearson_r, pearson_p = stats.pearsonr(pearson_disc, pearson_val)
    pearson_rho, pearson_rho_p = stats.spearmanr(pearson_disc, pearson_val)
    resid_r, resid_p = stats.pearsonr(resid_disc, resid_val)
    resid_rho, resid_rho_p = stats.spearmanr(resid_disc, resid_val)

    logger.info(f"\n  Cross-cohort Pearson correlation:")
    logger.info(f"    Prediction quality: r={pearson_r:.3f} (p={pearson_p:.2e}), "
                f"rho={pearson_rho:.3f}")
    logger.info(f"    Mean |residual|:    r={resid_r:.3f} (p={resid_p:.2e}), "
                f"rho={resid_rho:.3f}")

    # Summary stats
    logger.info(f"\n  Discovery: mean Pearson={np.mean(pearson_disc):.3f} "
                f"(sd={np.std(pearson_disc):.3f})")
    logger.info(f"  Validation: mean Pearson={np.mean(pearson_val):.3f} "
                f"(sd={np.std(pearson_val):.3f})")

    # Top genes: best in both
    bridge_df["mean_both"] = (bridge_df["pearson_discovery"] + bridge_df["pearson_validation"]) / 2
    top_both = bridge_df.nlargest(15, "mean_both")
    logger.info("\n  Top 15 bridge genes (best in both cohorts):")
    for _, row in top_both.iterrows():
        logger.info(
            f"    {row['gene']}: disc={row['pearson_discovery']:.3f}, "
            f"val={row['pearson_validation']:.3f}"
        )

    # Genes with biggest cross-cohort discrepancy
    bridge_df["abs_diff"] = np.abs(bridge_df["pearson_discovery"] - bridge_df["pearson_validation"])
    top_diff = bridge_df.nlargest(10, "abs_diff")
    logger.info("\n  Top 10 discrepant bridge genes:")
    for _, row in top_diff.iterrows():
        logger.info(
            f"    {row['gene']}: disc={row['pearson_discovery']:.3f}, "
            f"val={row['pearson_validation']:.3f} (diff={row['abs_diff']:.3f})"
        )

    # DE overlap on bridge genes
    logger.info("\n  Checking DE overlap on bridge genes...")
    de_overlap = {"bridge_genes_de": {}}
    for cohort_key in ["discovery", "validation"]:
        cohort_name = config["cohorts"][cohort_key]["name"]
        meta_path = (Path(config["output_dir"]) / "phase3" / "de" / cohort_name /
                     "meta_de_unmatched.csv")
        if meta_path.exists():
            meta_df = pd.read_csv(meta_path)
            bridge_de = meta_df[meta_df["gene"].isin(verified_bridge)]
            sig_genes = set(bridge_de[bridge_de["reproducibility"] >= 0.5]["gene"])
            de_overlap[f"{cohort_key}_sig_bridge_genes"] = sorted(sig_genes)
            logger.info(f"    {cohort_name}: {len(sig_genes)} bridge genes reproducibly DE")
        else:
            de_overlap[f"{cohort_key}_sig_bridge_genes"] = []

    # Overlap between cohort DE on bridge genes
    disc_sig = set(de_overlap.get("discovery_sig_bridge_genes", []))
    val_sig = set(de_overlap.get("validation_sig_bridge_genes", []))
    if disc_sig or val_sig:
        both_sig = disc_sig & val_sig
        either_sig = disc_sig | val_sig
        de_jaccard = len(both_sig) / len(either_sig) if either_sig else 0
        de_overlap["de_jaccard"] = float(de_jaccard)
        de_overlap["n_both_sig"] = len(both_sig)
        de_overlap["both_sig_genes"] = sorted(both_sig)
        logger.info(f"    DE overlap: {len(both_sig)} genes sig in both, "
                    f"Jaccard={de_jaccard:.3f}")

        # For overlapping DE genes, check direction consistency
        if both_sig:
            disc_meta = pd.read_csv(
                Path(config["output_dir"]) / "phase3" / "de" / config["cohorts"]["discovery"]["name"] /
                "meta_de_unmatched.csv"
            )
            val_meta = pd.read_csv(
                Path(config["output_dir"]) / "phase3" / "de" / config["cohorts"]["validation"]["name"] /
                "meta_de_unmatched.csv"
            )
            direction_consistent = 0
            for gene in both_sig:
                disc_fc = disc_meta[disc_meta["gene"] == gene]["median_log2fc"].values
                val_fc = val_meta[val_meta["gene"] == gene]["median_log2fc"].values
                if len(disc_fc) > 0 and len(val_fc) > 0:
                    if np.sign(disc_fc[0]) == np.sign(val_fc[0]):
                        direction_consistent += 1

            de_overlap["n_direction_consistent"] = direction_consistent
            de_overlap["direction_consistency"] = direction_consistent / len(both_sig) if both_sig else 0
            logger.info(f"    Direction consistency: {direction_consistent}/{len(both_sig)} "
                        f"({de_overlap['direction_consistency']:.0%})")

    # Save summary
    summary = {
        "n_bridge_genes": len(verified_bridge),
        "cross_cohort_pearson": {
            "prediction_quality_r": float(pearson_r),
            "prediction_quality_p": float(pearson_p),
            "prediction_quality_rho": float(pearson_rho),
            "mean_abs_resid_r": float(resid_r),
            "mean_abs_resid_p": float(resid_p),
            "mean_abs_resid_rho": float(resid_rho),
        },
        "discovery_mean_pearson": float(np.mean(pearson_disc)),
        "validation_mean_pearson": float(np.mean(pearson_val)),
        "de_overlap": de_overlap,
        "top_bridge_genes": top_both[["gene", "pearson_discovery", "pearson_validation"]].to_dict(
            orient="records"
        ),
    }

    with open(out_dir / "bridge_gene_correlation.json", "w") as f:
        json.dump(summary, f, indent=2)

    total_time = time.time() - overall_start
    logger.info(f"\nTotal time: {format_time(total_time)}")
    logger.info("Done.")


if __name__ == "__main__":
    main()

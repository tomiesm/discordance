#!/usr/bin/env python3
"""
Script 14: Encoder Consistency Analysis (Phase 4B)

Per-gene Pearson from each encoder independently (Ridge regressor).
Identifies:
  - "Morphologically visible" genes: high Pearson across all 3 encoders
  - "Encoder-specific" genes: high in one encoder, low in others
  - Overall agreement structure between encoder representations

Output:
    outputs/phase4/encoder_consistency/{cohort}/
        per_gene_pearson_by_encoder.csv
        encoder_agreement.json
        encoder_specific_genes.csv

Usage:
    python scripts/14_encoder_consistency.py
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


def compute_per_gene_pearson(pred_dir, cohort_name, encoder_name, n_folds):
    """Compute per-gene Pearson for a single encoder (Ridge), aggregated across folds."""
    all_targets = []
    all_preds = []

    for fold_idx in range(n_folds):
        fold_dir = pred_dir / cohort_name / encoder_name / "ridge" / f"fold{fold_idx}"
        targets = np.load(fold_dir / "test_targets.npy")
        preds = np.load(fold_dir / "test_predictions.npy")
        all_targets.append(targets)
        all_preds.append(preds)

    targets = np.vstack(all_targets)
    preds = np.vstack(all_preds)

    n_genes = targets.shape[1]
    pearsons = []
    for g in range(n_genes):
        yt = targets[:, g]
        yp = preds[:, g]
        if np.std(yt) > 1e-10 and np.std(yp) > 1e-10:
            r, _ = stats.pearsonr(yt, yp)
            if np.isnan(r):
                r = 0.0
        else:
            r = 0.0
        pearsons.append(r)

    return np.array(pearsons)


def main():
    parser = argparse.ArgumentParser(description="Phase 4B: Encoder consistency")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--cohort", type=str, default=None)
    args = parser.parse_args()

    config = load_v3_config(args.config)

    log_dir = Path(config["output_dir"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(
        log_level=config.get("log_level", "INFO"),
        log_file=str(log_dir / "14_encoder_consistency.log"),
    )

    logger.info("=" * 60)
    logger.info("Phase 4B: Encoder Consistency Analysis")
    logger.info("=" * 60)

    cohort_keys = [args.cohort] if args.cohort else ["discovery", "validation"]
    pred_dir = Path(config["output_dir"]) / "predictions"
    encoder_names = [e["name"] for e in config["encoders"]]
    overall_start = time.time()

    for cohort_key in cohort_keys:
        cohort_config = config["cohorts"][cohort_key]
        cohort_name = cohort_config["name"]
        n_folds = cohort_config["n_lopo_folds"]
        out_dir = Path(config["output_dir"]) / "phase4" / "encoder_consistency" / cohort_name
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\nCohort: {cohort_name} ({cohort_key})")

        # Load gene names
        gene_list_path = Path(config["v3_data_dir"]) / f"gene_list_{cohort_name}.json"
        with open(gene_list_path) as f:
            gene_names = json.load(f)

        # Per-gene Pearson for each encoder
        encoder_pearsons = {}
        for enc in encoder_names:
            pearsons = compute_per_gene_pearson(pred_dir, cohort_name, enc, n_folds)
            encoder_pearsons[enc] = pearsons
            logger.info(f"  {enc}: mean={np.mean(pearsons):.4f}, median={np.median(pearsons):.4f}")

        # Build DataFrame
        df = pd.DataFrame({"gene": gene_names})
        for enc in encoder_names:
            df[f"pearson_{enc}"] = encoder_pearsons[enc]
        df["mean_pearson"] = df[[f"pearson_{enc}" for enc in encoder_names]].mean(axis=1)
        df["min_pearson"] = df[[f"pearson_{enc}" for enc in encoder_names]].min(axis=1)
        df["max_pearson"] = df[[f"pearson_{enc}" for enc in encoder_names]].max(axis=1)
        df["range_pearson"] = df["max_pearson"] - df["min_pearson"]

        df.to_csv(out_dir / "per_gene_pearson_by_encoder.csv", index=False)

        # Pairwise encoder agreement
        agreement_records = []
        for i, enc1 in enumerate(encoder_names):
            for enc2 in encoder_names[i + 1:]:
                r, p = stats.pearsonr(encoder_pearsons[enc1], encoder_pearsons[enc2])
                rho, _ = stats.spearmanr(encoder_pearsons[enc1], encoder_pearsons[enc2])
                agreement_records.append({
                    "encoder1": enc1,
                    "encoder2": enc2,
                    "pearson_r": float(r),
                    "spearman_rho": float(rho),
                })
                logger.info(f"  {enc1} vs {enc2}: Pearson r={r:.3f}, Spearman={rho:.3f}")

        # Identify morphologically visible genes (high across all encoders)
        visible_threshold = df["mean_pearson"].quantile(0.75)
        morpho_visible = df[df["min_pearson"] > visible_threshold].sort_values(
            "mean_pearson", ascending=False
        )
        logger.info(f"\n  Morphologically visible genes (min > {visible_threshold:.3f}): "
                     f"{len(morpho_visible)}")
        if not morpho_visible.empty:
            for _, row in morpho_visible.head(10).iterrows():
                logger.info(
                    f"    {row['gene']}: mean={row['mean_pearson']:.3f}, "
                    f"range={row['range_pearson']:.3f}"
                )

        # Identify encoder-specific genes (high range, high in one)
        specific_threshold = df["range_pearson"].quantile(0.90)
        encoder_specific = df[df["range_pearson"] > specific_threshold].copy()

        # Determine which encoder is best for each specific gene
        specific_records = []
        for _, row in encoder_specific.iterrows():
            best_enc = max(encoder_names, key=lambda e: row[f"pearson_{e}"])
            worst_enc = min(encoder_names, key=lambda e: row[f"pearson_{e}"])
            specific_records.append({
                "gene": row["gene"],
                "best_encoder": best_enc,
                "best_pearson": float(row[f"pearson_{best_enc}"]),
                "worst_encoder": worst_enc,
                "worst_pearson": float(row[f"pearson_{worst_enc}"]),
                "range": float(row["range_pearson"]),
                "mean_pearson": float(row["mean_pearson"]),
            })

        specific_df = pd.DataFrame(specific_records).sort_values("range", ascending=False)
        specific_df.to_csv(out_dir / "encoder_specific_genes.csv", index=False)

        logger.info(f"\n  Encoder-specific genes (range > {specific_threshold:.3f}): "
                     f"{len(specific_df)}")
        if not specific_df.empty:
            # Count by best encoder
            for enc in encoder_names:
                n = (specific_df["best_encoder"] == enc).sum()
                logger.info(f"    Best in {enc}: {n} genes")
            for _, row in specific_df.head(5).iterrows():
                logger.info(
                    f"    {row['gene']}: best={row['best_encoder']} "
                    f"({row['best_pearson']:.3f}), worst={row['worst_encoder']} "
                    f"({row['worst_pearson']:.3f})"
                )

        # Summary
        summary = {
            "cohort": cohort_name,
            "n_genes": len(gene_names),
            "encoder_means": {enc: float(np.mean(encoder_pearsons[enc])) for enc in encoder_names},
            "encoder_medians": {enc: float(np.median(encoder_pearsons[enc])) for enc in encoder_names},
            "pairwise_agreement": agreement_records,
            "n_morpho_visible": len(morpho_visible),
            "n_encoder_specific": len(specific_df),
            "top_morpho_visible": morpho_visible.head(20)[
                ["gene", "mean_pearson", "min_pearson", "range_pearson"]
            ].to_dict(orient="records") if not morpho_visible.empty else [],
        }

        with open(out_dir / "encoder_agreement.json", "w") as f:
            json.dump(summary, f, indent=2)

    total_time = time.time() - overall_start
    logger.info(f"\nTotal time: {format_time(total_time)}")
    logger.info("Done.")


if __name__ == "__main__":
    main()

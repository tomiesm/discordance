#!/usr/bin/env python3
"""
Script 04: Compute Discordance Scores for v3

Aggregates residuals across LOPO folds (each spot appears in exactly one test fold)
and computes two discordance metrics per spot for each of 9 configs.

Metrics:
    D_raw:  mean |residual| across genes (v3 spec)
    D_cond: D_raw conditioned on total expression (depth-corrected)

Output:
    outputs/phase2/scores/{cohort}/{sample_id}_discordance.parquet

Usage:
    python scripts/04_discordance_scores.py
    python scripts/04_discordance_scores.py --cohort discovery
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.utils import setup_logging, format_time
from src.discordance import (
    compute_mean_absolute_discordance,
    compute_conditional_discordance,
)


def load_v3_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_spatial_coords(sample_id, hest_dir):
    """Load spatial coordinates from h5ad for a sample."""
    h5ad_path = Path(hest_dir) / "st" / f"{sample_id}.h5ad"
    adata = ad.read_h5ad(h5ad_path)
    coords = adata.obsm["spatial"]  # (N, 2)
    barcodes = list(adata.obs.index)
    coord_dict = {b: coords[i] for i, b in enumerate(barcodes)}
    return coord_dict


def aggregate_folds(pred_dir, cohort_name, encoder_name, reg_name, n_folds):
    """Aggregate residuals/targets across folds. Each spot appears in exactly one fold."""
    all_residuals = []
    all_targets = []
    all_spot_ids = []

    for fold_idx in range(n_folds):
        fold_dir = pred_dir / cohort_name / encoder_name / reg_name / f"fold{fold_idx}"
        residuals = np.load(fold_dir / "test_residuals.npy")
        targets = np.load(fold_dir / "test_targets.npy")
        with open(fold_dir / "test_spot_ids.json") as f:
            spots = json.load(f)

        all_residuals.append(residuals)
        all_targets.append(targets)
        all_spot_ids.extend(spots)

    return (
        np.vstack(all_residuals),
        np.vstack(all_targets),
        all_spot_ids,
    )


def main():
    parser = argparse.ArgumentParser(description="Compute discordance scores for v3")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--cohort", type=str, default=None)
    args = parser.parse_args()

    config = load_v3_config(args.config)

    log_dir = Path(config["output_dir"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(
        log_level=config.get("log_level", "INFO"),
        log_file=str(log_dir / "04_discordance_scores.log"),
    )

    cohort_keys = [args.cohort] if args.cohort else ["discovery", "validation"]
    encoders = config["encoders"]
    regressors = config["regressors"]
    pred_dir = Path(config["output_dir"]) / "predictions"
    phase2_config = config.get("phase2", {})
    n_bins = phase2_config.get("conditional_n_bins", 10)
    min_bin_size = phase2_config.get("conditional_min_bin_size", 30)

    logger.info("=" * 60)
    logger.info("v3 Phase 2: Discordance Score Computation")
    logger.info("=" * 60)

    overall_start = time.time()

    for cohort_key in cohort_keys:
        cohort_config = config["cohorts"][cohort_key]
        cohort_name = cohort_config["name"]
        n_folds = cohort_config["n_lopo_folds"]
        samples = cohort_config["samples"]

        logger.info(f"\nCohort: {cohort_name} ({cohort_key})")

        # Load spatial coordinates for all samples
        logger.info("  Loading spatial coordinates...")
        all_coords = {}
        for sid in samples:
            all_coords[sid] = load_spatial_coords(sid, config["hest_dir"])

        # For each config, aggregate folds and compute discordance
        config_names = []
        config_discordance = {}  # {config_name: {spot_id: (D_raw, D_cond, total_expr)}}

        for enc in encoders:
            for reg in regressors:
                config_name = f"{enc['name']}_{reg['name']}"
                config_names.append(config_name)
                logger.info(f"  Config: {config_name}")

                residuals, targets, spot_ids = aggregate_folds(
                    pred_dir, cohort_name, enc["name"], reg["name"], n_folds
                )
                logger.info(f"    Aggregated: {residuals.shape[0]} spots × {residuals.shape[1]} genes")

                # Compute discordance scores
                D_raw = compute_mean_absolute_discordance(residuals)
                total_expr = targets.sum(axis=1)
                D_cond = compute_conditional_discordance(
                    D_raw, total_expr, n_bins=n_bins, min_bin_size=min_bin_size
                )

                logger.info(
                    f"    D_raw: mean={D_raw.mean():.4f}, std={D_raw.std():.4f}"
                )
                logger.info(
                    f"    D_cond: mean={D_cond.mean():.4f}, std={D_cond.std():.4f}"
                )

                # Store per-spot
                for i, sid_spot in enumerate(spot_ids):
                    if sid_spot not in config_discordance:
                        config_discordance[sid_spot] = {}
                    config_discordance[sid_spot][config_name] = (
                        D_raw[i],
                        D_cond[i],
                        total_expr[i],
                    )

        # Build per-sample DataFrames and save
        out_dir = Path(config["output_dir"]) / "phase2" / "scores" / cohort_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for sid in samples:
            prefix = f"{sid}_"
            sample_spots = [s for s in config_discordance if s.startswith(prefix)]

            if not sample_spots:
                logger.warning(f"  No spots for {sid}")
                continue

            rows = []
            for spot_id in sample_spots:
                barcode = spot_id[len(prefix):]
                coord = all_coords[sid].get(barcode, np.array([np.nan, np.nan]))

                row = {
                    "spot_id": spot_id,
                    "sample_id": sid,
                    "barcode": barcode,
                    "x": coord[0],
                    "y": coord[1],
                }

                # Add total expression (same across configs, take from first)
                first_config = config_names[0]
                row["total_expr"] = config_discordance[spot_id][first_config][2]

                # Add discordance scores per config
                for cn in config_names:
                    d_raw, d_cond, _ = config_discordance[spot_id][cn]
                    row[f"D_raw_{cn}"] = d_raw
                    row[f"D_cond_{cn}"] = d_cond

                rows.append(row)

            df = pd.DataFrame(rows)
            out_path = out_dir / f"{sid}_discordance.parquet"
            df.to_parquet(out_path, index=False)
            logger.info(f"  Saved {sid}: {len(df)} spots -> {out_path}")

        logger.info(f"  Total samples saved: {len(samples)}")

    total_time = time.time() - overall_start
    logger.info(f"\nTotal time: {format_time(total_time)}")
    logger.info("Done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Script 07: Gate 2.3 — Dual-Track Concordance

Tests whether discordance is stable across gene subsets.

Gate 2.3a (Internal): Within each cohort, split 280 genes into two halves
(140/140, seed=42). Compute discordance from each half independently
using stored residuals (no retraining). Median Spearman ρ > 0.15.

This is the critical test that failed in v2 (ρ ≈ 0.04 with 25/25 genes).
With 140/140 genes, the signal-to-noise ratio should be dramatically better.

Output:
    outputs/phase2/gate2_3_dual_track.json

Usage:
    python scripts/07_dual_track.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.utils import setup_logging, format_time
from src.discordance import compute_mean_absolute_discordance


def load_v3_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def split_genes(n_genes, seed=42):
    """Randomly split gene indices into two halves."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_genes)
    mid = n_genes // 2
    return indices[:mid], indices[mid:]


def aggregate_residuals(pred_dir, cohort_name, encoder_name, reg_name, n_folds):
    """Aggregate residuals across folds, return (residuals, spot_ids)."""
    all_residuals = []
    all_spot_ids = []

    for fold_idx in range(n_folds):
        fold_dir = pred_dir / cohort_name / encoder_name / reg_name / f"fold{fold_idx}"
        residuals = np.load(fold_dir / "test_residuals.npy")
        with open(fold_dir / "test_spot_ids.json") as f:
            spots = json.load(f)
        all_residuals.append(residuals)
        all_spot_ids.extend(spots)

    return np.vstack(all_residuals), all_spot_ids


def main():
    parser = argparse.ArgumentParser(description="Gate 2.3: Dual-track concordance")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_v3_config(args.config)

    log_dir = Path(config["output_dir"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(
        log_level=config.get("log_level", "INFO"),
        log_file=str(log_dir / "07_dual_track.log"),
    )

    phase2_config = config.get("phase2", {})
    threshold = phase2_config.get("gate3_internal_spearman_threshold", 0.15)

    logger.info("=" * 60)
    logger.info("Gate 2.3: Dual-Track Concordance")
    logger.info("=" * 60)
    logger.info(f"Threshold (internal): median ρ > {threshold}")

    overall_start = time.time()
    pred_dir = Path(config["output_dir"]) / "predictions"
    seed = config["seed"]

    results = {
        "gate": "2.3",
        "threshold": threshold,
        "seed": seed,
        "cohorts": {},
    }
    gate_pass_overall = True

    for cohort_key in ["discovery", "validation"]:
        cohort_config = config["cohorts"][cohort_key]
        cohort_name = cohort_config["name"]
        n_folds = cohort_config["n_lopo_folds"]
        samples = cohort_config["samples"]

        logger.info(f"\nCohort: {cohort_name}")

        # Get gene count from first fold
        first_fold = pred_dir / cohort_name / config["encoders"][0]["name"] / config["regressors"][0]["name"] / "fold0"
        n_genes = np.load(first_fold / "test_residuals.npy").shape[1]
        genes_A, genes_B = split_genes(n_genes, seed=seed)
        logger.info(f"  Split: {len(genes_A)} / {len(genes_B)} genes")

        # Compute dual-track for each config × sample
        config_results = {}

        for enc in config["encoders"]:
            for reg in config["regressors"]:
                config_name = f"{enc['name']}_{reg['name']}"
                logger.info(f"\n  Config: {config_name}")

                # Aggregate residuals across folds
                residuals, spot_ids = aggregate_residuals(
                    pred_dir, cohort_name, enc["name"], reg["name"], n_folds
                )

                # Split residuals by gene halves
                residuals_A = residuals[:, genes_A]
                residuals_B = residuals[:, genes_B]

                # Compute discordance from each half
                D_A = compute_mean_absolute_discordance(residuals_A)
                D_B = compute_mean_absolute_discordance(residuals_B)

                # Per-sample Spearman
                sample_rhos = {}
                for sid in samples:
                    prefix = f"{sid}_"
                    mask = np.array([s.startswith(prefix) for s in spot_ids])

                    if mask.sum() < 10:
                        continue

                    rho, pval = spearmanr(D_A[mask], D_B[mask])
                    sample_rhos[sid] = {
                        "rho": float(rho),
                        "p_value": float(pval),
                        "n_spots": int(mask.sum()),
                    }
                    logger.info(f"    {sid}: ρ = {rho:.4f} (n={mask.sum()})")

                rho_values = [v["rho"] for v in sample_rhos.values()]
                median_rho = float(np.median(rho_values)) if rho_values else 0.0
                mean_rho = float(np.mean(rho_values)) if rho_values else 0.0

                config_results[config_name] = {
                    "samples": sample_rhos,
                    "median_rho": median_rho,
                    "mean_rho": mean_rho,
                    "n_genes_A": int(len(genes_A)),
                    "n_genes_B": int(len(genes_B)),
                }

                logger.info(f"    Median ρ = {median_rho:.4f}, Mean ρ = {mean_rho:.4f}")

        # Gate: median across all configs and samples
        all_rhos = []
        for cn, cr in config_results.items():
            for sid, sr in cr["samples"].items():
                all_rhos.append(sr["rho"])

        overall_median = float(np.median(all_rhos)) if all_rhos else 0.0
        cohort_pass = overall_median > threshold
        gate_pass_overall = gate_pass_overall and cohort_pass

        results["cohorts"][cohort_key] = {
            "cohort_name": cohort_name,
            "configs": config_results,
            "overall_median_rho": overall_median,
            "gate_pass": cohort_pass,
        }

        status = "PASS" if cohort_pass else "FAIL"
        logger.info(
            f"\n  {cohort_name}: overall median ρ = {overall_median:.4f} [{status}]"
        )

    results["gate_pass"] = gate_pass_overall
    status = "PASS" if gate_pass_overall else "FAIL"
    logger.info(f"\nGate 2.3a overall: [{status}]")

    if gate_pass_overall:
        logger.info("Dual-track concordance PASSES — discordance is gene-set-stable.")
        logger.info("This is the key improvement over v2 (which failed at ρ ≈ 0.04 with 25/25 genes).")
    else:
        logger.warning("Dual-track concordance FAILS — discordance may be gene-set-dependent.")
        logger.warning("Phase 3 should restrict to gene-level analysis only.")

    # Save
    out_path = Path(config["output_dir"]) / "phase2" / "gate2_3_dual_track.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved: {out_path}")

    total_time = time.time() - overall_start
    logger.info(f"Total time: {format_time(total_time)}")


if __name__ == "__main__":
    main()

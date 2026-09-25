#!/usr/bin/env python3
"""
Script 05: Gate 2.1 — Multi-Model Agreement

Tests whether discordance scores agree across the 9 configs
(3 encoders × 3 regressors). Uses conditional discordance (D_cond)
to avoid confounding with expression depth.

Gate criterion: Median pairwise Spearman ρ > 0.4 across 9 configs,
in >= 50% of samples within each cohort.

Output:
    outputs/phase2/gate2_1_agreement.json

Usage:
    python scripts/05_multimodel_agreement.py
"""

import argparse
import json
import sys
import time
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.utils import setup_logging, format_time


def load_v3_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Gate 2.1: Multi-model agreement")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_v3_config(args.config)

    log_dir = Path(config["output_dir"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(
        log_level=config.get("log_level", "INFO"),
        log_file=str(log_dir / "05_multimodel_agreement.log"),
    )

    phase2_config = config.get("phase2", {})
    threshold = phase2_config.get("gate1_spearman_threshold", 0.4)
    min_fraction = phase2_config.get("gate1_min_fraction", 0.50)

    logger.info("=" * 60)
    logger.info("Gate 2.1: Multi-Model Agreement")
    logger.info("=" * 60)
    logger.info(f"Threshold: median ρ > {threshold}")
    logger.info(f"Min fraction: {min_fraction}")

    overall_start = time.time()

    # Build config column names
    config_names = []
    for enc in config["encoders"]:
        for reg in config["regressors"]:
            config_names.append(f"{enc['name']}_{reg['name']}")

    d_cond_cols = [f"D_cond_{cn}" for cn in config_names]

    results = {"gate": "2.1", "threshold": threshold, "min_fraction": min_fraction, "cohorts": {}}
    gate_pass_overall = True

    for cohort_key in ["discovery", "validation"]:
        cohort_config = config["cohorts"][cohort_key]
        cohort_name = cohort_config["name"]
        samples = cohort_config["samples"]

        logger.info(f"\nCohort: {cohort_name}")
        scores_dir = Path(config["output_dir"]) / "phase2" / "scores" / cohort_name

        sample_results = {}
        n_pass = 0

        for sid in samples:
            parquet_path = scores_dir / f"{sid}_discordance.parquet"
            df = pd.read_parquet(parquet_path)

            # Extract D_cond columns
            D_matrix = df[d_cond_cols].values  # (n_spots, 9)

            # Pairwise Spearman between all 9 configs
            n_configs = len(config_names)
            pairwise_rhos = []
            for i, j in combinations(range(n_configs), 2):
                rho, _ = spearmanr(D_matrix[:, i], D_matrix[:, j])
                pairwise_rhos.append(rho)

            median_rho = float(np.median(pairwise_rhos))
            mean_rho = float(np.mean(pairwise_rhos))
            min_rho = float(np.min(pairwise_rhos))
            passed = median_rho > threshold

            if passed:
                n_pass += 1

            sample_results[sid] = {
                "median_rho": median_rho,
                "mean_rho": mean_rho,
                "min_rho": min_rho,
                "n_spots": len(df),
                "pass": passed,
            }

            status = "PASS" if passed else "FAIL"
            logger.info(
                f"  {sid}: median ρ = {median_rho:.4f} "
                f"(mean={mean_rho:.4f}, min={min_rho:.4f}) [{status}]"
            )

        fraction_pass = n_pass / len(samples)
        cohort_pass = fraction_pass >= min_fraction
        gate_pass_overall = gate_pass_overall and cohort_pass

        results["cohorts"][cohort_key] = {
            "cohort_name": cohort_name,
            "samples": sample_results,
            "n_pass": n_pass,
            "n_total": len(samples),
            "fraction_pass": fraction_pass,
            "gate_pass": cohort_pass,
        }

        status = "PASS" if cohort_pass else "FAIL"
        logger.info(
            f"\n  {cohort_name}: {n_pass}/{len(samples)} samples pass "
            f"({fraction_pass:.1%}) [{status}]"
        )

    results["gate_pass"] = gate_pass_overall
    status = "PASS" if gate_pass_overall else "FAIL"
    logger.info(f"\nGate 2.1 overall: [{status}]")

    # Save
    out_path = Path(config["output_dir"]) / "phase2" / "gate2_1_agreement.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved: {out_path}")

    total_time = time.time() - overall_start
    logger.info(f"Total time: {format_time(total_time)}")


if __name__ == "__main__":
    main()

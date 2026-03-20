#!/usr/bin/env python3
"""
Script 06: Gate 2.2 — Spatial Structure Beyond Geometry

Tests whether conditional discordance (D_cond) has spatial autocorrelation
beyond what tissue geometry predicts, using Moran's I with a Level 2 null
(geometry-preserving permutation via boundary rings).

Gate criterion: Moran's I p < 0.01 under Level 2 null,
in >= 50% of samples within each cohort.

Output:
    outputs/phase2/gate2_2_spatial.json

Usage:
    python scripts/06_spatial_structure.py
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
from src.utils import setup_logging, format_time
from src.spatial import (
    build_spatial_weights,
    morans_i_permutation,
    assign_boundary_rings,
)


def load_v3_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Gate 2.2: Spatial structure")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_v3_config(args.config)

    log_dir = Path(config["output_dir"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(
        log_level=config.get("log_level", "INFO"),
        log_file=str(log_dir / "06_spatial_structure.log"),
    )

    phase2_config = config.get("phase2", {})
    p_threshold = phase2_config.get("gate2_pvalue_threshold", 0.01)
    min_fraction = phase2_config.get("gate2_min_fraction", 0.50)
    n_neighbors = phase2_config.get("spatial_n_neighbors", 6)
    n_permutations = phase2_config.get("spatial_n_permutations", 999)

    logger.info("=" * 60)
    logger.info("Gate 2.2: Spatial Structure Beyond Geometry")
    logger.info("=" * 60)
    logger.info(f"Threshold: p < {p_threshold}")
    logger.info(f"Min fraction: {min_fraction}")
    logger.info(f"Neighbors: {n_neighbors}, Permutations: {n_permutations}")

    overall_start = time.time()

    # Use ridge as reference regressor (simplest, most stable)
    # Average D_cond across 3 encoders for robustness
    ref_encoders = [e["name"] for e in config["encoders"]]
    ref_reg = "ridge"

    results = {
        "gate": "2.2",
        "p_threshold": p_threshold,
        "min_fraction": min_fraction,
        "n_neighbors": n_neighbors,
        "n_permutations": n_permutations,
        "cohorts": {},
    }
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

            # Average D_cond across encoders (all using ridge)
            d_cond_cols = [f"D_cond_{enc}_{ref_reg}" for enc in ref_encoders]
            D_cond = df[d_cond_cols].mean(axis=1).values

            # Spatial coordinates
            coords = df[["x", "y"]].values

            # Drop spots with missing coords
            valid = ~np.isnan(coords).any(axis=1)
            coords = coords[valid]
            D_cond = D_cond[valid]

            logger.info(f"  {sid}: {len(D_cond)} spots")

            # Build spatial weights
            t0 = time.time()
            W = build_spatial_weights(coords, n_neighbors=n_neighbors)

            # Assign boundary rings for Level 2 null
            rings = assign_boundary_rings(coords)
            n_rings = len(np.unique(rings))

            # Moran's I with Level 2 permutation (within rings)
            I_obs, p_value = morans_i_permutation(
                D_cond, W,
                n_permutations=n_permutations,
                permutation_groups=rings,
                seed=config["seed"],
            )
            elapsed = time.time() - t0

            passed = p_value < p_threshold

            if passed:
                n_pass += 1

            sample_results[sid] = {
                "morans_i": float(I_obs),
                "p_value": float(p_value),
                "n_spots": int(len(D_cond)),
                "n_rings": int(n_rings),
                "time_s": float(elapsed),
                "pass": passed,
            }

            status = "PASS" if passed else "FAIL"
            logger.info(
                f"    Moran's I = {I_obs:.4f}, p = {p_value:.4f} "
                f"({n_rings} rings, {format_time(elapsed)}) [{status}]"
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
    logger.info(f"\nGate 2.2 overall: [{status}]")

    # Save
    out_path = Path(config["output_dir"]) / "phase2" / "gate2_2_spatial.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved: {out_path}")

    total_time = time.time() - overall_start
    logger.info(f"Total time: {format_time(total_time)}")


if __name__ == "__main__":
    main()

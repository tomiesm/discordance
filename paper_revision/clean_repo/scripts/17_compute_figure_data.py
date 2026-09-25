#!/usr/bin/env python3
"""
Script 17: Pre-compute data needed for figures.

Generates:
  1. Dual-track subsampling curve (Fig 2e): split genes at various counts,
     compute track A/B discordance correlation.
  2. Per-spot dual-track values (Fig 2d): track A/B D per spot for
     representative samples.
  3. Moran's I null distributions (Fig 2b): permutation values for
     representative samples.
  4. Per-gene within-patient residuals (Fig 5a): mean |residual| per gene
     per sample for multi-section patients.

Output:
    outputs/figure_data/
        subsampling_curve.csv
        dual_track_spots_{sample}.csv
        morans_null_{sample}.npy
        within_patient_per_gene.csv

Usage:
    python scripts/17_compute_figure_data.py
    python scripts/17_compute_figure_data.py --analysis subsampling
    python scripts/17_compute_figure_data.py --analysis dual_track
    python scripts/17_compute_figure_data.py --analysis morans
    python scripts/17_compute_figure_data.py --analysis within_patient
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
from src.spatial import build_spatial_weights, morans_i


def load_v3_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


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


# ============================================================
# Analysis 1: Dual-track subsampling curve
# ============================================================

def compute_subsampling_curve(config, logger):
    """For various gene counts, compute dual-track ρ with 100 random splits."""
    pred_dir = Path(config["output_dir"]) / "predictions"
    gene_counts = [10, 20, 30, 50, 75, 100, 140, 200, 280]
    n_splits = 100
    seed_base = 42

    records = []

    for cohort_key in ["discovery", "validation"]:
        cohort_config = config["cohorts"][cohort_key]
        cohort_name = cohort_config["name"]
        n_folds = cohort_config["n_lopo_folds"]
        samples = cohort_config["samples"]

        logger.info(f"\nCohort: {cohort_name}")

        # Use ridge with averaged encoder residuals (same as D_cond definition)
        encoder_names = [e["name"] for e in config["encoders"]]
        all_encoder_residuals = []
        spot_ids = None

        for enc in encoder_names:
            residuals, sids = aggregate_residuals(
                pred_dir, cohort_name, enc, "ridge", n_folds
            )
            all_encoder_residuals.append(residuals)
            if spot_ids is None:
                spot_ids = sids

        # Average across encoders
        residuals = np.mean(all_encoder_residuals, axis=0)
        n_genes = residuals.shape[1]
        logger.info(f"  {residuals.shape[0]} spots × {n_genes} genes")

        # Build sample masks
        sample_masks = {}
        for sid in samples:
            prefix = f"{sid}_"
            mask = np.array([s.startswith(prefix) for s in spot_ids])
            if mask.sum() > 0:
                sample_masks[sid] = mask

        for n_per_track in gene_counts:
            if 2 * n_per_track > n_genes:
                continue

            logger.info(f"  Gene count per track: {n_per_track}")
            rng = np.random.RandomState(seed_base)

            for split_idx in range(n_splits):
                # Random split
                indices = rng.permutation(n_genes)
                genes_A = indices[:n_per_track]
                genes_B = indices[n_per_track:2 * n_per_track]

                D_A = compute_mean_absolute_discordance(residuals[:, genes_A])
                D_B = compute_mean_absolute_discordance(residuals[:, genes_B])

                for sid, mask in sample_masks.items():
                    if mask.sum() < 50:
                        continue
                    rho, _ = spearmanr(D_A[mask], D_B[mask])
                    records.append({
                        "cohort": cohort_key,
                        "cohort_name": cohort_name,
                        "sample_id": sid,
                        "n_genes_per_track": n_per_track,
                        "split_idx": split_idx,
                        "spearman_rho": float(rho),
                    })

    df = pd.DataFrame(records)
    return df


# ============================================================
# Analysis 2: Per-spot dual-track values
# ============================================================

def compute_dual_track_spots(config, logger, representative_samples=None):
    """Compute per-spot track A/B discordance for representative samples."""
    pred_dir = Path(config["output_dir"]) / "predictions"
    seed = config["seed"]

    if representative_samples is None:
        representative_samples = {
            "discovery": "TENX193",
            "validation": "NCBI785",
        }

    results = {}

    for cohort_key, sample_id in representative_samples.items():
        cohort_config = config["cohorts"][cohort_key]
        cohort_name = cohort_config["name"]
        n_folds = cohort_config["n_lopo_folds"]

        logger.info(f"\n  {sample_id} ({cohort_name})")

        encoder_names = [e["name"] for e in config["encoders"]]
        all_encoder_residuals = []
        spot_ids = None

        for enc in encoder_names:
            residuals, sids = aggregate_residuals(
                pred_dir, cohort_name, enc, "ridge", n_folds
            )
            all_encoder_residuals.append(residuals)
            if spot_ids is None:
                spot_ids = sids

        residuals = np.mean(all_encoder_residuals, axis=0)
        n_genes = residuals.shape[1]

        # Split genes
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n_genes)
        mid = n_genes // 2
        genes_A, genes_B = indices[:mid], indices[mid:]

        D_A = compute_mean_absolute_discordance(residuals[:, genes_A])
        D_B = compute_mean_absolute_discordance(residuals[:, genes_B])
        D_full = compute_mean_absolute_discordance(residuals)

        # Filter to this sample
        prefix = f"{sample_id}_"
        mask = np.array([s.startswith(prefix) for s in spot_ids])

        df = pd.DataFrame({
            "spot_id": [spot_ids[i] for i in np.where(mask)[0]],
            "D_track_A": D_A[mask],
            "D_track_B": D_B[mask],
            "D_full": D_full[mask],
        })

        rho, _ = spearmanr(df["D_track_A"], df["D_track_B"])
        logger.info(f"    ρ = {rho:.4f} (n={len(df)} spots, "
                    f"{len(genes_A)}/{len(genes_B)} genes)")

        results[sample_id] = df

    return results


# ============================================================
# Analysis 3: Moran's I null distributions
# ============================================================

def compute_morans_null(config, logger, representative_samples=None):
    """Re-run Moran's I permutation test and save null distribution."""
    if representative_samples is None:
        representative_samples = {
            "discovery": "TENX193",
            "validation": "NCBI785",
        }

    n_permutations = 999
    n_neighbors = config.get("phase2", {}).get("spatial_n_neighbors", 6)
    seed = config["seed"]

    results = {}

    for cohort_key, sample_id in representative_samples.items():
        cohort_name = config["cohorts"][cohort_key]["name"]
        scores_dir = Path(config["output_dir"]) / "phase2" / "scores" / cohort_name

        logger.info(f"\n  {sample_id}: computing {n_permutations} permutations...")

        disc_df = pd.read_parquet(scores_dir / f"{sample_id}_discordance.parquet")

        # Compute average D_cond
        ridge_cols = [c for c in disc_df.columns
                      if c.startswith("D_cond_") and c.endswith("_ridge")]
        D_cond = disc_df[ridge_cols].mean(axis=1).values
        coords = disc_df[["x", "y"]].values

        # Build weights
        W = build_spatial_weights(coords, n_neighbors=n_neighbors)

        # Observed
        I_obs = morans_i(D_cond, W)

        # Permutations
        rng = np.random.RandomState(seed)
        null_values = np.zeros(n_permutations)
        for i in range(n_permutations):
            perm = rng.permutation(len(D_cond))
            null_values[i] = morans_i(D_cond[perm], W)

        p_value = (1 + np.sum(null_values >= I_obs)) / (1 + n_permutations)

        logger.info(f"    I_obs = {I_obs:.4f}, p = {p_value:.4f}")
        logger.info(f"    Null range: [{null_values.min():.4f}, {null_values.max():.4f}]")

        results[sample_id] = {
            "I_obs": I_obs,
            "null_values": null_values,
            "p_value": p_value,
        }

    return results


# ============================================================
# Analysis 4: Per-gene within-patient residuals
# ============================================================

def compute_within_patient_per_gene(config, logger):
    """Compute mean |residual| per gene per sample for multi-section patients."""
    pred_dir = Path(config["output_dir"]) / "predictions"
    encoder_names = [e["name"] for e in config["encoders"]]

    records = []

    for cohort_key in ["discovery", "validation"]:
        cohort_config = config["cohorts"][cohort_key]
        cohort_name = cohort_config["name"]
        n_folds = cohort_config["n_lopo_folds"]

        # Load gene names
        gene_list_path = Path(config["v3_data_dir"]) / f"gene_list_{cohort_name}.json"
        with open(gene_list_path) as f:
            gene_names = json.load(f)

        for patient_id, patient_samples in cohort_config["patient_mapping"].items():
            if len(patient_samples) < 2:
                continue

            logger.info(f"\n  {patient_id}: {patient_samples}")

            for sample_id in patient_samples:
                # Average residuals across encoders
                sample_prefix = f"{sample_id}_"
                all_residuals = {}

                for enc in encoder_names:
                    for fold_idx in range(n_folds):
                        fold_dir = pred_dir / cohort_name / enc / "ridge" / f"fold{fold_idx}"
                        residuals = np.load(fold_dir / "test_residuals.npy")
                        with open(fold_dir / "test_spot_ids.json") as f:
                            spots = json.load(f)

                        for i, sid in enumerate(spots):
                            if sid.startswith(sample_prefix):
                                barcode = sid[len(sample_prefix):]
                                if barcode not in all_residuals:
                                    all_residuals[barcode] = []
                                all_residuals[barcode].append(residuals[i])

                if not all_residuals:
                    continue

                barcodes = sorted(all_residuals.keys())
                residual_matrix = np.array(
                    [np.mean(all_residuals[b], axis=0) for b in barcodes]
                )

                # Mean |residual| per gene (across spots)
                mean_abs_per_gene = np.mean(np.abs(residual_matrix), axis=0)

                for g_idx, gene in enumerate(gene_names):
                    records.append({
                        "patient_id": patient_id,
                        "cohort": cohort_name,
                        "sample_id": sample_id,
                        "gene": gene,
                        "mean_abs_residual": float(mean_abs_per_gene[g_idx]),
                    })

                logger.info(f"    {sample_id}: {len(barcodes)} spots, "
                            f"{len(gene_names)} genes")

    df = pd.DataFrame(records)
    return df


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Pre-compute data for figures")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--analysis", type=str, nargs="*", default=None,
                        help="Which analyses to run (subsampling, dual_track, morans, within_patient)")
    args = parser.parse_args()

    config = load_v3_config(args.config)

    log_dir = Path(config["output_dir"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(
        log_level=config.get("log_level", "INFO"),
        log_file=str(log_dir / "17_compute_figure_data.log"),
    )

    logger.info("=" * 60)
    logger.info("Pre-computing data for figures")
    logger.info("=" * 60)

    out_dir = Path(config["output_dir"]) / "figure_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    analyses = args.analysis or ["subsampling", "dual_track", "morans", "within_patient"]
    overall_start = time.time()

    # 1. Subsampling curve
    if "subsampling" in analyses:
        logger.info("\n--- Dual-track subsampling curve ---")
        t0 = time.time()
        subsampling_df = compute_subsampling_curve(config, logger)
        subsampling_df.to_csv(out_dir / "subsampling_curve.csv", index=False)
        logger.info(f"  Saved: {out_dir / 'subsampling_curve.csv'} "
                    f"({len(subsampling_df)} rows, {format_time(time.time() - t0)})")

    # 2. Dual-track per-spot values
    if "dual_track" in analyses:
        logger.info("\n--- Per-spot dual-track values ---")
        t0 = time.time()
        dt_results = compute_dual_track_spots(config, logger)
        for sample_id, df in dt_results.items():
            path = out_dir / f"dual_track_spots_{sample_id}.csv"
            df.to_csv(path, index=False)
            logger.info(f"  Saved: {path}")
        logger.info(f"  Time: {format_time(time.time() - t0)}")

    # 3. Moran's I null distributions
    if "morans" in analyses:
        logger.info("\n--- Moran's I null distributions ---")
        t0 = time.time()
        morans_results = compute_morans_null(config, logger)
        for sample_id, data in morans_results.items():
            np.save(out_dir / f"morans_null_{sample_id}.npy", data["null_values"])
            # Also save observed value
            with open(out_dir / f"morans_obs_{sample_id}.json", "w") as f:
                json.dump({"I_obs": data["I_obs"], "p_value": data["p_value"]}, f)
            logger.info(f"  Saved: morans_null_{sample_id}.npy")
        logger.info(f"  Time: {format_time(time.time() - t0)}")

    # 4. Within-patient per-gene residuals
    if "within_patient" in analyses:
        logger.info("\n--- Within-patient per-gene residuals ---")
        t0 = time.time()
        wp_df = compute_within_patient_per_gene(config, logger)
        wp_df.to_csv(out_dir / "within_patient_per_gene.csv", index=False)
        logger.info(f"  Saved: {out_dir / 'within_patient_per_gene.csv'} "
                    f"({len(wp_df)} rows, {format_time(time.time() - t0)})")

    total_time = time.time() - overall_start
    logger.info(f"\nTotal time: {format_time(total_time)}")
    logger.info("Done.")


if __name__ == "__main__":
    main()

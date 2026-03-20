#!/usr/bin/env python3
"""
Script 12: Within-Patient Reproducibility (Phase 3E, RQ4)

For 7 multi-section patients, compare discordance across serial sections.

Analyses:
  1. Per-gene residual Pearson between sections from the same patient
     (do the same genes have high/low residuals across sections?)
  2. Spot-level discordance rank correlation (after coordinate normalization
     and spatial binning)
  3. Dice coefficient on top-quartile discordance regions (with binned coords)
  4. Summary: which patients show reproducible discordance patterns?

Multi-section patients:
  - P01: [TENX99, TENX98]         (2 sections)
  - P02: [TENX97, TENX95]         (2 sections)
  - P03: [TENX193, TENX192, TENX191]  (3 sections)
  - P04: [TENX196, TENX195]       (2 sections)
  - P05: [TENX199, TENX198, TENX197]  (3 sections)
  - P06: [TENX202, TENX201, TENX200]  (3 sections)
  - P07: [NCBI785, NCBI784]       (2 sections)

P08 (NCBI783) has only 1 section and is excluded.

Output:
    outputs/phase3/within_patient/
        gene_residual_correlation.csv   (per-patient, per-section-pair)
        spatial_binned_correlation.csv  (binned spot-level)
        dice_top_quartile.csv           (spatial overlap)
        within_patient_summary.json

Usage:
    python scripts/12_within_patient.py
"""

import argparse
import json
import sys
import time
from itertools import combinations
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


def load_residuals_for_sample(sample_id, cohort_name, pred_dir, n_folds):
    """Load aggregated residuals for a sample (ridge, all 3 encoders averaged)."""
    encoder_names = ["uni", "virchow2", "hoptimus0"]
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

    # Average across encoders (each spot has 3 residual vectors, one per encoder)
    barcodes = sorted(all_residuals.keys())
    residuals = np.array([np.mean(all_residuals[b], axis=0) for b in barcodes])
    return residuals, barcodes


def load_discordance_for_sample(sample_id, scores_dir):
    """Load discordance scores."""
    path = scores_dir / f"{sample_id}_discordance.parquet"
    return pd.read_parquet(path)


def normalize_coordinates(coords):
    """Normalize coordinates to [0, 1] range."""
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    return (coords - mins) / ranges


def spatial_binning(coords_norm, values, n_bins=20):
    """Bin spots into a spatial grid and average values within each bin.

    Returns: bin_values (n_bins, n_bins), bin_counts (n_bins, n_bins)
    """
    bins = np.linspace(0, 1, n_bins + 1)
    x_bins = np.digitize(coords_norm[:, 0], bins) - 1
    y_bins = np.digitize(coords_norm[:, 1], bins) - 1

    # Clip to valid range
    x_bins = np.clip(x_bins, 0, n_bins - 1)
    y_bins = np.clip(y_bins, 0, n_bins - 1)

    bin_sums = np.zeros((n_bins, n_bins))
    bin_counts = np.zeros((n_bins, n_bins))

    for i in range(len(values)):
        bin_sums[x_bins[i], y_bins[i]] += values[i]
        bin_counts[x_bins[i], y_bins[i]] += 1

    # Average
    bin_values = np.full((n_bins, n_bins), np.nan)
    mask = bin_counts > 0
    bin_values[mask] = bin_sums[mask] / bin_counts[mask]

    return bin_values, bin_counts


def compute_gene_level_correlation(residuals_a, residuals_b):
    """Compute per-gene correlation of mean |residual| between two sections.

    For each gene, compute mean |residual| across spots in each section,
    then correlate the gene-level profiles.
    """
    # Mean absolute residual per gene
    mean_abs_a = np.mean(np.abs(residuals_a), axis=0)
    mean_abs_b = np.mean(np.abs(residuals_b), axis=0)

    if np.std(mean_abs_a) < 1e-10 or np.std(mean_abs_b) < 1e-10:
        return 0.0, 1.0

    r, p = stats.pearsonr(mean_abs_a, mean_abs_b)
    return float(r), float(p)


def compute_binned_spatial_correlation(disc_df_a, disc_df_b, n_bins=20):
    """Correlate binned discordance maps between two sections."""
    ridge_cols = [c for c in disc_df_a.columns if c.startswith("D_cond_") and c.endswith("_ridge")]

    for df in [disc_df_a, disc_df_b]:
        df["D_cond_mean"] = df[ridge_cols].mean(axis=1)

    coords_a = disc_df_a[["x", "y"]].values
    coords_b = disc_df_b[["x", "y"]].values

    norm_a = normalize_coordinates(coords_a)
    norm_b = normalize_coordinates(coords_b)

    bins_a, counts_a = spatial_binning(norm_a, disc_df_a["D_cond_mean"].values, n_bins)
    bins_b, counts_b = spatial_binning(norm_b, disc_df_b["D_cond_mean"].values, n_bins)

    # Compare only bins that have data in both sections
    mask = (~np.isnan(bins_a)) & (~np.isnan(bins_b))
    n_common_bins = mask.sum()

    if n_common_bins < 10:
        return 0.0, 1.0, int(n_common_bins)

    r, p = stats.spearmanr(bins_a[mask], bins_b[mask])
    return float(r), float(p), int(n_common_bins)


def compute_dice_top_quartile(disc_df_a, disc_df_b, n_bins=20):
    """Dice coefficient on top-quartile discordance regions after spatial binning."""
    ridge_cols = [c for c in disc_df_a.columns if c.startswith("D_cond_") and c.endswith("_ridge")]

    for df in [disc_df_a, disc_df_b]:
        if "D_cond_mean" not in df.columns:
            df["D_cond_mean"] = df[ridge_cols].mean(axis=1)

    coords_a = disc_df_a[["x", "y"]].values
    coords_b = disc_df_b[["x", "y"]].values

    norm_a = normalize_coordinates(coords_a)
    norm_b = normalize_coordinates(coords_b)

    bins_a, counts_a = spatial_binning(norm_a, disc_df_a["D_cond_mean"].values, n_bins)
    bins_b, counts_b = spatial_binning(norm_b, disc_df_b["D_cond_mean"].values, n_bins)

    # Threshold each map at 75th percentile
    valid_a = ~np.isnan(bins_a)
    valid_b = ~np.isnan(bins_b)

    if valid_a.sum() < 5 or valid_b.sum() < 5:
        return 0.0

    q75_a = np.percentile(bins_a[valid_a], 75)
    q75_b = np.percentile(bins_b[valid_b], 75)

    high_a = valid_a & (bins_a >= q75_a)
    high_b = valid_b & (bins_b >= q75_b)

    intersection = (high_a & high_b).sum()
    union_sum = high_a.sum() + high_b.sum()

    dice = 2 * intersection / union_sum if union_sum > 0 else 0
    return float(dice)


def main():
    parser = argparse.ArgumentParser(description="Phase 3E: Within-patient reproducibility")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--n-bins", type=int, default=20, help="Number of spatial bins per axis")
    args = parser.parse_args()

    config = load_v3_config(args.config)

    log_dir = Path(config["output_dir"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(
        log_level=config.get("log_level", "INFO"),
        log_file=str(log_dir / "12_within_patient.log"),
    )

    logger.info("=" * 60)
    logger.info("Phase 3E: Within-Patient Reproducibility")
    logger.info("=" * 60)

    pred_dir = Path(config["output_dir"]) / "predictions"
    out_dir = Path(config["output_dir"]) / "phase3" / "within_patient"
    out_dir.mkdir(parents=True, exist_ok=True)

    overall_start = time.time()

    # Build patient -> (cohort_key, cohort_name, samples) mapping for multi-section patients
    multi_section_patients = []
    for cohort_key in ["discovery", "validation"]:
        cohort_config = config["cohorts"][cohort_key]
        cohort_name = cohort_config["name"]
        n_folds = cohort_config["n_lopo_folds"]
        scores_dir = Path(config["output_dir"]) / "phase2" / "scores" / cohort_name

        for patient_id, patient_samples in cohort_config["patient_mapping"].items():
            if len(patient_samples) >= 2:
                multi_section_patients.append({
                    "patient_id": patient_id,
                    "samples": patient_samples,
                    "cohort_key": cohort_key,
                    "cohort_name": cohort_name,
                    "n_folds": n_folds,
                    "scores_dir": scores_dir,
                })

    logger.info(f"Multi-section patients: {len(multi_section_patients)}")
    for p in multi_section_patients:
        logger.info(f"  {p['patient_id']}: {p['samples']} ({p['cohort_name']})")

    gene_corr_records = []
    spatial_corr_records = []
    dice_records = []

    for patient_info in multi_section_patients:
        patient_id = patient_info["patient_id"]
        patient_samples = patient_info["samples"]
        cohort_name = patient_info["cohort_name"]
        n_folds = patient_info["n_folds"]
        scores_dir = patient_info["scores_dir"]

        logger.info(f"\nPatient {patient_id} ({cohort_name}): {patient_samples}")

        # Load residuals and discordance for each section
        section_residuals = {}
        section_disc = {}

        for sid in patient_samples:
            residuals, barcodes = load_residuals_for_sample(
                sid, cohort_name, pred_dir, n_folds
            )
            section_residuals[sid] = (residuals, barcodes)
            section_disc[sid] = load_discordance_for_sample(sid, scores_dir)
            logger.info(f"  {sid}: {residuals.shape[0]} spots × {residuals.shape[1]} genes")

        # Pairwise comparisons
        for sid_a, sid_b in combinations(patient_samples, 2):
            logger.info(f"  Comparing {sid_a} vs {sid_b}:")

            # 1. Gene-level residual correlation
            res_a, _ = section_residuals[sid_a]
            res_b, _ = section_residuals[sid_b]
            gene_r, gene_p = compute_gene_level_correlation(res_a, res_b)
            gene_corr_records.append({
                "patient_id": patient_id,
                "cohort": cohort_name,
                "section_a": sid_a,
                "section_b": sid_b,
                "pearson_r": gene_r,
                "pvalue": gene_p,
                "n_genes": res_a.shape[1],
            })
            logger.info(f"    Gene residual Pearson: r={gene_r:.3f} (p={gene_p:.2e})")

            # 2. Binned spatial correlation
            disc_a = section_disc[sid_a].copy()
            disc_b = section_disc[sid_b].copy()
            spatial_r, spatial_p, n_bins_common = compute_binned_spatial_correlation(
                disc_a, disc_b, n_bins=args.n_bins
            )
            spatial_corr_records.append({
                "patient_id": patient_id,
                "cohort": cohort_name,
                "section_a": sid_a,
                "section_b": sid_b,
                "spearman_r": spatial_r,
                "pvalue": spatial_p,
                "n_common_bins": n_bins_common,
                "n_bins": args.n_bins,
            })
            logger.info(f"    Spatial binned Spearman: r={spatial_r:.3f} (p={spatial_p:.2e}, {n_bins_common} bins)")

            # 3. Dice coefficient on top quartile
            dice = compute_dice_top_quartile(disc_a, disc_b, n_bins=args.n_bins)
            dice_records.append({
                "patient_id": patient_id,
                "cohort": cohort_name,
                "section_a": sid_a,
                "section_b": sid_b,
                "dice_coeff": dice,
                "n_bins": args.n_bins,
            })
            logger.info(f"    Dice (top 25%): {dice:.3f}")

    # Save results
    gene_corr_df = pd.DataFrame(gene_corr_records)
    spatial_corr_df = pd.DataFrame(spatial_corr_records)
    dice_df = pd.DataFrame(dice_records)

    gene_corr_df.to_csv(out_dir / "gene_residual_correlation.csv", index=False)
    spatial_corr_df.to_csv(out_dir / "spatial_binned_correlation.csv", index=False)
    dice_df.to_csv(out_dir / "dice_top_quartile.csv", index=False)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)

    summary = {
        "n_patients": len(multi_section_patients),
        "n_section_pairs": len(gene_corr_records),
        "n_bins": args.n_bins,
    }

    # Per-patient summary
    patient_summaries = []
    for patient_info in multi_section_patients:
        pid = patient_info["patient_id"]
        gene_rows = gene_corr_df[gene_corr_df["patient_id"] == pid]
        spatial_rows = spatial_corr_df[spatial_corr_df["patient_id"] == pid]
        dice_rows = dice_df[dice_df["patient_id"] == pid]

        ps = {
            "patient_id": pid,
            "cohort": patient_info["cohort_name"],
            "n_sections": len(patient_info["samples"]),
            "n_pairs": len(gene_rows),
            "mean_gene_pearson": float(gene_rows["pearson_r"].mean()),
            "mean_spatial_spearman": float(spatial_rows["spearman_r"].mean()),
            "mean_dice": float(dice_rows["dice_coeff"].mean()),
        }
        patient_summaries.append(ps)

        logger.info(
            f"  {pid}: gene_r={ps['mean_gene_pearson']:.3f}, "
            f"spatial_r={ps['mean_spatial_spearman']:.3f}, "
            f"dice={ps['mean_dice']:.3f}"
        )

    summary["per_patient"] = patient_summaries
    summary["overall_mean_gene_pearson"] = float(gene_corr_df["pearson_r"].mean())
    summary["overall_mean_spatial_spearman"] = float(spatial_corr_df["spearman_r"].mean())
    summary["overall_mean_dice"] = float(dice_df["dice_coeff"].mean())

    logger.info(f"\n  Overall: gene_r={summary['overall_mean_gene_pearson']:.3f}, "
                f"spatial_r={summary['overall_mean_spatial_spearman']:.3f}, "
                f"dice={summary['overall_mean_dice']:.3f}")

    with open(out_dir / "within_patient_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    total_time = time.time() - overall_start
    logger.info(f"\nTotal time: {format_time(total_time)}")
    logger.info("Done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Script 09: Pathway Enrichment Analysis (Phase 3B)

For each sample and config:
  1. Compute signed residuals per gene per spot from stored predictions
  2. Compute pathway-level signed residual scores (mean signed residual per pathway)
  3. Compute studentized gene-level residuals and studentized pathway scores
  4. Test pathway scores in discordant vs concordant spots (Wilcoxon)
  5. Cross-sample consistency: direction and significance of each pathway

Uses Hallmark gene sets from MSigDB with min_overlap=5.

Output:
    outputs/phase3/pathways/{cohort}/
        per_sample/{sample_id}_pathway_scores.csv
        pathway_de.csv              (per-sample pathway-level DE)
        pathway_consistency.csv     (cross-sample direction consistency)
        pathway_overlap.csv         (gene overlap with panel)
        pathway_summary.json

Usage:
    python scripts/09_pathway_enrichment.py
    python scripts/09_pathway_enrichment.py --cohort discovery
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
from src.pathways import (
    load_gene_sets,
    compute_pathway_signed_residuals,
    compute_studentized_gene_residuals,
    compute_studentized_pathway_scores,
)
from src.de_analysis import partition_spots_by_discordance
from scipy import stats
from statsmodels.stats.multitest import multipletests


def load_v3_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def aggregate_residuals_for_sample(sample_id, cohort_name, pred_dir, n_folds):
    """Load and aggregate residuals + targets across folds for a sample.

    Uses ridge regressor averaged across 3 encoders.
    Returns: residuals (N, G), targets (N, G), gene_names, spot_ids
    """
    encoder_names = ["uni", "virchow2", "hoptimus0"]

    # First pass: get spot IDs and aggregate
    sample_prefix = f"{sample_id}_"
    all_residuals = {}  # spot_id -> list of residual vectors (one per encoder)

    for enc in encoder_names:
        enc_residuals = []
        enc_targets = []
        enc_spots = []

        for fold_idx in range(n_folds):
            fold_dir = pred_dir / cohort_name / enc / "ridge" / f"fold{fold_idx}"
            residuals = np.load(fold_dir / "test_residuals.npy")
            targets = np.load(fold_dir / "test_targets.npy")
            with open(fold_dir / "test_spot_ids.json") as f:
                spots = json.load(f)

            # Filter to this sample's spots
            for i, sid in enumerate(spots):
                if sid.startswith(sample_prefix):
                    enc_residuals.append(residuals[i])
                    enc_targets.append(targets[i])
                    enc_spots.append(sid)

        # Index by spot_id
        for i, sid in enumerate(enc_spots):
            if sid not in all_residuals:
                all_residuals[sid] = {"residuals": [], "target": enc_targets[i]}
            all_residuals[sid]["residuals"].append(enc_residuals[i])

    # Average residuals across encoders
    spot_ids = sorted(all_residuals.keys())
    residuals = np.array([np.mean(all_residuals[s]["residuals"], axis=0) for s in spot_ids])
    targets = np.array([all_residuals[s]["target"] for s in spot_ids])

    # Get gene names from metrics
    metrics_path = pred_dir / cohort_name / "uni" / "ridge" / "fold0" / "metrics.json"
    with open(metrics_path) as f:
        metrics = json.load(f)
    gene_names = [k for k in metrics.keys() if not k.startswith("__")]

    return residuals, targets, gene_names, spot_ids


def pathway_de_test(pathway_scores, disc_mask, conc_mask, pathway_name):
    """Wilcoxon rank-sum test on pathway scores between discordant and concordant."""
    disc_vals = pathway_scores[disc_mask]
    conc_vals = pathway_scores[conc_mask]

    mean_disc = np.mean(disc_vals)
    mean_conc = np.mean(conc_vals)

    try:
        stat, pval = stats.mannwhitneyu(disc_vals, conc_vals, alternative="two-sided")
    except ValueError:
        pval = 1.0

    # Effect size (pooled within-group SD, consistent with wilcoxon_de)
    n_d, n_c = len(disc_vals), len(conc_vals)
    var_d = np.var(disc_vals, ddof=1) if n_d > 1 else 0
    var_c = np.var(conc_vals, ddof=1) if n_c > 1 else 0
    pooled_var = ((var_d * (n_d - 1) + var_c * (n_c - 1)) / (n_d + n_c - 2)) if (n_d + n_c) > 2 else 0
    pooled_sd = np.sqrt(pooled_var) if pooled_var > 0 else 1e-6
    cohens_d = (mean_disc - mean_conc) / pooled_sd

    return {
        "pathway": pathway_name,
        "mean_disc": float(mean_disc),
        "mean_conc": float(mean_conc),
        "delta": float(mean_disc - mean_conc),
        "cohens_d": float(cohens_d),
        "pval": float(pval),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 3B: Pathway enrichment")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--cohort", type=str, default=None)
    args = parser.parse_args()

    config = load_v3_config(args.config)

    log_dir = Path(config["output_dir"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(
        log_level=config.get("log_level", "INFO"),
        log_file=str(log_dir / "09_pathway_enrichment.log"),
    )

    logger.info("=" * 60)
    logger.info("Phase 3B: Pathway Enrichment Analysis")
    logger.info("=" * 60)

    # Load gene sets
    gmt_path = config.get("phase3", {}).get("hallmark_gmt", "data/gene_sets/h.all.v2024.1.Hs.symbols.gmt")
    gene_sets = load_gene_sets(gmt_path)
    min_overlap = config.get("phase2", {}).get("pathway_min_overlap", 5)
    logger.info(f"Loaded {len(gene_sets)} Hallmark gene sets (min overlap: {min_overlap})")

    cohort_keys = [args.cohort] if args.cohort else ["discovery", "validation"]
    pred_dir = Path(config["output_dir"]) / "predictions"
    overall_start = time.time()

    for cohort_key in cohort_keys:
        cohort_config = config["cohorts"][cohort_key]
        cohort_name = cohort_config["name"]
        n_folds = cohort_config["n_lopo_folds"]
        samples = cohort_config["samples"]
        scores_dir = Path(config["output_dir"]) / "phase2" / "scores" / cohort_name
        out_dir = Path(config["output_dir"]) / "phase3" / "pathways" / cohort_name
        sample_dir = out_dir / "per_sample"
        sample_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\nCohort: {cohort_name} ({cohort_key}), {len(samples)} samples")

        # Compute pathway overlap with this cohort's gene panel
        gene_list_path = Path(config["v3_data_dir"]) / f"gene_list_{cohort_name}.json"
        with open(gene_list_path) as f:
            panel_genes = json.load(f)

        overlap_records = []
        for pname, pgenes in gene_sets.items():
            overlap = [g for g in pgenes if g in set(panel_genes)]
            overlap_records.append({
                "pathway": pname,
                "n_overlap": len(overlap),
                "n_total": len(pgenes),
                "frac_overlap": len(overlap) / len(pgenes) if pgenes else 0,
                "passes_threshold": len(overlap) >= min_overlap,
                "genes_used": ",".join(sorted(overlap)),
            })
        overlap_df = pd.DataFrame(overlap_records).sort_values("n_overlap", ascending=False)
        overlap_df.to_csv(out_dir / "pathway_overlap.csv", index=False)
        n_pass = overlap_df["passes_threshold"].sum()
        logger.info(f"  Pathways passing overlap threshold: {n_pass}/{len(gene_sets)}")

        # Per-sample analysis
        all_pathway_de = []

        for sample_id in samples:
            logger.info(f"  Sample: {sample_id}")

            # Load residuals and targets
            residuals, targets, gene_names, spot_ids = aggregate_residuals_for_sample(
                sample_id, cohort_name, pred_dir, n_folds
            )
            logger.info(f"    {residuals.shape[0]} spots × {residuals.shape[1]} genes")

            # Load D_cond for partitioning
            disc_df = pd.read_parquet(scores_dir / f"{sample_id}_discordance.parquet")
            ridge_cols = [c for c in disc_df.columns if c.startswith("D_cond_") and c.endswith("_ridge")]
            disc_df["D_cond_ridge_mean"] = disc_df[ridge_cols].mean(axis=1)

            # Align discordance with residuals (same spot order)
            disc_spot_to_dcond = dict(zip(disc_df["spot_id"], disc_df["D_cond_ridge_mean"]))
            aligned_dcond = np.array([disc_spot_to_dcond.get(s, np.nan) for s in spot_ids])
            valid = ~np.isnan(aligned_dcond)
            if valid.sum() < residuals.shape[0]:
                logger.info(f"    Dropping {(~valid).sum()} spots without D_cond")
                residuals = residuals[valid]
                targets = targets[valid]
                aligned_dcond = aligned_dcond[valid]
                spot_ids = [s for s, v in zip(spot_ids, valid) if v]

            # Partition
            disc_mask, conc_mask = partition_spots_by_discordance(aligned_dcond)

            # Compute pathway scores
            scores_df, _ = compute_pathway_signed_residuals(
                residuals, gene_names, gene_sets, min_overlap=min_overlap
            )

            # Compute studentized pathway scores
            stud_resid = compute_studentized_gene_residuals(residuals, targets)
            stud_pathway_scores = compute_studentized_pathway_scores(
                stud_resid, gene_names, gene_sets, min_overlap=min_overlap
            )

            # Save per-sample pathway scores (studentized)
            if stud_pathway_scores:
                stud_df = pd.DataFrame(stud_pathway_scores)
                stud_df.insert(0, "spot_id", spot_ids[:len(stud_df)])
                stud_df.to_csv(sample_dir / f"{sample_id}_pathway_scores.csv", index=False)

            # Pathway-level DE (studentized scores)
            sample_de_records = []
            for pname, pscores in stud_pathway_scores.items():
                result = pathway_de_test(pscores, disc_mask, conc_mask, pname)
                result["sample_id"] = sample_id
                sample_de_records.append(result)

            if sample_de_records:
                sample_de_df = pd.DataFrame(sample_de_records)
                _, fdr, _, _ = multipletests(sample_de_df["pval"], method="fdr_bh")
                sample_de_df["fdr"] = fdr
                all_pathway_de.append(sample_de_df)
                n_sig = (sample_de_df["fdr"] < 0.05).sum()
                logger.info(f"    Pathways tested: {len(sample_de_df)}, FDR<0.05: {n_sig}")

        # Aggregate pathway DE across samples
        if all_pathway_de:
            combined_de = pd.concat(all_pathway_de, ignore_index=True)
            combined_de.to_csv(out_dir / "pathway_de.csv", index=False)

            # Cross-sample consistency
            consistency_records = []
            for pname in combined_de["pathway"].unique():
                pdata = combined_de[combined_de["pathway"] == pname]
                n_samples = len(pdata)
                n_sig = (pdata["fdr"] < 0.05).sum()
                n_positive = (pdata["delta"] > 0).sum()
                n_negative = (pdata["delta"] < 0).sum()
                direction_consistency = max(n_positive, n_negative) / n_samples if n_samples > 0 else 0
                median_delta = pdata["delta"].median()
                mean_cohens_d = pdata["cohens_d"].mean()

                consistency_records.append({
                    "pathway": pname,
                    "n_samples": n_samples,
                    "n_sig_fdr05": n_sig,
                    "reproducibility": n_sig / n_samples if n_samples > 0 else 0,
                    "n_positive_delta": n_positive,
                    "n_negative_delta": n_negative,
                    "direction_consistency": direction_consistency,
                    "median_delta": float(median_delta),
                    "mean_cohens_d": float(mean_cohens_d),
                })

            consistency_df = pd.DataFrame(consistency_records).sort_values(
                "reproducibility", ascending=False
            )
            consistency_df.to_csv(out_dir / "pathway_consistency.csv", index=False)

            # Summary
            top_pathways = consistency_df.head(10).to_dict(orient="records")
            summary = {
                "cohort": cohort_name,
                "n_samples": len(samples),
                "n_pathways_tested": len(consistency_df),
                "n_pathways_passing_overlap": int(n_pass),
                "min_overlap": min_overlap,
                "top_10_pathways": top_pathways,
            }

            with open(out_dir / "pathway_summary.json", "w") as f:
                json.dump(summary, f, indent=2)

            logger.info(f"\n  Top reproducible pathways:")
            for rec in consistency_df.head(5).itertuples():
                logger.info(
                    f"    {rec.pathway}: repro={rec.reproducibility:.2f}, "
                    f"dir_cons={rec.direction_consistency:.2f}, "
                    f"d={rec.mean_cohens_d:.3f}"
                )

    total_time = time.time() - overall_start
    logger.info(f"\nTotal time: {format_time(total_time)}")
    logger.info("Done.")


if __name__ == "__main__":
    main()

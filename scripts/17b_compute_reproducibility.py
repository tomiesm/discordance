#!/usr/bin/env python3
"""
Script 17b: Compute within-patient reproducibility analyses.

Three complementary analyses replacing spatial serial-section analysis:
  1. Patient identification from discordance profiles (LOO-NN + clustering)
  2. Distributional reproducibility (KS statistic + overlap coefficient)
  3. Pathway-level reproducibility (continuous enrichment profiles)

Output:
    outputs/figure_data/reproducibility/
        patient_id_discovery.json
        patient_id_validation.json
        patient_id_cross_cohort.json
        distributional_reproducibility.csv
        distributional_summary.json
        pathway_repro_profiles.csv
        pathway_repro_pairwise.csv
        pathway_repro_summary.json

Usage:
    python scripts/17b_compute_reproducibility.py
    python scripts/17b_compute_reproducibility.py --analysis patient_id
    python scripts/17b_compute_reproducibility.py --analysis distributional
    python scripts/17b_compute_reproducibility.py --analysis pathway_repro
"""

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde, mannwhitneyu
from sklearn.metrics import adjusted_rand_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.utils import setup_logging, format_time


def load_v3_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_patient_for_sample(config, sample_id):
    """Look up patient ID for a given sample."""
    for cohort_key in ["discovery", "validation"]:
        mapping = config["cohorts"][cohort_key].get("patient_mapping", {})
        for patient_id, samples in mapping.items():
            if sample_id in samples:
                return patient_id
    return None


def get_cohort_for_sample(config, sample_id):
    """Look up cohort key for a given sample."""
    for cohort_key in ["discovery", "validation"]:
        if sample_id in config["cohorts"][cohort_key]["samples"]:
            return cohort_key
    return None


# ============================================================
# Analysis 1: Patient Identification from Discordance Profiles
# ============================================================

def loo_nn_classify(dist_matrix, labels):
    """Leave-one-out nearest-neighbor classification.

    Args:
        dist_matrix: (n, n) symmetric distance matrix.
        labels: (n,) array of true labels.

    Returns:
        predictions: (n,) predicted labels.
        accuracy: int, number correct.
    """
    n = len(labels)
    predictions = []
    for i in range(n):
        dists = dist_matrix[i].copy()
        dists[i] = np.inf  # exclude self
        nn_idx = np.argmin(dists)
        predictions.append(labels[nn_idx])
    accuracy = sum(p == t for p, t in zip(predictions, labels))
    return predictions, accuracy


def compute_patient_identification(config, logger, cohort_key, out_dir):
    """Analysis 1: Patient identification via residual profile clustering."""
    base = Path(config["output_dir"])
    cohort_config = config["cohorts"][cohort_key]
    cohort_name = cohort_config["name"]
    samples = cohort_config["samples"]
    patient_mapping = cohort_config["patient_mapping"]

    logger.info(f"\n=== Patient Identification: {cohort_key} ({cohort_name}) ===")

    # Load per-gene mean absolute residuals
    wp_path = base / "figure_data" / "within_patient_per_gene.csv"
    wp_df = pd.read_csv(wp_path)

    # Filter to this cohort
    cohort_df = wp_df[wp_df["cohort"] == cohort_name]
    cohort_samples = sorted(cohort_df["sample_id"].unique())
    logger.info(f"  Sections: {len(cohort_samples)}")

    # Build sample → patient mapping
    sample_to_patient = {}
    for pid, sids in patient_mapping.items():
        for sid in sids:
            sample_to_patient[sid] = pid

    # Filter to samples that have patient mappings (multi-section only)
    cohort_samples = [s for s in cohort_samples if s in sample_to_patient]

    # Pivot to (n_sections, n_genes) matrix
    pivot = cohort_df[cohort_df["sample_id"].isin(cohort_samples)].pivot(
        index="sample_id", columns="gene", values="mean_abs_residual"
    )
    pivot = pivot.loc[cohort_samples]  # enforce order
    gene_names = list(pivot.columns)
    section_labels = list(pivot.index)
    patient_labels = [sample_to_patient[s] for s in section_labels]

    logger.info(f"  Matrix shape: {pivot.shape}")
    logger.info(f"  Patients: {sorted(set(patient_labels))}")

    # Z-score per gene (column)
    z_matrix = pivot.values.copy().astype(np.float64)
    col_means = z_matrix.mean(axis=0)
    col_stds = z_matrix.std(axis=0)
    col_stds[col_stds == 0] = 1.0  # avoid div-by-zero for constant genes
    z_matrix = (z_matrix - col_means) / col_stds

    # Pairwise distance matrix (1 - Pearson correlation)
    corr_dist_condensed = pdist(z_matrix, metric='correlation')
    corr_dist = squareform(corr_dist_condensed)

    # Hierarchical clustering (Ward's method)
    linkage_matrix = linkage(corr_dist_condensed, method='ward')

    # LOO-NN classification
    predictions, accuracy = loo_nn_classify(corr_dist, patient_labels)
    n_total = len(patient_labels)
    logger.info(f"  LOO-NN accuracy: {accuracy}/{n_total} ({accuracy/n_total:.1%})")

    # Permutation null (10,000 shuffles)
    n_perm = 10000
    rng = np.random.RandomState(42)
    null_accuracies = []
    for _ in range(n_perm):
        shuffled = list(rng.permutation(patient_labels))
        _, perm_acc = loo_nn_classify(corr_dist, shuffled)
        null_accuracies.append(perm_acc)

    pvalue = (1 + sum(na >= accuracy for na in null_accuracies)) / (1 + n_perm)
    logger.info(f"  Permutation p-value: {pvalue:.4f}")

    # ARI: cut dendrogram at k = n_patients
    n_patients = len(set(patient_labels))
    cluster_labels = fcluster(linkage_matrix, t=n_patients, criterion='maxclust')
    ari = adjusted_rand_score(patient_labels, cluster_labels)
    logger.info(f"  ARI: {ari:.3f}")

    # Confusion matrix
    unique_patients = sorted(set(patient_labels))
    cm = confusion_matrix(patient_labels, predictions, labels=unique_patients)

    # Save results
    result = {
        "cohort": cohort_key,
        "cohort_name": cohort_name,
        "n_sections": n_total,
        "n_patients": n_patients,
        "section_labels": section_labels,
        "patient_labels": patient_labels,
        "gene_names": gene_names,
        "z_matrix": z_matrix.tolist(),
        "linkage_matrix": linkage_matrix.tolist(),
        "distance_matrix": corr_dist.tolist(),
        "loo_nn": {
            "accuracy": accuracy,
            "total": n_total,
            "predictions": predictions,
            "fraction": accuracy / n_total,
        },
        "permutation": {
            "n_permutations": n_perm,
            "null_accuracies": null_accuracies,
            "pvalue": pvalue,
        },
        "ari": ari,
        "confusion_matrix": cm.tolist(),
        "unique_patients": unique_patients,
    }

    out_path = out_dir / f"patient_id_{cohort_key}.json"
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f"  Saved: {out_path}")

    return result


def compute_patient_identification_cross_cohort(config, logger, out_dir):
    """Analysis 1 cross-cohort: all multi-section samples × 90 bridge genes."""
    base = Path(config["output_dir"])

    logger.info("\n=== Patient Identification: Cross-Cohort (bridge genes) ===")

    # Load bridge gene list
    bridge_path = Path("data/v3/gene_list_bridge.json")
    with open(bridge_path) as f:
        bridge_genes = json.load(f)
    logger.info(f"  Bridge genes: {len(bridge_genes)}")

    # Load per-gene residuals
    wp_path = base / "figure_data" / "within_patient_per_gene.csv"
    wp_df = pd.read_csv(wp_path)

    # Filter to bridge genes only
    wp_bridge = wp_df[wp_df["gene"].isin(bridge_genes)]

    # Build sample → patient + cohort mappings
    sample_to_patient = {}
    sample_to_cohort = {}
    for cohort_key in ["discovery", "validation"]:
        mapping = config["cohorts"][cohort_key].get("patient_mapping", {})
        for pid, sids in mapping.items():
            for sid in sids:
                sample_to_patient[sid] = pid
                sample_to_cohort[sid] = cohort_key

    # All multi-section samples
    all_samples = sorted(wp_bridge["sample_id"].unique())
    all_samples = [s for s in all_samples if s in sample_to_patient]

    # Pivot
    pivot = wp_bridge[wp_bridge["sample_id"].isin(all_samples)].pivot(
        index="sample_id", columns="gene", values="mean_abs_residual"
    )
    # Only keep genes present for all samples (bridge genes should be in both panels)
    pivot = pivot.dropna(axis=1)
    pivot = pivot.loc[all_samples]
    gene_names = list(pivot.columns)
    section_labels = list(pivot.index)
    patient_labels = [sample_to_patient[s] for s in section_labels]
    cohort_labels = [sample_to_cohort[s] for s in section_labels]

    logger.info(f"  Matrix shape: {pivot.shape}")
    logger.info(f"  Patients: {sorted(set(patient_labels))}")

    # Z-score per gene
    z_matrix = pivot.values.copy().astype(np.float64)
    col_means = z_matrix.mean(axis=0)
    col_stds = z_matrix.std(axis=0)
    col_stds[col_stds == 0] = 1.0
    z_matrix = (z_matrix - col_means) / col_stds

    # Pairwise distance, clustering, LOO-NN
    corr_dist_condensed = pdist(z_matrix, metric='correlation')
    corr_dist = squareform(corr_dist_condensed)
    linkage_matrix = linkage(corr_dist_condensed, method='ward')

    predictions, accuracy = loo_nn_classify(corr_dist, patient_labels)
    n_total = len(patient_labels)
    logger.info(f"  LOO-NN accuracy: {accuracy}/{n_total} ({accuracy/n_total:.1%})")

    # Permutation null
    n_perm = 10000
    rng = np.random.RandomState(42)
    null_accuracies = []
    for _ in range(n_perm):
        shuffled = list(rng.permutation(patient_labels))
        _, perm_acc = loo_nn_classify(corr_dist, shuffled)
        null_accuracies.append(perm_acc)
    pvalue = (1 + sum(na >= accuracy for na in null_accuracies)) / (1 + n_perm)
    logger.info(f"  Permutation p-value: {pvalue:.4f}")

    n_patients = len(set(patient_labels))
    cluster_labels = fcluster(linkage_matrix, t=n_patients, criterion='maxclust')
    ari = adjusted_rand_score(patient_labels, cluster_labels)
    logger.info(f"  ARI: {ari:.3f}")

    unique_patients = sorted(set(patient_labels))
    cm = confusion_matrix(patient_labels, predictions, labels=unique_patients)

    result = {
        "cohort": "cross_cohort",
        "n_sections": n_total,
        "n_patients": n_patients,
        "n_bridge_genes": len(gene_names),
        "section_labels": section_labels,
        "patient_labels": patient_labels,
        "cohort_labels": cohort_labels,
        "gene_names": gene_names,
        "z_matrix": z_matrix.tolist(),
        "linkage_matrix": linkage_matrix.tolist(),
        "distance_matrix": corr_dist.tolist(),
        "loo_nn": {
            "accuracy": accuracy,
            "total": n_total,
            "predictions": predictions,
            "fraction": accuracy / n_total,
        },
        "permutation": {
            "n_permutations": n_perm,
            "null_accuracies": null_accuracies,
            "pvalue": pvalue,
        },
        "ari": ari,
        "confusion_matrix": cm.tolist(),
        "unique_patients": unique_patients,
    }

    out_path = out_dir / "patient_id_cross_cohort.json"
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f"  Saved: {out_path}")

    return result


# ============================================================
# Analysis 2: Distributional Reproducibility
# ============================================================

def overlap_coefficient(x, y, n_points=1000):
    """Compute overlap coefficient between two distributions via KDE."""
    combined = np.concatenate([x, y])
    lo, hi = combined.min(), combined.max()
    margin = (hi - lo) * 0.05
    grid = np.linspace(lo - margin, hi + margin, n_points)
    kde_x = gaussian_kde(x)(grid)
    kde_y = gaussian_kde(y)(grid)
    return float(np.trapezoid(np.minimum(kde_x, kde_y), grid))


def rank_biserial(u_stat, n1, n2):
    """Compute rank-biserial correlation from Mann-Whitney U."""
    return 1 - (2 * u_stat) / (n1 * n2)


def compute_distributional_reproducibility(config, logger, out_dir):
    """Analysis 2: Distributional comparison of D_cond across sections."""
    base = Path(config["output_dir"])

    logger.info("\n=== Distributional Reproducibility ===")

    # Build sample → patient + cohort mappings
    sample_to_patient = {}
    sample_to_cohort = {}
    all_cohort_samples = {}
    for cohort_key in ["discovery", "validation"]:
        cohort_config = config["cohorts"][cohort_key]
        cohort_name = cohort_config["name"]
        mapping = cohort_config.get("patient_mapping", {})
        samples = cohort_config["samples"]
        all_cohort_samples[cohort_key] = (samples, cohort_name)
        for pid, sids in mapping.items():
            for sid in sids:
                sample_to_patient[sid] = pid
                sample_to_cohort[sid] = cohort_key

    # Load D_cond vectors for all samples
    dcond_vectors = {}
    for cohort_key, (samples, cohort_name) in all_cohort_samples.items():
        for sid in samples:
            pq_path = base / "phase2" / "scores" / cohort_name / f"{sid}_discordance.parquet"
            if not pq_path.exists():
                logger.warning(f"  Missing: {pq_path}")
                continue
            df = pd.read_parquet(pq_path)
            # Average D_cond across 3 ridge encoders
            ridge_cols = [c for c in df.columns if c.startswith('D_cond_') and c.endswith('_ridge')]
            dcond = df[ridge_cols].mean(axis=1).values
            dcond_vectors[sid] = dcond
            logger.info(f"  Loaded {sid}: {len(dcond)} spots")

    # Compute pairwise metrics within each cohort
    records = []
    for cohort_key, (samples, cohort_name) in all_cohort_samples.items():
        available = [s for s in samples if s in dcond_vectors]
        for s_a, s_b in combinations(available, 2):
            pat_a = sample_to_patient.get(s_a, "unknown")
            pat_b = sample_to_patient.get(s_b, "unknown")
            is_within = pat_a == pat_b

            ks_stat, ks_pval = stats.ks_2samp(dcond_vectors[s_a], dcond_vectors[s_b])
            ovl = overlap_coefficient(dcond_vectors[s_a], dcond_vectors[s_b])

            records.append({
                "section_a": s_a,
                "section_b": s_b,
                "cohort": cohort_key,
                "patient_a": pat_a,
                "patient_b": pat_b,
                "is_within_patient": is_within,
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_pval,
                "overlap_coefficient": ovl,
            })

    dist_df = pd.DataFrame(records)
    dist_df.to_csv(out_dir / "distributional_reproducibility.csv", index=False)
    logger.info(f"  Pairwise comparisons: {len(dist_df)} ({dist_df['is_within_patient'].sum()} within-patient)")

    # Within vs between comparison
    within = dist_df[dist_df["is_within_patient"]]
    between = dist_df[~dist_df["is_within_patient"]]

    summary = {}
    for metric, expected_dir in [("ks_statistic", "within_smaller"), ("overlap_coefficient", "within_larger")]:
        w_vals = within[metric].values
        b_vals = between[metric].values

        if len(w_vals) > 0 and len(b_vals) > 0:
            u_stat, u_pval = mannwhitneyu(w_vals, b_vals, alternative='two-sided')
            rb = rank_biserial(u_stat, len(w_vals), len(b_vals))
        else:
            u_stat, u_pval, rb = np.nan, np.nan, np.nan

        summary[metric] = {
            "within_patient": {
                "mean": float(np.mean(w_vals)) if len(w_vals) > 0 else None,
                "sd": float(np.std(w_vals)) if len(w_vals) > 0 else None,
                "n": int(len(w_vals)),
            },
            "between_patient": {
                "mean": float(np.mean(b_vals)) if len(b_vals) > 0 else None,
                "sd": float(np.std(b_vals)) if len(b_vals) > 0 else None,
                "n": int(len(b_vals)),
            },
            "mannwhitney_u": float(u_stat) if not np.isnan(u_stat) else None,
            "mannwhitney_p": float(u_pval) if not np.isnan(u_pval) else None,
            "rank_biserial": float(rb) if not np.isnan(rb) else None,
        }
        logger.info(f"  {metric}: within={summary[metric]['within_patient']['mean']:.3f}±{summary[metric]['within_patient']['sd']:.3f}, "
                     f"between={summary[metric]['between_patient']['mean']:.3f}±{summary[metric]['between_patient']['sd']:.3f}, "
                     f"MW p={u_pval:.4f}")

    with open(out_dir / "distributional_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  Saved distributional results")


# ============================================================
# Analysis 3: Pathway-Level Reproducibility
# ============================================================

def compute_pathway_reproducibility(config, logger, out_dir):
    """Analysis 3: Continuous pathway enrichment profile comparison.

    Uses existing per-sample pathway_de.csv (cohens_d of pathway-level
    signed residuals: discordant vs concordant spots). Compares the
    31-dimensional enrichment profiles between sections.
    """
    base = Path(config["output_dir"])

    logger.info("\n=== Pathway-Level Reproducibility ===")

    # Build sample mappings
    sample_to_patient = {}
    sample_to_cohort = {}
    for cohort_key in ["discovery", "validation"]:
        mapping = config["cohorts"][cohort_key].get("patient_mapping", {})
        for pid, sids in mapping.items():
            for sid in sids:
                sample_to_patient[sid] = pid
                sample_to_cohort[sid] = cohort_key

    # Load pathway_de.csv for each cohort (per-sample pathway enrichment)
    profile_records = []
    pairwise_records = []

    for cohort_key in ["discovery", "validation"]:
        cohort_config = config["cohorts"][cohort_key]
        cohort_name = cohort_config["name"]
        samples = cohort_config["samples"]

        pw_de_path = base / "phase3" / "pathways" / cohort_name / "pathway_de.csv"
        if not pw_de_path.exists():
            logger.warning(f"  Missing: {pw_de_path}")
            continue

        pw_de = pd.read_csv(pw_de_path)
        logger.info(f"\n  Cohort {cohort_key}: {len(pw_de)} pathway-sample records")

        # Pivot: rows=samples, cols=pathways, values=cohens_d
        pivot = pw_de.pivot(index="sample_id", columns="pathway", values="cohens_d")
        available = [s for s in samples if s in pivot.index]
        pivot = pivot.loc[available]
        pathways = list(pivot.columns)
        n_pathways = len(pathways)
        logger.info(f"  {len(available)} samples × {n_pathways} pathways")

        # Save profile records
        for sid in available:
            for pw in pathways:
                profile_records.append({
                    "section_id": sid,
                    "cohort": cohort_key,
                    "patient_id": sample_to_patient.get(sid, "unknown"),
                    "pathway": pw,
                    "cohens_d": float(pivot.loc[sid, pw]),
                })

        # Pairwise Pearson correlation of pathway profiles
        for s_a, s_b in combinations(available, 2):
            pat_a = sample_to_patient.get(s_a, "unknown")
            pat_b = sample_to_patient.get(s_b, "unknown")
            is_within = pat_a == pat_b

            r, p = stats.pearsonr(pivot.loc[s_a].values, pivot.loc[s_b].values)

            pairwise_records.append({
                "section_a": s_a,
                "section_b": s_b,
                "cohort": cohort_key,
                "patient_a": pat_a,
                "patient_b": pat_b,
                "is_within_patient": is_within,
                "pearson_r": float(r),
                "pearson_p": float(p),
            })

    # Save profile data (for heatmap visualization)
    profile_df = pd.DataFrame(profile_records)
    profile_df.to_csv(out_dir / "pathway_repro_profiles.csv", index=False)

    # Save pairwise comparisons
    pairwise_df = pd.DataFrame(pairwise_records)
    pairwise_df.to_csv(out_dir / "pathway_repro_pairwise.csv", index=False)

    # Within vs between comparison
    within_pw = pairwise_df[pairwise_df["is_within_patient"]]
    between_pw = pairwise_df[~pairwise_df["is_within_patient"]]

    w_vals = within_pw["pearson_r"].values
    b_vals = between_pw["pearson_r"].values

    if len(w_vals) > 0 and len(b_vals) > 0:
        u_stat, u_pval = mannwhitneyu(w_vals, b_vals, alternative='greater')
        rb = rank_biserial(u_stat, len(w_vals), len(b_vals))
    else:
        u_stat, u_pval, rb = np.nan, np.nan, np.nan

    summary = {
        "n_pathways_discovery": len(profile_df[profile_df["cohort"] == "discovery"]["pathway"].unique()),
        "n_pathways_validation": len(profile_df[profile_df["cohort"] == "validation"]["pathway"].unique()),
        "within_patient": {
            "mean_r": float(np.mean(w_vals)) if len(w_vals) > 0 else None,
            "sd_r": float(np.std(w_vals)) if len(w_vals) > 0 else None,
            "n": int(len(w_vals)),
        },
        "between_patient": {
            "mean_r": float(np.mean(b_vals)) if len(b_vals) > 0 else None,
            "sd_r": float(np.std(b_vals)) if len(b_vals) > 0 else None,
            "n": int(len(b_vals)),
        },
        "mannwhitney_u": float(u_stat) if not np.isnan(u_stat) else None,
        "mannwhitney_p": float(u_pval) if not np.isnan(u_pval) else None,
        "rank_biserial": float(rb) if not np.isnan(rb) else None,
    }

    with open(out_dir / "pathway_repro_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"  Pathway profile r: within={summary['within_patient']['mean_r']:.3f}±{summary['within_patient']['sd_r']:.3f}, "
                f"between={summary['between_patient']['mean_r']:.3f}±{summary['between_patient']['sd_r']:.3f}, "
                f"MW p={u_pval:.4f}")
    logger.info(f"  Saved pathway reproducibility results")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Compute within-patient reproducibility analyses")
    parser.add_argument("--analysis", nargs="*",
                        choices=["patient_id", "distributional", "pathway_repro"],
                        help="Which analyses to run (default: all)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_v3_config(args.config)
    logger = setup_logging("INFO")

    out_dir = Path(config["output_dir"]) / "figure_data" / "reproducibility"
    out_dir.mkdir(parents=True, exist_ok=True)

    analyses = args.analysis or ["patient_id", "distributional", "pathway_repro"]
    t0 = time.time()

    if "patient_id" in analyses:
        compute_patient_identification(config, logger, "discovery", out_dir)
        compute_patient_identification(config, logger, "validation", out_dir)
        compute_patient_identification_cross_cohort(config, logger, out_dir)

    if "distributional" in analyses:
        compute_distributional_reproducibility(config, logger, out_dir)

    if "pathway_repro" in analyses:
        compute_pathway_reproducibility(config, logger, out_dir)

    logger.info(f"\nTotal time: {format_time(time.time() - t0)}")
    logger.info("Done.")


if __name__ == "__main__":
    main()

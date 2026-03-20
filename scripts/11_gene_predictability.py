#!/usr/bin/env python3
"""
Script 11: Gene Predictability Regression (Phase 3D)

OLS regression with N_genes observations per cohort (280).
Response variable: mean Pearson correlation per gene (averaged across folds and encoders, ridge).
Predictors:
  - mean_expression: mean log1p expression across all spots
  - cv_expression: coefficient of variation of expression
  - spatial_autocorrelation: Moran's I of raw gene expression
  - subcellular_localization: one-hot from UniProt (Secreted, Membrane, etc.)
  - functional_category: one-hot from GO/keyword classification
  - pathway_membership_count: number of Hallmark pathways containing the gene

VIF check for multicollinearity.

Output:
    outputs/phase3/gene_predictability/{cohort}/
        gene_features.csv           (full feature table)
        ols_results.json            (coefficients, R², p-values)
        vif.csv                     (variance inflation factors)
        gene_predictability_summary.json

Usage:
    python scripts/11_gene_predictability.py
    python scripts/11_gene_predictability.py --cohort discovery
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
from src.spatial import build_spatial_weights, morans_i
from src.pathways import load_gene_sets
from src.gene_annotations import fetch_uniprot_localization, fetch_go_slim


def load_v3_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_gene_pearson_per_gene(pred_dir, cohort_name, n_folds, gene_names):
    """Compute mean Pearson per gene, averaged across encoders (ridge only)."""
    encoder_names = ["uni", "virchow2", "hoptimus0"]
    gene_pearsons = {g: [] for g in gene_names}

    for enc in encoder_names:
        # Aggregate across folds
        all_targets = []
        all_preds = []

        for fold_idx in range(n_folds):
            fold_dir = pred_dir / cohort_name / enc / "ridge" / f"fold{fold_idx}"
            targets = np.load(fold_dir / "test_targets.npy")
            preds = np.load(fold_dir / "test_predictions.npy")
            all_targets.append(targets)
            all_preds.append(preds)

        targets = np.vstack(all_targets)
        preds = np.vstack(all_preds)

        for g_idx, gene in enumerate(gene_names):
            yt = targets[:, g_idx]
            yp = preds[:, g_idx]
            if np.std(yt) > 1e-10 and np.std(yp) > 1e-10:
                r, _ = stats.pearsonr(yt, yp)
                if np.isnan(r):
                    r = 0.0
            else:
                r = 0.0
            gene_pearsons[gene].append(r)

    # Average across encoders
    mean_pearsons = {g: np.mean(rs) for g, rs in gene_pearsons.items()}
    return mean_pearsons


def compute_gene_expression_stats(sample_ids, hest_dir, gene_list):
    """Compute mean expression and CV per gene across all samples in cohort."""
    import scanpy as sc

    all_means = []
    all_vars = []
    total_spots = 0

    for sid in sample_ids:
        adata = sc.read_h5ad(Path(hest_dir) / "st" / f"{sid}.h5ad")
        available = [g for g in gene_list if g in adata.var_names]
        adata = adata[:, available]
        sc.pp.log1p(adata)
        X = adata.X
        if hasattr(X, 'toarray'):
            X = X.toarray()
        all_means.append(X.mean(axis=0) * X.shape[0])
        all_vars.append(np.var(X, axis=0) * X.shape[0])
        total_spots += X.shape[0]

    mean_expr = np.sum(all_means, axis=0) / total_spots
    # Approximate global variance (ignoring between-sample component for simplicity)
    var_expr = np.sum(all_vars, axis=0) / total_spots
    cv = np.sqrt(var_expr) / (mean_expr + 1e-6)

    return dict(zip(available, mean_expr)), dict(zip(available, cv))


def compute_spatial_autocorrelation_per_gene(sample_ids, hest_dir, gene_list, n_neighbors=6):
    """Compute average Moran's I per gene across samples."""
    import scanpy as sc

    gene_morans = {g: [] for g in gene_list}

    for sid in sample_ids:
        adata = sc.read_h5ad(Path(hest_dir) / "st" / f"{sid}.h5ad")
        available = [g for g in gene_list if g in adata.var_names]
        adata = adata[:, available]
        sc.pp.log1p(adata)
        X = adata.X
        if hasattr(X, 'toarray'):
            X = X.toarray()
        coords = adata.obsm["spatial"]

        W = build_spatial_weights(coords, n_neighbors=n_neighbors)

        for g_idx, gene in enumerate(available):
            I = morans_i(X[:, g_idx], W)
            gene_morans[gene].append(I)

    # Average across samples
    mean_morans = {}
    for gene in gene_list:
        vals = gene_morans.get(gene, [])
        mean_morans[gene] = np.mean(vals) if vals else 0.0

    return mean_morans


def compute_pathway_membership(gene_list, gene_sets):
    """Count how many pathways each gene belongs to."""
    counts = {g: 0 for g in gene_list}
    for pname, pgenes in gene_sets.items():
        for g in pgenes:
            if g in counts:
                counts[g] += 1
    return counts


def compute_vif(X_df):
    """Compute Variance Inflation Factor for each predictor."""
    from numpy.linalg import lstsq

    vif_records = []
    X = X_df.values
    n_cols = X.shape[1]

    for i in range(n_cols):
        y = X[:, i]
        X_other = np.delete(X, i, axis=1)
        X_other = np.column_stack([np.ones(X_other.shape[0]), X_other])

        beta, residuals, _, _ = lstsq(X_other, y, rcond=None)
        y_pred = X_other @ beta
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        vif = 1 / (1 - r_sq) if r_sq < 1 else float("inf")

        vif_records.append({
            "feature": X_df.columns[i],
            "r_squared": float(r_sq),
            "vif": float(vif),
        })

    return pd.DataFrame(vif_records)


def main():
    parser = argparse.ArgumentParser(description="Phase 3D: Gene predictability regression")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--cohort", type=str, default=None)
    parser.add_argument("--skip-spatial", action="store_true",
                        help="Skip Moran's I computation (slow)")
    args = parser.parse_args()

    config = load_v3_config(args.config)

    log_dir = Path(config["output_dir"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(
        log_level=config.get("log_level", "INFO"),
        log_file=str(log_dir / "11_gene_predictability.log"),
    )

    logger.info("=" * 60)
    logger.info("Phase 3D: Gene Predictability Regression")
    logger.info("=" * 60)

    cohort_keys = [args.cohort] if args.cohort else ["discovery", "validation"]
    pred_dir = Path(config["output_dir"]) / "predictions"
    cache_dir = Path(config["output_dir"]) / "phase3" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    overall_start = time.time()

    # Load gene sets
    gmt_path = config.get("phase3", {}).get("hallmark_gmt", "data/gene_sets/h.all.v2024.1.Hs.symbols.gmt")
    gene_sets = load_gene_sets(gmt_path)
    logger.info(f"Loaded {len(gene_sets)} Hallmark gene sets")

    for cohort_key in cohort_keys:
        cohort_config = config["cohorts"][cohort_key]
        cohort_name = cohort_config["name"]
        n_folds = cohort_config["n_lopo_folds"]
        samples = cohort_config["samples"]
        out_dir = Path(config["output_dir"]) / "phase3" / "gene_predictability" / cohort_name
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\nCohort: {cohort_name} ({cohort_key})")

        # Load gene list
        gene_list_path = Path(config["v3_data_dir"]) / f"gene_list_{cohort_name}.json"
        with open(gene_list_path) as f:
            gene_list = json.load(f)
        n_genes = len(gene_list)
        logger.info(f"  {n_genes} genes")

        # 1. Response: mean Pearson per gene
        logger.info("  Computing mean Pearson per gene...")
        mean_pearsons = compute_gene_pearson_per_gene(pred_dir, cohort_name, n_folds, gene_list)

        # 2. Expression stats
        logger.info("  Computing expression statistics...")
        mean_expr, cv_expr = compute_gene_expression_stats(
            samples, config["hest_dir"], gene_list
        )

        # 3. Spatial autocorrelation (subsample for speed)
        if not args.skip_spatial:
            logger.info("  Computing spatial autocorrelation (Moran's I per gene)...")
            # Use up to 3 representative samples to speed up
            sample_subset = samples[:min(3, len(samples))]
            spatial_auto = compute_spatial_autocorrelation_per_gene(
                sample_subset, config["hest_dir"], gene_list
            )
        else:
            logger.info("  Skipping spatial autocorrelation")
            spatial_auto = {g: np.nan for g in gene_list}

        # 4. Pathway membership
        logger.info("  Computing pathway membership counts...")
        pathway_counts = compute_pathway_membership(gene_list, gene_sets)

        # 5. Subcellular localization (with caching)
        logger.info("  Fetching subcellular localization...")
        loc_cache = str(cache_dir / f"uniprot_localization_{cohort_name}.csv")
        loc_df = fetch_uniprot_localization(gene_list, cache_path=loc_cache)

        # 6. Functional category (with caching)
        logger.info("  Fetching functional categories...")
        go_cache = str(cache_dir / f"go_slim_{cohort_name}.csv")
        go_df = fetch_go_slim(gene_list, cache_path=go_cache)

        # Build feature table
        logger.info("  Building feature table...")
        features = pd.DataFrame({"gene": gene_list})
        features["mean_pearson"] = features["gene"].map(mean_pearsons)
        features["mean_expression"] = features["gene"].map(mean_expr)
        features["cv_expression"] = features["gene"].map(cv_expr)
        features["spatial_autocorrelation"] = features["gene"].map(spatial_auto)
        features["pathway_count"] = features["gene"].map(pathway_counts)

        # Merge annotations
        features = features.merge(loc_df[["gene", "primary_localization"]], on="gene", how="left")
        features = features.merge(go_df[["gene", "primary_function"]], on="gene", how="left")

        features["primary_localization"] = features["primary_localization"].fillna("Unknown")
        features["primary_function"] = features["primary_function"].fillna("Other")

        features.to_csv(out_dir / "gene_features.csv", index=False)
        logger.info(f"  Feature table: {features.shape}")

        # OLS regression
        logger.info("  Running OLS regression...")

        # Prepare X matrix with one-hot encoding
        y = features["mean_pearson"].values

        # Continuous features (z-scored)
        continuous_cols = ["mean_expression", "cv_expression", "pathway_count"]
        if not args.skip_spatial:
            continuous_cols.append("spatial_autocorrelation")

        X_continuous = features[continuous_cols].copy()
        # Z-score continuous features
        for col in continuous_cols:
            vals = X_continuous[col].values
            valid = ~np.isnan(vals)
            if valid.sum() > 0 and np.std(vals[valid]) > 0:
                X_continuous.loc[:, col] = (vals - np.nanmean(vals)) / np.nanstd(vals)
            X_continuous[col] = X_continuous[col].fillna(0)

        # One-hot for localization (drop first to avoid collinearity)
        loc_dummies = pd.get_dummies(features["primary_localization"], prefix="loc", drop_first=True)
        func_dummies = pd.get_dummies(features["primary_function"], prefix="func", drop_first=True)

        X_df = pd.concat([X_continuous, loc_dummies, func_dummies], axis=1)
        # Ensure all columns are numeric (one-hot dummies may be bool/object)
        X_df = X_df.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float64)
        X = X_df.values

        # Add intercept
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        feature_names = ["intercept"] + list(X_df.columns)

        # Drop rows with NaN in y
        valid = ~np.isnan(y)
        if valid.sum() < X.shape[0]:
            logger.info(f"  Dropping {(~valid).sum()} genes with NaN Pearson")
            X_with_intercept = X_with_intercept[valid]
            y = y[valid]

        # OLS via normal equations
        from numpy.linalg import lstsq

        beta, residuals_ols, _, _ = lstsq(X_with_intercept, y, rcond=None)
        y_pred = X_with_intercept @ beta

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        n = len(y)
        p = X_with_intercept.shape[1]
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p) if n > p else r_squared  # p includes intercept

        # Standard errors
        mse = ss_res / (n - p) if n > p else ss_res
        try:
            cov_beta = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
            se = np.sqrt(np.diag(cov_beta))
            t_stats = beta / se
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - p))
        except np.linalg.LinAlgError:
            se = np.full_like(beta, np.nan)
            t_stats = np.full_like(beta, np.nan)
            p_values = np.full_like(beta, np.nan)

        logger.info(f"  OLS R² = {r_squared:.4f}, Adj R² = {adj_r_squared:.4f}")
        logger.info(f"  N = {n}, p = {p}")

        # Save OLS results
        coef_records = []
        for i, name in enumerate(feature_names):
            coef_records.append({
                "feature": name,
                "coefficient": float(beta[i]),
                "std_error": float(se[i]) if not np.isnan(se[i]) else None,
                "t_statistic": float(t_stats[i]) if not np.isnan(t_stats[i]) else None,
                "p_value": float(p_values[i]) if not np.isnan(p_values[i]) else None,
            })

        ols_results = {
            "cohort": cohort_name,
            "n_genes": int(n),
            "n_features": int(p),
            "r_squared": float(r_squared),
            "adj_r_squared": float(adj_r_squared),
            "coefficients": coef_records,
        }

        # Log significant predictors
        for rec in coef_records:
            if rec["p_value"] is not None and rec["p_value"] < 0.05:
                logger.info(
                    f"    {rec['feature']}: β={rec['coefficient']:.4f}, "
                    f"t={rec['t_statistic']:.2f}, p={rec['p_value']:.4f}"
                )

        with open(out_dir / "ols_results.json", "w") as f:
            json.dump(ols_results, f, indent=2)

        # VIF check
        vif_df = compute_vif(X_df)
        vif_df.to_csv(out_dir / "vif.csv", index=False)
        high_vif = vif_df[vif_df["vif"] > 5]
        if not high_vif.empty:
            logger.info(f"  WARNING: High VIF features:")
            for _, row in high_vif.iterrows():
                logger.info(f"    {row['feature']}: VIF={row['vif']:.1f}")

        # Summary
        summary = {
            "cohort": cohort_name,
            "n_genes": int(n),
            "r_squared": float(r_squared),
            "adj_r_squared": float(adj_r_squared),
            "significant_predictors": [
                r for r in coef_records
                if r["p_value"] is not None and r["p_value"] < 0.05
            ],
            "n_high_vif": len(high_vif),
        }
        with open(out_dir / "gene_predictability_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    total_time = time.time() - overall_start
    logger.info(f"\nTotal time: {format_time(total_time)}")
    logger.info("Done.")


if __name__ == "__main__":
    main()

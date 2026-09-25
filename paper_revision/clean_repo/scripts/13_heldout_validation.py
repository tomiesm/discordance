#!/usr/bin/env python3
"""
Script 13: Held-Out Gene Validation (Phase 4A)

5-fold gene-level cross-validation: train on 224 genes, test on 56.
Question: Can gene-level features (expression level, CV, spatial autocorrelation,
localization, function) predict which held-out genes will be well/poorly predicted?

This tests whether gene predictability is a *gene-intrinsic* property that can
be anticipated from gene characteristics, without seeing the morphology predictions.

Method:
  1. Load gene features from Phase 3D (gene_features.csv)
  2. 5-fold CV on gene index: train OLS on 224 genes, predict Pearson for 56
  3. Report: overall correlation between predicted and actual Pearson, MAE

Output:
    outputs/phase4/heldout_validation/{cohort}/
        cv_predictions.csv          (gene, actual_pearson, predicted_pearson, fold)
        heldout_summary.json

Usage:
    python scripts/13_heldout_validation.py
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
from sklearn.model_selection import KFold
from src.utils import setup_logging, format_time


def load_v3_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Phase 4A: Held-out gene validation")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--cohort", type=str, default=None)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    config = load_v3_config(args.config)

    log_dir = Path(config["output_dir"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(
        log_level=config.get("log_level", "INFO"),
        log_file=str(log_dir / "13_heldout_validation.log"),
    )

    logger.info("=" * 60)
    logger.info("Phase 4A: Held-Out Gene Validation")
    logger.info("=" * 60)

    cohort_keys = [args.cohort] if args.cohort else ["discovery", "validation"]
    overall_start = time.time()

    for cohort_key in cohort_keys:
        cohort_config = config["cohorts"][cohort_key]
        cohort_name = cohort_config["name"]
        features_dir = Path(config["output_dir"]) / "phase3" / "gene_predictability" / cohort_name
        out_dir = Path(config["output_dir"]) / "phase4" / "heldout_validation" / cohort_name
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\nCohort: {cohort_name} ({cohort_key})")

        # Load gene features from Phase 3D
        features_path = features_dir / "gene_features.csv"
        if not features_path.exists():
            logger.warning(f"  Gene features not found at {features_path}, skipping")
            continue

        features = pd.read_csv(features_path)
        logger.info(f"  Loaded {len(features)} genes with {features.shape[1]} features")

        # Prepare X and y
        y = features["mean_pearson"].values

        # Continuous features
        continuous_cols = ["mean_expression", "cv_expression", "pathway_count"]
        if "spatial_autocorrelation" in features.columns:
            continuous_cols.append("spatial_autocorrelation")

        X_continuous = features[continuous_cols].copy()
        for col in continuous_cols:
            vals = X_continuous[col].values.astype(float)
            valid = ~np.isnan(vals)
            if valid.sum() > 0 and np.std(vals[valid]) > 0:
                X_continuous.loc[:, col] = (vals - np.nanmean(vals)) / np.nanstd(vals)
            X_continuous[col] = X_continuous[col].fillna(0)

        # One-hot
        loc_dummies = pd.get_dummies(
            features["primary_localization"], prefix="loc", drop_first=True
        )
        func_dummies = pd.get_dummies(
            features["primary_function"], prefix="func", drop_first=True
        )

        X_df = pd.concat([X_continuous, loc_dummies, func_dummies], axis=1)
        X_df = X_df.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float64)
        X = X_df.values

        # Drop NaN y
        valid = ~np.isnan(y)
        X = X[valid]
        y = y[valid]
        gene_names = features["gene"].values[valid]

        logger.info(f"  Using {len(y)} genes, {X.shape[1]} features")

        # 5-fold gene-level CV
        kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
        all_predictions = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Add intercept
            X_train_i = np.column_stack([np.ones(len(X_train)), X_train])
            X_test_i = np.column_stack([np.ones(len(X_test)), X_test])

            # OLS
            from numpy.linalg import lstsq
            beta, _, _, _ = lstsq(X_train_i, y_train, rcond=None)
            y_pred = X_test_i @ beta

            for i, idx in enumerate(test_idx):
                all_predictions.append({
                    "gene": gene_names[idx],
                    "actual_pearson": float(y_test[i]),
                    "predicted_pearson": float(y_pred[i]),
                    "fold": fold_idx,
                })

            r, _ = stats.pearsonr(y_test, y_pred)
            mae = np.mean(np.abs(y_test - y_pred))
            logger.info(f"  Fold {fold_idx}: r={r:.3f}, MAE={mae:.4f} ({len(test_idx)} genes)")

        # Overall metrics
        pred_df = pd.DataFrame(all_predictions)
        overall_r, overall_p = stats.pearsonr(
            pred_df["actual_pearson"], pred_df["predicted_pearson"]
        )
        overall_mae = np.mean(np.abs(
            pred_df["actual_pearson"] - pred_df["predicted_pearson"]
        ))
        overall_spearman, _ = stats.spearmanr(
            pred_df["actual_pearson"], pred_df["predicted_pearson"]
        )

        logger.info(f"\n  Overall: Pearson r={overall_r:.3f} (p={overall_p:.2e}), "
                     f"Spearman={overall_spearman:.3f}, MAE={overall_mae:.4f}")

        # Save
        pred_df.to_csv(out_dir / "cv_predictions.csv", index=False)

        # Find best/worst predicted genes
        pred_df["error"] = pred_df["actual_pearson"] - pred_df["predicted_pearson"]
        best_predicted = pred_df.nsmallest(10, "error", keep="first")[
            ["gene", "actual_pearson", "predicted_pearson", "error"]
        ].to_dict(orient="records")
        worst_predicted = pred_df.nlargest(10, "error", keep="first")[
            ["gene", "actual_pearson", "predicted_pearson", "error"]
        ].to_dict(orient="records")

        summary = {
            "cohort": cohort_name,
            "n_genes": len(pred_df),
            "n_features": X.shape[1],
            "n_folds": args.n_folds,
            "overall_pearson_r": float(overall_r),
            "overall_pearson_p": float(overall_p),
            "overall_spearman": float(overall_spearman),
            "overall_mae": float(overall_mae),
            "overpredicted_genes": worst_predicted,
            "underpredicted_genes": best_predicted,
        }

        with open(out_dir / "heldout_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    total_time = time.time() - overall_start
    logger.info(f"\nTotal time: {format_time(total_time)}")
    logger.info("Done.")


if __name__ == "__main__":
    main()

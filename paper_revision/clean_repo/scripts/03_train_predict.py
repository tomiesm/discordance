#!/usr/bin/env python3
"""
Script 03: Train & Predict for v3 (Phase 1)

Per-cohort LOPO-CV: 2 cohorts × 4 folds × 3 encoders × 3 regressors = 72 runs.
Each run: StandardScaler → Regressor(PCA + model) → per-gene Pearson.

Output:
    outputs/predictions/{cohort}/{encoder}/{regressor}/fold{i}/
        test_predictions.npy, test_targets.npy, test_residuals.npy,
        test_spot_ids.json, metrics.json

Usage:
    python scripts/03_train_predict.py
    python scripts/03_train_predict.py --cohort discovery --encoder uni --regressor ridge
    python scripts/03_train_predict.py --cohort validation --fold 0
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import h5py
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.utils import setup_logging, seed_everything, format_time
from src.regressors import get_regressor


def load_v3_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_expression_for_cohort(cohort_config, config, logger):
    """Load expression data for all samples in a cohort."""
    from src.data import load_v3_task

    cohort_name = cohort_config["name"]
    gene_list_path = Path(config["v3_data_dir"]) / f"gene_list_{cohort_name}.json"

    expr_df, gene_names, spot_to_sample = load_v3_task(
        sample_ids=cohort_config["samples"],
        hest_dir=config["hest_dir"],
        gene_list_path=str(gene_list_path),
        normalize=True,
    )
    logger.info(f"  Expression: {expr_df.shape[0]} spots × {expr_df.shape[1]} genes")

    return expr_df, gene_names, spot_to_sample


def load_embeddings_for_samples(sample_ids, encoder_name, config):
    """Load embeddings for multiple samples, return {sample_id: (embeddings, spot_ids)}."""
    embed_dir = Path(config["output_dir"]) / "embeddings"
    result = {}

    for sid in sample_ids:
        path = embed_dir / sid / f"{encoder_name}_embeddings.h5"
        with h5py.File(path, "r") as f:
            embeddings = f["embeddings"][:]
            spot_ids = [
                s.decode() if isinstance(s, bytes) else s for s in f["spot_ids"][:]
            ]
        result[sid] = (embeddings, spot_ids)

    return result


def align_expression_embeddings(expr_df, spot_to_sample, embed_data, sample_ids):
    """Align expression and embedding data by matching spot IDs.

    expr_df has spot IDs like "{sample_id}_{barcode}".
    embed_data has spot IDs as raw barcodes per sample.

    Returns aligned (X_embed, Y_expr, aligned_spot_ids).
    """
    aligned_embeds = []
    aligned_exprs = []
    aligned_spots = []

    for sid in sample_ids:
        if sid not in embed_data:
            continue

        embeddings, embed_spot_ids = embed_data[sid]
        prefix = f"{sid}_"

        # Get expression spots for this sample
        sample_expr_spots = {
            s: s[len(prefix) :] for s in expr_df.index if s.startswith(prefix)
        }
        # Invert: barcode -> prefixed spot ID
        barcode_to_prefixed = {v: k for k, v in sample_expr_spots.items()}

        # Build index for embeddings
        embed_idx = {b: i for i, b in enumerate(embed_spot_ids)}

        # Find common spots (preserving embedding order), skip NaN embeddings
        for barcode in embed_spot_ids:
            if barcode in barcode_to_prefixed:
                emb = embeddings[embed_idx[barcode]]
                if np.any(np.isnan(emb)):
                    continue
                prefixed = barcode_to_prefixed[barcode]
                aligned_embeds.append(emb)
                aligned_exprs.append(expr_df.loc[prefixed].values)
                aligned_spots.append(prefixed)

    X = np.array(aligned_embeds, dtype=np.float32)
    Y = np.array(aligned_exprs, dtype=np.float32)

    # Final NaN check on expression data
    nan_mask = np.isnan(Y).any(axis=1) | np.isnan(X).any(axis=1)
    if nan_mask.any():
        n_drop = nan_mask.sum()
        X = X[~nan_mask]
        Y = Y[~nan_mask]
        aligned_spots = [s for s, m in zip(aligned_spots, nan_mask) if not m]

    return X, Y, aligned_spots


def compute_per_gene_pearson(Y_true, Y_pred, gene_names):
    """Compute Pearson correlation per gene."""
    results = {}
    pearsons = []

    for i, gene in enumerate(gene_names):
        y_t = Y_true[:, i]
        y_p = Y_pred[:, i]

        # Skip constant genes
        if np.std(y_t) < 1e-10 or np.std(y_p) < 1e-10:
            r = 0.0
            p = 1.0
        else:
            r, p = pearsonr(y_t, y_p)
            if np.isnan(r):
                r = 0.0

        results[gene] = {"pearson": float(r), "pvalue": float(p)}
        pearsons.append(r)

    pearsons = np.array(pearsons)
    results["__mean_pearson__"] = float(np.mean(pearsons))
    results["__median_pearson__"] = float(np.median(pearsons))
    results["__n_positive__"] = int(np.sum(pearsons > 0))
    results["__n_genes__"] = len(gene_names)
    results["__n_spots__"] = int(Y_true.shape[0])

    return results


def run_one_fold(
    X_train,
    Y_train,
    X_test,
    Y_test,
    test_spots,
    regressor_config,
    gene_names,
    out_dir,
    seed,
    logger,
):
    """Train regressor, predict, save outputs for one fold."""
    reg_name = regressor_config["name"]
    reg_type = regressor_config["type"]

    # StandardScaler (applied before regressor's internal PCA)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Create regressor
    reg = get_regressor(regressor_config)

    # For MLP, create validation split for early stopping
    if reg_type == "mlp":
        X_tr, X_val, Y_tr, Y_val = train_test_split(
            X_train_sc, Y_train, test_size=0.15, random_state=seed
        )
        logger.info(
            f"      MLP train/val split: {X_tr.shape[0]}/{X_val.shape[0]} spots"
        )
        t0 = time.time()
        reg.fit(X_tr, Y_tr, X_val=X_val, Y_val=Y_val)
        train_time = time.time() - t0
    else:
        t0 = time.time()
        reg.fit(X_train_sc, Y_train)
        train_time = time.time() - t0

    # Predict
    t0 = time.time()
    Y_pred = reg.predict(X_test_sc)
    pred_time = time.time() - t0

    logger.info(f"      Train: {format_time(train_time)}, Predict: {format_time(pred_time)}")

    # Metrics
    metrics = compute_per_gene_pearson(Y_test, Y_pred, gene_names)
    logger.info(
        f"      Mean Pearson: {metrics['__mean_pearson__']:.4f} "
        f"(median: {metrics['__median_pearson__']:.4f}, "
        f"{metrics['__n_positive__']}/{metrics['__n_genes__']} positive)"
    )

    # Residuals
    residuals = Y_test - Y_pred

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "test_predictions.npy", Y_pred.astype(np.float32))
    np.save(out_dir / "test_targets.npy", Y_test.astype(np.float32))
    np.save(out_dir / "test_residuals.npy", residuals.astype(np.float32))

    with open(out_dir / "test_spot_ids.json", "w") as f:
        json.dump(test_spots, f)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Clean up GPU memory for MLP
    if reg_type == "mlp":
        del reg
        import torch

        torch.cuda.empty_cache()

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train & predict for v3 (Phase 1)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--cohort",
        type=str,
        default=None,
        help="Process specific cohort only (discovery/validation)",
    )
    parser.add_argument("--encoder", type=str, default=None)
    parser.add_argument("--regressor", type=str, default=None)
    parser.add_argument("--fold", type=int, default=None)
    args = parser.parse_args()

    config = load_v3_config(args.config)

    log_dir = Path(config["output_dir"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(
        log_level=config.get("log_level", "INFO"),
        log_file=str(log_dir / "03_train_predict.log"),
    )

    seed_everything(config["seed"])

    # Determine what to run
    cohort_keys = [args.cohort] if args.cohort else ["discovery", "validation"]
    encoder_configs = (
        [e for e in config["encoders"] if e["name"] == args.encoder]
        if args.encoder
        else config["encoders"]
    )
    regressor_configs = (
        [r for r in config["regressors"] if r["name"] == args.regressor]
        if args.regressor
        else config["regressors"]
    )

    # Count total runs
    total_runs = 0
    for ck in cohort_keys:
        n_folds = config["cohorts"][ck]["n_lopo_folds"]
        folds = [args.fold] if args.fold is not None else list(range(n_folds))
        total_runs += len(folds) * len(encoder_configs) * len(regressor_configs)

    logger.info("=" * 60)
    logger.info("v3 Phase 1: Train & Predict")
    logger.info("=" * 60)
    logger.info(f"Cohorts: {cohort_keys}")
    logger.info(f"Encoders: {[e['name'] for e in encoder_configs]}")
    logger.info(f"Regressors: {[r['name'] for r in regressor_configs]}")
    logger.info(f"Total runs: {total_runs}")

    completed = 0
    failed = 0
    skipped = 0
    overall_start = time.time()

    for cohort_key in cohort_keys:
        cohort_config = config["cohorts"][cohort_key]
        cohort_name = cohort_config["name"]
        n_folds = cohort_config["n_lopo_folds"]
        folds = [args.fold] if args.fold is not None else list(range(n_folds))

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Cohort: {cohort_name} ({cohort_key})")
        logger.info(f"{'=' * 60}")

        # Load expression data for entire cohort (once)
        logger.info("Loading expression data...")
        expr_df, gene_names, spot_to_sample = load_expression_for_cohort(
            cohort_config, config, logger
        )

        for encoder_config in encoder_configs:
            encoder_name = encoder_config["name"]

            logger.info(f"\n  Encoder: {encoder_name}")

            # Load embeddings for all samples in cohort (once per encoder)
            logger.info("  Loading embeddings...")
            embed_data = load_embeddings_for_samples(
                cohort_config["samples"], encoder_name, config
            )
            total_embed_spots = sum(e[0].shape[0] for e in embed_data.values())
            embed_dim = next(iter(embed_data.values()))[0].shape[1]
            logger.info(f"  Embeddings: {total_embed_spots} spots × {embed_dim} dims")

            for fold_idx in folds:
                # Load fold split
                splits_dir = (
                    Path(config["v3_data_dir"]) / f"lopo_splits_{cohort_name}"
                )
                fold_path = splits_dir / f"fold_{fold_idx}.json"
                with open(fold_path) as f:
                    fold = json.load(f)

                train_samples = fold["train_samples"]
                test_samples = fold["test_samples"]
                test_patient = fold["test_patient"]

                # Align expression with embeddings
                X_train, Y_train, train_spots = align_expression_embeddings(
                    expr_df, spot_to_sample, embed_data, train_samples
                )
                X_test, Y_test, test_spots = align_expression_embeddings(
                    expr_df, spot_to_sample, embed_data, test_samples
                )

                logger.info(
                    f"\n  Fold {fold_idx} (test: {test_patient}): "
                    f"train={X_train.shape[0]}, test={X_test.shape[0]}"
                )

                for reg_config in regressor_configs:
                    reg_name = reg_config["name"]

                    out_dir = (
                        Path(config["output_dir"])
                        / "predictions"
                        / cohort_name
                        / encoder_name
                        / reg_name
                        / f"fold{fold_idx}"
                    )

                    # Skip if already done
                    metrics_path = out_dir / "metrics.json"
                    if metrics_path.exists():
                        logger.info(f"    [{reg_name}] Already exists, skipping.")
                        skipped += 1
                        continue

                    logger.info(f"    [{reg_name}]")

                    try:
                        run_one_fold(
                            X_train,
                            Y_train,
                            X_test,
                            Y_test,
                            test_spots,
                            reg_config,
                            gene_names,
                            out_dir,
                            config["seed"],
                            logger,
                        )
                        completed += 1
                    except Exception as e:
                        logger.error(f"      FAILED: {e}", exc_info=True)
                        failed += 1

                    logger.info(
                        f"      Progress: {completed + failed + skipped}/{total_runs}"
                    )

            # Free memory between encoders
            del embed_data
            gc.collect()

        # Free memory between cohorts
        del expr_df, spot_to_sample
        gc.collect()

    total_time = time.time() - overall_start

    logger.info(f"\n{'=' * 60}")
    logger.info("Summary")
    logger.info(f"{'=' * 60}")
    logger.info(f"Completed: {completed}")
    logger.info(f"Skipped: {skipped}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total time: {format_time(total_time)}")

    # Quality gate: aggregate metrics
    logger.info(f"\n{'=' * 60}")
    logger.info("Quality Gate: Mean Pearson across folds")
    logger.info(f"{'=' * 60}")

    pred_dir = Path(config["output_dir"]) / "predictions"
    all_pass = True

    for cohort_key in cohort_keys:
        cohort_name = config["cohorts"][cohort_key]["name"]
        n_folds = config["cohorts"][cohort_key]["n_lopo_folds"]
        logger.info(f"\n  {cohort_name}:")

        for enc in encoder_configs:
            for reg in regressor_configs:
                fold_pearsons = []
                for fi in range(n_folds):
                    mp = (
                        pred_dir
                        / cohort_name
                        / enc["name"]
                        / reg["name"]
                        / f"fold{fi}"
                        / "metrics.json"
                    )
                    if mp.exists():
                        with open(mp) as f:
                            m = json.load(f)
                        fold_pearsons.append(m["__mean_pearson__"])

                if fold_pearsons:
                    mean_p = np.mean(fold_pearsons)
                    status = "PASS" if mean_p > 0 else "FAIL"
                    if mean_p <= 0:
                        all_pass = False
                    logger.info(
                        f"    {enc['name']}/{reg['name']}: "
                        f"mean={mean_p:.4f} (folds: {[f'{p:.4f}' for p in fold_pearsons]}) [{status}]"
                    )

    if not all_pass:
        logger.warning("Some configs have negative mean Pearson. Flagged but not excluded.")

    if failed > 0:
        logger.error(f"{failed} runs failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Held-out prediction diagnostics against training-only constant baselines."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
REPO = ROOT / "paper_revision/clean_repo"
sys.path.insert(0, str(REPO))


def readj(p):
    return json.loads(p.read_text())


def summary(y, pred, mean, median):
    r = y - pred
    raw = np.abs(r).mean(axis=1)
    babs = np.abs(y - median).mean(axis=1)
    mse = np.square(r).mean()
    bmse = np.square(y - mean).mean()
    return {"n_spots": len(y), "mae": float(raw.mean()), "rmse": float(np.sqrt(mse)),
            "mean_signed_error": float(r.mean()), "mean_absolute_gene_bias": float(np.abs(r.mean(axis=0)).mean()),
            "negative_prediction_fraction": float(np.mean(pred < 0)),
            "baseline_median_mae": float(babs.mean()), "baseline_mean_rmse": float(np.sqrt(bmse)),
            "mae_improvement_fraction": float(1 - raw.mean() / babs.mean()),
            "mse_improvement_fraction": float(1 - mse / bmse),
            "spots_better_than_median_baseline_fraction": float(np.mean(raw < babs))}


def gene_summary(y, p, mean, median):
    r = y - p
    yc = y - y.mean(axis=0); pc = p - p.mean(axis=0)
    den = np.sqrt(np.sum(yc ** 2, axis=0) * np.sum(pc ** 2, axis=0))
    corr = np.divide(np.sum(yc * pc, axis=0), den, out=np.full(y.shape[1], np.nan), where=den > 0)
    return pd.DataFrame({"pearson": corr, "mae": np.abs(r).mean(axis=0),
                         "rmse": np.sqrt(np.square(r).mean(axis=0)), "bias": r.mean(axis=0),
                         "baseline_median_mae": np.abs(y - median).mean(axis=0),
                         "baseline_mean_rmse": np.sqrt(np.square(y - mean).mean(axis=0))})


def main():
    cfg = yaml.safe_load((REPO / "config.yaml").read_text())
    fold_results, sections, genes_all, quartiles, diagnostics, pca_rows, checks = [], [], [], [], [], [], []
    for cc in cfg["cohorts"].values():
        cohort = cc["name"]
        base = REPO / "outputs/predictions" / cohort
        genes = readj(REPO / "data/v3" / f"gene_list_{cohort}.json")
        ys, ids_all = [], []
        for fold in range(cc["n_lopo_folds"]):
            fd = base / "uni/ridge" / f"fold{fold}"
            ys.append(np.load(fd / "test_targets.npy"))
            ids_all.append(readj(fd / "test_spot_ids.json"))
        for fold, y32 in enumerate(ys):
            split = readj(REPO / "data/v3" / f"lopo_splits_{cohort}" / f"fold_{fold}.json")
            train = np.concatenate([v for i, v in enumerate(ys) if i != fold], axis=0)
            mean = train.mean(axis=0, dtype=np.float64)
            median = np.concatenate([np.median(train[:, j:j+32].astype(np.float64), axis=0)
                                     for j in range(0, len(genes), 32)])
            del train
            y = y32.astype(np.float64)
            ids = ids_all[fold]
            sample_ids = np.array([next(s for s in split["test_samples"] if v.startswith(s + "_")) for v in ids])
            bmae = np.abs(y - median).mean(axis=1)
            bmse = np.square(y - mean).mean(axis=1)
            encoder_errors, encoder_preds = [], []
            for enc in [e["name"] for e in cfg["encoders"]]:
                for reg in [r["name"] for r in cfg["regressors"]]:
                    fd = base / enc / reg / f"fold{fold}"
                    mids = readj(fd / "test_spot_ids.json")
                    targets = np.load(fd / "test_targets.npy", mmap_mode="r")
                    same = mids == ids and np.array_equal(targets, y32)
                    checks.append({"check": f"{cohort}/{enc}/{reg}/{fold}:target_and_id_identity", "pass": same})
                    assert same
                    del targets
                    pred = np.load(fd / "test_predictions.npy").astype(np.float64)
                    assert np.isfinite(pred).all()
                    identity = {"cohort": cohort, "patient": split["test_patient"], "fold": fold, "encoder": enc, "regressor": reg}
                    metric = summary(y, pred, mean, median)
                    gs = gene_summary(y, pred, mean, median)
                    metric["mean_gene_pearson"] = float(gs.pearson.mean())
                    metric["genes_better_than_median_baseline_fraction"] = float((gs.mae < gs.baseline_median_mae).mean())
                    fold_results.append({**identity, **metric})
                    gs["gene"] = genes
                    for k,v in identity.items(): gs[k] = v
                    genes_all.append(gs)
                    for sid in split["test_samples"]:
                        mask = sample_ids == sid
                        sections.append({**identity, "sample": sid, **summary(y[mask], pred[mask], mean, median)})
                    if reg == "ridge":
                        saved = np.load(fd / "training_expression_mean.npy")
                        error = float(np.max(np.abs(saved - mean)))
                        checks.append({"check": f"{cohort}/{enc}/{fold}:training_mean", "pass": error < 1e-10, "max_abs": error})
                        assert error < 1e-10
                        encoder_errors.append(np.abs(y - pred).mean(axis=1))
                        encoder_preds.append(pred)
                        with warnings.catch_warnings(record=True) as ws:
                            warnings.simplefilter("always")
                            model = joblib.load(fd / "calibrated_model.joblib")
                        pca_rows.append({**identity,
                            "variance_retained": float(model["regressor"].pca.explained_variance_ratio_.sum()),
                            "components": int(model["regressor"].pca.n_components_),
                            "load_warnings": " | ".join(str(w.message) for w in ws)})
                        del model
                    del pred
            avg_error = np.mean(encoder_errors, axis=0)
            mean_prediction = np.mean(encoder_preds, axis=0)
            ensemble_error = np.abs(y - mean_prediction).mean(axis=1)
            del encoder_preds, mean_prediction
            for sid in split["test_samples"]:
                mask = sample_ids == sid
                si = np.flatnonzero(mask)
                ds = pd.read_parquet(REPO / "outputs/phase2/scores" / cohort / f"{sid}_discordance.parquet").set_index("spot_id")
                ds = ds.loc[np.array(ids)[mask]]
                score = ds[[f"D_cond_{e['name']}_ridge" for e in cfg["encoders"]]].mean(axis=1).to_numpy()
                saved_raw = ds[[f"D_raw_{e['name']}_ridge" for e in cfg["encoders"]]].mean(axis=1).to_numpy()
                # Original scoring used float32 residual arithmetic before averaging.
                rawerr = float(np.max(np.abs(saved_raw - avg_error[mask])))
                checks.append({"check": f"{sid}:raw_score_reconstruction", "pass": rawerr < 2e-6, "max_abs": rawerr})
                assert rawerr < 2e-6
                edges = np.quantile(score, [.25, .5, .75])
                assert edges[0] < edges[1] < edges[2]
                q = np.select([score <= edges[0], score <= edges[1], score < edges[2]], [1,2,3], default=4)
                f = pd.DataFrame({"spot_id": np.array(ids)[mask], "sample": sid, "patient": split["test_patient"],
                     "cohort": cohort, "quartile": q, "D_cond": score,
                     "ridge_encoder_mean_mae": avg_error[mask], "ridge_ensemble_mae": ensemble_error[mask],
                     "baseline_median_mae": bmae[mask], "baseline_mean_mse": bmse[mask],
                     "sum_log1p_expression": y[mask].sum(axis=1), "detected_genes": (y[mask] > 0).sum(axis=1)})
                diagnostics.append(f)
                for quartile, group in f.groupby("quartile"):
                    quartiles.append({"sample": sid, "patient": split["test_patient"], "cohort": cohort,
                         "quartile": int(quartile), "n_spots": len(group),
                         "mean_mae": group.ridge_encoder_mean_mae.mean(),
                         "baseline_median_mae": group.baseline_median_mae.mean(),
                         "mae_improvement_fraction": 1-group.ridge_encoder_mean_mae.mean()/group.baseline_median_mae.mean(),
                         "spots_better_than_baseline_fraction": (group.ridge_encoder_mean_mae < group.baseline_median_mae).mean(),
                         "sum_log1p_expression_mean": group.sum_log1p_expression.mean(),
                         "detected_genes_mean": group.detected_genes.mean()})
            print(f"{cohort} {split['test_patient']}: completed 9 model comparisons", flush=True)
    folds = pd.DataFrame(fold_results)
    folds.to_csv(OUT / "patient_model_metrics.csv", index=False)
    pd.DataFrame(sections).to_csv(OUT / "section_model_metrics.csv", index=False)
    gene_table = pd.concat(genes_all, ignore_index=True)
    gene_table.to_csv(OUT / "gene_model_metrics.csv", index=False)
    # Descriptive follow-up prompted by the broad atlas: exact MSE decomposition,
    # not a proposed correction using held-out outcomes.
    bias = gene_table.loc[gene_table.regressor.eq('ridge')].copy()
    bias['bias_squared'] = bias['bias'] ** 2
    bias['mse'] = bias['rmse'] ** 2
    decomposition = bias.groupby(['cohort','patient','encoder'])[['bias_squared','mse']].mean()
    decomposition['MSE_fraction_due_to_gene_mean_error'] = decomposition.bias_squared / decomposition.mse
    decomposition.reset_index().to_csv(OUT / 'patient_bias_mse_decomposition.csv', index=False)
    pd.DataFrame(pca_rows).to_csv(OUT / "ridge_pca_variance.csv", index=False)
    pd.DataFrame(quartiles).to_csv(OUT / "quartile_prediction_quality.csv", index=False)
    pd.concat(diagnostics, ignore_index=True).to_parquet(OUT / "spot_prediction_diagnostics.parquet", index=False)
    cols = [c for c in folds.select_dtypes("number").columns if c not in ["fold", "n_spots"]]
    cohort_summary = folds.groupby(["cohort", "encoder", "regressor"])[cols].mean().reset_index()
    cohort_summary.to_csv(OUT / "cohort_patient_mean_metrics.csv", index=False)
    (OUT / "checks.json").write_text(json.dumps({"completed_utc": datetime.now(timezone.utc).isoformat(),
                            "status": "pass" if all(c["pass"] for c in checks) else "failed",
                            "checks": checks}, indent=2) + "\n")
    print(cohort_summary[["cohort","encoder","regressor","mae","baseline_median_mae","mae_improvement_fraction","mean_gene_pearson"]].to_string(index=False))


if __name__ == "__main__":
    main()

"""Independent audit: raw HEST -> original folds -> a direct ridge solution.

Does not import revision code. Writes only beneath independent_audit/.
The unchanged original regressor is used for two positive-control refits.
"""
import os
for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[variable] = '4'
import json
import argparse
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import yaml
from scipy.linalg import solve
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OLD = ROOT / 'outputs_v3'
NEW = ROOT / 'paper_revision/clean_repo/outputs'
sys.path.insert(0, str(ROOT / 'clean_repo'))
from src.regressors import FixedAlphaRidgeRegressor  # unchanged ORIGINAL only


def differences(a, b):
    delta = np.asarray(a, dtype=np.float64) - b
    return dict(max_abs=float(np.max(np.abs(delta))),
                mean_abs=float(np.mean(np.abs(delta))),
                rmse=float(np.sqrt(np.mean(delta ** 2))))


def work(cohort_key, encoder):
    with threadpool_limits(4):
        return work_limited(cohort_key, encoder)


def work_limited(cohort_key, encoder):
    cfg = yaml.safe_load((ROOT / 'config_v3.yaml').read_text())['cohorts'][cohort_key]
    family = cfg['name']
    genes = json.loads((ROOT / f'data/v3/gene_list_{family}.json').read_text())
    data = {}
    for sid in cfg['samples']:
        a = ad.read_h5ad(ROOT / f'data/hest/st/{sid}.h5ad')
        assert a.var_names.is_unique and a.obs_names.is_unique
        counts = a[:, genes].X
        if hasattr(counts, 'toarray'):
            counts = counts.toarray()
        # Scanpy converts integer input to Python float (float64) before log1p.
        # NumPy's uint16 ufunc loop otherwise selects float32 and rounds differently.
        if not np.issubdtype(counts.dtype, np.floating):
            counts = counts.astype(float)
        y = np.log1p(counts).astype(np.float32)
        index = {str(b): i for i, b in enumerate(a.obs_names)}
        with h5py.File(OLD / f'embeddings/{sid}/{encoder}_embeddings.h5', 'r') as f:
            x = f['embeddings'][:].astype(np.float32)
            barcodes = [b.decode() if isinstance(b, bytes) else str(b) for b in f['spot_ids'][:]]
        assert len(set(barcodes)) == len(barcodes)
        keep = [i for i, b in enumerate(barcodes) if b in index and np.isfinite(x[i]).all()
                and np.isfinite(y[index[b]]).all()]
        data[sid] = (x[keep], y[[index[barcodes[i]] for i in keep]],
                     [f'{sid}_{barcodes[i]}' for i in keep])
        del a, counts, x, y
    records = []
    for fold in range(4):
        start = time.time()
        split = json.loads((ROOT / f'data/v3/lopo_splits_{family}/fold_{fold}.json').read_text())
        assert not set(split['train_samples']) & set(split['test_samples'])
        assert not set(split['train_patients']) & {split['test_patient']}
        def stack(samples):
            return (np.concatenate([data[s][0] for s in samples]),
                    np.concatenate([data[s][1] for s in samples]),
                    sum([data[s][2] for s in samples], []))
        x, y, train_ids = stack(split['train_samples'])
        xt, yt, ids = stack(split['test_samples'])
        old_dir = OLD / f'predictions/{family}/{encoder}/ridge/fold{fold}'
        new_dir = NEW / f'predictions/{family}/{encoder}/ridge/fold{fold}'
        out = HERE / f'models/{family}/{encoder}/fold{fold}'
        out.mkdir(parents=True, exist_ok=True)
        archived_y = np.load(old_dir / 'test_targets.npy')
        assert ids == json.loads((old_dir / 'test_spot_ids.json').read_text())
        assert np.array_equal(yt, archived_y), (family, encoder, fold, 'raw target mismatch')
        assert genes == [g for g in json.loads((old_dir / 'metrics.json').read_text()) if not g.startswith('__')]
        scaler = StandardScaler()
        xs = scaler.fit_transform(x)
        xts = scaler.transform(xt)
        pca = PCA(n_components=256, random_state=42)
        z32 = pca.fit_transform(xs)
        zt = pca.transform(xts).astype(np.float64)
        z = z32.astype(np.float64)
        mean_y = y.mean(axis=0, dtype=np.float64)
        mean_z = z.mean(axis=0)
        zc = z - mean_z
        yc = y.astype(np.float64) - mean_y
        alpha = 100.0 / (256 * len(genes))
        # Direct normal equations, in float64, without sklearn's ridge or new wrapper.
        gram = zc.T @ zc + alpha * np.eye(256)
        rhs = zc.T @ yc
        coefficients = solve(gram, rhs, assume_a='pos')
        direct = ((zt - mean_z) @ coefficients + mean_y).astype(np.float32)
        old_pred = np.load(old_dir / 'test_predictions.npy')
        revised_pred = np.load(new_dir / 'test_predictions.npy')
        # A separate baseline-restoration control, using the archived slopes.
        restored = (old_pred.astype(np.float64) + mean_y).astype(np.float32)
        record = dict(cohort=cohort_key, family=family, encoder=encoder, fold=fold,
                      raw_targets_exact=True, spot_ids_exact=True, genes_exact=True,
                      n_train=len(y), n_test=len(yt), alpha=alpha,
                      original_mae=float(np.mean(np.abs(yt.astype(float) - old_pred))),
                      corrected_mae=float(np.mean(np.abs(yt.astype(float) - revised_pred))),
                      independent_mae=float(np.mean(np.abs(yt.astype(float) - direct))),
                      direct_vs_revision=differences(direct, revised_pred),
                      baseline_restoration_vs_revision=differences(restored, revised_pred),
                      train_baseline_vs_saved=differences(mean_y, np.load(new_dir / 'training_expression_mean.npy')),
                      normal_equation_relative_error=float(np.linalg.norm(gram @ coefficients-rhs) / np.linalg.norm(rhs)),
                      training_pca_max_abs_mean=float(np.abs(mean_z).max()))
        if encoder == 'uni' and fold == 0:
            original = FixedAlphaRidgeRegressor(pca_components=256)
            original.fit(xs, y)
            control = original.predict(xts)
            record['original_class_fit_intercept'] = bool(original.model.fit_intercept)
            record['original_refit_vs_archive'] = differences(control, old_pred)
            # Third formulation: explicitly center y and use intercept-free LSQR.
            # Float64 arithmetic separates target centering from float32 roundoff.
            manual = Ridge(alpha=alpha, fit_intercept=False, solver='lsqr')
            manual.fit(zc, yc)
            manual_pred = manual.predict(zt - mean_z) + mean_y
            record['manual_centered_lsqr_vs_revision'] = differences(manual_pred, revised_pred)
            record['manual_centered_lsqr_vs_direct'] = differences(manual_pred, direct)
        np.save(out / 'direct_predictions.npy', direct)
        np.save(out / 'baseline_restored_predictions.npy', restored)
        (out / 'test_spot_ids.json').write_text(json.dumps(ids))
        record['seconds'] = time.time() - start
        (out / 'checks.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(record), flush=True)
        records.append(record)
        del x, y, xt, yt, xs, xts, z, z32, zt, zc, yc, direct, restored
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohorts', nargs='+', default=['discovery', 'validation'])
    args = parser.parse_args()
    all_records = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        jobs = [pool.submit(work, cohort, encoder)
                for cohort in args.cohorts
                for encoder in ['uni', 'virchow2', 'hoptimus0']]
        for future in as_completed(jobs):
            all_records.extend(future.result())
    all_records = [json.loads(p.read_text()) for p in sorted((HERE / 'models').glob('*/*/fold*/checks.json'))]
    (HERE / 'model_audit.json').write_text(json.dumps(all_records, indent=2) + '\n')
    print(f'{len(all_records)} INDEPENDENT MODELS COMPLETE', flush=True)

#!/usr/bin/env python3
"""Audit preservation, saved refits, and input identity for correction 01."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import joblib
import numpy as np
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
REVISION = REPO.parent
PROJECT = REVISION.parent
OUT = REPO / 'outputs'
sys.path.insert(0, str(REPO))


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((REVISION / 'original_snapshot_manifest.json').read_text())
    checks = {'status': 'partial' if args.partial else 'complete', 'original_files_unchanged': {},
              'copied_manuscript_files_unchanged': 0, 'reused_prediction_files_identical': 0,
              'folds': []}
    for folder, files in manifest.items():
        for name, info in files.items():
            assert digest(PROJECT / folder / name) == info['sha256'], f'Original changed: {folder}/{name}'
            if folder in ['CIBM_submission', 'Reviews']:
                assert digest(REVISION / folder / name) == info['sha256'], f'Copied manuscript/review changed: {name}'
                checks['copied_manuscript_files_unchanged'] += 1
        checks['original_files_unchanged'][folder] = len(files)
    for path in (OUT / 'predictions').glob('*/*/*/fold*/*'):
        if path.is_file() and path.parents[1].name in ['mlp', 'xgboost']:
            archive = PROJECT / 'outputs_v3' / path.relative_to(OUT)
            assert digest(path) == digest(archive), path
            checks['reused_prediction_files_identical'] += 1
    markers = list((OUT / 'predictions').glob('*/*/ridge/fold*/calibration.json'))
    markers += list((OUT / 'coad/predictions').glob('*/fold*/calibration.json'))
    markers += list((OUT / 'idc_visium/predictions').glob('*/fold*/calibration.json'))
    if not args.partial:
        assert len(markers) == 66, len(markers)
    for marker in sorted(markers):
        info = json.loads(marker.read_text())
        folder = marker.parent
        archive = Path(info['archived_fold'])
        assert digest(folder / 'test_spot_ids.json') == info['test_ids_sha256']
        assert digest(folder / 'test_targets.npy') == digest(archive / 'test_targets.npy')
        y = np.load(folder / 'test_targets.npy', mmap_mode='r')
        pred = np.load(folder / 'test_predictions.npy', mmap_mode='r')
        residual = np.load(folder / 'test_residuals.npy', mmap_mode='r')
        assert y.shape == pred.shape == residual.shape == (info['n_test'], info['n_genes'])
        for i in range(0, y.shape[1], 256):
            sl = np.s_[:, i:i + 256]
            assert np.isfinite(pred[sl]).all() and np.isfinite(residual[sl]).all()
            assert np.array_equal(y[sl] - pred[sl], residual[sl]), folder
        bundle = joblib.load(folder / 'calibrated_model.joblib')
        reg = bundle['regressor']
        assert reg.model.fit_intercept and reg.model.solver == 'lsqr'
        assert np.isclose(reg.model.alpha, 100 / (256 * y.shape[1]))
        assert not (set(bundle['train_samples']) & set(bundle['test_samples']))
        assert bundle['train_samples'] == info['train_samples']
        train_mean = np.load(folder / 'training_expression_mean.npy')
        baseline_error = float(np.max(np.abs(reg.model.intercept_ - train_mean)))
        assert baseline_error < .002, (folder, baseline_error)
        ids = json.loads((folder / 'test_spot_ids.json').read_text())[:8]
        sid = ids[0].split('_', 1)[0]
        embedding_root = OUT / 'embeddings' if info['family'] in ['biomarkers', '10x_janesick'] else OUT / info['family'] / 'embeddings'
        with h5py.File(embedding_root / sid / f"{info['encoder']}_embeddings.h5", 'r') as handle:
            index = {b.decode() if isinstance(b, bytes) else b: i for i, b in enumerate(handle['spot_ids'][:])}
            x = handle['embeddings'][:][[index[spot[len(sid) + 1:]] for spot in ids]].astype(np.float32)
        with threadpool_limits(4):
            reloaded = reg.predict(bundle['scaler'].transform(x))
        error = float(np.max(np.abs(reloaded - pred[:8])))
        assert np.allclose(reloaded, pred[:8], atol=2e-4, rtol=2e-5), (folder, error)
        checks['folds'].append({'family': info['family'], 'encoder': info['encoder'], 'fold': info['fold'],
                               'training_baseline_max_abs_error': baseline_error,
                               'reloaded_predictions_max_abs_error': error,
                               'target_identity_and_residuals': 'pass'})
        print(f"PASS {info['family']}/{info['encoder']}/fold{info['fold']}", flush=True)
    checks['n_models_verified'] = len(checks['folds'])
    path = OUT / 'revision_01' / ('validation_partial.json' if args.partial else 'validation.json')
    path.write_text(json.dumps(checks, indent=2) + '\n')
    print(f'Validation passed: {path}', flush=True)


if __name__ == '__main__':
    main()

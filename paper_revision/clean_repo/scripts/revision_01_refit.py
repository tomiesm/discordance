#!/usr/bin/env python3
"""Refit the archived ridge folds after correcting the missing intercept.

Uses the exact archived target rows and existing image embeddings. No new data,
gene selection, patient split, image encoding, or test-patient calibration is used.
All writes are restricted to this revision repository's outputs directory.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import h5py
import joblib
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import yaml

REPO = Path(__file__).resolve().parents[1]
REVISION = REPO.parent
PROJECT = REVISION.parent
ARCHIVE = PROJECT / 'outputs_v3'
OUTPUT = REPO / 'outputs'
sys.path.insert(0, str(REPO))

from src.regressors import FixedAlphaRidgeRegressor

ENCODERS = ['uni', 'virchow2', 'hoptimus0']
EXTERNAL_SAMPLES = {
    'coad': ['TENX111', 'TENX147', 'TENX148', 'TENX149'],
    'idc_visium': ['TENX13', 'TENX14', 'TENX39', 'TENX53', 'TENX68',
                   'NCBI776', 'NCBI681', 'NCBI682', 'NCBI683', 'NCBI684'],
}


def definitions(family, encoder):
    config = yaml.safe_load((REPO / 'config.yaml').read_text())
    if family in ('biomarkers', '10x_janesick'):
        cohort = next(c for c in config['cohorts'].values() if c['name'] == family)
        folds = [json.loads((PROJECT / 'data/v3' / f'lopo_splits_{family}' /
                             f'fold_{i}.json').read_text()) for i in range(4)]
        genes = json.loads((PROJECT / 'data/v3' / f'gene_list_{family}.json').read_text())
        source = ARCHIVE / 'predictions' / family / encoder / 'ridge'
        dest = OUTPUT / 'predictions' / family / encoder / 'ridge'
        embeddings = ARCHIVE / 'embeddings'
        samples = cohort['samples']
    else:
        samples = EXTERNAL_SAMPLES[family]
        folds = [{'fold': i, 'test_samples': [sid],
                  'train_samples': [s for s in samples if s != sid]}
                 for i, sid in enumerate(samples)]
        genes = json.loads((ARCHIVE / family / 'gene_panel.json').read_text())
        source = ARCHIVE / family / 'predictions' / encoder
        dest = OUTPUT / family / 'predictions' / encoder
        embeddings = ARCHIVE / family / 'embeddings'
    return samples, folds, genes, source, dest, embeddings


def calibration(y, pred):
    # Float64 reductions, with gene chunks to bound temporary allocation.
    n, g = y.shape
    total = n * g
    absolute = squared = signed = positive = negative_predictions = 0.0
    gene_bias = []
    for start in range(0, g, 256):
        yy = np.asarray(y[:, start:start + 256], dtype=np.float64)
        pp = np.asarray(pred[:, start:start + 256], dtype=np.float64)
        residual = yy - pp
        absolute += np.abs(residual).sum()
        squared += np.square(residual).sum()
        signed += residual.sum()
        positive += (residual > 0).sum()
        negative_predictions += (pp < 0).sum()
        gene_bias.extend(residual.mean(axis=0).tolist())
    return {'mae': float(absolute / total), 'rmse': float(np.sqrt(squared / total)),
            'mean_signed_residual': float(signed / total),
            'fraction_positive_residuals': float(positive / total),
            'fraction_negative_predictions': float(negative_predictions / total),
            'mean_absolute_gene_bias': float(np.mean(np.abs(gene_bias)))}


def gene_metrics(y, pred, genes):
    result = {}
    correlations = []
    for j, gene in enumerate(genes):
        a, b = y[:, j], pred[:, j]
        if np.std(a) < 1e-10 or np.std(b) < 1e-10:
            r, p = 0.0, 1.0
        else:
            r, p = pearsonr(a, b)
            if np.isnan(r):
                r = 0.0
        result[gene] = {'pearson': float(r), 'pvalue': float(p)}
        correlations.append(r)
    result.update({'__mean_pearson__': float(np.mean(correlations)),
                   '__median_pearson__': float(np.median(correlations)),
                   '__n_positive__': int(np.sum(np.array(correlations) > 0)),
                   '__n_genes__': len(genes), '__n_spots__': len(y)})
    return result


def run_group(family, encoder, selected_fold=None):
    with threadpool_limits(limits=4):
        return _run_group(family, encoder, selected_fold)


def _run_group(family, encoder, selected_fold=None):
    samples, folds, genes, source, dest, embeddings = definitions(family, encoder)
    if dest.is_symlink() or not dest.resolve().is_relative_to(OUTPUT.resolve()):
        raise RuntimeError(f'Unsafe destination: {dest}')
    requested = range(len(folds)) if selected_fold is None else [selected_fold]
    if all((dest / f'fold{i}' / 'calibration.json').exists() for i in requested):
        return f'{family}/{encoder}: already completed'

    # Recover each sample's exact original training/test row set and gene order.
    sample_targets, sample_ids = {}, {}
    source_gene_order = [g for g in json.loads((source / 'fold0/metrics.json').read_text())
                         if not g.startswith('__')]
    assert source_gene_order == genes, (family, 'gene order differs')
    for i, fold in enumerate(folds):
        old = source / f'fold{i}'
        ids = json.loads((old / 'test_spot_ids.json').read_text())
        targets = np.load(old / 'test_targets.npy', mmap_mode='r')
        assert targets.shape == (len(ids), len(genes))
        id_samples = np.array([s.split('_', 1)[0] for s in ids])
        assert set(id_samples) == set(fold['test_samples'])
        for sid in fold['test_samples']:
            assert sid not in sample_targets
            mask = id_samples == sid
            sample_targets[sid] = np.asarray(targets[mask])
            sample_ids[sid] = [s for s, use in zip(ids, mask) if use]

    sample_features = {}
    for sid in samples:
        with h5py.File(embeddings / sid / f'{encoder}_embeddings.h5', 'r') as f:
            names = [s.decode() if isinstance(s, bytes) else s for s in f['spot_ids'][:]]
            index = {b: i for i, b in enumerate(names)}
            ordered = [index[s[len(sid) + 1:]] for s in sample_ids[sid]]
            sample_features[sid] = f['embeddings'][:][ordered].astype(np.float32)
        assert np.isfinite(sample_features[sid]).all()
        assert np.isfinite(sample_targets[sid]).all()

    completed = 0
    for i, fold in enumerate(folds):
        if selected_fold is not None and i != selected_fold:
            continue
        out = dest / f'fold{i}'
        if (out / 'calibration.json').exists():
            continue
        started = time.monotonic()
        train_samples, test_samples = fold['train_samples'], fold['test_samples']
        assert not (set(train_samples) & set(test_samples))
        x_train = np.concatenate([sample_features[s] for s in train_samples])
        y_train = np.concatenate([sample_targets[s] for s in train_samples])
        x_test = np.concatenate([sample_features[s] for s in test_samples])
        y_test = np.concatenate([sample_targets[s] for s in test_samples])
        test_ids = sum([sample_ids[s] for s in test_samples], [])
        old = source / f'fold{i}'
        assert test_ids == json.loads((old / 'test_spot_ids.json').read_text())
        assert np.array_equal(y_test, np.load(old / 'test_targets.npy', mmap_mode='r'))

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        reg = FixedAlphaRidgeRegressor(pca_components=256)
        print(f'START {family}/{encoder}/fold{i}: '
              f'{len(y_train)} training spots, {len(genes)} targets', flush=True)
        reg.fit(x_train, y_train)
        pred = reg.predict(x_test).astype(np.float32)
        assert np.isfinite(pred).all()
        metrics = gene_metrics(y_test, pred, genes)
        old_pred = np.load(old / 'test_predictions.npy', mmap_mode='r')
        old_metrics = json.loads((old / 'metrics.json').read_text())
        new_diagnostics = calibration(y_test, pred)
        old_diagnostics = calibration(y_test, old_pred)
        train_mean = np.mean(y_train, axis=0, dtype=np.float64)

        out.mkdir(parents=True, exist_ok=True)
        np.save(out / 'test_predictions.npy', pred)
        np.save(out / 'test_residuals.npy', (y_test - pred).astype(np.float32))
        shutil.copy2(old / 'test_targets.npy', out / 'test_targets.npy')
        shutil.copy2(old / 'test_spot_ids.json', out / 'test_spot_ids.json')
        (out / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
        np.save(out / 'training_expression_mean.npy', train_mean)
        joblib.dump({'scaler': scaler, 'regressor': reg, 'gene_names': genes,
                     'train_samples': train_samples, 'test_samples': test_samples},
                    out / 'calibrated_model.joblib', compress=0)
        # Completion marker written last, after all artifacts are present.
        provenance = {'correction': '01_ridge_intercept', 'family': family,
                      'encoder': encoder, 'fold': i, 'fit_intercept': True,
                      'alpha': float(reg.model.alpha), 'solver': reg.model.solver,
                      'pca_components': 256, 'pca_random_state': 42,
                      'n_train': len(y_train), 'n_test': len(y_test), 'n_genes': len(genes),
                      'train_samples': train_samples, 'test_samples': test_samples,
                      'archived_fold': str(old),
                      'test_ids_sha256': hashlib.sha256((old / 'test_spot_ids.json').read_bytes()).hexdigest(),
                      'old': old_diagnostics, 'corrected': new_diagnostics,
                      'old_mean_pearson': old_metrics['__mean_pearson__'],
                      'corrected_mean_pearson': metrics['__mean_pearson__'],
                      'seconds': time.monotonic() - started}
        (out / 'calibration.json').write_text(json.dumps(provenance, indent=2) + '\n')
        print(f'{family}/{encoder}/fold{i}: MAE {old_diagnostics["mae"]:.4f} -> '
              f'{new_diagnostics["mae"]:.4f}; {provenance["seconds"]:.1f}s', flush=True)
        completed += 1
        del x_train, y_train, x_test, y_test, pred, reg, scaler
    return f'{family}/{encoder}: {completed} folds refitted'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--families', nargs='+', default=['biomarkers', '10x_janesick', 'coad', 'idc_visium'])
    parser.add_argument('--encoders', nargs='+', default=ENCODERS)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--parallel-folds', action='store_true',
                        help='Schedule individual folds independently; statistical fit is unchanged.')
    args = parser.parse_args()
    jobs = [(f, e, i) for f in args.families
            for i in (range(10 if f == 'idc_visium' else 4) if args.parallel_folds else [None])
            for e in args.encoders]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_group, *job): job for job in jobs}
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == '__main__':
    main()

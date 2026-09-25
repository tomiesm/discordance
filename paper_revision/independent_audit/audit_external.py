"""Check external raw targets and independently solve representative ridge fits.

All 42 external held-out target matrices are checked against raw HEST. Direct
solutions cover all 12 COAD fits and fold 0 of each whole-transcriptome encoder.
Does not import the revision or original model implementation.
"""
import os
for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[variable] = '4'
import ast
import json
from pathlib import Path
import anndata as ad
import h5py
import numpy as np
from scipy.linalg import solve
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OLD = ROOT / 'outputs_v3'
NEW = ROOT / 'paper_revision/clean_repo/outputs'


def constants(path, name):
    tree = ast.parse(path.read_text())
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))


def main():
    records = []
    for family, script, constant in [('coad', '20_coad_generalization.py', 'COAD_SAMPLES'),
                                      ('idc_visium', '21_idc_visium_generalization.py', 'VISIUM_SAMPLES')]:
        samples = constants(ROOT / 'clean_repo/scripts' / script, constant)
        genes = json.loads((OLD / family / 'gene_panel.json').read_text())
        raw = {}
        for sid in samples:
            a = ad.read_h5ad(ROOT / f'data/hest/st/{sid}.h5ad')
            x = a[:, genes].X
            if hasattr(x, 'toarray'):
                x = x.toarray()
            if not np.issubdtype(x.dtype, np.floating):
                x = x.astype(float)
            raw[sid] = (np.log1p(x).astype(np.float32), list(map(str, a.obs_names)))
            del a, x
        for encoder in ['uni', 'virchow2', 'hoptimus0']:
            data = {}
            for sid in samples:
                y, barcodes = raw[sid]
                by_barcode = {b: i for i, b in enumerate(barcodes)}
                with h5py.File(OLD / f'{family}/embeddings/{sid}/{encoder}_embeddings.h5', 'r') as f:
                    emb = f['embeddings'][:].astype(np.float32)
                    names = [b.decode() if isinstance(b, bytes) else str(b) for b in f['spot_ids'][:]]
                assert len(set(names)) == len(names)
                keep = [i for i, b in enumerate(names) if b in by_barcode and np.isfinite(emb[i]).all()
                        and np.isfinite(y[by_barcode[b]]).all()]
                data[sid] = (emb[keep], y[[by_barcode[names[i]] for i in keep]],
                             [f'{sid}_{names[i]}' for i in keep])
            for fold, sid in enumerate(samples):
                old = OLD / f'{family}/predictions/{encoder}/fold{fold}'
                new = NEW / f'{family}/predictions/{encoder}/fold{fold}'
                xt, yt, ids = data[sid]
                assert ids == json.loads((old / 'test_spot_ids.json').read_text())
                assert np.array_equal(yt, np.load(old / 'test_targets.npy', mmap_mode='r'))
                record = dict(family=family, encoder=encoder, fold=fold, n_genes=len(genes),
                              raw_targets_exact=True, raw_ids_exact=True)
                if family == 'coad' or fold == 0:
                    train = [s for s in samples if s != sid]
                    x = np.concatenate([data[s][0] for s in train])
                    y = np.concatenate([data[s][1] for s in train])
                    scaler = StandardScaler()
                    xs = scaler.fit_transform(x)
                    pca = PCA(n_components=256, random_state=42)
                    z = pca.fit_transform(xs).astype(float)
                    zt = pca.transform(scaler.transform(xt)).astype(float)
                    mu_z = z.mean(axis=0)
                    mu_y = y.mean(axis=0, dtype=float)
                    z -= mu_z
                    zt -= mu_z
                    alpha = 100/(256*len(genes))
                    gram = z.T@z + alpha*np.eye(256)
                    saved = np.load(new / 'test_predictions.npy', mmap_mode='r')
                    max_diff, abs_diff, squares, direct_error = 0., 0., 0., 0.
                    n = len(yt)*len(genes)
                    # Chunk gene targets; direct solve never calls sklearn Ridge.
                    for start in range(0, len(genes), 512):
                        sl = slice(start, start+512)
                        rhs = z.T @ (y[:, sl].astype(float)-mu_y[sl])
                        coefficients = solve(gram, rhs, assume_a='pos')
                        prediction = (zt@coefficients + mu_y[sl]).astype(np.float32)
                        delta = prediction.astype(float)-saved[:, sl]
                        max_diff = max(max_diff, float(np.abs(delta).max()))
                        abs_diff += np.abs(delta).sum()
                        squares += (delta**2).sum()
                        direct_error += np.abs(yt[:, sl].astype(float)-prediction).sum()
                    record.update(direct_solve=True, alpha=alpha,
                                  max_prediction_difference=max_diff,
                                  mean_prediction_difference=float(abs_diff/n),
                                  prediction_rmse=float(np.sqrt(squares/n)),
                                  independent_mae=float(direct_error/n),
                                  training_baseline_max_difference=float(np.abs(mu_y-np.load(new/'training_expression_mean.npy')).max()))
                    del x, y, xs, z, zt, saved
                records.append(record)
                print(json.dumps(record), flush=True)
                (HERE / 'external_audit.json').write_text(json.dumps(records, indent=2)+'\n')
            del data
        del raw
    print('ALL 42 EXTERNAL TARGET MATRICES VERIFIED; 15 DIRECT SOLUTIONS COMPLETE', flush=True)


if __name__ == '__main__':
    with threadpool_limits(4):
        main()

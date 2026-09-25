"""Read-only refits with the original research loader, alignment, and regressor.

Run in the `torch` environment named in the historical training log. This
environment may have been updated since submission; do not imply bit identity.
"""
import os
for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[variable] = '4'
import importlib.util
import json
import logging
import sys
from pathlib import Path
import numpy as np
import scipy
import sklearn
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('original_training', ROOT / 'scripts_v3/03_train_predict.py')
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)
config = training.load_v3_config(str(ROOT / 'config_v3.yaml'))
for key in ['hest_dir', 'v3_data_dir', 'output_dir']:
    config[key] = str(ROOT / config[key])
records = []
with threadpool_limits(4):
    for cohort in ['discovery', 'validation']:
        cfg = config['cohorts'][cohort]
        family = cfg['name']
        expr, genes, mapping = training.load_expression_for_cohort(cfg, config, logging.getLogger('audit'))
        embeddings = training.load_embeddings_for_samples(cfg['samples'], 'uni', config)
        split = json.loads((ROOT / f'data/v3/lopo_splits_{family}/fold_0.json').read_text())
        x, y, train_ids = training.align_expression_embeddings(expr, mapping, embeddings, split['train_samples'])
        xt, yt, ids = training.align_expression_embeddings(expr, mapping, embeddings, split['test_samples'])
        old = ROOT / f'outputs_v3/predictions/{family}/uni/ridge/fold0'
        assert ids == json.loads((old / 'test_spot_ids.json').read_text())
        assert np.array_equal(yt, np.load(old / 'test_targets.npy'))
        scaler = StandardScaler()
        xs, xts = scaler.fit_transform(x), scaler.transform(xt)
        reg = training.get_regressor(config['regressors'][0])
        reg.fit(xs, y)
        pred = reg.predict(xts)
        archived = np.load(old / 'test_predictions.npy')
        delta = pred.astype(float)-archived
        record = dict(cohort=cohort, encoder='uni', fold=0,
                      original_loader_targets_exact=True, original_loader_ids_exact=True,
                      python=sys.version, numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                      fit_intercept=bool(reg.model.fit_intercept),
                      maximum_prediction_difference=float(np.abs(delta).max()),
                      mean_prediction_difference=float(np.abs(delta).mean()),
                      prediction_rmse=float(np.sqrt(np.mean(delta**2))),
                      original_mae=float(np.abs(yt.astype(float)-archived).mean()),
                      refit_mae=float(np.abs(yt.astype(float)-pred).mean()))
        records.append(record)
        print(json.dumps(record), flush=True)
        del expr, embeddings, x, y, xt, yt, xs, xts, reg, pred, archived, delta
(HERE / 'original_environment_controls.json').write_text(json.dumps(records, indent=2)+'\n')

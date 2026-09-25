"""Propagate saved diagnostic vectors through fixed ridge models; no refits."""
import json
from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml

ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT / 'paper_revision/clean_repo'
OUT = Path(__file__).resolve().parent / 'embedding_numerics'
sys.path.insert(0, str(REPO))


def main():
    cfg = yaml.safe_load((REPO / 'config.yaml').read_text())
    locations = pd.read_csv(OUT / 'locations.csv')
    rows, warning_text = [], []
    for ec in cfg['encoders']:
        name = ec['name']
        vectors = np.load(OUT / f'{name}_vectors.npz')
        full = np.load(OUT / f'{name}_original_batch_vectors.npz')
        full_map = dict(zip(full['indices'].tolist(), full['recomputed']))
        for cc in cfg['cohorts'].values():
            for fold, (patient, samples) in enumerate(cc['patient_mapping'].items()):
                fd = REPO / 'outputs/predictions' / cc['name'] / name / 'ridge' / f'fold{fold}'
                with warnings.catch_warnings(record=True) as ws:
                    warnings.simplefilter('always')
                    model = joblib.load(fd / 'calibrated_model.joblib')
                warning_text.extend(str(w.message) for w in ws)
                assert set(samples) == set(model['test_samples'])
                ids = json.loads((fd / 'test_spot_ids.json').read_text())
                lookup = {s:i for i,s in enumerate(ids)}
                saved = np.load(fd / 'test_predictions.npy', mmap_mode='r')
                targets = np.load(fd / 'test_targets.npy', mmap_mode='r')
                selected = locations.index[locations['sample'].isin(samples)].to_numpy()
                old = model['regressor'].predict(model['scaler'].transform(vectors['archived'][selected]))
                new = model['regressor'].predict(model['scaler'].transform(vectors['deterministic_batch8'][selected]))
                for pos, k in enumerate(selected):
                    loc = locations.loc[k].to_dict()
                    j = lookup[loc['sample']+'_'+loc['barcode']]
                    y = targets[j].astype(float)
                    modes = {'deterministic_batch8':new[pos]}
                    if k in full_map:
                        modes['original_contiguous_batch'] = model['regressor'].predict(
                            model['scaler'].transform(full_map[k][None]))[0]
                    for mode,pred in modes.items():
                        delta = pred.astype(float)-old[pos].astype(float)
                        rows.append({**loc,'cohort':cc['name'],'patient':patient,'encoder':name,'mode':mode,
                            'saved_prediction_reconstruction_max_error':float(np.max(np.abs(old[pos]-saved[j]))),
                            'mean_absolute_prediction_change':float(np.abs(delta).mean()),
                            'max_absolute_prediction_change':float(np.max(np.abs(delta))),
                            'archived_vector_mae':float(np.abs(y-old[pos]).mean()),
                            'recomputed_vector_mae':float(np.abs(y-pred).mean()),
                            'absolute_mae_change':float(abs(np.abs(y-old[pos]).mean()-np.abs(y-pred).mean()))})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'prediction_impact.csv',index=False)
    summary = {'sklearn_version':sklearn.__version__,'model_load_warnings':warning_text,
               'n_spots':len(locations),'n_comparisons':len(df),
               'saved_prediction_reconstruction_max_error':float(df.saved_prediction_reconstruction_max_error.max()),
               'mean_absolute_prediction_change':float(df.mean_absolute_prediction_change.mean()),
               'max_absolute_prediction_change':float(df.max_absolute_prediction_change.max()),
               'max_absolute_spot_mae_change':float(df.absolute_mae_change.max()),
               'mean_absolute_spot_mae_change':float(df.absolute_mae_change.mean()),
               'scope':'diagnostic sample, fixed fitted ridge models; no full-cohort rank-stability certification'}
    assert not warning_text
    assert summary['saved_prediction_reconstruction_max_error'] < 1e-4
    (OUT / 'prediction_impact_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()

"""Load the fixed prediction folds and their spatial coordinates."""
import json
import numpy as np
import pandas as pd
from .paths import PROJECT_ROOT as REPO, ANALYSIS_ROOT as AUDIT
ENCODERS = ['uni', 'virchow2', 'hoptimus0']

def readj(p):
    return json.loads(p.read_text())

def load_cohort(cohort):
    diag = pd.read_parquet(AUDIT / 'scores/spot_score_diagnostics.parquet').set_index('spot_id')
    ys, ids, rs = [], [], []
    for fold in range(4):
        base = REPO / 'outputs/predictions' / cohort
        fd = base / 'uni/ridge' / f'fold{fold}'
        yy = np.load(fd / 'test_targets.npy')
        ii = readj(fd / 'test_spot_ids.json')
        rr = []
        for enc in ENCODERS:
            d = base / enc / 'ridge' / f'fold{fold}'
            assert readj(d / 'test_spot_ids.json') == ii
            assert np.array_equal(np.load(d / 'test_targets.npy'), yy)
            rr.append(yy.astype(float) - np.load(d / 'test_predictions.npy').astype(float))
        ys.append(yy); ids.extend(ii); rs.append(np.stack(rr))
    y = np.concatenate(ys)
    r = np.concatenate(rs, axis=1)
    loc = diag.loc[ids].reset_index()
    assert loc.spot_id.is_unique and (loc.cohort == cohort).all()
    genes = readj(REPO / 'data/v3' / f'gene_list_{cohort}.json')
    return y, r, loc, genes

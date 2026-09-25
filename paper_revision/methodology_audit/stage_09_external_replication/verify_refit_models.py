"""Recreate actual PCA training scores for an independent centered-ridge check."""
import argparse
import json
from pathlib import Path
import sys
import h5py
import joblib
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

OUT=Path(__file__).resolve().parent;AUDIT=OUT.parent;ROOT=AUDIT.parents[1]
REPO=ROOT/'paper_revision/clean_repo';sys.path.insert(0,str(REPO))
BASE=REPO/'outputs/idc_visium'
SAMPLES=['TENX13','TENX14','TENX39','TENX53','TENX68','NCBI776','NCBI681','NCBI682','NCBI683','NCBI684']


def main(enc):
    assert sklearn.__version__=='1.4.0'
    dest=OUT/'visium_replacement'/enc
    bundle=joblib.load(dest/'model.joblib');model=bundle['regressor'];scaler=bundle['scaler']
    xs={};ys={}
    for s in SAMPLES:
        fd=BASE/'predictions'/enc/f'fold{SAMPLES.index(s)}';ids=json.loads((fd/'test_spot_ids.json').read_text())
        ys[s]=np.load(fd/'test_targets.npy')
        with h5py.File(BASE/'embeddings'/s/f'{enc}_embeddings.h5','r') as f:
            bars=[v.decode() if isinstance(v,bytes) else str(v) for v in f['spot_ids'][:]];lookup={v:i for i,v in enumerate(bars)}
            xs[s]=f['embeddings'][:][[lookup[v[len(s)+1:]] for v in ids]].astype(np.float32)
    train=bundle['train_samples'];test=bundle['test_samples'];assert not set(train)&set(test)
    x=scaler.transform(np.concatenate([xs[s] for s in train]));xt=scaler.transform(np.concatenate([xs[s] for s in test]))
    y=np.concatenate([ys[s] for s in train]);del xs,ys
    # fit_transform uses U*S for randomized PCA. transform(training_X) uses X*V,
    # which is a different approximation. Recreate U*S with identical seed/data.
    pca=clone(model.pca);xp=pca.fit_transform(x)
    component_error=float(np.max(np.abs(pca.components_-model.pca.components_)))
    assert component_error==0,component_error
    stored=np.load(dest/'test_predictions.npy');reproduced=model.predict(xt).astype(np.float32)
    assert np.array_equal(stored,reproduced)
    sel=np.linspace(0,y.shape[1]-1,17,dtype=int)
    # Retain the float32 reduction order of the full original training matrix.
    ym=y.mean(axis=0)[sel];xm=xp.mean(axis=0)
    ref=Ridge(alpha=model.model.alpha,fit_intercept=False,solver='lsqr',tol=model.model.tol)
    ref.fit(xp-xm,y[:,sel]-ym)
    independent=ref.predict(model.pca.transform(xt)-xm)+ym
    difference=float(np.max(np.abs(independent-stored[:,sel])))
    coeff_error=float(np.max(np.abs(ref.coef_-model.model.coef_[sel])))
    intercept_error=float(np.max(np.abs((ym-ref.coef_@xm)-model.model.intercept_[sel])))
    # Fixed numerical tolerance declared before the new comparison was run.
    passed=difference<1e-4 and coeff_error<1e-4 and intercept_error<1e-4
    diagnostic=dict(status='pass' if passed else 'fail',encoder=enc,n_genes=17,
        selected_gene_indices=sel.tolist(),pca_components_max_difference=component_error,
        training_fit_transform_vs_transform_max_difference=float(np.max(np.abs(xp-model.pca.transform(x)))),
        independent_centered_prediction_max_difference=difference,coefficient_max_difference=coeff_error,
        intercept_max_difference=intercept_error,tolerance=1e-4,saved_full_prediction_reproduction='exact',
        explanation='Original preliminary diagnostic used PCA.transform on training inputs; this check recreates the actual deterministic PCA.fit_transform training scores.')
    (dest/'independent_model_checks.json').write_text(json.dumps(diagnostic,indent=2)+'\n')
    print(json.dumps(diagnostic,indent=2),flush=True)
    assert passed,diagnostic


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--encoder',choices=['uni','virchow2','hoptimus0'],required=True)
    with threadpool_limits(limits=1):main(p.parse_args().encoder)

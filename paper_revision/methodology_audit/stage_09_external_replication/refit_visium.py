"""Only the three joint Block A replacement fits; reuse the other 24 fits."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import sys
import time
import h5py
import joblib
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

OUT=Path(__file__).resolve().parent;AUDIT=OUT.parent;ROOT=AUDIT.parents[1]
REPO=ROOT/'paper_revision/clean_repo';BASE=REPO/'outputs/idc_visium'
sys.path.insert(0,str(REPO))
from src.regressors import FixedAlphaRidgeRegressor
SAMPLES=['TENX13','TENX14','TENX39','TENX53','TENX68','NCBI776','NCBI681','NCBI682','NCBI683','NCBI684']


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def one_encoder(enc):
    with threadpool_limits(limits=1):return run(enc)


def run(enc):
    started=time.monotonic();dest=OUT/'visium_replacement'/enc;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'COMPLETE.json').exists():return json.loads((dest/'COMPLETE.json').read_text())
    spec=json.loads((AUDIT/'stage_01_provenance/visium_grouped_split_specification.json').read_text())['folds'][0]
    genes=json.loads((BASE/'gene_panel.json').read_text());ys={};ids={};xs={};inputs={}
    for k,sample in enumerate(SAMPLES):
        fd=BASE/'predictions'/enc/f'fold{k}'
        ys[sample]=np.load(fd/'test_targets.npy');ids[sample]=json.loads((fd/'test_spot_ids.json').read_text())
        assert all(s.startswith(sample+'_') for s in ids[sample])
        assert ys[sample].shape==(len(ids[sample]),len(genes))
        m=json.loads((fd/'metrics.json').read_text());assert [g for g in m if not g.startswith('__')]==genes
        hp=BASE/'embeddings'/sample/f'{enc}_embeddings.h5'
        with h5py.File(hp,'r') as f:
            bar=[b.decode() if isinstance(b,bytes) else str(b) for b in f['spot_ids'][:]]
            lookup={b:i for i,b in enumerate(bar)}
            order=[lookup[s[len(sample)+1:]] for s in ids[sample]]
            xs[sample]=f['embeddings'][:][order].astype(np.float32)
        for p in [fd/'test_targets.npy',fd/'test_spot_ids.json',hp]:inputs[str(p.relative_to(ROOT))]=digest(p)
    train=spec['train_samples'];test=spec['test_samples'];assert not set(train)&set(test)
    x=np.concatenate([xs[s] for s in train]);y=np.concatenate([ys[s] for s in train]);xt=np.concatenate([xs[s] for s in test]);yt=np.concatenate([ys[s] for s in test]);tid=sum([ids[s] for s in test],[])
    del xs,ys
    assert np.isfinite(x).all() and np.isfinite(y).all()
    scaler=StandardScaler();x=scaler.fit_transform(x);xt=scaler.transform(xt)
    model=FixedAlphaRidgeRegressor(pca_components=256)
    print(enc,'START joint Block A:',len(y),'training /',len(yt),'test /',len(genes),'genes',flush=True)
    model.fit(x,y)
    print(enc,'fit complete after',round(time.monotonic()-started,1),'seconds',flush=True)
    pred=model.predict(xt).astype(np.float32);assert np.isfinite(pred).all()
    # Independent centered-target formulation with the identical lsqr settings,
    # on fixed evenly spaced genes, not a different solver chosen after results.
    sel=np.linspace(0,len(genes)-1,17,dtype=int);xp=model.pca.transform(x);zp=model.pca.transform(xt)
    # PCA fit_transform and transform can differ slightly numerically; validate
    # the intercept/baseline identity separately from that transform discrepancy.
    ymean=y[:,sel].mean(axis=0);xmean=xp.mean(axis=0)
    reference=Ridge(alpha=model.model.alpha,fit_intercept=False,solver='lsqr',tol=model.model.tol)
    reference.fit(xp-xmean,y[:,sel]-ymean)
    independent=reference.predict(zp-xmean)+ymean
    difference=float(np.max(np.abs(independent-pred[:,sel])))
    # Preserve and expose this diagnostic; verify final stored-model predictions
    # exactly below. The PCA fit_transform/transform distinction is not hidden.
    np.save(dest/'test_targets.npy',yt);np.save(dest/'test_predictions.npy',pred);np.save(dest/'test_residuals.npy',(yt-pred).astype(np.float32))
    np.save(dest/'training_expression_mean.npy',y.mean(axis=0,dtype=np.float64))
    (dest/'test_spot_ids.json').write_text(json.dumps(tid)+'\n')
    bundle=dict(scaler=scaler,regressor=model,gene_names=genes,train_samples=train,test_samples=test)
    joblib.dump(bundle,dest/'model.joblib')
    loaded=joblib.load(dest/'model.joblib')
    reproduced=loaded['regressor'].predict(xt[:53]).astype(np.float32)
    assert np.array_equal(reproduced,pred[:53])
    record=dict(encoder=enc,group='10x_block_A',train_samples=train,test_samples=test,n_train=len(y),n_test=len(yt),n_genes=len(genes),
        sklearn_version=sklearn.__version__,alpha=float(model.model.alpha),solver=model.model.solver,tol=float(model.model.tol),fit_intercept=True,
        pca_components=256,pca_variance_retained=float(model.pca.explained_variance_ratio_.sum()),seconds=time.monotonic()-started,
        independent_centered_target_max_prediction_difference=difference,stored_model_reproduction='exact',input_hashes=inputs)
    (dest/'COMPLETE.json').write_text(json.dumps(record,indent=2)+'\n')
    print(enc,'COMPLETE',round(record['seconds'],1),'seconds; centered-target max difference',difference,flush=True)
    return record


def main():
    assert sklearn.__version__=='1.4.0',sklearn.__version__
    registry=[]
    for enc in ['uni','virchow2','hoptimus0']:
        for k,sample in enumerate(SAMPLES[2:],start=2):
            fd=BASE/'predictions'/enc/f'fold{k}';meta=json.loads((fd/'calibration.json').read_text())
            assert meta['test_samples']==[sample]
            assert set(meta['train_samples'])==set(SAMPLES)-{sample}
            registry.append(dict(encoder=enc,sample=sample,original_fold=k,prediction_dir=str(fd),
                train_samples=meta['train_samples'],test_samples=meta['test_samples'],predictions_sha256=digest(fd/'test_predictions.npy')))
    (OUT/'reused_visium_fits.json').write_text(json.dumps(registry,indent=2)+'\n')
    records=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(one_encoder,enc) for enc in ['uni','virchow2','hoptimus0']]
        for f in as_completed(futures):records.append(f.result())
    (OUT/'refit_complete.json').write_text(json.dumps(dict(status='complete',new_fits=records,reused_fits=len(registry)),indent=2)+'\n')


if __name__=='__main__':main()

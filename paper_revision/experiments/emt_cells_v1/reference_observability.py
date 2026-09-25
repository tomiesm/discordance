"""Donor-held-out test of a restricted panel against disjoint reference genes."""
from common import *
from datetime import datetime,timezone
import shutil
import anndata as ad
import pandas as pd
import joblib
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


def rho(a,b):
    return float(spearmanr(a,b).statistic) if np.ptp(a)>0 and np.ptp(b)>0 else np.nan


def fit_weighted(x,y,donors):
    unique,counts=np.unique(donors,return_counts=True)
    weights=np.array([len(x)/(len(unique)*counts[np.flatnonzero(unique==d)[0]]) for d in donors])
    scaler=StandardScaler().fit(x,sample_weight=weights)
    model=Ridge(alpha=100).fit(scaler.transform(x),y,sample_weight=weights)
    return scaler,model,weights


def main():
    frozen=HERE/'reference_frozen';frozen.mkdir(exist_ok=True)
    for name in ['protocol.json','reference_observability.py','common.py']:
        shutil.copy2(HERE/name,frozen/name)
    dump(frozen/'manifest.json',{'utc':datetime.now(timezone.utc).isoformat(),'sha256':{p.name:sha(p) for p in frozen.iterdir() if p.is_file() and p.name!='manifest.json'}})
    reference=ad.read_h5ad(RESULTS/'wu_reference.h5ad')
    genes=np.array(reference.var_names);donors=reference.obs['orig.ident'].to_numpy()
    tumor=reference.obs.celltype_major.to_numpy()=='Cancer Epithelial'
    full=normalize(reference.X)
    programs=reference.obs[['orig.ident','subtype','celltype_major','celltype_minor']].copy()
    programs['full_hallmark_emt']=gene_score(full,genes,hallmark_genes())
    programs['full_epithelial']=gene_score(full,genes,PROTOCOL['epithelial_genes'])
    cell_counts=pd.Series(donors[tumor]).value_counts()
    eligible=sorted(cell_counts[cell_counts>=100].index)
    train_pool=[]
    rng=np.random.default_rng(PROTOCOL['seed'])
    for donor in eligible:
        ix=np.flatnonzero(tumor & (donors==donor))
        train_pool.extend(np.sort(rng.choice(ix,min(len(ix),3000),replace=False)).tolist())
    train_pool=np.array(train_pool,dtype=int)
    rows=[];summaries=[];coverage=[]
    for representative,panel in [('NCBI785','panel313'),('NCBI783','panel280')]:
        _,source_genes,_=read_10x(SOURCES/'vendor'/representative/'cell_feature_matrix.h5')
        panel_genes=source_genes[biological_mask(source_genes)]
        present=panel_genes[np.isin(panel_genes,genes)]
        index=pd.Index(genes).get_indexer(present)
        raw=reference.X[:,index]
        x=normalize(raw).toarray()
        target_genes=[g for g in hallmark_genes() if g in genes and g not in set(panel_genes)]
        assert not set(target_genes)&set(panel_genes) and len(target_genes)>=20
        y=gene_score(full,genes,target_genes)
        direct=gene_score(sparse.csr_matrix(x),present,hallmark_genes())
        programs[panel+'_disjoint_target']=y
        programs[panel+'_measured_hallmark']=direct
        flags,ntf,nepi=coexpression_flags(raw,present)
        programs[panel+'_tf_candidate']=flags
        programs[panel+'_tf_detected']=ntf
        heldout=np.full(len(reference),np.nan)
        fold_specs=[]
        for donor in eligible:
            training=train_pool[donors[train_pool]!=donor]
            testing=np.flatnonzero(tumor & (donors==donor))
            assert donor not in set(donors[training])
            scaler,model,weights=fit_weighted(x[training],y[training],donors[training])
            predicted=model.predict(scaler.transform(x[testing]));heldout[testing]=predicted
            rows.append(dict(panel=panel,donor=donor,n_test=len(testing),n_train=len(training),
                             rho_learned=rho(y[testing],predicted),rho_direct=rho(y[testing],direct[testing]),
                             rho_depth=rho(y[testing],reference.obs.nCount_RNA.iloc[testing]),
                             rmse=float(np.sqrt(np.mean((predicted-y[testing])**2)))))
            fold_specs.append(dict(donor=donor,train_donors=sorted(set(donors[training])),n_training=len(training),
                                   scaler_mean=scaler.mean_.tolist(),scaler_scale=scaler.scale_.tolist(),
                                   coefficients=model.coef_.tolist(),intercept=float(model.intercept_)))
            if donor==eligible[0]:
                design=np.column_stack([scaler.transform(x[training]),np.ones(len(training))])
                penalty=np.eye(design.shape[1])*100;penalty[-1,-1]=0
                independent=np.linalg.solve(design.T@(weights[:,None]*design)+penalty,design.T@(weights*y[training]))
                check=np.column_stack([scaler.transform(x[testing]),np.ones(len(testing))])@independent
                np.testing.assert_allclose(check,predicted,rtol=1e-9,atol=1e-9)
            print('REFERENCE FOLD',panel,donor,'rho',rows[-1]['rho_learned'],flush=True)
        programs[panel+'_heldout_prediction']=heldout
        scaler,model,_=fit_weighted(x[train_pool],y[train_pool],donors[train_pool])
        # All-cell predictions are descriptive and not held out for training donors;
        # phenotype-specific held-out performance above is the primary evaluation.
        programs[panel+'_full_fit_prediction']=model.predict(scaler.transform(x))
        joblib.dump(dict(genes=present,scaler=scaler,model=model,target_genes=target_genes,
                         train_donors=eligible,protocol_sha256=sha(HERE/'protocol.json')),
                    RESULTS/f'{panel}_reference_model.joblib')
        frame=pd.DataFrame([r for r in rows if r['panel']==panel])
        fraction=float((frame.rho_learned>=.3).mean());median=float(frame.rho_learned.median())
        passed=len(frame)>=10 and median>=.3 and fraction>=.7
        summaries.append(dict(panel=panel,n_donors=len(frame),median_rho=median,
                              fraction_donors_rho_at_least_0_3=fraction,passed=bool(passed),
                              target_genes=target_genes,measured_hallmark_genes=[g for g in present if g in hallmark_genes()],
                              interpretation='Operational observability of a transcriptomic proxy; not validation of EMT biology or Xenium transfer'))
        for program,signature in [('hallmark_emt',hallmark_genes()),('epithelial',PROTOCOL['epithelial_genes']),('canonical_tfs',PROTOCOL['canonical_emt_tfs'])]:
            available=[g for g in present if g in signature]
            coverage.append(dict(panel=panel,program=program,n_genes=len(available),genes=';'.join(available)))
        dump(RESULTS/f'{panel}_reference_folds.json',fold_specs)
    pd.DataFrame(rows).to_csv(RESULTS/'reference_donor_performance.csv',index=False)
    pd.DataFrame(coverage).to_csv(RESULTS/'reference_program_coverage.csv',index=False)
    programs.to_parquet(RESULTS/'reference_cell_programs.parquet')
    dump(RESULTS/'REFERENCE_OBSERVABILITY.json',summaries)
    print('REFERENCE OBSERVABILITY COMPLETE',summaries,flush=True)


if __name__=='__main__':main()

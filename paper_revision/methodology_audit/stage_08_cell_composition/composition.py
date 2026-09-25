"""Fixed-group composition and source-lineage associations in verified cell data."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import anndata as ad

OUT=Path(__file__).resolve().parent;AUDIT=OUT.parent;ROOT=AUDIT.parents[1]
sys.path.insert(0,str(AUDIT/'stage_02_patient_calibration'))
from calibration import load_cohort
from src.discordance import compute_conditional_discordance
CELLS=ROOT/'paper_revision/experiments/emt_cells_v1/results'
ZONES=ROOT/'paper_revision/experiments/emt_zone_residual_v1/results'
SAMPLES=['NCBI785','NCBI784','NCBI783']


def groups(score):
    q=np.quantile(score,[.25,.75]);return score<=q[0],score>=q[1]


def design(exposed,cov):
    sd=cov.std(axis=0);cov=cov[:,sd>1e-10]
    cov=(cov-cov.mean(axis=0))/cov.std(axis=0) if cov.shape[1] else cov
    return np.column_stack([np.ones(len(exposed)),exposed,cov])


def fit(x,y,xy,origin,bootstrap=True):
    assert np.isfinite(origin).all() and np.isfinite(xy).all(), 'Spatial bootstrap requires finite coordinates and origin'
    beta,_,rank,_=np.linalg.lstsq(x,y,rcond=None)
    out=dict(estimate=beta[1],rank=int(rank),n_columns=x.shape[1],condition=float(np.linalg.cond(x)),n=len(x))
    verification=float(np.max(np.abs(beta-np.linalg.pinv(x.T@x)@x.T@y)))
    out['solver_error']=verification
    if rank<x.shape[1]:out['estimate']=np.full(y.shape[1],np.nan)
    intervals=[]
    if bootstrap:
        for width in [800,1600]:
            _,block=np.unique(np.floor((xy-origin)/width).astype(int),axis=0,return_inverse=True)
            nb=block.max()+1;xx=np.zeros((nb,x.shape[1],x.shape[1]));yy=np.zeros((nb,x.shape[1],y.shape[1]))
            assert nb>=2, 'A one-block interval is not estimable'
            for k in range(nb):
                mask=block==k;xx[k]=x[mask].T@x[mask];yy[k]=x[mask].T@y[mask]
            rng=np.random.default_rng(20260920);w=rng.multinomial(nb,np.full(nb,1/nb),size=499)
            lhs=np.einsum('bg,gij->bij',w,xx);rhs=np.einsum('bg,gij->bij',w,yy)
            eig=np.linalg.eigvalsh(lhs);ok=eig[:,0]>np.maximum(eig[:,-1],1)*1e-10
            vals=np.full((499,y.shape[1]),np.nan)
            if ok.any():vals[ok]=np.linalg.solve(lhs[ok],rhs[ok])[:,1,:]
            ci=np.nanquantile(vals,[.025,.975],axis=0) if ok.sum()>=475 else np.full((2,y.shape[1]),np.nan)
            # One explicit repeat-block fit verifies aggregation rather than just
            # solving the same normal equations twice.
            if ok.any():
                k=int(np.flatnonzero(ok)[0]);repeat=np.repeat(np.arange(len(x)),w[k,block])
                direct=np.linalg.lstsq(x[repeat],y[repeat],rcond=None)[0][1]
                assert np.max(np.abs(direct-vals[k]))<1e-7
            intervals.append(dict(width_um=width,n_blocks=int(nb),valid_draws=int(ok.sum()),ci_low=ci[0],ci_high=ci[1]))
    return out,intervals


def main():
    y32,r,loc,genes=load_cohort('10x_janesick');y=y32.astype(float);sr=r.mean(axis=0);ab=np.abs(r).mean(axis=0);del r
    counts=np.rint(np.expm1(y));total=y.sum(axis=1);rawsum=ab.sum(axis=1)
    coverage=pd.read_csv(AUDIT/'stage_05_biology/coverage.csv');coverage=coverage[(coverage.cohort=='10x_janesick')&coverage.eligible]
    marker=pd.read_csv(AUDIT/'stage_07_gene_audit/marker_coverage.csv')
    marker=sorted(marker[(marker.cohort=='10x_janesick')&marker.measured].gene.unique())
    endpoints={}
    for row in coverage.itertuples():
        a=np.load(AUDIT/'stage_05_biology/arrays/10x_janesick'/f'{row.pathway}.npz')
        members=row.genes.split(';')
        endpoints[row.pathway]=dict(members=members,score=a['program_excluded_conditional'],
            observed=a['observed'],signed=a['signed'],absolute=a['absolute'],
            outside_counts=a['outside_total_counts'],outside_detect=a['outside_detected_genes'])
    for gene in marker:
        j=genes.index(gene)
        endpoints[gene]=dict(members=[gene],score=compute_conditional_discordance((rawsum-ab[:,j])/(len(genes)-1),total-y[:,j]),
            observed=y[:,j],signed=sr[:,j],absolute=ab[:,j],outside_counts=counts.sum(axis=1)-counts[:,j],outside_detect=(y>0).sum(axis=1)-(y[:,j]>0))
    rows=[];mixture=[];within=[];support=[];checks=[]
    (OUT/'arrays').mkdir(exist_ok=True)
    for sample in SAMPLES:
        ci=np.flatnonzero(loc.sample_id.eq(sample).to_numpy());ii=loc.iloc[ci].spot_id.to_numpy()
        full=pd.read_parquet(ZONES/sample/'full_spots.parquet');mapped=pd.read_parquet(ZONES/sample/'mapped_cells.parquet')
        a=ad.read_h5ad(CELLS/sample/'measured_cells.h5ad')
        assert np.array_equal(a.obs_names.astype(str),mapped.index.astype(str))
        X=a.X.tocsr().astype(float);LX=X.copy();LX.data=np.log1p(LX.data)
        idx=mapped.full_spot_index.to_numpy(int);qc=mapped.qc_pass.to_numpy()&(idx>=0)
        analysis=mapped.source_group.astype(str).to_numpy().copy()
        analysis[mapped.source_label.astype(str).str.contains('macrophage',case=False).to_numpy()]='Macrophage'
        groupset=sorted(np.unique(analysis))
        nfull=len(full)
        def agg(values,mask=qc):return np.bincount(idx[mask],weights=np.asarray(values)[mask],minlength=nfull)
        def mean(values,mask=qc):
            n=agg(np.ones(len(mapped)),mask);v=agg(values,mask)
            return np.divide(v,n,out=np.full(nfull,np.nan),where=n>0)
        n=agg(np.ones(len(mapped)));frac={g:np.divide(agg(np.ones(len(mapped)),qc&(analysis==g)),n,out=np.zeros(nfull),where=n>0) for g in groupset}
        assert np.max(np.abs(sum(frac.values())[n>0]-1))<1e-12
        full_ids=sample+'_'+full.barcode.astype(str)
        si=pd.Index(full_ids).get_indexer(ii);assert (si>=0).all()
        f=pd.DataFrame({'spot_id':ii,'patient':loc.iloc[ci].patient.to_numpy(),'n_cells':n[si],
            'mean_area':mean(mapped.cell_area.to_numpy(float))[si],
            'x_um':full.native_x.to_numpy()[si],'y_um':full.native_y.to_numpy()[si]})
        for g,v in frac.items():f[g+'_fraction']=v[si]
        prior=pd.read_parquet(ZONES/sample/'spot_zone_evidence.parquet').loc[ii]
        checks.append(dict(check=sample+':prior_stromal_fraction',max_abs=float(np.nanmax(np.abs(f.Stromal_fraction-prior.stromal_fraction.to_numpy()))),passed=True))
        assert checks[-1]['max_abs']<1e-12
        f.to_parquet(OUT/'arrays'/f'{sample}_composition.parquet',index=False)
        section_origin=np.nanmin(f[['x_um','y_um']].to_numpy(),axis=0)
        lo,hi=groups(loc.iloc[ci].conditional.to_numpy());eligible=f.n_cells.to_numpy()>=20
        for kind in ['group','label']:
            labels=analysis if kind=='group' else mapped.source_label.astype(str).to_numpy()
            for label in sorted(np.unique(labels)):
                z=np.divide(agg(np.ones(len(mapped)),qc&(labels==label)),n,out=np.full(nfull,np.nan),where=n>0)[si]
                mask=(lo|hi)&eligible&np.isfinite(z)
                out,ints=fit(design(hi[mask].astype(float),np.empty((mask.sum(),0))),z[mask,None],f[['x_um','y_um']].to_numpy()[mask],section_origin)
                row=dict(sample=sample,patient=f.patient.iloc[0],kind=kind,label=label,n_Q1=int((mask&lo).sum()),n_Q4=int((mask&hi).sum()),
                    Q1_fraction=float(z[mask&lo].mean()),Q4_fraction=float(z[mask&hi].mean()),difference=float(out['estimate'][0]))
                for iv in ints:mixture.append(row|{k:(float(v[0]) if isinstance(v,np.ndarray) else v) for k,v in iv.items()})
        cell_total=np.asarray(X.sum(axis=1)).ravel();cell_detect=np.asarray((X>0).sum(axis=1)).ravel()
        for name,ep in endpoints.items():
            lo,hi=groups(ep['score'][ci]);mask=(lo|hi)&eligible
            tech=np.column_stack([np.log1p(ep['outside_counts'][ci]),np.log1p(ep['outside_detect'][ci]),np.log1p(f.n_cells),np.log1p(f.mean_area)])
            comp=np.column_stack([f[g+'_fraction'] for g in groupset if g!='Tumor'])
            yy=np.stack([ep[k][ci] for k in ['observed','signed','absolute']],axis=1).astype(float)
            mask&=np.isfinite(tech).all(axis=1)&np.isfinite(yy).all(axis=1)
            ident=dict(sample=sample,patient=f.patient.iloc[0],endpoint=name)
            support.append(dict(**ident,analysis='whole_spot',n_Q1=int((mask&lo).sum()),n_Q4=int((mask&hi).sum()),n_cells=int(f.n_cells[mask].sum())))
            if min((mask&lo).sum(),(mask&hi).sum())>=30:
                for adjustment,cov in [('unadjusted',np.empty((len(f),0))),('technical',tech),('composition',np.column_stack([tech,comp]))]:
                    xx=design(hi[mask].astype(float),cov[mask]);out,ints=fit(xx,yy[mask],f[['x_um','y_um']].to_numpy()[mask],section_origin)
                    checks.append(dict(check=f'{sample}/{name}/{adjustment}:solver',max_abs=out['solver_error'],passed=out['solver_error']<1e-7))
                    for k,outcome in enumerate(['observed','signed','absolute']):
                        for iv in ints:rows.append(dict(**ident,adjustment=adjustment,outcome=outcome,estimate=float(out['estimate'][k]),rank=out['rank'],n_columns=out['n_columns'],condition=out['condition'],
                            n_Q1=int((mask&lo).sum()),n_Q4=int((mask&hi).sum()),**{key:(float(v[k]) if isinstance(v,np.ndarray) else v) for key,v in iv.items()}))
            ji=pd.Index(a.var_names).get_indexer(ep['members']);assert (ji>=0).all()
            cell_out=np.asarray(LX[:,ji].mean(axis=1)).ravel()
            cell_det=np.asarray((X[:,ji]>0).mean(axis=1)).ravel()
            outside=cell_total-np.asarray(X[:,ji].sum(axis=1)).ravel();outside_det=cell_detect-np.asarray((X[:,ji]>0).sum(axis=1)).ravel()
            for lineage in ['Tumor','Stromal','Macrophage']:
                lm=qc&(analysis==lineage);nc=agg(np.ones(len(mapped)),lm)[si]
                yl=np.column_stack([mean(cell_out,lm)[si],mean(cell_det,lm)[si]])
                cov=np.column_stack([np.log1p(mean(outside,lm)[si]),np.log1p(mean(outside_det,lm)[si]),np.log1p(mean(mapped.cell_area.to_numpy(float),lm)[si])])
                if lineage=='Tumor':cov=np.column_stack([cov,mean(mapped.source_label.astype(str).str.startswith('DCIS').to_numpy(float),lm)[si]])
                for minimum in [5,20]:
                    use=(lo|hi)&(nc>=minimum)&np.isfinite(yl).all(axis=1)&np.isfinite(cov).all(axis=1)
                    ns=[int((use&lo).sum()),int((use&hi).sum())]
                    support.append(dict(**ident,analysis=lineage,minimum_cells=minimum,n_Q1=ns[0],n_Q4=ns[1],n_cells=int(nc[use].sum())))
                    if min(ns)<30:continue
                    for adjustment,z in [('unadjusted',np.empty((len(f),0))),('within_lineage',cov)]:
                        out,ints=fit(design(hi[use].astype(float),z[use]),yl[use],f[['x_um','y_um']].to_numpy()[use],section_origin,bootstrap=minimum==5)
                        if not ints:ints=[dict(width_um=0,n_blocks=0,valid_draws=0,ci_low=np.full(2,np.nan),ci_high=np.full(2,np.nan))]
                        for k,outcome in enumerate(['mean_cell_log_expression','cell_detection_fraction']):
                            for iv in ints:within.append(dict(**ident,lineage=lineage,minimum_cells=minimum,adjustment=adjustment,outcome=outcome,estimate=float(out['estimate'][k]),
                                Q1_mean=float(yl[use&lo,k].mean()),Q4_mean=float(yl[use&hi,k].mean()),
                                Q1_cell_weighted=float(np.average(yl[use&lo,k],weights=nc[use&lo])),Q4_cell_weighted=float(np.average(yl[use&hi,k],weights=nc[use&hi])),
                                n_Q1=ns[0],n_Q4=ns[1],rank=out['rank'],n_columns=out['n_columns'],**{key:(float(v[k]) if isinstance(v,np.ndarray) else v) for key,v in iv.items()}))
            if name.startswith('HALLMARK_'):print(sample,name,'completed',flush=True)
        # Independently reconstruct a few cell sums using dense selected rows.
        selected=np.flatnonzero(qc)[:101];cols=np.arange(min(5,X.shape[1]))
        dense=X[selected][:,cols].toarray();assert np.allclose(np.log1p(dense).sum(axis=1),np.asarray(LX[selected][:,cols].sum(axis=1)).ravel())
        checks.append(dict(check=sample+':direct_cell_log_aggregation',max_abs=0.,passed=True))
        print(sample,'all endpoints complete',flush=True)
        for fname,data in [('whole_spot_effects',rows),('cell_mixture',mixture),('within_lineage_effects',within),('support',support)]:pd.DataFrame(data).to_csv(OUT/f'{fname}.csv',index=False)
        del a,X,LX,mapped
    assert all(c['passed'] for c in checks)
    (OUT/'checks.json').write_text(json.dumps(dict(status='pass',checks=checks),indent=2)+'\n')
    (OUT/'endpoint_register.json').write_text(json.dumps({k:v['members'] for k,v in endpoints.items()},indent=2)+'\n')


if __name__=='__main__':main()

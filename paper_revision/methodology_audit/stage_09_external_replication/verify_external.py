"""Independent score bins, raw program reconstruction and stratified contrasts."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent;AUDIT=OUT.parent;ROOT=AUDIT.parents[1]
CHECK_PROGRAMS=['HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION','HALLMARK_COMPLEMENT','HALLMARK_E2F_TARGETS']


def centered(raw,expression):
    edges=np.quantile(expression,np.arange(1,10)/10)
    b=np.searchsorted(edges,expression,side='right')
    counts=np.bincount(b,minlength=10)
    # The frozen cohorts have no small occupied bins, so no merge is needed.
    assert counts[counts>0].min()>=30
    sums=np.bincount(b,weights=raw,minlength=10)
    means=np.divide(sums,counts,out=np.zeros(10),where=counts>0)
    return raw-means[b]


def main(family):
    dest=OUT/family; loc=pd.read_parquet(dest/'scores.parquet')
    registry=pd.read_csv(dest/'prediction_registry.csv')
    genes=json.loads((ROOT/'paper_revision/clean_repo/outputs'/family/'gene_panel.json').read_text())
    n=len(loc);g=len(genes);ys=np.empty((n,g),dtype=np.float32)
    ae=np.zeros((n,g));sr=np.zeros((n,g));checks=[]
    def check(label,value,tol=1e-9):
        assert np.isfinite(value) and value<tol,(label,value,tol)
        checks.append(dict(check=label,max_abs=float(value),tolerance=tol,passed=True))
    for row in registry.itertuples():
        fd=Path(row.prediction_source);ids=json.loads((fd/'test_spot_ids.json').read_text())
        if (fd/'metrics.json').exists():
            metric_keys=json.loads((fd/'metrics.json').read_text())
            assert [k for k in metric_keys if not k.startswith('__')]==genes
            meta=json.loads((fd/'calibration.json').read_text())
            assert set(meta['test_samples'])=={row.sample}
            assert set(meta['train_samples'])==set(loc['sample'].unique())-{row.sample}
        else:
            meta=json.loads((fd/'COMPLETE.json').read_text())
            assert set(meta['test_samples'])=={'TENX13','TENX14'}
            assert set(meta['train_samples'])==set(loc['sample'].unique())-set(meta['test_samples'])
            assert meta['n_genes']==len(genes)
        wanted=loc['sample'].eq(row.sample).to_numpy();order=pd.Index(ids).get_indexer(loc.loc[wanted,'spot_id'])
        assert (order>=0).all() and len(set(ids))==len(ids)
        yy=np.load(fd/'test_targets.npy',mmap_mode='r')[order]
        pp=np.load(fd/'test_predictions.npy',mmap_mode='r')[order]
        rr=np.load(fd/'test_residuals.npy',mmap_mode='r')[order]
        check(f'{row.sample}/{row.encoder}:saved_float32_residual',np.max(np.abs(rr-(yy-pp))),1e-20)
        if row.encoder=='uni':ys[wanted]=yy
        else:check(f'{row.sample}/{row.encoder}:target_identity',np.max(np.abs(ys[wanted]-yy)),1e-20)
        err=yy.astype(float)-pp.astype(float)
        ae[wanted]+=np.abs(err)/3;sr[wanted]+=err/3
    y=ys.astype(float);raw=ae.mean(axis=1)
    check('raw_score_from_all_predictions',np.max(np.abs(raw-loc.raw.to_numpy())))
    check('independent_conditional_bins',np.max(np.abs(centered(raw,ys.sum(axis=1))-loc.conditional.to_numpy())))
    gm=pd.read_csv(dest/'gene_metrics.csv');sel=np.linspace(0,g-1,17,dtype=int)
    for group,gg in loc.groupby('specimen_group'):
        train=loc.specimen_group.ne(group).to_numpy();median=np.median(y[train][:,sel],axis=0)
        for sample,ss in gg.groupby('sample'):
            vals=np.abs(y[ss.index][:,sel]-median).mean(axis=0)
            for j,expected in zip(sel,vals):
                saved=gm[gm['sample'].eq(sample)&gm.gene.eq(genes[j])].baseline_mae.to_numpy()
                assert len(saved)==3
                check(f'{sample}/{genes[j]}:independent_training_median',np.max(np.abs(saved-expected)))
    count=np.rint(np.expm1(y));effects=pd.read_csv(dest/'section_program_effects.csv')
    for name in CHECK_PROGRAMS:
        path=dest/'arrays'/f'{name}.npz'
        if not path.exists():continue
        a=np.load(path);members=json.loads((dest/'arrays'/f'{name}_members.json').read_text())
        js=np.array([genes.index(s) for s in members['measured']]);outside=np.ones(g,dtype=bool);outside[js]=False
        # Explicit outside-column means avoid using subtraction from a cached total.
        oc=count[:,outside].sum(axis=1);od=(y[:,outside]>0).sum(axis=1);ox=y[:,outside].sum(axis=1)
        excluded=centered(ae[:,outside].mean(axis=1),ox)
        for k,arr in [('outside_counts',oc),('outside_detected',od),('outside_sum_log_expression',ox),('program_excluded_conditional',excluded)]:
            check(f'{name}/{k}:raw_reconstruction',np.max(np.abs(arr-a[k])),1e-7)
        outcomes={}
        for label,columns in [('full',members['measured'])]+list(members['common'].items()):
            if len(columns)<5:continue
            ji=[genes.index(s) for s in columns]
            for outcome,arr in [('observed',y),('signed',sr),('absolute',ae)]:
                key=outcome if label=='full' else f'common_{label}_{outcome}'
                outcomes[key]=arr[:,ji].mean(axis=1)
                check(f'{name}/{key}:raw_reconstruction',np.max(np.abs(outcomes[key]-a[key])))
        for sample,ss in loc.groupby('sample'):
            ix=ss.index.to_numpy()
            probabilities=np.linspace(0,1,6)[1:-1]
            cc=np.searchsorted(np.quantile(oc[ix],probabilities),oc[ix],side='right')
            dd=np.searchsorted(np.quantile(od[ix],probabilities),od[ix],side='right')
            strata=list(zip(cc,dd))
            for grouping,sc in [('full',loc.conditional.to_numpy()[ix]),('program_excluded',excluded[ix])]:
                for tail in ([.25] if grouping=='full' else [.2,.25,.3]):
                    lo=sc<=np.quantile(sc,tail);hi=sc>=np.quantile(sc,1-tail)
                    weights1=np.zeros(len(ix));weights4=np.zeros(len(ix))
                    for key in set(strata):
                        m=np.array([v==key for v in strata]);n1=int((m&lo).sum());n4=int((m&hi).sum())
                        if min(n1,n4)<10:continue
                        h=2*n1*n4/(n1+n4)
                        weights1[m&lo]=h/n1;weights4[m&hi]=h/n4
                    for outcome,v in outcomes.items():
                        vals=v[ix];sd=np.sqrt(((lo.sum()-1)*np.var(vals[lo],ddof=1)+(hi.sum()-1)*np.var(vals[hi],ddof=1))/(lo.sum()+hi.sum()-2))
                        for adjustment,w1,w4 in [('unadjusted',lo.astype(float),hi.astype(float)),('overlap_adjusted',weights1,weights4)]:
                            expected=np.dot(w4,vals)/w4.sum()-np.dot(w1,vals)/w1.sum()
                            saved=effects[effects['sample'].eq(sample)&effects.pathway.eq(name)&effects.grouping.eq(grouping)&effects['tail'].eq(tail)&effects.outcome.eq(outcome)&effects.adjustment.eq(adjustment)]
                            assert len(saved)==1
                            check(f'{sample}/{name}/{grouping}/{tail}/{outcome}/{adjustment}',abs(expected-saved.effect.iloc[0]))
                            check(f'{sample}/{name}/{grouping}/{tail}/{outcome}/{adjustment}:standardized',abs(expected/sd-saved.standardized_effect.iloc[0]))
    unit=pd.read_csv(dest/'specimen_program_effects.csv')
    keys=['family','specimen_group','pathway','grouping','tail','adjustment','outcome']
    expected=effects.groupby(keys)[['effect','standardized_effect']].mean().sort_index()
    saved=unit.set_index(keys).sort_index();assert expected.index.equals(saved.index)
    check('equal_section_within_specimen',np.max(np.abs(expected.to_numpy()-saved[expected.columns].to_numpy())))
    for row in pd.read_csv(dest/'cohort_program_summary.csv').itertuples():
        v=unit[unit.pathway.eq(row.pathway)&unit.grouping.eq(row.grouping)&unit['tail'].eq(row.tail)&unit.adjustment.eq(row.adjustment)&unit.outcome.eq(row.outcome)]
        if row.scope=='excluding_P07':v=v[v.specimen_group.ne('NCBI776')]
        assert len(v)==row.n_groups
        check(f'{row.scope}/{row.pathway}/{row.grouping}/{row.tail}/{row.adjustment}/{row.outcome}:group_mean',abs(v.standardized_effect.mean()-row.mean))
    (dest/'independent_checks.json').write_text(json.dumps(dict(status='pass',n_checks=len(checks),checks=checks),indent=2)+'\n')
    print(family,'independent checks:',len(checks))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--family',choices=['coad','idc_visium'],required=True);main(p.parse_args().family)

"""Verify raw cell-to-spot expression aggregation and within-lineage OLS."""
import json
from pathlib import Path
import sys
import anndata as ad
import numpy as np
import pandas as pd
import statsmodels.api as sm

OUT=Path(__file__).resolve().parent;AUDIT=OUT.parent;ROOT=AUDIT.parents[1]
sys.path.insert(0,str(AUDIT/'stage_02_patient_calibration'))
from calibration import load_cohort
from src.discordance import compute_conditional_discordance


def main():
    yy,r,loc,genes=load_cohort('10x_janesick');y=yy.astype(float);ae=np.abs(r).mean(axis=0);del r
    register=json.loads((OUT/'endpoint_register.json').read_text());saved=pd.read_csv(OUT/'within_lineage_effects.csv')
    saved=saved[saved.width_um.eq(800)&saved.minimum_cells.eq(5)];checks=[]
    def check(label,value):
        assert np.isfinite(value) and value<1e-8,(label,value)
        checks.append(dict(check=label,max_abs=float(value),passed=True))
    for sample in ['NCBI785','NCBI784','NCBI783']:
        base=ROOT/'paper_revision/experiments/emt_zone_residual_v1/results'/sample
        mapped=pd.read_parquet(base/'mapped_cells.parquet');full=pd.read_parquet(base/'full_spots.parquet')
        a=ad.read_h5ad(ROOT/'paper_revision/experiments/emt_cells_v1/results'/sample/'measured_cells.h5ad')
        assert np.array_equal(mapped.index.astype(str),a.obs_names.astype(str))
        # Dense matrix here deliberately differs from the sparse production path.
        counts=a.X.toarray().astype(float);labels=mapped.source_group.astype(str).copy()
        labels[mapped.source_label.str.contains('macrophage',case=False)]='Macrophage'
        ci=np.flatnonzero(loc.sample_id.eq(sample).to_numpy());ids=sample+'_'+full.barcode.astype(str)
        spot_order=pd.Index(ids).get_indexer(loc.iloc[ci].spot_id);assert (spot_order>=0).all()
        xy=full[['native_x','native_y']].to_numpy()[spot_order]
        origin=np.nanmin(xy,axis=0);assert np.isfinite(origin).all()
        for endpoint in ['EPCAM','CD163','HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION']:
            ji=pd.Index(a.var_names).get_indexer(register[endpoint]);assert (ji>=0).all()
            selected=counts[:,ji];outside=np.ones(counts.shape[1],bool);outside[ji]=False
            dat=pd.DataFrame(dict(full_spot_index=mapped.full_spot_index.to_numpy(),qc=mapped.qc_pass.to_numpy(),
                lineage=labels.to_numpy(),expression=np.log1p(selected).mean(axis=1),detection=(selected>0).mean(axis=1),
                counts=counts[:,outside].sum(axis=1),detected=(counts[:,outside]>0).sum(axis=1),
                area=mapped.cell_area.to_numpy(),dcis=mapped.source_label.str.startswith('DCIS').astype(float).to_numpy()))
            if endpoint.startswith('HALLMARK_'):
                score=np.load(AUDIT/'stage_05_biology/arrays/10x_janesick'/f'{endpoint}.npz')['program_excluded_conditional'][ci]
            else:
                j=genes.index(endpoint);oi=np.arange(len(genes))!=j
                score=compute_conditional_discordance(ae[:,oi].mean(axis=1),y[:,oi].sum(axis=1))[ci]
            lo=score<=np.quantile(score,.25);hi=score>=np.quantile(score,.75)
            for lineage in ['Tumor','Stromal','Macrophage']:
                rows=dat[dat.qc&dat.full_spot_index.ge(0)&dat.lineage.eq(lineage)]
                grouped=rows.groupby('full_spot_index');z=grouped[['expression','detection','counts','detected','area','dcis']].mean().reindex(spot_order)
                nc=grouped.size().reindex(spot_order,fill_value=0).to_numpy()
                cov=np.log1p(z[['counts','detected','area']].to_numpy())
                if lineage=='Tumor':cov=np.column_stack([cov,z.dcis])
                val=z[['expression','detection']].to_numpy();use=(lo|hi)&(nc>=5)&np.isfinite(val).all(axis=1)&np.isfinite(cov).all(axis=1)
                if min((use&lo).sum(),(use&hi).sum())<30:continue
                for adjustment,cv in [('unadjusted',np.empty((len(z),0))),('within_lineage',cov)]:
                    cv=cv[use];cv=cv[:,cv.std(axis=0)>1e-10]
                    if cv.shape[1]:cv=(cv-cv.mean(axis=0))/cv.std(axis=0)
                    design=np.column_stack([np.ones(use.sum()),hi[use],cv])
                    for k,outcome in enumerate(['mean_cell_log_expression','cell_detection_fraction']):
                        ref=saved[saved['sample'].eq(sample)&saved.endpoint.eq(endpoint)&saved.lineage.eq(lineage)&saved.adjustment.eq(adjustment)&saved.outcome.eq(outcome)]
                        assert len(ref)==1
                        ref=ref.iloc[0];prefix=f'{sample}/{endpoint}/{lineage}/{adjustment}/{outcome}'
                        assert (use&lo).sum()==ref.n_Q1 and (use&hi).sum()==ref.n_Q4
                        expected_blocks=len(np.unique(np.floor((xy[use]-origin)/800).astype(int),axis=0))
                        assert expected_blocks==ref.n_blocks and expected_blocks>=2
                        check(prefix+':Q1_raw_cell_mean',abs(val[use&lo,k].mean()-ref.Q1_mean))
                        check(prefix+':Q4_raw_cell_mean',abs(val[use&hi,k].mean()-ref.Q4_mean))
                        check(prefix+':independent_OLS',abs(sm.OLS(val[use,k],design).fit().params[1]-ref.estimate))
    # The origin correction must leave every estimate and support count unchanged.
    for name,key in [('whole_spot_effects','estimate'),('within_lineage_effects','estimate'),('cell_mixture','difference')]:
        before=pd.read_csv(OUT/'before_fixed_block_origin'/f'{name}.csv');after=pd.read_csv(OUT/f'{name}.csv')
        assert len(before)==len(after)
        check(name+':unchanged_estimates',np.max(np.abs(before[key]-after[key])))
        assert np.array_equal(before[['n_Q1','n_Q4']].to_numpy(),after[['n_Q1','n_Q4']].to_numpy())
    (OUT/'lineage_checks.json').write_text(json.dumps(dict(status='pass',n_checks=len(checks),checks=checks),indent=2)+'\n')
    print('Independent raw-cell/lineage checks:',len(checks))


if __name__=='__main__':main()

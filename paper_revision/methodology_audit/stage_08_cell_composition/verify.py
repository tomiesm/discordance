"""Independent pandas cell aggregation and statsmodels whole-spot fits."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

OUT=Path(__file__).resolve().parent;AUDIT=OUT.parent;ROOT=AUDIT.parents[1]
sys.path.insert(0,str(AUDIT/'stage_02_patient_calibration'))
from calibration import load_cohort
from src.discordance import compute_conditional_discordance


def main():
    yy,r,loc,genes=load_cohort('10x_janesick');y=yy.astype(float);sr=r.mean(axis=0);ab=np.abs(r).mean(axis=0);del r
    rawcounts=np.rint(np.expm1(y));saved=pd.read_csv(OUT/'whole_spot_effects.csv');saved=saved[saved.width_um.eq(800)]
    checks=[]
    def check(label,value,tol=1e-9):
        assert np.isfinite(value) and value<tol,(label,value)
        checks.append(dict(check=label,max_abs=float(value),passed=True))
    for sample in ['NCBI785','NCBI784','NCBI783']:
        base=ROOT/'paper_revision/experiments/emt_zone_residual_v1/results'/sample
        cells=pd.read_parquet(base/'mapped_cells.parquet');full=pd.read_parquet(base/'full_spots.parquet')
        cells=cells[cells.qc_pass & cells.full_spot_index.ge(0)].copy()
        cells['analysis_group']=cells.source_group.astype(str)
        cells.loc[cells.source_label.str.contains('macrophage',case=False),'analysis_group']='Macrophage'
        grouped=cells.groupby(['full_spot_index','analysis_group']).size().unstack(fill_value=0).reindex(full.index,fill_value=0)
        denominator=grouped.sum(axis=1);fractions=grouped.div(denominator.replace(0,np.nan),axis=0).fillna(0)
        ids=sample+'_'+full.barcode.astype(str);fractions.index=ids;denominator.index=ids
        f=pd.read_parquet(OUT/'arrays'/f'{sample}_composition.parquet').set_index('spot_id')
        for group in fractions:
            check(f'{sample}/{group}:independent_cell_fraction',np.max(np.abs(f[group+'_fraction']-fractions.loc[f.index,group])))
        check(sample+':independent_n_cells',np.max(np.abs(f.n_cells-denominator.loc[f.index])))
        ci=pd.Index(loc.spot_id).get_indexer(f.index);assert (ci>=0).all()
        for endpoint in ['EPCAM','CD163','MKI67','SNAI1','HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION','HALLMARK_COMPLEMENT']:
            if endpoint.startswith('HALLMARK_'):
                z=np.load(AUDIT/'stage_05_biology/arrays/10x_janesick'/f'{endpoint}.npz')
                score=z['program_excluded_conditional'][ci];vals=np.stack([z[k][ci] for k in ['observed','signed','absolute']],axis=1)
                oc=z['outside_total_counts'][ci];od=z['outside_detected_genes'][ci]
            else:
                j=genes.index(endpoint);oi=np.arange(len(genes))!=j
                score=compute_conditional_discordance(ab[:,oi].mean(axis=1),y[:,oi].sum(axis=1))[ci]
                vals=np.stack([y[ci,j],sr[ci,j],ab[ci,j]],axis=1);oc=rawcounts[ci][:,oi].sum(axis=1);od=(y[ci][:,oi]>0).sum(axis=1)
            lo=score<=np.quantile(score,.25);hi=score>=np.quantile(score,.75)
            tech=np.column_stack([np.log1p(oc),np.log1p(od),np.log1p(f.n_cells),np.log1p(f.mean_area)])
            comp=fractions.loc[f.index,[g for g in sorted(fractions.columns) if g!='Tumor']].to_numpy()
            use=(lo|hi)&f.n_cells.ge(20).to_numpy()&np.isfinite(tech).all(axis=1)&np.isfinite(vals).all(axis=1)
            for adjustment,cov in [('unadjusted',np.empty((len(f),0))),('technical',tech),('composition',np.column_stack([tech,comp]))]:
                cv=cov[use];cv=cv[:,cv.std(axis=0)>1e-10]
                if cv.shape[1]:cv=(cv-cv.mean(axis=0))/cv.std(axis=0)
                xx=np.column_stack([np.ones(use.sum()),hi[use],cv])
                for k,outcome in enumerate(['observed','signed','absolute']):
                    ref=saved[(saved['sample']==sample)&(saved.endpoint==endpoint)&(saved.adjustment==adjustment)&(saved.outcome==outcome)].iloc[0]
                    val=sm.OLS(vals[use,k],xx).fit().params[1]
                    check(f'{sample}/{endpoint}/{adjustment}/{outcome}:statsmodels',abs(val-ref.estimate),1e-8)
    (OUT/'independent_checks.json').write_text(json.dumps(dict(status='pass',n_checks=len(checks),checks=checks),indent=2)+'\n')
    print('Independent composition checks:',len(checks))


if __name__=='__main__':main()

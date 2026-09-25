"""Direct EMT residual contrast in source-tumor-rich fixed quartile subsets."""
import sys
from common import *
sys.path.insert(0,str(AUDIT/'stage_08_cell_composition'))
from composition import design,fit

def main():
    track(AUDIT/'stage_08_cell_composition/composition.py')
    loc=csv(AUDIT/'stage_05_biology/arrays/10x_janesick/locations.csv')
    path=AUDIT/'stage_05_biology/arrays/10x_janesick'/f'{EMT}.npz'
    with np.load(track(path)) as data:a={k:data[k] for k in data.files}
    rows=[];support=[];arrays={}
    for sample in ['NCBI785','NCBI784','NCBI783']:
        ix=np.flatnonzero(loc.sample_id.eq(sample));f=parquet(AUDIT/'stage_08_cell_composition/arrays'/f'{sample}_composition.parquet').set_index('spot_id').loc[loc.iloc[ix].spot_id].reset_index()
        lo,hi=tails(a['program_excluded_conditional'][ix]);v=np.stack([a[o][ix] for o in OUTCOMES],axis=1).astype(float)
        tech=np.column_stack([np.log1p(a['outside_total_counts'][ix]),np.log1p(a['outside_detected_genes'][ix]),np.log1p(f.n_cells),np.log1p(f.mean_area)]);fractions=[c for c in f.columns if c.endswith('_fraction') and c!='Tumor_fraction'];comp=f[fractions].to_numpy();xy=f[['x_um','y_um']].to_numpy();origin=np.nanmin(xy,axis=0)
        base=(lo|hi)&f.n_cells.ge(20).to_numpy()&np.isfinite(tech).all(axis=1)&np.isfinite(v).all(axis=1)&np.isfinite(xy).all(axis=1)&np.isfinite(comp).all(axis=1)
        for minimum in [0.,.5,.75]:
            mask=base&f.Tumor_fraction.ge(minimum).to_numpy();n1=int((mask&lo).sum());n4=int((mask&hi).sum());ident=dict(sample=sample,patient=f.patient.iloc[0],minimum_tumor_fraction=minimum,n_Q1=n1,n_Q4=n4)
            support.append(ident|dict(estimable=min(n1,n4)>=30,Q1_tumor_mean=f.loc[mask&lo,'Tumor_fraction'].mean(),Q4_tumor_mean=f.loc[mask&hi,'Tumor_fraction'].mean(),Q1_stromal_mean=f.loc[mask&lo,'Stromal_fraction'].mean(),Q4_stromal_mean=f.loc[mask&hi,'Stromal_fraction'].mean()))
            if min(n1,n4)<30:continue
            for adjustment,cov in [('unadjusted',np.empty((len(f),0))),('technical',tech),('composition',np.column_stack([tech,comp]))]:
                x=design(hi[mask].astype(float),cov[mask]);out,ints=fit(x,v[mask],xy[mask],origin)
                # Independent Frisch–Waugh–Lovell coefficient via residualizing exposure/outcomes.
                nuisance=np.delete(x,1,axis=1);e=x[:,1]-nuisance@np.linalg.lstsq(nuisance,x[:,1],rcond=None)[0];yy=v[mask]-nuisance@np.linalg.lstsq(nuisance,v[mask],rcond=None)[0]
                direct=(e@yy)/(e@e);check(f'{sample}/{minimum}/{adjustment}:FWL',np.nanmax(abs(direct-out['estimate'])),1e-8)
                arrays[f'{sample}__{minimum}__{adjustment}__x']=x;arrays[f'{sample}__{minimum}__{adjustment}__y']=v[mask];arrays[f'{sample}__{minimum}__{adjustment}__xy']=xy[mask]
                for k,outcome in enumerate(OUTCOMES):
                    for interval in ints:rows.append(ident|dict(adjustment=adjustment,outcome=outcome,estimate=float(out['estimate'][k]),rank=out['rank'],n_columns=out['n_columns'],condition=out['condition'],Q1_mean=float(v[mask&lo,k].mean()),Q4_mean=float(v[mask&hi,k].mean()),**{key:(float(value[k]) if isinstance(value,np.ndarray) else value) for key,value in interval.items()}))
        print(sample,'tumor-rich EMT complete',flush=True)
    d=pd.DataFrame(rows);reference=csv(AUDIT/'stage_08_cell_composition/whole_spot_effects.csv');reference=reference[reference.endpoint.eq(EMT)]
    merged=d[d.minimum_tumor_fraction.eq(0)].merge(reference,on=['sample','patient','adjustment','outcome','width_um'],suffixes=('_new','_old'),validate='one_to_one');assert len(merged)==54
    for col in ['estimate','ci_low','ci_high','n_Q1','n_Q4','n_blocks','valid_draws']:check('unrestricted_reproduction:'+col,np.nanmax(abs(merged[col+'_new']-merged[col+'_old'])),1e-8)
    d.to_csv(OUT/'tumor_rich_emt_effects.csv',index=False);pd.DataFrame(support).to_csv(OUT/'tumor_rich_emt_support.csv',index=False);np.savez_compressed(OUT/'tumor_rich_emt_designs.npz',**arrays);finish('tumor')

if __name__=='__main__':main()

"""Separate score/bin/weighted-effect checks and archived contrast correspondence."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent;AUDIT=OUT.parent
sys.path.insert(0,str(AUDIT/'stage_02_patient_calibration'))
from calibration import load_cohort


def main():
    saved=pd.read_csv(OUT/'section_effects.csv').set_index(['cohort','sample','gene','grouping','adjustment','outcome'])
    checks=[]
    def check(label,err,tol=1e-9):
        assert np.isfinite(err) and err<tol,(label,err,tol)
        checks.append(dict(check=label,max_abs=float(err),passed=True))
    old=pd.read_csv(AUDIT/'stage_05_biology/gene_characterization.csv')
    for row in old.itertuples():
        for outcome,col in [('observed_log','observed_delta'),('signed','signed_delta'),('absolute','absolute_delta')]:
            new=saved.loc[(row.cohort,row.sample,row.gene,'full','unadjusted',outcome)]
            check(f'{row.sample}/{row.gene}/{outcome}:archived',abs(new.effect-getattr(row,col)),2e-6)
    for cohort in ['biomarkers','10x_janesick']:
        yy,r,loc,genes=load_cohort(cohort);y=yy.astype(float)
        sr=r.mean(axis=0);ae=np.abs(r).mean(axis=0);del r
        counts=np.rint(np.expm1(y))
        selected=sorted(set(np.linspace(0,len(genes)-1,10,dtype=int))|{genes.index(g) for g in ['EPCAM','CD163','MKI67','SNAI1','ZEB1','VIM','HSPG2'] if g in genes})
        for j in selected:
            oi=np.array([k for k in range(len(genes)) if k!=j]);raw=ae[:,oi].mean(axis=1);total=y[:,oi].sum(axis=1)
            bins=np.digitize(total,np.percentile(total,np.arange(10,100,10)))
            mean=np.bincount(bins,weights=raw)/np.bincount(bins);score=raw-mean[bins]
            for sample,g in loc.groupby('sample_id',sort=False):
                ix=g.index.to_numpy();v=score[ix];full=g.conditional.to_numpy()
                oc=counts[ix][:,oi].sum(axis=1);od=(y[ix][:,oi]>0).sum(axis=1)
                grid=np.linspace(0,1,6)[1:-1];strata=5*np.digitize(oc,np.quantile(oc,grid))+np.digitize(od,np.quantile(od,grid))
                vals=np.stack([y[ix,j],counts[ix,j],(y[ix,j]>0).astype(float),y[ix,j]-sr[ix,j],sr[ix,j],ae[ix,j]],axis=1)
                for grouping,s in [('full',full),('gene_excluded',v)]:
                    lo=s<=np.percentile(s,25);hi=s>=np.percentile(s,75)
                    d=vals[hi].mean(axis=0)-vals[lo].mean(axis=0)
                    weights=[];ds=[]
                    for st in np.unique(strata):
                        a=lo&(strata==st);b=hi&(strata==st);na=a.sum();nb=b.sum()
                        if min(na,nb)<10:continue
                        weights.append(2*na*nb/(na+nb));ds.append(vals[b].mean(axis=0)-vals[a].mean(axis=0))
                    adj=np.average(ds,weights=weights,axis=0) if weights else np.full(6,np.nan)
                    for adjustment,effects in [('unadjusted',d),('overlap_adjusted',adj)]:
                        for k,name in enumerate(['observed_log','observed_counts','detected','predicted_log','signed','absolute']):
                            ref=saved.loc[(cohort,sample,genes[j],grouping,adjustment,name)]
                            if np.isnan(effects[k]):assert np.isnan(ref.effect)
                            else:check(f'{sample}/{genes[j]}/{grouping}/{adjustment}/{name}:direct',abs(ref.effect-effects[k]),1e-8)
        print(cohort,'independent reconstruction completed',flush=True)
    patient=pd.read_csv(OUT/'patient_effects.csv').set_index(['cohort','patient','gene','grouping','adjustment','outcome'])
    direct=saved.reset_index().groupby(['cohort','patient','gene','grouping','adjustment','outcome']).effect.mean()
    check('equal_section_patient_averaging',np.nanmax(np.abs(direct-patient.effect)))
    (OUT/'independent_checks.json').write_text(json.dumps(dict(status='pass',n_checks=len(checks),checks=checks),indent=2)+'\n')
    print('Independent checks:',len(checks),flush=True)


if __name__=='__main__':main()

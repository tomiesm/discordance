"""Independent raw expression/residual verification of common-member IDC effects."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent;AUDIT=OUT.parent
sys.path.insert(0,str(AUDIT/'stage_02_patient_calibration'))
from calibration import load_cohort


def main():
    coverage=pd.read_csv(OUT/'common_member_coverage.csv');effects=pd.read_csv(OUT/'idc_common_member_section_effects.csv');checks=[]
    def check(name,value):
        assert np.isfinite(value) and value<1e-8,(name,value)
        checks.append(dict(check=name,max_abs=float(value),passed=True))
    for cohort in ['biomarkers','10x_janesick']:
        y32,r,loc,genes=load_cohort(cohort);y=y32.astype(float);sr=r.mean(axis=0);ae=np.abs(r).mean(axis=0);del r
        stored=pd.read_csv(AUDIT/'stage_05_biology/arrays'/cohort/'locations.csv')
        assert np.array_equal(loc.spot_id,stored.spot_id)
        sub=coverage[coverage.idc_cohort.eq(cohort)&coverage.eligible]
        sub=sub[sub.family.eq('coad')|sub.pathway.isin(['HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION','HALLMARK_COMPLEMENT','HALLMARK_E2F_TARGETS'])]
        for row in sub.itertuples():
            js=[genes.index(g) for g in row.genes.split(';')]
            values=np.stack([a[:,js].mean(axis=1) for a in [y,sr,ae]],axis=1)
            a=np.load(AUDIT/'stage_05_biology/arrays'/cohort/f'{row.pathway}.npz')
            for sample,ss in loc.groupby('sample_id'):
                ix=ss.index.to_numpy();score=a['program_excluded_conditional'][ix]
                lo=score<=np.quantile(score,.25);hi=score>=np.quantile(score,.75)
                count=a['outside_total_counts'][ix];detect=a['outside_detected_genes'][ix]
                # Use the frozen floating-point quantile probabilities. Decimal
                # .6 and linspace's 0.6000000000000001 can split an exact tie.
                probabilities=np.linspace(0,1,6)[1:-1]
                strata=pd.DataFrame(dict(c=np.searchsorted(np.quantile(count,probabilities),count,side='right'),
                    d=np.searchsorted(np.quantile(detect,probabilities),detect,side='right'),lo=lo,hi=hi))
                w1=np.zeros(len(ix));w4=np.zeros(len(ix))
                for _,group in strata.groupby(['c','d']):
                    n1=group.lo.sum();n4=group.hi.sum()
                    if min(n1,n4)<10:continue
                    mass=2*n1*n4/(n1+n4);w1[group.index[group.lo]]=mass/n1;w4[group.index[group.hi]]=mass/n4
                vals=values[ix];sd=np.sqrt(((lo.sum()-1)*vals[lo].var(axis=0,ddof=1)+(hi.sum()-1)*vals[hi].var(axis=0,ddof=1))/(lo.sum()+hi.sum()-2))
                for adjustment,p,q in [('unadjusted',lo.astype(float),hi.astype(float)),('overlap_adjusted',w1,w4)]:
                    diff=(q@vals)/q.sum()-(p@vals)/p.sum()
                    for k,outcome in enumerate(['observed','signed','absolute']):
                        saved=effects[effects.family.eq(row.family)&effects.idc_cohort.eq(cohort)&effects['sample'].eq(sample)&effects.pathway.eq(row.pathway)&effects.adjustment.eq(adjustment)&effects.outcome.eq(outcome)]
                        assert len(saved)==1
                        label=f'{row.family}/{cohort}/{sample}/{row.pathway}/{adjustment}/{outcome}'
                        check(label,abs(diff[k]-saved.effect.iloc[0]));check(label+':standardized',abs(diff[k]/sd[k]-saved.standardized_effect.iloc[0]))
    keys=['family','idc_cohort','patient','pathway','outcome','adjustment'];cols=['effect','standardized_effect']
    expected=effects.groupby(keys)[cols].mean().sort_index();actual=pd.read_csv(OUT/'idc_common_member_patient_effects.csv').set_index(keys).sort_index()
    assert actual.index.equals(expected.index);check('equal_section_patient_summaries',np.max(np.abs(expected.to_numpy()-actual[cols].to_numpy())))
    keys.remove('patient');expected=actual.groupby(keys)[cols].mean().sort_index();actual=pd.read_csv(OUT/'idc_common_member_cohort_effects.csv').set_index(keys).sort_index()
    assert actual.index.equals(expected.index);check('equal_patient_cohort_summaries',np.max(np.abs(expected.to_numpy()-actual[cols].to_numpy())))
    (OUT/'idc_comparator_independent_checks.json').write_text(json.dumps(dict(status='pass',n_checks=len(checks),checks=checks),indent=2)+'\n')
    print('Independent common-member IDC checks:',len(checks))


if __name__=='__main__':main()

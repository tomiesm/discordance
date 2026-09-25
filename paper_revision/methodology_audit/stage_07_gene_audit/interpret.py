"""Predictability links, all-shared-gene comparisons and named-marker register."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata,spearmanr

OUT=Path(__file__).resolve().parent


def correlation(x,y):
    keep=np.isfinite(x)&np.isfinite(y);x=np.asarray(x)[keep];y=np.asarray(y)[keep]
    return float(spearmanr(x,y).statistic) if len(x)>3 and np.std(x)>0 and np.std(y)>0 else np.nan


def main():
    pred=pd.read_csv(OUT/'prediction_quality.csv')
    # Correct an output-label typo from the initial writer; the calculation is
    # Spearman and was never used for any Pearson-based inference.
    if 'ensemble_quartile_pearson' in pred:
        pred=pred.rename(columns={'ensemble_quartile_pearson':'ensemble_quartile_spearman'})
        pred.to_csv(OUT/'prediction_quality.csv',index=False)
    p=pred.groupby(['cohort','patient','gene','grouping','quartile']).mean(numeric_only=True).reset_index()
    p.to_csv(OUT/'patient_prediction_quality.csv',index=False)
    effects=pd.read_csv(OUT/'patient_effects.csv')
    rows=[]
    for (cohort,patient),g in p[(p.grouping=='gene_excluded')&(p.quartile=='Q1')].groupby(['cohort','patient']):
        for adjustment in ['unadjusted','overlap_adjusted']:
            for outcome in ['observed_log','signed','absolute']:
                e=effects[(effects.cohort==cohort)&(effects.patient==patient)&(effects.grouping=='gene_excluded')&(effects.adjustment==adjustment)&(effects.outcome==outcome)]
                a=g.merge(e[['gene','effect','standardized_effect']],on='gene')
                for name,y in [('signed_contrast',a.standardized_effect.to_numpy()),('absolute_contrast',a.standardized_effect.abs().to_numpy())]:
                    x=a.whole_section_gene_pearson.to_numpy()
                    z=a[['whole_section_mean_log','whole_section_detection','whole_section_sd']].to_numpy()
                    keep=np.isfinite(x)&np.isfinite(y)&np.isfinite(z).all(axis=1)
                    xx=rankdata(x[keep]);yy=rankdata(y[keep]);zz=np.stack([np.ones(keep.sum())]+[rankdata(z[keep,k]) for k in range(3)],axis=1)
                    rx=xx-zz@np.linalg.lstsq(zz,xx,rcond=None)[0];ry=yy-zz@np.linalg.lstsq(zz,yy,rcond=None)[0]
                    partial=float(np.corrcoef(rx,ry)[0,1]) if np.std(rx)>0 and np.std(ry)>0 else np.nan
                    rows.append(dict(cohort=cohort,patient=patient,adjustment=adjustment,outcome=outcome,contrast=name,
                        n_genes=int(keep.sum()),spearman=correlation(x,y),partial_rank_correlation=partial))
    pd.DataFrame(rows).to_csv(OUT/'predictability_effect_relationship.csv',index=False)
    cohort=pd.read_csv(OUT/'cohort_effects.csv')
    bridge=[]
    for key,g in cohort.groupby(['grouping','adjustment','outcome','metric']):
        a=g[g.cohort.eq('biomarkers')].set_index('gene');b=g[g.cohort.eq('10x_janesick')].set_index('gene')
        for gene in sorted(set(a.index)&set(b.index)):
            x=a.loc[gene];y=b.loc[gene]
            bridge.append(dict(zip(['grouping','adjustment','outcome','metric'],key))|dict(gene=gene,discovery_mean=x['mean'],validation_mean=y['mean'],
                direction_agreement=bool(x['mean']*y['mean']>0),discovery_positive=int(x.n_positive),validation_positive=int(y.n_positive),
                discovery_negative=int(x.n_negative),validation_negative=int(y.n_negative),
                discovery_ci_low=x.ci_low,discovery_ci_high=x.ci_high,validation_ci_low=y.ci_low,validation_ci_high=y.ci_high))
    bridge=pd.DataFrame(bridge);bridge.to_csv(OUT/'all_bridge_gene_comparisons.csv',index=False)
    summary=[]
    for key,g in bridge.groupby(['grouping','adjustment','outcome','metric']):
        consistent=((g.discovery_positive>=3)&(g.validation_positive>=3))|((g.discovery_negative>=3)&(g.validation_negative>=3))
        summary.append(dict(zip(['grouping','adjustment','outcome','metric'],key))|dict(n_shared=len(g),n_same_direction=int(g.direction_agreement.sum()),n_at_least_3_of_4_same_direction_each=int(consistent.sum()),spearman=correlation(g.discovery_mean,g.validation_mean)))
    pd.DataFrame(summary).to_csv(OUT/'bridge_summary.csv',index=False)
    cov=pd.read_csv(OUT/'marker_coverage.csv')
    named=cov[cov.measured].merge(cohort[(cohort.grouping=='gene_excluded')&(cohort.metric=='standardized_effect')],on=['cohort','gene'],how='left')
    named.to_csv(OUT/'named_marker_evidence.csv',index=False)
    # Full versus excluded on the same adjustment/measurement scale.
    a=cohort[cohort.grouping.eq('full')];b=cohort[cohort.grouping.eq('gene_excluded')]
    keys=['cohort','gene','adjustment','outcome','metric']
    attenuation=a.merge(b,on=keys,suffixes=('_full','_excluded'))
    attenuation['effect_change']=attenuation.mean_excluded-attenuation.mean_full
    attenuation['absolute_effect_reduction_fraction']=np.where(attenuation.mean_full.abs()>1e-12,1-attenuation.mean_excluded.abs()/attenuation.mean_full.abs(),np.nan)
    attenuation['sign_changed']=attenuation.mean_full*attenuation.mean_excluded<0
    attenuation.to_csv(OUT/'gene_exclusion_attenuation.csv',index=False)
    # Ranking is by mean Q1 baseline gain, not by a preferred biological result.
    good=p[(p.grouping=='full')&(p.quartile=='Q1')].groupby(['cohort','gene']).agg(mean_gain=('relative_mae_gain','mean'),
        n_patients_better=('relative_mae_gain',lambda v:int((v>0).sum())),mean_mae=('mae','mean'),mean_baseline_mae=('baseline_mae','mean'),
        mean_correlation=('whole_section_gene_pearson','mean')).reset_index()
    good.sort_values(['cohort','mean_gain'],ascending=[True,False]).to_csv(OUT/'Q1_predictable_genes.csv',index=False)
    print(pd.DataFrame(summary).query("grouping=='gene_excluded' and adjustment=='overlap_adjusted' and metric=='standardized_effect'").to_string(index=False))


if __name__=='__main__':main()

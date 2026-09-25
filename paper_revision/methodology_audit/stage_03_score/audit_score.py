"""Descriptive checks of the current magnitude score, without biological tests."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import yaml

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
REPO = ROOT / 'paper_revision/clean_repo'
sys.path.insert(0, str(REPO))
from src.discordance import compute_conditional_discordance


def rho(x, y):
    return float(spearmanr(x, y).statistic) if np.std(x)>0 and np.std(y)>0 else np.nan


def groups(x, tail=.25):
    lo,hi = np.quantile(x,[tail,1-tail])
    return x<=lo, x>=hi, float(lo),float(hi)


def compare(x,y):
    a,b,_,_ = groups(x); c,d,_,_ = groups(y)
    return {'spearman':rho(x,y),'Q1_overlap_fraction':float(np.sum(a&c)/np.sum(a)),
            'Q4_overlap_fraction':float(np.sum(b&d)/np.sum(b)),
            'Q1_jaccard':float(np.sum(a&c)/np.sum(a|c)),
            'Q4_jaccard':float(np.sum(b&d)/np.sum(b|d))}


def independent_center(raw, total):
    edges = np.percentile(total,np.linspace(0,100,11))
    labels = np.searchsorted(edges[1:-1],total,side='right')
    assert np.bincount(labels,minlength=10).min() >= 30
    means = np.array([raw[labels==j].mean() for j in range(10)])
    return raw.astype(float)-means[labels], labels, edges, means


def main():
    cfg = yaml.safe_load((REPO / 'config.yaml').read_text())
    base = pd.read_parquet(OUT.parent/'stage_02_prediction/spot_prediction_diagnostics.parquet').set_index('spot_id')
    coverage = pd.read_parquet(OUT.parent/'stage_01_provenance/spot_coverage.parquet')
    coverage['spot_id'] = coverage['sample']+'_'+coverage['barcode']
    coverage=coverage.set_index('spot_id')
    correlations, comparisons, depths, binrows, cutoffs, contrasts, genelist, concentrations, allspots, checks = ([] for _ in range(10))
    covariates=['total_expr','panel_counts','noncontrol_counts','panel_genes_detected','tissue_fraction']
    for cc in cfg['cohorts'].values():
        cohort=cc['name']
        s = pd.concat([pd.read_parquet(REPO/'outputs/phase2/scores'/cohort/f'{sid}_discordance.parquet') for sid in cc['samples']],ignore_index=True)
        s=s.set_index('spot_id')
        assert s.index.is_unique
        s['patient']=base.loc[s.index,'patient']
        s['cohort']=cohort
        for c in covariates[1:]: s[c]=coverage.loc[s.index,c]
        s['baseline_raw']=base.loc[s.index,'baseline_median_mae']
        rcols=[f"D_raw_{e['name']}_ridge" for e in cfg['encoders']]
        ccols=[f"D_cond_{e['name']}_ridge" for e in cfg['encoders']]
        s['raw']=s[rcols].mean(axis=1);s['conditional']=s[ccols].mean(axis=1)
        s['baseline_conditional']=compute_conditional_discordance(s.baseline_raw.to_numpy(),s.total_expr.to_numpy())
        for enc in cfg['encoders']:
            raw=s[f"D_raw_{enc['name']}_ridge"].to_numpy()
            got,labels,edges,means=independent_center(raw,s.total_expr.to_numpy())
            saved=s[f"D_cond_{enc['name']}_ridge"].to_numpy()
            err=float(np.max(np.abs(got-saved)))
            ref=compute_conditional_discordance(raw,s.total_expr.to_numpy())
            checks.append({'check':f'{cohort}/{enc["name"]}:conditional_reconstruction','max_abs':err,'pass':err<2e-6 and np.allclose(got,ref,atol=1e-12,rtol=0)})
            assert checks[-1]['pass']
        s['pooled_expression_bin']=labels
        for b,g in s.groupby('pooled_expression_bin'):
            for (sid,patient),h in g.groupby(['sample_id','patient']):
                binrows.append({'cohort':cohort,'bin':b,'lower_edge':edges[b],'upper_edge':edges[b+1],
                    'sample':sid,'patient':patient,'n':len(h),'fraction_of_pooled_bin':len(h)/len(g),
                    'mean_raw':h.raw.mean(),'mean_conditional':h.conditional.mean(),
                    'sd_conditional':h.conditional.std(),'mean_baseline_conditional':h.baseline_conditional.mean()})
        genes=json.loads((REPO/'data/v3'/f'gene_list_{cohort}.json').read_text())
        for fold in range(4):
            fd=REPO/'outputs/predictions'/cohort/'uni/ridge'/f'fold{fold}'
            ids=json.loads((fd/'test_spot_ids.json').read_text())
            errors=[]
            for ec in cfg['encoders']:
                r=np.load(REPO/'outputs/predictions'/cohort/ec['name']/'ridge'/f'fold{fold}'/'test_residuals.npy')
                errors.append(np.abs(r).astype(float))
            error=np.mean(errors,axis=0);del errors
            f=s.loc[ids]
            for sid,g in f.groupby('sample_id',sort=False):
                m=f.sample_id.to_numpy()==sid; e=error[m]
                q1,q4,_,_=groups(g.conditional.to_numpy())
                mean=e.mean(axis=0); delta=e[q4].mean(axis=0)-e[q1].mean(axis=0)
                share=mean/mean.sum();positive=np.maximum(delta,0);share_delta=positive/positive.sum()
                concentrations.append({'cohort':cohort,'sample':sid,'patient':g.patient.iloc[0],
                    'n_genes':len(genes),'largest_gene_error_share':share.max(),'top10_gene_error_share':np.sort(share)[-10:].sum(),
                    'effective_n_error_genes':1/np.square(share).sum(),
                    'top10_positive_Q4_Q1_error_difference_share':np.sort(share_delta)[-10:].sum(),
                    'n_genes_with_positive_error_difference':int((delta>0).sum())})
                for j,gene in enumerate(genes):
                    genelist.append({'cohort':cohort,'sample':sid,'patient':g.patient.iloc[0],'gene':gene,
                        'mean_abs_error':mean[j],'mean_error_share':share[j],
                        'Q4_Q1_abs_error_difference':delta[j],
                        'positive_difference_share':share_delta[j]})
            del error
        for sid,g in s.groupby('sample_id',sort=False):
            g=g.copy();identity={'cohort':cohort,'sample':sid,'patient':g.patient.iloc[0]}
            g['section_centered_diagnostic']=compute_conditional_discordance(g.raw.to_numpy(),g.total_expr.to_numpy())
            for metric in ['raw','conditional','baseline_raw','baseline_conditional','section_centered_diagnostic']:
                for c in covariates:
                    correlations.append({**identity,'score':metric,'covariate':c,'spearman':rho(g[metric],g[c])})
            for a,b in [('conditional','raw'),('conditional','baseline_conditional'),('raw','baseline_raw'),('conditional','section_centered_diagnostic')]:
                comparisons.append({**identity,'score_a':a,'score_b':b,**compare(g[a].to_numpy(),g[b].to_numpy())})
            q1,q4,_,_=groups(g.conditional.to_numpy())
            for c in covariates:
                a=g[c].to_numpy();pooled=np.sqrt((np.var(a[q1],ddof=1)+np.var(a[q4],ddof=1))/2)
                contrasts.append({**identity,'covariate':c,'Q1_mean':np.mean(a[q1]),'Q4_mean':np.mean(a[q4]),
                    'Q1_median':np.median(a[q1]),'Q4_median':np.median(a[q4]),
                    'Q4_Q1_standardized_difference':(np.mean(a[q4])-np.mean(a[q1]))/pooled if pooled else np.nan})
            for tail in [.2,.25,.3]:
                for metric in ['raw','conditional','baseline_conditional','section_centered_diagnostic']:
                    a,b,lo,hi=groups(g[metric].to_numpy(),tail)
                    cutoffs.append({**identity,'score':metric,'tail_fraction':tail,'lower_threshold':lo,'upper_threshold':hi,
                        'n_lower':int(a.sum()),'n_upper':int(b.sum()),'ties_lower':int((g[metric]==lo).sum()),
                        'ties_upper':int((g[metric]==hi).sum()),'disjoint':not bool(np.any(a&b)),
                        'primary_Q1_recovered_fraction':float(np.sum(a&q1)/np.sum(q1)),
                        'primary_Q4_recovered_fraction':float(np.sum(b&q4)/np.sum(q4))})
                    assert not np.any(a&b)
            dec=np.searchsorted(np.quantile(g.total_expr,np.linspace(0,1,11))[1:-1],g.total_expr,side='right')
            g['section_expression_bin']=dec
            g['Q4']=q4
            for b,h in g.groupby('section_expression_bin'):
                depths.append({**identity,'section_bin':b,'n':len(h),'expression_mean':h.total_expr.mean(),
                    'raw_mean':h.raw.mean(),'raw_sd':h.raw.std(),'conditional_mean':h.conditional.mean(),
                    'conditional_sd':h.conditional.std(),'Q4_fraction':h.Q4.mean(),
                    'panel_counts_median':h.panel_counts.median(),'detected_genes_mean':h.panel_genes_detected.mean()})
            allspots.append(g.reset_index()[['spot_id','sample_id','patient','cohort','raw','conditional','baseline_raw',
                'baseline_conditional','section_centered_diagnostic','total_expr','panel_counts','noncontrol_counts',
                'panel_genes_detected','tissue_fraction','pooled_expression_bin','section_expression_bin']])
        print(cohort,'completed',len(s),'spots',flush=True)
    for name,values in [('score_covariate_correlations',correlations),('score_comparisons',comparisons),
        ('section_depth_profiles',depths),('pooled_bin_composition',binrows),('cutoffs_and_tail_sensitivities',cutoffs),
        ('quartile_covariate_contrasts',contrasts),('gene_error_contributions',genelist),('gene_concentration',concentrations)]:
        pd.DataFrame(values).to_csv(OUT/f'{name}.csv',index=False)
    pd.concat(allspots).to_parquet(OUT/'spot_score_diagnostics.parquet',index=False)
    (OUT/'checks.json').write_text(json.dumps({'status':'pass' if all(c['pass'] for c in checks) else 'failed','checks':checks},indent=2)+'\n')


if __name__=='__main__':
    main()

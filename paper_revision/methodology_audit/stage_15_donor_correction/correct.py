"""Propagate primary-source donor identities without changing fixed predictions."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd
from scipy.stats import t, spearmanr, pearsonr
import yaml

OUT=Path(__file__).resolve().parent; A=OUT.parent; R=A.parent/'clean_repo'
T=OUT/'tables'; T.mkdir(exist_ok=True)
SOURCES={}; CHECKS=[]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def track(p):
    p=Path(p); SOURCES[str(p.resolve())]=sha(p); return p
def read(p): return pd.read_csv(track(A/p))
def save(d,n): d.to_csv(T/(n+'.csv'),index=False); return d
def check(name,passed):
    CHECKS.append(dict(name=name,passed=bool(passed))); assert passed,name

clinical=read('clinical_metadata_check_10/discovery_clinical_metadata.csv').sort_values('sample')
donors=dict(zip(clinical['sample'],[f'D{i:02}' for i in range(1,12)]))
cfg=yaml.safe_load(track(R/'config.yaml').read_text())
for p,ss in cfg['cohorts']['validation']['patient_mapping'].items():
    donors.update({s:p for s in ss})
def relabel(d):
    d=d.copy()
    sample='sample' if 'sample' in d else 'sample_id'
    for col in ['patient','unit']:
        if col in d:
            d['inherited_'+col]=d[col]
            d[col]=d[sample].map(donors).fillna(d[col])
    d['donor']=d[sample].map(donors)
    return d
def summaries(d,keys,values,unit='patient'):
    units=d.groupby(keys+[unit],dropna=False)[values].mean().reset_index()
    rows=[]
    for key,g in units.groupby(keys,dropna=False):
        if not isinstance(key,tuple):key=(key,)
        ident=dict(zip(keys,key))
        for metric in values:
            x=g[metric].dropna().to_numpy();n=len(x);mean=x.mean() if n else np.nan
            half=t.ppf(.975,n-1)*x.std(ddof=1)/np.sqrt(n) if n>1 else np.nan
            rows.append(ident|dict(metric=metric,mean=mean,n_units=n,n_positive=int((x>0).sum()),n_negative=int((x<0).sum()),ci_low=mean-half,ci_high=mean+half,minimum=x.min() if n else np.nan,maximum=x.max() if n else np.nan))
    return units,pd.DataFrame(rows)

def registry():
    s=relabel(read('stage_01_inventory/section_inventory.csv'))
    s=s.merge(clinical[['sample','source_specimen','source_disease','source_TNM','source_grade','source_receptors']],on='sample',how='left')
    s['holdout_group']=s.inherited_patient.replace({'P03':'B01','P04':'B02','P05':'B03','P06':'B04'})
    s['repeat_section_available']=s.patient.map(s.patient.value_counts()).gt(1)
    save(s,'specimen_registry')
    rows=[]
    for cohort in ['biomarkers','10x_janesick']:
        seen=[]
        for k in range(4):
            p=track(R/'data/v3'/f'lopo_splits_{cohort}'/f'fold_{k}.json');f=json.loads(p.read_text())
            tr={donors[x] for x in f['train_samples']};te={donors[x] for x in f['test_samples']}
            check(f'{cohort}/{k}:donor_disjoint',not tr&te)
            check(f'{cohort}/{k}:section_disjoint',not set(f['train_samples'])&set(f['test_samples']))
            check(f'{cohort}/{k}:complete_partition',set(f['train_samples']+f['test_samples'])==set(s[s.cohort.eq(cohort)]['sample']))
            seen+=f['test_samples']
            rows.append(dict(cohort=cohort,fold=k,holdout_group=s[s['sample'].eq(f['test_samples'][0])].holdout_group.iloc[0],train_samples=';'.join(f['train_samples']),test_samples=';'.join(f['test_samples']),train_donors=';'.join(sorted(tr)),test_donors=';'.join(sorted(te)),n_train_donors=len(tr),n_test_donors=len(te),donor_overlap=0))
        check(cohort+':test_once',len(seen)==len(set(seen)))
    save(pd.DataFrame(rows),'fold_registry')

def genes():
    d=relabel(read('stage_07_gene_audit/section_effects.csv')); save(d,'gene_section_effects')
    keys=['cohort','gene','grouping','adjustment','outcome']; vals=['effect','standardized_effect','Q1_mean','Q4_mean','Q1_retained','Q4_retained']
    u,c=summaries(d,keys,vals); save(u,'gene_donor_effects'); save(c,'gene_cohort_effects')
    _,b=summaries(d,keys,vals,unit='inherited_patient');save(b,'gene_split_group_sensitivity')
    p=relabel(read('stage_07_gene_audit/prediction_quality.csv'));save(p,'gene_section_quality')
    numeric=[x for x in p.select_dtypes('number') if x!='n_spots']
    q=p.groupby(['cohort','patient','gene','grouping','quartile'])[numeric].mean().reset_index();save(q,'gene_donor_quality')
    a=c[c.grouping.eq('gene_excluded')&c.adjustment.eq('overlap_adjusted')&c.metric.eq('standardized_effect')]
    bridge=a.pivot(index=['gene','outcome'],columns='cohort',values='mean').dropna().reset_index()
    bridge['same_direction']=bridge.biomarkers*bridge['10x_janesick']>0;save(bridge,'bridge_genes')
    rows=[]
    for outcome,g in bridge.groupby('outcome'):
        rows.append(dict(outcome=outcome,n_genes=len(g),same_direction=int(g.same_direction.sum()),spearman=spearmanr(g.biomarkers,g['10x_janesick']).statistic))
    save(pd.DataFrame(rows),'bridge_summary')
    qp=relabel(read('stage_07_gene_audit/quartile_profiles.csv'))
    sec=qp.groupby(['cohort','patient','sample','quartile'])[['absolute','baseline_mae','observed_log','signed']].mean().reset_index();save(sec,'section_quartile_quality')
    pa=sec.groupby(['cohort','patient','quartile'])[['absolute','baseline_mae','observed_log','signed']].mean().reset_index();pa['relative_mae_gain']=1-pa.absolute/pa.baseline_mae;save(pa,'donor_quartile_quality')
    links=[]
    effects=u[u.grouping.eq('gene_excluded')&u.adjustment.eq('overlap_adjusted')&u.outcome.eq('observed_log')]
    for (cohort,donor),g in q[q.grouping.eq('full')&q.quartile.eq('Q1')].groupby(['cohort','patient']):
        z=g.merge(effects[effects.cohort.eq(cohort)&effects.patient.eq(donor)][['gene','standardized_effect']],on='gene')
        links.append(dict(cohort=cohort,donor=donor,n_genes=len(z),spearman_predictability_vs_absolute_expression_effect=spearmanr(z.whole_section_gene_pearson,z.standardized_effect.abs()).statistic))
    save(pd.DataFrame(links),'predictability_expression_relationship')

def programs():
    sections=[];units=[];cohorts=[];splits=[]
    for family in ['biomarkers','10x_janesick','coad','idc_visium']:
        d=relabel(read(f'stage_10_decile_sensitivity/{family}/section_program_effects.csv'))
        d=d[d['tail'].isin([.2,.25,.3])];sections.append(d)
        keys=['family','pathway','grouping','tail','adjustment','outcome']; vals=['effect','standardized_effect','lower_mean','upper_mean','lower_retained','upper_retained']
        u,c=summaries(d,keys,vals,unit='unit');units.append(u);cohorts.append(c)
        if family=='biomarkers':
            _,b=summaries(d,keys,vals,unit='inherited_unit');splits.append(b)
    save(pd.concat(sections),'program_section_effects');save(pd.concat(units),'program_unit_effects');save(pd.concat(cohorts),'program_cohort_effects');save(pd.concat(splits),'program_split_group_sensitivity')
    d=relabel(read('stage_14_remaining_analyses/boundary_section_effects.csv'));save(d,'boundary_section_effects')
    keys=['family','kind','endpoint','grouping','adjustment','band_um','outcome']
    u,c=summaries(d,keys,['effect','fixed_standardized_effect','Q1_retained','Q4_retained'],unit='unit');save(u,'boundary_unit_effects');save(c,'boundary_cohort_effects')

def predictions():
    d=relabel(read('stage_02_prediction/section_model_metrics.csv'));save(d,'model_section_quality')
    vals=['mae','rmse','baseline_median_mae','baseline_mean_rmse','mean_signed_error']
    q=d.groupby(['cohort','patient','encoder','regressor'])[vals].mean().reset_index();q['relative_mae_gain']=1-q.mae/q.baseline_median_mae;save(q,'model_donor_quality')
    rows=[]
    for cc in cfg['cohorts'].values():
        cohort=cc['name']; genes=json.loads(track(R/'data/v3'/f'gene_list_{cohort}.json').read_text())
        for k in range(4):
            fd=R/'outputs/predictions'/cohort/'uni/ridge'/f'fold{k}'
            y=np.load(track(fd/'test_targets.npy')).astype(float);ids=json.loads(track(fd/'test_spot_ids.json').read_text())
            f=json.loads((R/'data/v3'/f'lopo_splits_{cohort}'/f'fold_{k}.json').read_text())
            mask={s:np.array([i.startswith(s+'_') for i in ids]) for s in f['test_samples']}
            check(f'{cohort}/{k}:IDs_covered',np.stack(list(mask.values())).sum(axis=0).min()==1 and np.stack(list(mask.values())).sum(axis=0).max()==1)
            for enc in [e['name'] for e in cfg['encoders']]:
                for reg in [e['name'] for e in cfg['regressors']]:
                    p=R/'outputs/predictions'/cohort/enc/reg/f'fold{k}'
                    check(f'{cohort}/{k}/{enc}/{reg}:ID_order',json.loads(track(p/'test_spot_ids.json').read_text())==ids)
                    pred=np.load(track(p/'test_predictions.npy')).astype(float)
                    for s,ix in mask.items():
                        a=y[ix]-y[ix].mean(axis=0);b=pred[ix]-pred[ix].mean(axis=0);den=np.sqrt((a*a).sum(axis=0)*(b*b).sum(axis=0))
                        r=np.divide((a*b).sum(axis=0),den,out=np.full(len(genes),np.nan),where=den>0)
                        rows.extend(dict(cohort=cohort,patient=donors[s],sample=s,fold=k,encoder=enc,regressor=reg,gene=g,pearson=v) for g,v in zip(genes,r))
            print(cohort,k,'prediction section correlations complete',flush=True)
    d=pd.DataFrame(rows);save(d,'model_gene_section_quality')
    q=d.groupby(['cohort','patient','encoder','regressor','gene']).pearson.mean().reset_index();save(q,'model_gene_donor_quality')
    q=q.groupby(['cohort','encoder','regressor','gene']).pearson.mean().reset_index();save(q,'model_gene_cohort_quality')
    save(q.groupby(['cohort','encoder','regressor']).pearson.mean().reset_index(),'model_cohort_correlations')

def features():
    import importlib.util
    p=track(A/'stage_11_gene_feature_model/audit.py');spec=importlib.util.spec_from_file_location('feature_audit',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    metrics=[];coefs=[];preds=[];allfeatures=[];influence=[]
    for cohort in ['biomarkers','10x_janesick']:
        sec=relabel(read(f'stage_11_gene_feature_model/{cohort}/section_features.csv'))
        old=read(f'stage_11_gene_feature_model/{cohort}/revised_features.csv')
        f,u=m.features_from_sections(sec,old);f['cohort']=cohort;allfeatures.append(f)
        for name,cols in [('mean_only',m.NUM[:1]),('expression_pathway',m.NUM[:3]),('numeric_with_spatial',m.NUM)]:
            x,_,names,_=m.encode(f,f,cols,False);b=np.linalg.lstsq(x,f.mean_pearson,rcond=None)[0];out,baseline,fold,stat,di=m.cv(f,cols,False)
            metrics.append(dict(cohort=cohort,model=name,n_genes=len(f),in_sample_R2=1-np.sum((f.mean_pearson-x@b)**2)/np.sum((f.mean_pearson-f.mean_pearson.mean())**2),**stat))
            coefs.extend(dict(cohort=cohort,model=name,feature=n,coefficient=v) for n,v in zip(names,b))
            preds.extend(dict(cohort=cohort,model=name,gene=g,actual=a,predicted=p,baseline=c,fold=k) for g,a,p,c,k in zip(f.gene,f.mean_pearson,out,baseline,fold))
        for donor in sec.patient.unique():
            fi,_=m.features_from_sections(sec,old,donor);x,_,names,_=m.encode(fi,fi,m.NUM,False);b=np.linalg.lstsq(x,fi.mean_pearson,rcond=None)[0]
            influence.append(dict(cohort=cohort,omitted_donor=donor,spatial_coefficient=b[names.index('spatial_autocorrelation')]))
    save(pd.concat(allfeatures),'gene_features');save(pd.DataFrame(metrics),'gene_feature_metrics');save(pd.DataFrame(coefs),'gene_feature_coefficients');save(pd.DataFrame(preds),'gene_feature_cv_predictions');save(pd.DataFrame(influence),'gene_feature_influence')

def external():
    d=relabel(read('stage_09_external_replication/idc_common_member_section_effects.csv'))
    u,c=summaries(d,['family','idc_cohort','pathway','outcome','adjustment'],['standardized_effect']);save(u,'idc_common_member_donor_effects');save(c,'idc_common_member_cohort_effects')
    z=read('stage_09_external_replication/program_comparisons.csv');program=pd.read_csv(T/'program_cohort_effects.csv')
    for i,r in z.iterrows():
        if r.idc_cohort!='biomarkers':continue
        if r.comparison=='exact_common_members':
            a=c[c.family.eq(r.family)&c.idc_cohort.eq(r.idc_cohort)&c.pathway.eq(r.pathway)&c.outcome.eq(r.outcome)&c.adjustment.eq(r.adjustment)]
        else:
            a=program[program.family.eq(r.idc_cohort)&program.pathway.eq(r.pathway)&program.outcome.eq(r.outcome)&program.adjustment.eq(r.adjustment)&program.grouping.eq('program_excluded')&program['tail'].eq(.25)&program.metric.eq('standardized_effect')]
        assert len(a)==1,(i,len(a));a=a.iloc[0];z.loc[i,'idc_standardized_effect']=a['mean'];z.loc[i,'idc_positive']=a.n_positive;z.loc[i,'idc_negative']=a.n_negative
    z['same_direction']=z.idc_standardized_effect*z.external_standardized_effect>0;save(z,'external_program_comparisons')
    iq=pd.read_csv(T/'gene_donor_quality.csv');iq=iq[iq.grouping.eq('full')&iq.quartile.eq('Q1')]
    g=read('stage_09_external_replication/gene_predictability_pairs.csv')
    lookup=iq.groupby(['cohort','gene']).whole_section_gene_pearson.mean()
    g['idc_pearson']=[lookup.loc[(r.idc_cohort,r.gene)] for r in g.itertuples()];save(g,'external_gene_pairs')
    rows=[]
    for key,a in g.groupby(['family','idc_cohort','scope']):rows.append(dict(zip(['family','idc_cohort','scope'],key))|dict(n_genes=len(a),pearson=pearsonr(a.idc_pearson,a.external_pearson).statistic,spearman=spearmanr(a.idc_pearson,a.external_pearson).statistic))
    save(pd.DataFrame(rows),'external_gene_correlations')

if __name__=='__main__':
    for f in [registry,genes,programs,predictions,features,external]:
        f();print(f.__name__,'complete',flush=True)
    (OUT/'sources.json').write_text(json.dumps(SOURCES,indent=2)+'\n')
    (OUT/'checks.json').write_text(json.dumps(dict(status='pass',checks=CHECKS),indent=2)+'\n')

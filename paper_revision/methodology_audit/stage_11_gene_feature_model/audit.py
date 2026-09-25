"""Original gene-feature reproduction and patient-balanced all-section reassessment."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr,spearmanr
from sklearn.model_selection import KFold
import statsmodels.api as sm
import yaml

OUT=Path(__file__).resolve().parent;AUDIT=OUT.parent;ROOT=AUDIT.parents[1];REPO=ROOT/'paper_revision/clean_repo'
sys.path.insert(0,str(REPO))
from src.spatial import build_spatial_weights,morans_i
from src.pathways import load_gene_sets
NUM=['mean_expression','cv_expression','pathway_count','spatial_autocorrelation'];CAT=['primary_localization','primary_function']


def encode(train,test,columns,categorical):
    means=train[columns].mean();sd=train[columns].std(ddof=0);sd=sd.where(sd>0,1)
    a=(train[columns].fillna(means)-means)/sd;b=(test[columns].fillna(means)-means)/sd
    names=list(columns);xa=[a.to_numpy()];xb=[b.to_numpy()];unseen=[]
    if categorical:
        for col in CAT:
            levels=sorted(train[col].unique());unseen += [f'{col}:{v}' for v in sorted(set(test[col])-set(levels))]
            for level in levels[1:]:
                xa.append(train[col].eq(level).to_numpy()[:,None]);xb.append(test[col].eq(level).to_numpy()[:,None]);names.append(col+'='+level)
    return np.column_stack([np.ones(len(train)),*xa]).astype(float),np.column_stack([np.ones(len(test)),*xb]).astype(float),['intercept']+names,unseen


def cv(features,columns,categorical,global_scaling=False):
    y=features.mean_pearson.to_numpy();out=np.empty(len(y));baseline=np.empty(len(y));folds=np.zeros(len(y),int);diagnostics=[]
    if global_scaling:whole,_,_,_=encode(features,features,columns,categorical)
    for k,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=42).split(features)):
        if global_scaling:a=whole[tr];b=whole[te];unseen=[]
        else:a,b,_,unseen=encode(features.iloc[tr],features.iloc[te],columns,categorical)
        beta,_,rank,_=np.linalg.lstsq(a,y[tr],rcond=None);out[te]=b@beta;baseline[te]=y[tr].mean();folds[te]=k
        independent=sm.OLS(y[tr],a).fit().predict(b);assert np.max(np.abs(out[te]-independent))<1e-9
        diagnostics.append(dict(fold=k,n_train=len(tr),n_test=len(te),rank=int(rank),n_columns=a.shape[1],unseen_categories=';'.join(unseen)))
    mse=np.mean((y-out)**2);bmse=np.mean((y-baseline)**2)
    metrics=dict(pearson=float(pearsonr(y,out).statistic),spearman=float(spearmanr(y,out).statistic),mae=float(np.mean(np.abs(y-out))),
        baseline_mae=float(np.mean(np.abs(y-baseline))),mse=float(mse),baseline_mse=float(bmse),relative_MSE_gain=float(1-mse/bmse),
        predictive_R2=float(1-np.sum((y-out)**2)/np.sum((y-y.mean())**2)),n_rank_deficient_folds=sum(d['rank']<d['n_columns'] for d in diagnostics))
    return out,baseline,folds,metrics,diagnostics


def features_from_sections(section,static,exclude=None):
    s=section[section.patient.ne(exclude)] if exclude else section
    p=s.groupby(['patient','gene'])[['expression_mean','second_moment','morans_i','prediction_pearson']].mean().reset_index()
    a=p.groupby('gene')[['expression_mean','second_moment','morans_i','prediction_pearson']].mean().reindex(static.gene).reset_index()
    f=static.copy();f['mean_pearson']=a.prediction_pearson.to_numpy();f['mean_expression']=a.expression_mean.to_numpy()
    f['cv_expression']=np.sqrt(np.maximum(a.second_moment.to_numpy()-a.expression_mean.to_numpy()**2,0))/(a.expression_mean.to_numpy()+1e-6)
    f['spatial_autocorrelation']=a.morans_i.to_numpy()
    return f,p


def main():
    checks=[];metrics=[];coefs=[];cvrows=[];foldrows=[];annotations=[];changes=[];influence=[]
    sets=load_gene_sets(str(REPO/'data/gene_sets/h.all.v2024.1.Hs.symbols.gmt'));cfg=yaml.safe_load((REPO/'config.yaml').read_text())
    quality=pd.read_csv(AUDIT/'stage_07_gene_audit/prediction_quality.csv');quality=quality[quality.grouping.eq('full')&quality.quartile.eq('Q1')]
    for cohort in ['biomarkers','10x_janesick']:
        dest=OUT/cohort;dest.mkdir(exist_ok=True)
        olddir=REPO/'outputs/phase3/gene_predictability'/cohort;old=pd.read_csv(olddir/'gene_features.csv')
        assert old.gene.is_unique and np.isfinite(old[NUM+['mean_pearson']].to_numpy()).all()
        original,_,names,_=encode(old,old,NUM,True);beta,_,rank,_=np.linalg.lstsq(original,old.mean_pearson,rcond=None)
        saved=json.loads((olddir/'ols_results.json').read_text());sb=np.array([r['coefficient'] for r in saved['coefficients']])
        err=float(np.max(np.abs(beta-sb)));assert err<1e-9
        checks.append(dict(check=cohort+':archived_OLS',max_abs=err,passed=True))
        op,ob,fold,om,od=cv(old,NUM,True,True)
        archived=pd.read_csv(REPO/'outputs/phase4/heldout_validation'/cohort/'cv_predictions.csv').set_index('gene').loc[old.gene]
        err=float(np.max(np.abs(op-archived.predicted_pearson.to_numpy())));assert err<1e-9
        checks.append(dict(check=cohort+':archived_CV',max_abs=err,passed=True))
        # Audit annotations as cached; no live reannotation or changed categories.
        cache=REPO/'outputs/phase3/cache'
        for kind,fn in [('function','go_slim_'),('localization','uniprot_localization_')]:
            a=pd.read_csv(cache/f'{fn}{cohort}.csv');field='primary_function' if kind=='function' else 'primary_localization'
            assert a.gene.is_unique and a.set_index('gene').loc[old.gene,field].to_numpy().tolist()==old[field].tolist()
            for label,n in a[field].value_counts().items():annotations.append(dict(cohort=cohort,kind=kind,label=label,n=int(n)))
            if kind=='function':
                namespace=a.all_functions.astype(str).str.contains('biological_process|molecular_function|cellular_component').sum()
                annotations.append(dict(cohort=cohort,kind='function_provenance',label='GO namespaces rather than detailed terms',n=int(namespace)))
        counts=[sum(g in values for values in sets.values()) for g in old.gene];assert np.array_equal(counts,old.pathway_count)
        loc=pd.read_csv(AUDIT/'stage_05_biology/arrays'/cohort/'locations.csv');base=REPO/'outputs/predictions'/cohort/'uni/ridge'
        y=np.concatenate([np.load(base/f'fold{k}/test_targets.npy') for k in range(4)]).astype(float)
        ids=sum([json.loads((base/f'fold{k}/test_spot_ids.json').read_text()) for k in range(4)],[]);assert np.array_equal(ids,loc.spot_id)
        genes=json.loads((REPO/'data/v3'/f'gene_list_{cohort}.json').read_text());assert genes==old.gene.tolist()
        sections=[];spot_weights=np.empty(len(y))
        for sample,ss in loc.groupby('sample_id',sort=False):
            ix=ss.index.to_numpy();v=y[ix];mean=v.mean(axis=0);second=np.mean(v*v,axis=0);z=v-mean
            w=build_spatial_weights(ss[['x_um','y_um']].to_numpy(),n_neighbors=6)
            denom=(z*z).sum(axis=0);moran=np.divide(len(v)/w.sum()*np.sum(z*(w@z),axis=0),denom,out=np.zeros(len(genes)),where=denom>0)
            for j in [0,37,101,179,279]:
                expected=morans_i(v[:,j],w);err=abs(expected-moran[j]);assert err<1e-10
                checks.append(dict(check=f'{cohort}/{sample}/{genes[j]}:scalar_Moran',max_abs=float(err),passed=True))
            pred=quality[quality.cohort.eq(cohort)&quality['sample'].eq(sample)].set_index('gene').loc[genes,'whole_section_gene_pearson'].to_numpy()
            for j,g in enumerate(genes):sections.append(dict(cohort=cohort,sample=sample,patient=ss.patient.iloc[0],gene=g,expression_mean=mean[j],second_moment=second[j],morans_i=moran[j],prediction_pearson=pred[j]))
            patient=ss.patient.iloc[0];nsec=loc[loc.patient.eq(patient)].sample_id.nunique();spot_weights[ix]=1/loc.patient.nunique()/nsec/len(ix)
        section=pd.DataFrame(sections);section.to_csv(dest/'section_features.csv',index=False)
        revised,patient=features_from_sections(section,old);patient.to_csv(dest/'patient_features.csv',index=False);revised.to_csv(dest/'revised_features.csv',index=False)
        direct=spot_weights@y;err=float(np.max(np.abs(direct-revised.mean_expression)));assert err<1e-10
        checks.append(dict(check=cohort+':direct_weighted_mean',max_abs=err,passed=True))
        variance=spot_weights@(y*y)-direct**2;expected_cv=np.sqrt(np.maximum(variance,0))/(direct+1e-6)
        err=float(np.max(np.abs(expected_cv-revised.cv_expression)));assert err<1e-10
        checks.append(dict(check=cohort+':direct_weighted_variance_CV',max_abs=err,passed=True))
        response=old.copy();response['mean_pearson']=revised.mean_pearson
        for col in NUM+['mean_pearson']:
            changes.append(dict(cohort=cohort,feature=col,old_new_pearson=float(pearsonr(old[col],revised[col]).statistic) if old[col].std()>0 else np.nan,
                median_absolute_change=float(np.median(np.abs(old[col]-revised[col]))),max_absolute_change=float(np.max(np.abs(old[col]-revised[col])))))
        models={'mean_only':(['mean_expression'],False),'expression_pathway':(NUM[:3],False),'numeric_with_spatial':(NUM,False),'full_cached_annotations':(NUM,True)}
        for version,f in [('archived',old),('response_only',response),('revised_all_sections',revised)]:
            for model,(columns,categorical) in models.items():
                x,_,cn,_=encode(f,f,columns,categorical);b,_,rank,_=np.linalg.lstsq(x,f.mean_pearson,rcond=None);fitted=x@b
                checkbeta=sm.OLS(f.mean_pearson,x).fit().params.to_numpy();assert np.max(np.abs(b-checkbeta))<1e-9
                out,baseline,fold,stat,diagnostics=cv(f,columns,categorical)
                metrics.append(dict(cohort=cohort,version=version,model=model,n_genes=len(f),n_columns=x.shape[1],rank=int(rank),
                    in_sample_R2=float(1-np.sum((f.mean_pearson-fitted)**2)/np.sum((f.mean_pearson-f.mean_pearson.mean())**2)),**stat))
                coefs += [dict(cohort=cohort,version=version,model=model,feature=n,coefficient=float(v)) for n,v in zip(cn,b)]
                cvrows += [dict(cohort=cohort,version=version,model=model,gene=g,actual=float(a),predicted=float(p),baseline=float(c),fold=int(k)) for g,a,p,c,k in zip(f.gene,f.mean_pearson,out,baseline,fold)]
                foldrows += [dict(cohort=cohort,version=version,model=model,**d) for d in diagnostics]
                if version=='archived' and model=='full_cached_annotations':
                    checks.append(dict(check=cohort+':training_only_vs_global_standardization',max_abs=float(np.max(np.abs(out-op))),passed=True,
                        interpretation='Measured equivalence/difference; not assumed to be prediction leakage'))
        for omitted in patient.patient.unique():
            f,_=features_from_sections(section,old,omitted);_,_,_,m,_=cv(f,NUM,True)
            x,_,names,_=encode(f,f,NUM,True);b=np.linalg.lstsq(x,f.mean_pearson,rcond=None)[0]
            influence.append(dict(cohort=cohort,omitted_patient=omitted,spatial_coefficient=float(b[names.index('spatial_autocorrelation')]),**m))
        print(cohort,'gene-feature audit complete',flush=True)
    for name,rows in [('model_metrics',metrics),('coefficients',coefs),('cv_predictions',cvrows),('fold_diagnostics',foldrows),('annotation_audit',annotations),('feature_changes',changes),('patient_influence',influence)]:pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False)
    (OUT/'checks.json').write_text(json.dumps(dict(status='pass',n_checks=len(checks),checks=checks),indent=2)+'\n')
    (OUT/'COMPLETE.json').write_text(json.dumps(dict(status='complete',n_cohorts=2,n_genes=560,n_models=len(metrics)),indent=2)+'\n')


if __name__=='__main__':main()

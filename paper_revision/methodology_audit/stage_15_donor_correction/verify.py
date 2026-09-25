"""Independent aggregation, sampled correlation, provenance and preservation checks.

Does not import correct.py or its aggregation helpers.
"""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, t

O=Path(__file__).resolve().parent; A=O.parent; REV=A.parent; ROOT=REV.parent
T=O/'tables'; checks=[]
def record(name, ok, **details):
    checks.append(dict(name=name,passed=bool(ok),**details)); assert ok,(name,details)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while chunk:=f.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()
def read(name):return pd.read_csv(T/(name+'.csv'))

registry=read('specimen_registry'); discovery=registry[registry.cohort.eq('biomarkers')]
record('eleven_discovery_donors',len(discovery)==11 and discovery.patient.nunique()==11)
mapping=dict(zip(registry['sample'],registry.patient))
record('registry_matches_source',set(discovery.source_specimen)==set(pd.read_csv(A/'clinical_metadata_check_10/discovery_clinical_metadata.csv').source_specimen))
record('no_discovery_repeats',not discovery.repeat_section_available.any())
folds=read('fold_registry')
for cohort,g in folds.groupby('cohort'):
    seen=[]
    for r in g.itertuples():
        train=set(r.train_samples.split(';'));test=set(r.test_samples.split(';'))
        record(f'{cohort}/{r.fold}:independent_fold_check',not train&test and not {mapping[x] for x in train}&{mapping[x] for x in test})
        seen+=list(test)
    record(cohort+':each_specimen_tested_once',len(seen)==len(set(seen)) and set(seen)==set(registry[registry.cohort.eq(cohort)]['sample']))

original=pd.read_csv(A/'stage_07_gene_audit/section_effects.csv')
current=read('gene_section_effects')
numeric=list(original.select_dtypes('number'))
roundtrip_error=float(np.nanmax(abs(current[numeric].to_numpy()-original[numeric].to_numpy())))
record('all_section_gene_numbers_preserved_to_csv_precision',np.allclose(current[numeric],original[numeric],rtol=1e-14,atol=1e-12,equal_nan=True),max_csv_roundtrip_difference=roundtrip_error)
keys=['cohort','gene','grouping','adjustment','outcome']
new=read('gene_cohort_effects').set_index(keys+['metric'])
independent=original.copy();independent['patient']=independent['sample'].map(mapping)
max_error=0.;ci_error=0.;n_cases=0
for key,g in independent.groupby(keys):
    for metric in ['effect','standardized_effect']:
        by_donor=np.array([q[metric].mean() for _,q in g.groupby('patient')]);by_donor=by_donor[np.isfinite(by_donor)]
        row=new.loc[key+(metric,)];mean=np.mean(by_donor)
        half=t.ppf(.975,len(by_donor)-1)*np.std(by_donor,ddof=1)/np.sqrt(len(by_donor))
        max_error=max(max_error,abs(row['mean']-mean));ci_error=max(ci_error,abs(row.ci_low-(mean-half)),abs(row.ci_high-(mean+half)));n_cases+=1
record('independent_gene_aggregation_and_intervals',max_error<1e-10 and ci_error<1e-10,n_cases=n_cases,max_mean_error=max_error,max_interval_error=ci_error)
old=pd.read_csv(A/'stage_07_gene_audit/cohort_effects.csv').set_index(keys+['metric'])
val=old[old.index.get_level_values('cohort')=='10x_janesick']
err=np.max(np.abs(val['mean']-new.loc[val.index,'mean']))
record('validation_gene_means_unchanged',err<1e-10,max_error=float(err))
split=read('gene_split_group_sensitivity').set_index(keys+['metric'])
err=np.max(np.abs(old['mean']-split.loc[old.index,'mean']))
record('historical_split_averages_reproduced',err<1e-10,max_error=float(err))

b=read('boundary_cohort_effects');b=b[b.kind.eq('program')&b.grouping.eq('excluded')&b.adjustment.eq('overlap_adjusted')&b.outcome.eq('absolute')&b.metric.eq('fixed_standardized_effect')]
for (family,band),g in b.groupby(['family','band_um']):
    record(f'{family}:boundary_{band}:all_program_error_means_positive',g['mean'].gt(0).all(),n_programs=len(g))

sec=read('model_gene_section_quality').set_index(['cohort','sample','encoder','regressor','gene'])
for cohort in ['biomarkers','10x_janesick']:
    genes=json.loads((REV/'clean_repo/data/v3'/f'gene_list_{cohort}.json').read_text())
    for fold in [0,3]:
        base=REV/'clean_repo/outputs/predictions'/cohort/'uni/ridge'/f'fold{fold}'
        y=np.load(base/'test_targets.npy');p=np.load(base/'test_predictions.npy');ids=json.loads((base/'test_spot_ids.json').read_text())
        sample=ids[0].split('_')[0];mask=np.array([x.startswith(sample+'_') for x in ids])
        for j in [0,137,279]:
            actual=pearsonr(y[mask,j].astype(float),p[mask,j].astype(float)).statistic
            expected=sec.loc[(cohort,sample,'uni','ridge',genes[j]),'pearson']
            record(f'{cohort}/{fold}/{genes[j]}:scipy_correlation',abs(actual-expected)<1e-10)

influence=read('gene_feature_influence')
record('positive_spatial_coefficient_each_donor_omission',influence.spatial_coefficient.gt(0).all(),n_omissions=len(influence))

inputs=json.loads((O/'sources.json').read_text())
record('all_stage15_inputs_unchanged',all(sha(p)==h for p,h in inputs.items()),n_files=len(inputs))
snap=json.loads((REV/'original_snapshot_manifest.json').read_text());n_original=n_copy=0
for folder,items in snap.items():
    for rel,meta in items.items():
        p=ROOT/folder/rel
        record('original_preserved:'+str(p.relative_to(ROOT)),p.is_file() and sha(p)==meta['sha256']);n_original+=1
        if folder=='CIBM_submission':
            copy=REV/folder/rel;record('copied_manuscript_preserved:'+rel,copy.is_file() and sha(copy)==meta['sha256']);n_copy+=1

result=dict(status='pass',checks=checks,original_files_verified=n_original,copied_manuscript_files_verified=n_copy)
(O/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(status='pass',n_checks=len(checks),original_files_verified=n_original,copied_manuscript_files_verified=n_copy),indent=2))

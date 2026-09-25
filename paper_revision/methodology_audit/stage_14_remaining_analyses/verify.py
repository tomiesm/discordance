"""Alternative-formula checks of the new analyses and frozen source hashes."""
from itertools import permutations,combinations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from common import OUT,AUDIT,FAMILIES,IDC,EMT,OUTCOMES,check,finish,sha,CHECKS

def main():
    section=pd.read_csv(OUT/'boundary_section_effects.csv');boundary=pd.read_parquet(AUDIT/'stage_12_boundary_diagnostic/spot_boundary_diagnostics.parquet').set_index('spot_id')
    old=pd.read_csv(AUDIT/'stage_12_boundary_diagnostic/interior_sensitivity.csv')
    unique=section[section.grouping.eq('full')&section.band_um.gt(0)].drop_duplicates(['sample','band_um'])
    v=unique.merge(old,on=['sample','band_um'],validate='one_to_one');assert len(v)==64
    for new,ref in [('Q1_retained','lower_25_retained'),('Q4_retained','upper_25_retained'),('interior_fraction','fraction_retained')]:check('Stage12_fixed_groups:'+new,np.max(abs(v[new]-v[ref])))
    # Harmonic-overlap means equal the exposure coefficient in a regression
    # with a separate intercept for every eligible stratum.
    for family in FAMILIES:
        loc=pd.read_parquet(AUDIT/'stage_10_decile_sensitivity'/family/'locations.parquet');folder=AUDIT/('stage_05_biology/arrays/'+family if family in IDC else 'stage_09_external_replication/'+family+'/arrays');ck,dk=('outside_total_counts','outside_detected_genes') if family in IDC else ('outside_counts','outside_detected')
        for name in [EMT,'HALLMARK_COMPLEMENT','HALLMARK_E2F_TARGETS']:
            path=folder/f'{name}.npz'
            if not path.exists():continue
            a=np.load(path)
            for sample,ss in loc.groupby('sample'):
                ix=ss.index.to_numpy();b=boundary.loc[ss.spot_id];signal=a['program_excluded_conditional'][ix];cut=np.quantile(signal,[.25,.75]);q1=signal<=cut[0];q4=signal>=cut[1]
                count=a[ck][ix];detect=a[dk][ix];edges=[np.quantile(z,np.linspace(0,1,6)[1:-1]) for z in [count,detect]];s=np.sum(count[:,None]>=edges[0],axis=1)*5+np.sum(detect[:,None]>=edges[1],axis=1)
                for band in [200,400]:
                    interior=(b.tissue_distance_um.to_numpy()>band)&(b.image_distance_um.to_numpy()>band)&(b.expression_extent_distance_um.to_numpy()>band);lo=q1&interior;hi=q4&interior
                    eligible=[k for k in np.unique(s) if np.sum(lo&(s==k))>=10 and np.sum(hi&(s==k))>=10];keep=(lo|hi)&np.isin(s,eligible)
                    ref=section[section.family.eq(family)&section['sample'].eq(sample)&section.endpoint.eq(name)&section.grouping.eq('excluded')&section.adjustment.eq('overlap_adjusted')&section.band_um.eq(band)].set_index('outcome').loc[OUTCOMES]
                    if not ref.estimable.all():continue
                    x=np.column_stack([hi[keep].astype(float)]+[(s[keep]==k).astype(float) for k in eligible]);y=np.column_stack([a[o][ix][keep].astype(float) for o in OUTCOMES]);beta=np.linalg.lstsq(x,y,rcond=None)[0];check(f'{sample}/{name}/{band}:stratum_fixed_effects',np.max(abs(beta[0]-ref.effect.to_numpy())),1e-8)
    # All 7! section permutations independently reproduce both validation
    # reference tails, including source preservation; no unique-partition helper.
    d=pd.read_csv(AUDIT/'stage_10_decile_sensitivity/10x_janesick/section_program_effects.csv');d=d[d['tail'].eq(.25)&d.grouping.eq('program_excluded')&d.adjustment.eq('overlap_adjusted')&d.outcome.eq('observed')];meta=d[['sample','unit']].drop_duplicates().sort_values('sample');samples=meta['sample'].tolist();u=meta.unit.to_numpy();z=d.pivot(index='sample',columns='pathway',values='standardized_effect').loc[samples].to_numpy();c=np.corrcoef(z);groups=[np.flatnonzero(u==p) for p in sorted(set(u))];sources=np.array([x.startswith('NCBI') for x in samples])
    def statistic(order):
        gs=[np.asarray(order)[g] for g in groups];within=[np.mean([c[a,b] for a,b in combinations(g,2)]) for g in gs if len(g)>1];between=[c[np.ix_(g,h)].mean() for g,h in combinations(gs,2)];return np.mean(within)-np.mean(between)
    observed=statistic(np.arange(7));allvals=[];restricted=[]
    for order in permutations(range(7)):
        value=statistic(order);allvals.append(value)
        if np.array_equal(sources,sources[list(order)]):restricted.append(value)
    r=pd.read_csv(OUT/'patient_profile_summary.csv');r=r[r.family.eq('10x_janesick')&r.grouping.eq('program_excluded')&r.adjustment.eq('overlap_adjusted')&r.outcome.eq('observed')].set_index('reference')
    for reference,vals in [('source_preserving',restricted),('cohort_only_sensitivity',allvals)]:
        row=r.loc[reference];check('brute_7_factorial:'+reference,abs(float(np.mean(np.array(vals)>=observed-1e-12))-row.reference_p_one_sided),1e-12)
    # Direct statsmodels fits from exported tumor-rich designs check the
    # independent FWL and production estimators for every computed model.
    import statsmodels.api as sm
    a=np.load(OUT/'tumor_rich_emt_designs.npz');effects=pd.read_csv(OUT/'tumor_rich_emt_effects.csv')
    for key in a.files:
        if not key.endswith('__x'):continue
        stem=key[:-3];sample,minimum,adjustment=stem.split('__');x=a[key];y=a[stem+'__y'];ref=effects[effects['sample'].eq(sample)&effects.minimum_tumor_fraction.eq(float(minimum))&effects.adjustment.eq(adjustment)&effects.width_um.eq(800)].set_index('outcome').loc[OUTCOMES]
        coef=np.array([sm.OLS(y[:,j],x).fit().params[1] for j in range(3)]);check(stem+':statsmodels',np.max(abs(coef-ref.estimate)),1e-8)
    # Every declared input still has its originally recorded bytes/hash.
    hashes={}
    for name in ['boundary','profiles','tumor']:
        recorded=json.loads((OUT/f'{name}_sources.json').read_text())
        for path,meta in recorded.items():
            if path in hashes:assert hashes[path]==meta
            hashes[path]=meta
    for path,meta in hashes.items():assert sha(Path(path))==meta['sha256'],path
    check('unchanged_source_files',0,0);finish('independent')
    (OUT/'verification_summary.json').write_text(json.dumps(dict(status='pass',source_files_unchanged=len(hashes),independent_checks=len(CHECKS),boundary_rows=len(section),validation_brute_permutations=len(allvals),validation_source_preserving_permutations=len(restricted)),indent=2)+'\n')
    print('Independent verification passed',len(CHECKS),'checks;',len(hashes),'source files unchanged',flush=True)

if __name__=='__main__':main()

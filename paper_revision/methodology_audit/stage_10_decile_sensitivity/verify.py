"""Independent contrasts/graph counts and exact earlier-threshold regression checks."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

OUT=Path(__file__).resolve().parent;AUDIT=OUT.parent;ROOT=AUDIT.parents[1]


def main():
    checks=[]
    def check(name,value,tol=1e-8):
        assert np.isfinite(value) and value<tol,(name,value,tol)
        checks.append(dict(check=name,max_abs=float(value),passed=True))
    for family in ['biomarkers','10x_janesick','coad','idc_visium']:
        dest=OUT/family;loc=pd.read_parquet(dest/'locations.parquet');new=pd.read_csv(dest/'section_program_effects.csv')
        if family in ['biomarkers','10x_janesick']:
            old=pd.read_csv(AUDIT/'stage_06_inference/section_inference.csv')
            old=old[old.cohort.eq(family)&old.bins_per_covariate.eq(5)&old.block_width_um.eq(800)&old.outcome.isin(['observed','signed','absolute'])]
            old=old.rename(columns={'estimate':'effect','standardized_estimate':'standardized_effect'});old['grouping']='program_excluded';old['tail']=.25
            folder=AUDIT/'stage_05_biology/arrays'/family;ck='outside_total_counts';dk='outside_detected_genes'
        else:
            old=pd.read_csv(AUDIT/'stage_09_external_replication'/family/'section_program_effects.csv')
            old=old[old.outcome.isin(['observed','signed','absolute'])&~(old.grouping.eq('full')&old.adjustment.eq('overlap_adjusted'))]
            folder=AUDIT/'stage_09_external_replication'/family/'arrays';ck='outside_counts';dk='outside_detected'
            previous=pd.read_csv(AUDIT/'stage_09_external_replication'/family/'quartile_quality.csv')
            quality=pd.read_csv(dest/'tail_quality.csv')
            for sample,ss in loc.groupby('sample'):
                for name,q in [('lower',1),('upper',4)]:
                    a=quality[quality['sample'].eq(sample)&quality['tail'].eq(.25)&quality.group.eq(name)].iloc[0]
                    b=previous[previous['sample'].eq(sample)&previous.quartile.eq(q)].iloc[0]
                    check(f'{family}/{sample}/{name}:old_baseline',abs(a.baseline_mae-b.baseline_mae));check(f'{family}/{sample}/{name}:old_MAE',abs(a.mae-b.mae))
        keys=['sample','pathway','grouping','tail','adjustment','outcome']
        merged=old.merge(new,on=keys,suffixes=('_old','_new'),validate='one_to_one');assert len(merged)==len(old)
        for metric in ['effect','standardized_effect']:
            check(f'{family}:all_previous_{metric}',np.nanmax(np.abs(merged[metric+'_old']-merged[metric+'_new'])))
        for name in ['HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION','HALLMARK_COMPLEMENT','HALLMARK_E2F_TARGETS','HALLMARK_G2M_CHECKPOINT','HALLMARK_ESTROGEN_RESPONSE_EARLY']:
            path=folder/f'{name}.npz'
            if not path.exists():continue
            a=np.load(path)
            for sample,ss in loc.groupby('sample',sort=False):
                ix=ss.index.to_numpy();counts=a[ck][ix];det=a[dk][ix];probs=np.linspace(0,1,6)[1:-1]
                strata=5*np.searchsorted(np.quantile(counts,probs),counts,side='right')+np.searchsorted(np.quantile(det,probs),det,side='right')
                for grouping,score in [('full',ss.conditional.to_numpy()),('program_excluded',a['program_excluded_conditional'][ix])]:
                    low=score<=np.percentile(score,10);high=score>=np.percentile(score,90)
                    for outcome in ['observed','signed','absolute']:
                        v=a[outcome][ix].astype(float);pooled=np.sqrt(((low.sum()-1)*v[low].var(ddof=1)+(high.sum()-1)*v[high].var(ddof=1))/(low.sum()+high.sum()-2))
                        for adjustment in (['unadjusted'] if grouping=='full' else ['unadjusted','overlap_adjusted']):
                            if adjustment=='unadjusted':d=v[high].mean()-v[low].mean()
                            else:
                                differences=[];weights=[]
                                for s in np.unique(strata):
                                    l=low&(strata==s);h=high&(strata==s);n1=l.sum();n4=h.sum()
                                    if min(n1,n4)>=10:differences.append(v[h].mean()-v[l].mean());weights.append(2*n1*n4/(n1+n4))
                                d=np.average(differences,weights=weights) if len(weights) else np.nan
                            row=new[new['sample'].eq(sample)&new.pathway.eq(name)&new.grouping.eq(grouping)&new['tail'].eq(.1)&new.adjustment.eq(adjustment)&new.outcome.eq(outcome)].iloc[0]
                            assert np.isfinite(d)==np.isfinite(row.effect)
                            if np.isfinite(d):
                                check(f'{family}/{sample}/{name}/{grouping}/{outcome}/{adjustment}:decile_effect',abs(d-row.effect))
                                check(f'{family}/{sample}/{name}/{grouping}/{outcome}/{adjustment}:tail_SD',abs(d/pooled-row.standardized_effect))
                                check(f'{family}/{sample}/{name}/{grouping}/{outcome}/{adjustment}:fixed_section_SD',abs(d/np.std(v,ddof=1)-row.section_standardized_effect))
        spatial=pd.read_csv(dest/'spatial_coherence.csv');reference=pd.read_csv(dest/'density_reference.csv')
        for si,(sample,ss) in enumerate(loc.groupby('sample',sort=False)):
            v=ss.conditional.to_numpy();n=len(ss)
            for radius in [100,150]:
                pairs=cKDTree(ss[['x_um','y_um']].to_numpy()).query_pairs(radius,output_type='ndarray')
                degrees=np.bincount(pairs.ravel(),minlength=n)
                for group,use in [('lower',v<=np.percentile(v,10)),('upper',v>=np.percentile(v,90))]:
                    hit=use[pairs[:,0]]&use[pairs[:,1]];edge_count=hit.sum();ratio=2*edge_count/degrees[use].sum()
                    # Explicit adjacency traversal checks sparse connected-components output.
                    adjacency={i:[] for i in np.flatnonzero(use)}
                    for a,b in pairs[hit]:adjacency[a].append(b);adjacency[b].append(a)
                    seen=set();sizes=[]
                    for node in adjacency:
                        if node in seen:continue
                        todo=[node];seen.add(node);size=0
                        while todo:
                            u=todo.pop();size+=1
                            for w in adjacency[u]:
                                if w not in seen:seen.add(w);todo.append(w)
                        sizes.append(size)
                    r=spatial[spatial['sample'].eq(sample)&spatial.radius_um.eq(radius)&spatial['tail'].eq(.1)&spatial.group.eq(group)].iloc[0]
                    check(f'{sample}/{radius}/{group}:join_count',abs(r.same_tail_neighbor_fraction-ratio))
                    check(f'{sample}/{radius}/{group}:components',abs(r.components-len(sizes)))
                    check(f'{sample}/{radius}/{group}:largest_component',abs(r.largest_component_fraction-max(sizes)/use.sum()))
                    if si==0:
                        quart=v<=np.percentile(v,25) if group=='lower' else v>=np.percentile(v,75)
                        rng=np.random.default_rng(20260920);values=[]
                        for _ in range(199):
                            draw=np.zeros(n,bool);draw[rng.choice(np.flatnonzero(quart),int(use.sum()),replace=False)]=True
                            values.append(2*np.sum(draw[pairs[:,0]]&draw[pairs[:,1]])/degrees[draw].sum())
                        r=reference[reference['sample'].eq(sample)&reference.radius_um.eq(radius)&reference.group.eq(group)&reference.metric.eq('same_tail_neighbor_fraction')].iloc[0]
                        check(f'{sample}/{radius}/{group}:density_reference_mean',abs(np.mean(values)-r.thinned_q25_mean))
                        check(f'{sample}/{radius}/{group}:density_reference_p975',abs(np.quantile(values,.975)-r.thinned_q25_p975))
        # Direct section-to-unit aggregation for every effect/threshold.
        keys=['family','unit','pathway','grouping','tail','adjustment','outcome'];cols=['effect','standardized_effect','section_standardized_effect']
        expected=new.groupby(keys)[cols].mean().sort_index();got=pd.read_csv(dest/'unit_program_effects.csv').set_index(keys).sort_index()
        assert expected.index.equals(got.index)
        check(family+':all_unit_effects',np.nanmax(np.abs(expected.to_numpy()-got[cols].to_numpy())))
        print(family,'independent tail/graph checks passed',flush=True)
    (OUT/'independent_checks.json').write_text(json.dumps(dict(status='pass',n_checks=len(checks),checks=checks),indent=2)+'\n')


if __name__=='__main__':main()

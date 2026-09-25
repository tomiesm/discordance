"""Whole-profile grouping reference; never treat section pairs as independent."""
from itertools import combinations,product
from math import factorial
from common import *

def partitions(items,sizes):
    items=tuple(sorted(items));sizes=tuple(sorted(sizes))
    if len(sizes)==1:
        assert len(items)==sizes[0];yield (items,);return
    size=sizes[0]
    # If all sizes match, the smallest remaining item anchors the first group.
    # Otherwise groups of the smallest size are unique in our actual designs.
    if len(set(sizes))==1:
        options=((items[0],)+z for z in combinations(items[1:],size-1))
    else:
        assert sizes.count(size)==1
        options=combinations(items,size)
    for group in options:
        other=tuple(i for i in items if i not in group)
        for rest in partitions(other,sizes[1:]):yield (tuple(group),)+rest

def coefficients(groups,edges):
    lookup={tuple(e):j for j,e in enumerate(edges)};w=np.zeros(len(edges));b=np.zeros(len(edges));nw=sum(len(g)>1 for g in groups);nb=len(groups)*(len(groups)-1)//2
    for g in groups:
        if len(g)>1:
            for pair in combinations(sorted(g),2):w[lookup[pair]]+=1/(nw*(len(g)*(len(g)-1)/2))
    for one,two in combinations(groups,2):
        for a in one:
            for c in two:b[lookup[tuple(sorted((a,c)))]]+=1/(nb*len(one)*len(two))
    assert np.isclose(w.sum(),1) and np.isclose(b.sum(),1)
    return w,b

def direct_stat(matrix,groups):
    within=[np.mean([matrix[a,b] for a,b in combinations(g,2)]) for g in groups if len(g)>1]
    between=[np.mean([matrix[a,b] for a in g for b in h]) for g,h in combinations(groups,2)]
    return float(np.mean(within)),float(np.mean(between))

def main():
    allrows=[];pairrows=[];unitrows=[];loo=[];nulls={}
    for family in IDC:
        d=csv(AUDIT/'stage_10_decile_sensitivity'/family/'section_program_effects.csv');d=d[d['tail'].eq(.25)]
        meta=d[['sample','unit']].drop_duplicates().sort_values('sample');samples=meta['sample'].tolist();units=meta.unit.tolist();n=len(samples);edges=list(combinations(range(n),2));groups=tuple(tuple(i for i,u in enumerate(units) if u==p) for p in sorted(set(units)));sizes=sorted(map(len,groups));w,b=coefficients(groups,edges)
        allparts=list(partitions(range(n),sizes));expected=factorial(n)
        for k in sizes:expected//=factorial(k)
        for k in set(sizes):expected//=factorial(sizes.count(k))
        assert len(allparts)==expected
        assert len({tuple(sorted(tuple(sorted(g)) for g in x)) for x in allparts})==expected
        sources=['Janesick' if s.startswith('NCBI') else '10x_public' for s in samples]
        if family=='biomarkers':sources=['Biomarkers']*n
        sourceparts=[]
        for source in sorted(set(sources)):
            ix=[i for i,s in enumerate(sources) if s==source];sz=sorted(sum(units[i]==u for i in ix) for u in sorted(set(units[i] for i in ix)));sourceparts.append(list(partitions(ix,sz)))
        restricted=[tuple(g for block in combination for g in block) for combination in product(*sourceparts)]
        refs={'source_preserving':restricted}
        if family!='biomarkers':refs['cohort_only_sensitivity']=allparts
        check(f'{family}:partition_count',abs(len(restricted)-(15400 if family=='biomarkers' else 9)),0)
        coeff={name:np.stack([np.subtract(*coefficients(g,edges)) for g in pats]) for name,pats in refs.items()}
        config=[('full','unadjusted'),('program_excluded','unadjusted'),('program_excluded','overlap_adjusted')]
        for grouping,adjustment in config:
            for outcome in OUTCOMES:
                z=d[d.grouping.eq(grouping)&d.adjustment.eq(adjustment)&d.outcome.eq(outcome)];frame=z.pivot(index='sample',columns='pathway',values='standardized_effect').loc[samples];assert frame.notna().all().all();assert np.all(frame.std(axis=1)>0)
                c=np.corrcoef(frame.to_numpy());v=np.array([c[a,b] for a,b in edges]);wi=float(w@v);be=float(b@v);stat=wi-be;ident=dict(family=family,grouping=grouping,adjustment=adjustment,outcome=outcome,n_programs=frame.shape[1],n_sections=n,n_patients=len(groups))
                dw,db=direct_stat(c,groups);check(f'{family}/{grouping}/{adjustment}/{outcome}:direct_stat',max(abs(dw-wi),abs(db-be)))
                for a,k in edges:pairrows.append(ident|dict(sample1=samples[a],sample2=samples[k],patient1=units[a],patient2=units[k],same_patient=units[a]==units[k],pearson=float(c[a,k])))
                for patient in sorted(set(units)):
                    ix=[i for i,u in enumerate(units) if u==patient];pc=[c[a,k] for a,k in combinations(ix,2)];unitrows.append(ident|dict(patient=patient,n_sections_patient=len(ix),n_within_pairs=len(pc),within_mean=float(np.mean(pc)) if pc else np.nan))
                    remain=tuple(g for g in groups if units[g[0]]!=patient);lw,lb=direct_stat(c,remain);loo.append(ident|dict(omitted_patient=patient,within_mean=lw,between_mean=lb,difference=lw-lb))
                for ref,pats in refs.items():
                    vals=coeff[ref]@v;prob=float(np.mean(vals>=stat-1e-12));key='__'.join([family,grouping,adjustment,outcome,ref]);nulls[key]=vals
                    # Directly check sampled partition statistics and observed partition inclusion.
                    for i in [0,len(pats)//2,len(pats)-1]:
                        nw,nb=direct_stat(c,pats[i]);check(key+f':enumeration_{i}',abs(vals[i]-(nw-nb)))
                    assert np.min(abs(vals-stat))<1e-10
                    allrows.append(ident|dict(reference=ref,n_partitions=len(vals),within_mean=wi,between_mean=be,difference=stat,reference_p_one_sided=prob,null_mean=vals.mean(),null_q025=np.quantile(vals,.025),null_q975=np.quantile(vals,.975),primary=grouping=='program_excluded' and adjustment=='overlap_adjusted' and outcome=='observed' and ref=='source_preserving'))
        print(family,'patient profile comparison complete',flush=True)
    result=pd.DataFrame(allrows);result['primary_holm_p']=np.nan;idx=result[result.primary].sort_values('reference_p_one_sided').index;assert len(idx)==2
    vals=result.loc[idx,'reference_p_one_sided'].to_numpy();adj=np.minimum(1,np.maximum.accumulate(vals*np.array([2,1])));result.loc[idx,'primary_holm_p']=adj
    result.to_csv(OUT/'patient_profile_summary.csv',index=False);pd.DataFrame(pairrows).to_csv(OUT/'patient_profile_pairs.csv',index=False);pd.DataFrame(unitrows).to_csv(OUT/'patient_profile_patient_means.csv',index=False);pd.DataFrame(loo).to_csv(OUT/'patient_profile_leave_one_out.csv',index=False);np.savez_compressed(OUT/'patient_profile_reference_distributions.npz',**nulls);finish('profiles')

if __name__=='__main__':main()

"""Exploratory region-size scan with selection and localization-test correction."""
from spatial_test import *
from scipy.sparse.csgraph import connected_components


def component_sizes(coords, mask):
    points=coords[mask]
    if not len(points):return np.array([],dtype=int)
    pairs=cKDTree(points).query_pairs(100,output_type='ndarray')
    adjacency=sparse.csr_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(len(points),len(points)))
    _,labels=connected_components(adjacency,directed=False)
    return np.bincount(labels)


def main():
    frozen=HERE/'extent_frozen'
    frozen.mkdir(exist_ok=False)
    for name in ['extent_addendum.json','component_extent.py']:
        shutil.copy2(HERE/name,frozen/name)
    dump(frozen/'manifest.json',{'utc':datetime.now(timezone.utc).isoformat(),'sha256':{p.name:sha(p) for p in frozen.iterdir() if p.is_file()}})
    rows=[];regions=[]
    for si,(sample,patient) in enumerate(PROTOCOL['samples'].items()):
        obs=pd.read_parquet(OUT/f'{sample}_tumor_inputs.parquet')
        coords=obs[['x_um','y_um']].to_numpy(float)
        a,_=graph(coords,100);totals=np.asarray(a.sum(axis=1)).ravel()
        y=obs.tumor_tf_candidate.to_numpy(float)
        count=a@y
        observed_mask=(totals>=20)&(count>=5)&(count/totals>=.1)
        sizes=component_sizes(coords,observed_mask)
        original=obs.loc[obs.candidate_component_100um>=0].groupby('candidate_component_100um',observed=True).size()
        np.testing.assert_array_equal(np.sort(sizes),np.sort(original.to_numpy()))
        for ni,null in enumerate(['global','technical','context']):
            groups=[np.flatnonzero(obs[f'{null}_stratum'].to_numpy()==s) for s in sorted(obs[f'{null}_stratum'].unique())]
            positive=np.array([int(y[ix].sum()) for ix in groups])
            rng=np.random.default_rng(PROTOCOL['seed']+1000000*si+1000*100+10*ni)
            old=np.load(OUT/f'{sample}_100um_{null}_coexpression_null.npz')['statistics']
            maxima=np.empty(PROTOCOL['permutations'],int)
            for start in range(0,len(maxima),PROTOCOL['batch_size']):
                batch=min(PROTOCOL['batch_size'],len(maxima)-start)
                draws=permuted_marks(groups,positive,len(obs),batch,rng)
                count=a@draws
                eligible=(totals[:,None]>=20)&(count>=5)&(count/totals[:,None]>=.1)
                np.testing.assert_array_equal(eligible.sum(axis=0),old[start:start+batch,1])
                np.testing.assert_array_equal(.5*np.sum(draws*(count-draws),axis=0),old[start:start+batch,0])
                for b in range(batch):
                    sizes_null=component_sizes(coords,eligible[:,b])
                    maxima[start+b]=sizes_null.max() if len(sizes_null) else 0
            np.save(OUT/f'{sample}_100um_{null}_max_component_extent.npy',maxima)
            observed=int(sizes.max()) if len(sizes) else 0
            rows.append(dict(sample=sample,patient=patient,null=null,observed_maximum_centers=observed,
                null_mean=float(maxima.mean()),null_025=float(np.quantile(maxima,.025)),null_975=float(np.quantile(maxima,.975)),
                p_permutation=pvalue(maxima,observed)))
            for component,size in original.items():
                p=pvalue(maxima,int(size))
                regions.append(dict(sample=sample,patient=patient,null=null,component=int(component),n_centers=int(size),
                    p_extent_scan_section=p,p_extent_three_sections=min(1,3*p),p_extent_two_statistics_three_sections=min(1,6*p)))
            print('EXTENT',sample,null,'observed',observed,'null mean',maxima.mean(),'p',rows[-1]['p_permutation'],flush=True)
    table=pd.DataFrame(rows)
    table['p_holm_nine_extent_configurations']=holm(table.p_permutation.to_numpy())
    earlier=pd.read_csv(OUT/'statistics.csv')
    peak=earlier.loc[earlier.statistic.eq('maximum_local_z')].copy()
    combined=holm(np.r_[peak.p_permutation,table.p_permutation])
    peak['p_holm_all_39_localization_configurations']=combined[:len(peak)]
    table['p_holm_all_39_localization_configurations']=combined[len(peak):]
    peak.to_csv(OUT/'peak_localization_with_extent_multiplicity.csv',index=False)
    table.to_csv(OUT/'component_extent_statistics.csv',index=False)
    regions=pd.DataFrame(regions)
    regions.to_csv(OUT/'component_extent_evidence.csv',index=False)
    dump(OUT/'EXTENT_COMPLETE.json',{'status':'complete','configurations':9,'permutations_per_configuration':1999,
         'primary_pair_and_center_statistics_reproduced_for_every_draw':True})
    print('ALL EXTENT TESTS COMPLETE',flush=True)


if __name__=='__main__':main()

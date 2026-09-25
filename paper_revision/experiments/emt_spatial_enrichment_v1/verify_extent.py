"""Check component extents with independent hierarchical single linkage."""
from spatial_test import *
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from statsmodels.stats.multitest import multipletests


def extent(coords):
    if len(coords)<2:return len(coords)
    return int(np.bincount(fcluster(linkage(pdist(coords),method='single'),100,criterion='distance')).max())


def main():
    stats=pd.read_csv(OUT/'component_extent_statistics.csv')
    regions=pd.read_csv(OUT/'component_extent_evidence.csv')
    checked=0
    for si,sample in enumerate(PROTOCOL['samples']):
        obs=pd.read_parquet(OUT/f'{sample}_tumor_inputs.parquet')
        coords=obs[['x_um','y_um']].to_numpy(float)
        a,_=graph(coords,100);totals=np.asarray(a.sum(axis=1)).ravel()
        y=obs.tumor_tf_candidate.to_numpy(float)
        observed=extent(coords[obs.candidate_component_100um>=0])
        for ni,null in enumerate(['global','technical','context']):
            groups=[np.flatnonzero(obs[f'{null}_stratum'].to_numpy()==s) for s in sorted(obs[f'{null}_stratum'].unique())]
            positive=[int(y[ix].sum()) for ix in groups]
            rng=np.random.default_rng(PROTOCOL['seed']+1000000*si+1000*100+10*ni)
            draws=permuted_marks(groups,positive,len(obs),32,rng)
            counts=a@draws
            masks=(totals[:,None]>=20)&(counts>=5)&(counts/totals[:,None]>=.1)
            null_extent=np.load(OUT/f'{sample}_100um_{null}_max_component_extent.npy')
            for j in [0,3,11,19,31]:
                assert extent(coords[masks[:,j]])==null_extent[j]
                checked+=1
            row=stats.loc[stats['sample'].eq(sample)&stats.null.eq(null)].iloc[0]
            assert observed==row.observed_maximum_centers
            assert row.p_permutation==(1+int((null_extent>=observed).sum()))/2000
            for _,r in regions.loc[regions['sample'].eq(sample)&regions.null.eq(null)].iterrows():
                p=(1+int((null_extent>=r.n_centers).sum()))/2000
                np.testing.assert_allclose(p,r.p_extent_scan_section,atol=1e-14)
                np.testing.assert_allclose(min(1,6*p),r.p_extent_two_statistics_three_sections,atol=1e-14)
        print('EXTENT INDEPENDENT CHECK PASS',sample,flush=True)
    np.testing.assert_allclose(multipletests(stats.p_permutation,method='holm')[1],stats.p_holm_nine_extent_configurations,atol=1e-14)
    peaks=pd.read_csv(OUT/'peak_localization_with_extent_multiplicity.csv')
    expected=multipletests(np.r_[peaks.p_permutation,stats.p_permutation],method='holm')[1]
    np.testing.assert_allclose(expected,np.r_[peaks.p_holm_all_39_localization_configurations,stats.p_holm_all_39_localization_configurations],atol=1e-14)
    snapshot=json.loads((HERE/'extent_frozen/manifest.json').read_text())
    for name,digest in snapshot['sha256'].items():assert sha(HERE/'extent_frozen'/name)==digest
    dump(OUT/'EXTENT_VERIFICATION.json',{'status':'pass','independent_null_component_checks':checked,
        'method':'Hierarchical single linkage of all pair distances, independent of spatial adjacency connected-components algorithm',
        'observed_extents_verified':3,'all_component_pvalues_verified':len(regions),'multiplicity_verified':True})
    print('ALL EXTENT VERIFICATION PASSED',flush=True)


if __name__=='__main__':main()

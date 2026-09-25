"""Verify saved inference using direct distances and separate formulae."""
import os
for v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[v] = '4'
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
OUT = HERE/'results'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    protocol = json.loads((HERE/'protocol.json').read_text())
    table = pd.read_csv(OUT/'statistics.csv')
    assert len(table) == 90
    checked = []
    rng = np.random.default_rng(107)
    for sample in protocol['samples']:
        obs = pd.read_parquet(OUT/f'{sample}_tumor_inputs.parquet')
        coordinates = obs[['x_um', 'y_um']].to_numpy(float)
        original = pd.read_parquet(HERE.parent/f'emt_cells_v1/results/{sample}/cell_evidence.parquet').loc[obs.index]
        np.testing.assert_array_equal(obs.tumor_tf_candidate, original.tumor_tf_candidate)
        np.testing.assert_array_equal(coordinates, original[['x_um','y_um']])
        fixed = rng.choice(len(obs), 64, replace=False)
        for radius in protocol['radii_um']:
            edges = cKDTree(coordinates).query_pairs(radius, output_type='ndarray')
            sections = table.loc[table['sample'].eq(sample) & table.radius_um.eq(radius)]
            for (null, mark), frame in sections.groupby(['null','mark']):
                flag = obs['tumor_tf_candidate' if mark == 'coexpression' else 'tumor_nuclear_tf_candidate'].to_numpy(bool)
                direct_pairs = np.count_nonzero(pdist(coordinates[flag]) <= radius)
                pair_row = frame.loc[frame.statistic.eq('positive_pairs')].iloc[0]
                assert direct_pairs == pair_row.observed
                stratum = obs[f'{null}_stratum'].to_numpy(int)
                n = np.bincount(stratum)
                k = np.bincount(stratum, weights=flag.astype(int))
                probability = k/n
                left, right = stratum[edges[:,0]], stratum[edges[:,1]]
                expected_contribution = probability[left]*probability[right]
                same = left == right
                s = left[same]
                expected_contribution[same] = k[s]*(k[s]-1)/(n[s]*(n[s]-1))
                independent_expected = expected_contribution.sum()
                np.testing.assert_allclose(independent_expected, pair_row.expected_pairs_analytic, rtol=1e-12, atol=1e-8)
                tag = f'{sample}_{radius}um_{null}_{mark}'
                saved = np.load(OUT/f'{tag}_null.npz')
                assert saved['statistics'].shape == (1999,3)
                null_pairs = saved['statistics'][:,0]
                se = null_pairs.std(ddof=1)/np.sqrt(len(null_pairs))
                z_mean = abs(null_pairs.mean()-independent_expected)/se if se > 0 else 0
                assert z_mean < 6, (tag,z_mean)
                for j, statistic in enumerate(['positive_pairs','qualifying_centers','maximum_local_z']):
                    row = frame.loc[frame.statistic.eq(statistic)].iloc[0]
                    observed = saved['observed'][j]
                    null_values = saved['statistics'][:,j]
                    expected_p = (1+sum(float(t) >= observed-(1e-10 if j == 2 else 0) for t in null_values))/2000
                    np.testing.assert_allclose(row.p_permutation, expected_p, atol=1e-14)
                if radius == 100 and mark == 'coexpression':
                    local = pd.read_parquet(OUT/f'{tag}_local.parquet').set_index('cell_id').loc[obs.index]
                    for i in fixed:
                        near = np.sum((coordinates-coordinates[i])**2,axis=1) <= radius**2
                        assert near.sum() == local.iloc[i].n_tumor_neighbors
                        assert np.count_nonzero(near & flag) == local.iloc[i].observed_candidates
                        sizes_in_ball = np.bincount(stratum[near], minlength=len(n))
                        mean = sum(int(m)*float(ks)/int(ns) for m,ks,ns in zip(sizes_in_ball,k,n))
                        var = sum(int(m)*(ks/ns)*(1-ks/ns)*(ns-m)/(ns-1) for m,ks,ns in zip(sizes_in_ball,k,n) if ns > 1)
                        np.testing.assert_allclose(mean, local.iloc[i].null_expected_candidates, atol=1e-10)
                        np.testing.assert_allclose(var, local.iloc[i].null_variance, atol=1e-10)
                        if local.iloc[i].eligible:
                            z = (local.iloc[i].observed_candidates-mean)/np.sqrt(var)
                            np.testing.assert_allclose(z,local.iloc[i].z,atol=1e-10)
                            p = (1+np.count_nonzero(saved['statistics'][:,2] >= z-1e-10))/2000
                            np.testing.assert_allclose(p,local.iloc[i].p_scan_section,atol=1e-12)
                    original_zones = original.candidate_component_100um.to_numpy() >= 0
                    repeated_zones = (local.n_tumor_neighbors.to_numpy() >= 20) & (local.observed_candidates.to_numpy() >= 5) & (local.observed_candidates/local.n_tumor_neighbors >= .1).to_numpy()
                    np.testing.assert_array_equal(repeated_zones, original_zones)
                checked.append(dict(configuration=tag, analytic_vs_mc_mean_standard_errors=float(z_mean)))
        print('INDEPENDENT SPATIAL CHECKS PASS',sample,flush=True)
    for statistic, frame in table.groupby('statistic'):
        np.testing.assert_allclose(multipletests(frame.p_permutation, method='holm')[1],frame.p_holm_all_configurations_per_statistic,atol=1e-12)
    primary = table.loc[table.p_holm_primary_three_sections.notna()]
    np.testing.assert_allclose(multipletests(primary.p_permutation,method='holm')[1],primary.p_holm_primary_three_sections,atol=1e-12)
    manifest = json.loads((HERE/'input_manifest.json').read_text())
    assert sha(HERE/'protocol.json') == manifest['protocol_sha256']
    for path, digest in manifest['files'].items():
        assert sha(path) == digest, path
    for name, digest in json.loads((HERE/'frozen/manifest.json').read_text()).items():
        assert sha(HERE/'frozen'/name) == digest
    originals = json.loads((PROJECT/'paper_revision/original_snapshot_manifest.json').read_text())
    for root, entries in originals.items():
        for name, item in entries.items():
            assert sha(PROJECT/root/name) == item['sha256']
    old_outputs = json.loads((HERE.parent/'emt_cells_v1/output_manifest.json').read_text())
    for name, item in old_outputs.items():
        assert sha(HERE.parent/'emt_cells_v1'/name) == item['sha256']
    result = dict(status='pass',configurations_checked=len(checked),
        check_details=checked,local_neighborhood_checks=64*3*3,
        original_files_unchanged={root:len(entries) for root,entries in originals.items()},
        previous_experiment_outputs_unchanged=len(old_outputs),
        checks=['positive pairs by exhaustive pair distances','analytic expected pairs by direct edge probabilities',
            'Monte Carlo means compatible with exact expectations','local hypergeometric moments by direct sums',
            'local maximum-statistic p-values','original neighborhood rule reproduced','Holm against statsmodels',
            'source/input/protocol hashes'])
    (OUT/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print('ALL INDEPENDENT CHECKS PASS',flush=True)


if __name__ == '__main__':
    main()

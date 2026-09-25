"""Conditional random-labeling tests on fixed source-tumor cell positions."""
from src.paper.paths import PROJECT_ROOT, STAGE_SOURCE, CELL_SOURCE, ANALYSIS_ROOT, CELL_ROOT, stage_dir, cell_dir
import os
for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[name] = '4'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json
import hashlib
import shutil
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from scipy.spatial import cKDTree
HERE = cell_dir('emt_spatial_enrichment_v1')
PROJECT = PROJECT_ROOT
PREVIOUS = HERE.parent / 'emt_cells_v1'
OUT = HERE / 'results'
PROTOCOL = json.loads((CELL_SOURCE / 'emt_spatial_enrichment_v1' / 'protocol.json').read_text())

def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def split_groups(groups, values, bins, minimum=30):
    result = []
    for ix in groups:
        if len(ix) < bins * minimum:
            result.append(ix)
            continue
        cuts = np.unique(np.quantile(values[ix], np.arange(1, bins) / bins))
        labels = np.searchsorted(cuts, values[ix], side='right')
        children = [ix[labels == c] for c in np.unique(labels)]
        result.extend(children if min(map(len, children)) >= minimum else [ix])
    return result

def make_strata(obs):
    groups = [np.flatnonzero(obs.source_label.to_numpy() == label) for label in sorted(obs.source_label.unique())]
    for column, bins in [('nonmarker_transcripts', 5), ('cell_area', 3), ('nonmarker_detected_genes', 3)]:
        groups = split_groups(groups, obs[column].to_numpy(float), bins)
    technical = groups
    for column, bins in [('local_stromal_fraction', 3), ('local_myoepithelial_fraction', 2)]:
        groups = split_groups(groups, obs[column].to_numpy(float), bins)
    return {'global': [np.arange(len(obs))], 'technical': technical, 'context': groups}

def graph(coords, radius):
    pairs = cKDTree(coords).query_pairs(radius, output_type='ndarray')
    n = len(coords)
    rows = np.r_[pairs[:, 0], pairs[:, 1], np.arange(n)]
    cols = np.r_[pairs[:, 1], pairs[:, 0], np.arange(n)]
    a = sparse.csr_matrix((np.ones(len(rows), dtype=np.float64), (rows, cols)), shape=(n, n))
    return (a, pairs)

def null_moments(a, groups, y):
    n = len(y)
    labels = np.empty(n, dtype=int)
    sizes = np.array([len(ix) for ix in groups], dtype=int)
    positives = np.array([int(y[ix].sum()) for ix in groups], dtype=int)
    for j, ix in enumerate(groups):
        labels[ix] = j
    membership = sparse.csr_matrix((np.ones(n), (np.arange(n), labels)), shape=(n, len(groups)))
    counts = (a @ membership).tocsr()
    p = positives / sizes
    mean = np.asarray(counts @ p).ravel()
    factor = np.divide(p * (1 - p), sizes - 1, out=np.zeros_like(p), where=sizes > 1)
    variance = np.asarray(counts @ (factor * sizes) - counts.multiply(counts) @ factor).ravel()
    variance = np.maximum(variance, 0)
    node_p = p[labels]
    expected_pairs = 0.5 * (node_p @ (a @ node_p) - node_p @ node_p)
    internal_degree = np.asarray(counts[np.arange(n), labels]).ravel() - 1
    internal_pairs = 0.5 * np.bincount(labels, weights=internal_degree, minlength=len(groups))
    pair_probability = np.divide(positives * (positives - 1), sizes * (sizes - 1), out=np.zeros(len(groups)), where=sizes > 1)
    expected_pairs += np.sum(internal_pairs * (pair_probability - p * p))
    return (mean, variance, float(expected_pairs), sizes, positives)

def permuted_marks(groups, positives, n, batch, rng):
    draws = np.zeros((n, batch), dtype=np.float64)
    for ix, k in zip(groups, positives):
        if k == 0:
            continue
        if k == len(ix):
            draws[ix, :] = 1
        else:
            for b in range(batch):
                draws[rng.choice(ix, int(k), replace=False), b] = 1
    return draws

def summarize_stats(a, y, totals, mean, variance):
    count = np.asarray(a @ y)
    if y.ndim == 1:
        y = y[:, None]
        count = count[:, None]
    pairs = 0.5 * np.sum(y * (count - y), axis=0)
    qualifying = ((totals[:, None] >= 20) & (count >= 5) & (count / totals[:, None] >= 0.1)).sum(axis=0)
    eligible = (totals >= 20) & (variance > 1e-12)
    z = np.full(count.shape, -np.inf)
    z[eligible] = (count[eligible] - mean[eligible, None]) / np.sqrt(variance[eligible, None])
    maxima = z.max(axis=0)
    return (count, np.column_stack([pairs, qualifying, maxima]), z)

def pvalue(null, observed, tolerance=0):
    return float((1 + np.count_nonzero(null >= observed - tolerance)) / (len(null) + 1))

def holm(values):
    values = np.asarray(values, float)
    order = np.argsort(values)
    result = np.empty(len(values))
    result[order] = np.minimum(1, np.maximum.accumulate(values[order] * (len(values) - np.arange(len(values)))))
    return result

def prepare(sample):
    all_cells = pd.read_parquet(PREVIOUS / f'results/{sample}/cell_evidence.parquet')
    use = all_cells.qc_pass & all_cells.source_group.eq('Tumor')
    obs = all_cells.loc[use].copy()
    matrix = ad.read_h5ad(PREVIOUS / f'results/{sample}/measured_cells.h5ad')
    np.testing.assert_array_equal(matrix.obs_names, all_cells.index)
    previous_protocol = json.loads((CELL_SOURCE / 'emt_cells_v1/protocol.json').read_text())
    excluded = previous_protocol['epithelial_genes'] + previous_protocol['canonical_emt_tfs']
    keep = ~np.isin(matrix.var_names, excluded)
    counts = matrix.X[use.to_numpy()][:, keep].tocsr()
    obs['nonmarker_transcripts'] = np.asarray(counts.sum(axis=1)).ravel()
    obs['nonmarker_detected_genes'] = np.asarray((counts > 0).sum(axis=1)).ravel()
    coords = obs[['x_um', 'y_um']].to_numpy(float)
    all_qc = all_cells.loc[all_cells.qc_pass]
    total_local = cKDTree(all_qc[['x_um', 'y_um']].to_numpy(float)).query_ball_point(coords, 100, return_length=True, workers=4)
    obs['local_stromal_fraction'] = obs.stromal_neighbors_100um.to_numpy() / total_local
    obs['local_myoepithelial_fraction'] = obs.myoepithelial_neighbors_100um.to_numpy() / total_local
    obs.to_parquet(OUT / f'{sample}_tumor_inputs.parquet')
    return (obs, coords)

def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    component_rows = []
    balance_rows = []
    configurations = []
    for sample_index, (sample, patient) in enumerate(PROTOCOL['samples'].items()):
        obs, coords = prepare(sample)
        groups_by_null = make_strata(obs)
        stratum_summary = []
        for null, groups in groups_by_null.items():
            labels = np.empty(len(obs), int)
            for j, ix in enumerate(groups):
                labels[ix] = j
            obs[f'{null}_stratum'] = labels
            stratum_summary.append(dict(null=null, n_strata=len(groups), minimum_size=min(map(len, groups)), median_size=float(np.median(list(map(len, groups)))), maximum_size=max(map(len, groups))))
        obs.to_parquet(OUT / f'{sample}_tumor_inputs.parquet')
        dump(OUT / f'{sample}_strata.json', stratum_summary)
        covariate_names = ['log_nonmarker_transcripts', 'log_cell_area', 'nonmarker_detected_genes', 'local_stromal_fraction', 'local_myoepithelial_fraction']
        covariates = np.column_stack([np.log1p(obs.nonmarker_transcripts), np.log1p(obs.cell_area), obs.nonmarker_detected_genes, obs.local_stromal_fraction, obs.local_myoepithelial_fraction])
        for radius in PROTOCOL['radii_um']:
            a, pairs = graph(coords, radius)
            totals = np.asarray(a.sum(axis=1)).ravel()
            np.testing.assert_array_equal(totals, obs[f'tumor_neighbors_{radius}um'])
            main_y = obs.tumor_tf_candidate.to_numpy(float)
            np.testing.assert_array_equal(a @ main_y, obs[f'tf_candidate_neighbors_{radius}um'])
            for null_index, (null, groups) in enumerate(groups_by_null.items()):
                for mark in ['coexpression', 'nuclear'] if radius == 100 and null == 'technical' else ['coexpression']:
                    tag = f'{sample}_{radius}um_{null}_{mark}'
                    path = OUT / f'{tag}_null.npz'
                    y = obs['tumor_tf_candidate' if mark == 'coexpression' else 'tumor_nuclear_tf_candidate'].to_numpy(float)
                    mean, variance, expected_pairs, sizes, positive = null_moments(a, groups, y)
                    counts, observed_array, observed_z = summarize_stats(a, y, totals, mean, variance)
                    observed = observed_array[0]
                    configuration = dict(sample=sample, patient=patient, radius_um=radius, null=null, mark=mark, n_cells=len(obs), n_positive=int(y.sum()), n_strata=len(groups), n_positive_in_frozen_strata=int(positive[positive == sizes].sum()), expected_pairs_analytic=expected_pairs)
                    configurations.append(configuration)
                    seed = PROTOCOL['seed'] + 1000000 * sample_index + 1000 * radius + 10 * null_index + (mark == 'nuclear')
                    rng = np.random.default_rng(seed)
                    b = PROTOCOL['permutations']
                    null_stats = np.empty((b, 3))
                    balance = np.empty((b, len(covariate_names)))
                    for start in range(0, b, PROTOCOL['batch_size']):
                        batch = min(PROTOCOL['batch_size'], b - start)
                        draws = permuted_marks(groups, positive, len(obs), batch, rng)
                        np.testing.assert_array_equal(draws.sum(axis=0), np.full(batch, y.sum()))
                        if start == 0:
                            for ix, k in zip(groups, positive):
                                np.testing.assert_array_equal(draws[ix].sum(axis=0), np.full(batch, k))
                        _, statistics, _ = summarize_stats(a, draws, totals, mean, variance)
                        null_stats[start:start + batch] = statistics
                        balance[start:start + batch] = draws.T @ covariates / y.sum()
                    np.savez_compressed(path, statistics=null_stats, observed=observed, covariate_means=balance)
                    for j, name in enumerate(['positive_pairs', 'qualifying_centers', 'maximum_local_z']):
                        values = null_stats[:, j]
                        row = dict(**configuration, statistic=name, observed=float(observed[j]), null_mean=float(values.mean()), null_sd=float(values.std(ddof=1)), null_025=float(np.quantile(values, 0.025)), null_975=float(np.quantile(values, 0.975)), observed_over_null_mean=float(observed[j] / values.mean()) if values.mean() > 0 else None, p_permutation=pvalue(values, observed[j], 1e-10 if j == 2 else 0))
                        rows.append(row)
                    if radius == 100:
                        for j, name in enumerate(covariate_names):
                            balance_rows.append(dict(**configuration, covariate=name, observed_positive_mean=float(y @ covariates[:, j] / y.sum()), null_mean=float(balance[:, j].mean()), null_025=float(np.quantile(balance[:, j], 0.025)), null_975=float(np.quantile(balance[:, j], 0.975))))
                    if radius == 100 and mark == 'coexpression':
                        z = observed_z[:, 0]
                        ordered = np.sort(null_stats[:, 2])
                        pscan = (1 + b - np.searchsorted(ordered, z - 1e-10, side='left')) / (b + 1)
                        eligible = (totals >= 20) & (variance > 1e-12)
                        pscan[~eligible] = 1
                        local = pd.DataFrame({'cell_id': obs.index, 'x_um': coords[:, 0], 'y_um': coords[:, 1], 'n_tumor_neighbors': totals.astype(int), 'observed_candidates': counts[:, 0].astype(int), 'null_expected_candidates': mean, 'null_variance': variance, 'z': z, 'p_scan_section': pscan, 'p_scan_three_sections': np.minimum(1, 3 * pscan), 'old_component': obs.candidate_component_100um.to_numpy(), 'eligible': eligible})
                        local.to_parquet(OUT / f'{tag}_local.parquet', index=False)
                        for component in sorted(set(local.old_component) - {-1}):
                            subset = local.loc[local.old_component == component]
                            best = subset.loc[subset.z.idxmax()]
                            component_rows.append(dict(sample=sample, patient=patient, null=null, component=int(component), n_original_centers=len(subset), max_z=float(best.z), p_scan_section=float(best.p_scan_section), p_scan_three_sections=float(best.p_scan_three_sections), n_centers_scan_p_le_005=int((subset.p_scan_three_sections <= 0.05).sum()), best_center_cell_id=str(best.cell_id), x_um=float(best.x_um), y_um=float(best.y_um)))
                    print('FINISHED', tag, 'pair ratio', observed[0] / expected_pairs, 'p', rows[-3]['p_permutation'], flush=True)
                    pd.DataFrame(rows).to_csv(OUT / 'statistics_partial.csv', index=False)
    table = pd.DataFrame(rows)
    table['p_holm_all_configurations_per_statistic'] = np.nan
    for statistic, frame in table.groupby('statistic'):
        table.loc[frame.index, 'p_holm_all_configurations_per_statistic'] = holm(frame.p_permutation.to_numpy())
    primary = (table.radius_um == 100) & table.null.eq('technical') & table.mark.eq('coexpression') & table.statistic.eq('positive_pairs')
    table['p_holm_primary_three_sections'] = np.nan
    table.loc[primary, 'p_holm_primary_three_sections'] = holm(table.loc[primary, 'p_permutation'].to_numpy())
    table.to_csv(OUT / 'statistics.csv', index=False)
    pd.DataFrame(component_rows).to_csv(OUT / 'component_scan_evidence.csv', index=False)
    pd.DataFrame(balance_rows).to_csv(OUT / 'covariate_balance.csv', index=False)
    dump(OUT / 'configurations.json', configurations)
    dump(OUT / 'ANALYSIS_COMPLETE.json', {'status': 'complete', 'utc': datetime.now(timezone.utc).isoformat(), 'configurations': len(configurations), 'permutations_per_configuration': PROTOCOL['permutations'], 'primary_rows': table.loc[primary].dropna(axis=1).to_dict('records')})
    print('ALL SPATIAL TESTS COMPLETE', flush=True)
if __name__ == '__main__':
    main()

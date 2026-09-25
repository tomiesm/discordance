"""Raw/conditional disjoint-gene stability and spatial baseline comparisons."""
from src.paper.paths import PROJECT_ROOT, STAGE_SOURCE, CELL_SOURCE, ANALYSIS_ROOT, CELL_ROOT, stage_dir, cell_dir
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
import yaml
ROOT = PROJECT_ROOT
REPO = PROJECT_ROOT
OUT = stage_dir('reliability')
from src.discordance import compute_conditional_discordance
from src.spatial import build_spatial_weights, morans_i

def compare(a, b):
    qa = a >= np.quantile(a, 0.75)
    qb = b >= np.quantile(b, 0.75)
    return {'spearman': float(spearmanr(a, b).statistic), 'Q4_overlap_fraction': float(np.sum(qa & qb) / np.sum(qa)), 'Q4_jaccard': float(np.sum(qa & qb) / np.sum(qa | qb))}

def main():
    cfg = yaml.safe_load((REPO / 'config.yaml').read_text())
    gate = json.loads((REPO / 'outputs/phase2/gate2_3_dual_track.json').read_text())
    spatial_gate = json.loads((REPO / 'outputs/phase2/gate2_2_spatial.json').read_text())
    cov = pd.read_csv(OUT.parent / 'coverage/coverage_summary.csv').set_index('sample')
    scores = pd.read_parquet(OUT.parent / 'scores/spot_score_diagnostics.parquet').set_index('spot_id')
    checks, rows, spatial_rows, graph_rows, partitions = ([], [], [], [], {})
    for ck, cc in cfg['cohorts'].items():
        cohort = cc['name']
        base = REPO / 'outputs/predictions' / cohort
        ids = []
        ys = []
        for fold in range(4):
            fd = base / 'uni/ridge' / f'fold{fold}'
            ids += json.loads((fd / 'test_spot_ids.json').read_text())
            ys.append(np.load(fd / 'test_targets.npy'))
        y = np.concatenate(ys)
        sc = scores.loc[ids]
        indices = {sid: np.flatnonzero(sc.sample_id.to_numpy() == sid) for sid in cc['samples']}
        genes = json.loads((REPO / 'data/v3' / f'gene_list_{cohort}.json').read_text())
        n = y.shape[1]
        perm = np.random.RandomState(42).permutation(n)
        ga, gb = (perm[:n // 2], perm[n // 2:])
        ridge = []
        for ec in cfg['encoders']:
            for rc in cfg['regressors']:
                name = ec['name'] + '_' + rc['name']
                residual = np.concatenate([np.load(base / ec['name'] / rc['name'] / f'fold{fold}' / 'test_residuals.npy') for fold in range(4)])
                a = np.abs(residual[:, ga]).mean(axis=1)
                b = np.abs(residual[:, gb]).mean(axis=1)
                for sid, ix in indices.items():
                    got = float(spearmanr(a[ix], b[ix]).statistic)
                    ref = gate['cohorts'][ck]['configs'][name]['samples'][sid]['rho']
                    checks.append({'check': f'{cohort}/{name}/{sid}:original_raw_half', 'max_abs': abs(got - ref), 'pass': abs(got - ref) < 1e-10})
                if rc['name'] == 'ridge':
                    ridge.append(np.abs(residual).astype(float))
                del residual
        mean_abs = np.mean(ridge, axis=0)
        del ridge
        baseline = []
        for fold in range(4):
            train = np.concatenate([v for k, v in enumerate(ys) if k != fold])
            median = np.median(train, axis=0).astype(float)
            baseline.append(np.abs(ys[fold].astype(float) - median))
        baseline = np.concatenate(baseline)
        np.testing.assert_allclose(baseline.mean(axis=1), sc.baseline_raw, atol=2e-07, rtol=0)
        total_full = y.sum(axis=1)
        for seed in range(42, 62):
            ix = np.random.RandomState(seed).permutation(n)
            aidx, bidx = (ix[:n // 2], ix[n // 2:])
            partitions[f'{cohort}/{seed}'] = {'A': [genes[i] for i in aidx], 'B': [genes[i] for i in bidx]}
            sa = y[:, aidx].sum(axis=1)
            sb = y[:, bidx].sum(axis=1)
            for source, errors in [('ridge_encoder_mean', mean_abs), ('median_baseline', baseline)]:
                a = errors[:, aidx].mean(axis=1)
                b = errors[:, bidx].mean(axis=1)
                versions = {'raw': (a, b), 'conditional_own_half': (compute_conditional_discordance(a, sa), compute_conditional_discordance(b, sb)), 'conditional_shared_full': (compute_conditional_discordance(a, total_full), compute_conditional_discordance(b, total_full))}
                for mode, (va, vb) in versions.items():
                    for sid, idx in indices.items():
                        rows.append({'cohort': cohort, 'patient': sc.iloc[idx[0]].patient, 'sample': sid, 'seed': seed, 'source': source, 'mode': mode, **compare(va[idx], vb[idx])})
        print(cohort, 'half-gene checks complete', flush=True)
        for sid, idx in indices.items():
            df = pd.read_parquet(REPO / 'outputs/phase2/scores' / cohort / f'{sid}_discordance.parquet').set_index('spot_id')
            v = scores.loc[df.index]
            xy = df[['x', 'y']].to_numpy()
            coords = xy * cov.loc[sid, 'pixel_size_um']
            w = build_spatial_weights(xy, n_neighbors=6)
            wr, wc = w.nonzero()
            distance = np.linalg.norm(coords[wr] - coords[wc], axis=1)
            pairs = cKDTree(coords).query_pairs(150, output_type='ndarray')
            radius = sparse.csr_matrix((np.ones(2 * len(pairs)), (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]])), shape=w.shape)
            sums = np.asarray(radius.sum(axis=1)).ravel()
            radius = sparse.diags(1 / np.maximum(sums, 1)) @ radius
            identity = {'cohort': cohort, 'patient': v.patient.iloc[0], 'sample': sid}
            graph_rows.append({**identity, 'n_spots': len(v), 'k6_edge_median_um': np.median(distance), 'k6_edge_max_um': distance.max(), 'k6_edge_fraction_gt_150um': np.mean(distance > 150), 'k6_connected_components': connected_components(w, directed=False, return_labels=False), 'radius_connected_components': connected_components(radius, directed=False, return_labels=False), 'radius_isolated_spots': int((sums == 0).sum())})
            for graph, weights in [('original_k6', w), ('radius_150um', radius)]:
                for metric in ['raw', 'conditional', 'baseline_conditional', 'section_centered_diagnostic', 'total_expr', 'panel_genes_detected']:
                    value = v[metric].to_numpy(dtype=np.float64)
                    obs = morans_i(value, weights)
                    z = value - value.mean()
                    coo = weights.tocoo()
                    direct = len(z) / weights.sum() * np.sum(z[coo.row] * z[coo.col] * coo.data) / np.sum(z * z)
                    assert abs(obs - direct) < 1e-12
                    spatial_rows.append({**identity, 'graph': graph, 'score': metric, 'morans_i': obs})
                    if graph == 'original_k6' and metric == 'conditional':
                        ref = spatial_gate['cohorts'][ck]['samples'][sid]['morans_i']
                        checks.append({'check': f'{sid}:original_moran', 'max_abs': abs(obs - ref), 'pass': abs(obs - ref) < 1e-12})
        del mean_abs, baseline, y, ys
    assert all((c['pass'] for c in checks))
    pd.DataFrame(rows).to_csv(OUT / 'gene_half_stability.csv', index=False)
    pd.DataFrame(spatial_rows).to_csv(OUT / 'spatial_structure.csv', index=False)
    pd.DataFrame(graph_rows).to_csv(OUT / 'graph_diagnostics.csv', index=False)
    (OUT / 'gene_partitions.json').write_text(json.dumps(partitions, indent=2) + '\n')
    (OUT / 'checks.json').write_text(json.dumps({'status': 'pass', 'checks': checks}, indent=2) + '\n')
if __name__ == '__main__':
    main()

"""Audit existing UNI morphology matching without changing matches or DE outputs."""
from src.paper.paths import PROJECT_ROOT, STAGE_SOURCE, CELL_SOURCE, ANALYSIS_ROOT, CELL_ROOT, stage_dir, cell_dir
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import yaml
ROOT = PROJECT_ROOT
REPO = PROJECT_ROOT
OUT = stage_dir('matching')

def main():
    cfg = yaml.safe_load((REPO / 'config.yaml').read_text())
    cov = pd.read_parquet(OUT.parent / 'scores/spot_score_diagnostics.parquet').set_index('spot_id')
    rows, balances, checks = ([], [], [])
    for cc in cfg['cohorts'].values():
        cohort = cc['name']
        for sid in cc['samples']:
            df = pd.read_parquet(REPO / 'outputs/phase2/scores' / cohort / f'{sid}_discordance.parquet')
            score = df[[f"D_cond_{e['name']}_ridge" for e in cfg['encoders']]].mean(axis=1).to_numpy()
            low = np.flatnonzero(score <= np.quantile(score, 0.25))
            high = np.flatnonzero(score >= np.quantile(score, 0.75))
            with h5py.File(REPO / 'outputs/embeddings' / sid / 'uni_embeddings.h5') as f:
                names = [b.decode() if isinstance(b, bytes) else str(b) for b in f['spot_ids'][:]]
                lookup = {sid + '_' + b: i for i, b in enumerate(names)}
                x = f['embeddings'][:][[lookup[b] for b in df.spot_id]]
            x = normalize(x, norm='l2', axis=1)
            conc = x[low]
            disc = x[high]
            rng = np.random.RandomState(42)
            sub = conc[rng.choice(len(conc), 1000, replace=False)] if len(conc) > 1000 else conc
            within = NearestNeighbors(n_neighbors=min(11, len(sub)), metric='euclidean').fit(sub).kneighbors(sub)[0]
            distances, neighbors = NearestNeighbors(n_neighbors=min(10, len(conc)), metric='euclidean').fit(conc).kneighbors(disc)
            features = cov.loc[df.spot_id]
            for k in [1, 5, 10]:
                tau = np.percentile(within[:, 1:k + 1].ravel(), 90)
                use = distances[:, 0] <= tau
                chosen = low[neighbors[use, :k]]
                retained = high[use]
                usage = np.bincount(chosen.ravel(), minlength=len(df))
                active = usage[usage > 0]
                identity = {'cohort': cohort, 'sample': sid, 'patient': features.patient.iloc[0], 'k': k}
                rows.append({**identity, 'n_Q4': len(high), 'n_Q1': len(low), 'retained_Q4': len(retained), 'retained_Q4_fraction': float(use.mean()), 'euclidean_caliper': float(tau), 'cosine_caliper_equivalent': float(tau * tau / 2), 'retained_pairs_beyond_caliper_fraction': float((distances[use, :k] > tau).mean()), 'retained_Q4_with_any_neighbor_beyond_caliper_fraction': float((distances[use, :k].max(axis=1) > tau).mean()), 'unique_controls': len(active), 'max_control_reuse': int(active.max()), 'control_usage_effective_n': float(active.sum() ** 2 / np.square(active.astype(float)).sum()), 'max_pair_distance': float(distances[use, :k].max()), 'embedding_mean_difference_norm_before': float(np.linalg.norm(disc.mean(axis=0) - conc.mean(axis=0))), 'embedding_mean_difference_norm_after': float(np.linalg.norm(x[retained].mean(axis=0) - x[chosen].mean(axis=(0, 1))))})
                for name in ['total_expr', 'panel_counts', 'panel_genes_detected', 'tissue_fraction']:
                    v = features[name].to_numpy(dtype=float)
                    scale = np.sqrt((v[high].var(ddof=1) + v[low].var(ddof=1)) / 2)
                    balances.append({**identity, 'covariate': name, 'unmatched_standardized_difference': float((v[high].mean() - v[low].mean()) / scale) if scale else np.nan, 'matched_standardized_difference': float((v[retained].mean() - v[chosen].mean()) / scale) if scale else np.nan})
                if k == 5:
                    old = json.loads((REPO / 'outputs/phase3/de' / cohort / 'per_sample' / f'{sid}_matching_quality.json').read_text())
                    delta = abs(tau - old['distance_threshold'])
                    checks.append({'check': sid, 'caliper_difference': float(delta), 'retained_count_matches': len(retained) == old['n_matched'], 'pass': bool(delta < 1e-05 and len(retained) == old['n_matched'])})
            print(cohort, sid, 'matching audit complete', flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'matching_diagnostics.csv', index=False)
    pd.DataFrame(balances).to_csv(OUT / 'matching_covariate_balance.csv', index=False)
    (OUT / 'matching_checks.json').write_text(json.dumps({'status': 'pass' if all((x['pass'] for x in checks)) else 'requires_review', 'checks': checks}, indent=2) + '\n')
    assert all((x['pass'] for x in checks))
if __name__ == '__main__':
    main()

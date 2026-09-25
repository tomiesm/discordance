"""Fixed extreme-tail quality, program, spatial and encoder comparisons."""
from src.paper.paths import PROJECT_ROOT, STAGE_SOURCE, CELL_SOURCE, ANALYSIS_ROOT, CELL_ROOT, stage_dir, cell_dir
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.stats import t
OUT = stage_dir('cutoffs')
AUDIT = ANALYSIS_ROOT
ROOT = PROJECT_ROOT
REPO = PROJECT_ROOT
from src.discordance import compute_conditional_discordance
from src.paper.genes.gene_audit import overlap_weights
FAMILIES = ['biomarkers', '10x_janesick', 'coad', 'idc_visium']
ENC = ['uni', 'virchow2', 'hoptimus0']
TAILS = [0.2, 0.25, 0.3]

def partition(v, tail):
    a, b = np.quantile(v, [tail, 1 - tail])
    assert a < b
    return (v <= a, v >= b, a, b)

def load_locations(family):
    if family in FAMILIES[:2]:
        diag = pd.read_parquet(AUDIT / 'scores/spot_score_diagnostics.parquet').set_index('spot_id')
        loc = pd.read_csv(AUDIT / 'programs/arrays' / family / 'locations.csv').rename(columns={'sample_id': 'sample', 'patient': 'unit'})
        ref = diag.loc[loc.spot_id]
        for k in ['raw', 'conditional', 'baseline_raw']:
            loc[k] = ref[k].to_numpy()
        for sample, ss in loc.groupby('sample'):
            f = pd.read_parquet(REPO / 'outputs/phase2/scores' / family / f'{sample}_discordance.parquet').set_index('spot_id').loc[ss.spot_id]
            for enc in ENC:
                loc.loc[ss.index, enc] = f[f'D_cond_{enc}_ridge'].to_numpy()
    else:
        base = AUDIT / 'external' / family
        loc = pd.read_parquet(base / 'scores.parquet').rename(columns={'specimen_group': 'unit'})
        metadata = pd.read_csv(AUDIT / 'coverage/coverage_summary.csv').set_index('sample')
        scale = loc['sample'].map(metadata.pixel_size_um).to_numpy()
        assert np.isfinite(scale).all()
        loc['x_um'] = loc.x * scale
        loc['y_um'] = loc.y * scale
        registry = pd.read_csv(base / 'prediction_registry.csv')
        g = len(json.loads((REPO / 'outputs' / family / 'gene_panel.json').read_text()))
        y = np.empty((len(loc), g), np.float32)
        raw = {enc: np.zeros(len(loc)) for enc in ENC}
        for row in registry.itertuples():
            fd = Path(row.prediction_source)
            ids = json.loads((fd / 'test_spot_ids.json').read_text())
            ii = np.flatnonzero(loc['sample'].eq(row.sample))
            order = pd.Index(ids).get_indexer(loc.iloc[ii].spot_id)
            assert (order >= 0).all()
            yy = np.load(fd / 'test_targets.npy', mmap_mode='r')
            pp = np.load(fd / 'test_predictions.npy', mmap_mode='r')
            for start in range(0, len(ii), 512):
                dest = ii[start:start + 512]
                ix = order[start:start + 512]
                a = yy[ix]
                if row.encoder == 'uni':
                    y[dest] = a
                else:
                    assert np.array_equal(y[dest], a)
                raw[row.encoder][dest] = np.abs(a.astype(float) - pp[ix].astype(float)).mean(axis=1)
        for enc in ENC:
            loc[enc] = compute_conditional_discordance(raw[enc], y.sum(axis=1))
        assert np.max(np.abs(loc[ENC].mean(axis=1) - loc.conditional)) < 1e-09
        assert np.max(np.abs(np.mean(list(raw.values()), axis=0) - loc.raw)) < 1e-09

        def baseline(group):
            test = np.flatnonzero(loc.unit.eq(group))
            train = loc.unit.ne(group).to_numpy()
            b = np.zeros(len(test))
            for start in range(0, g, 128):
                z = y[:, start:start + 128].astype(float)
                med = np.median(z[train], axis=0)
                b += np.abs(z[test] - med).sum(axis=1)
            return (test, b / g)
        loc['baseline_raw'] = np.nan
        with ThreadPoolExecutor(max_workers=4) as pool:
            for ix, b in pool.map(baseline, loc.unit.unique()):
                loc.loc[ix, 'baseline_raw'] = b
        print(family, 'per-spot baselines complete', flush=True)
    loc['family'] = family
    assert loc.spot_id.is_unique
    assert np.isfinite(loc[['raw', 'conditional', 'baseline_raw', 'x_um', 'y_um']].to_numpy()).all()
    assert np.max(np.abs(loc[ENC].mean(axis=1) - loc.conditional)) < 2e-06
    return loc

def quality_and_stability(loc):
    rows = []
    profiles = []
    agree = []
    for sample, ss in loc.groupby('sample', sort=False):
        score = ss.conditional.to_numpy()
        ident = dict(family=ss.family.iloc[0], sample=sample, unit=ss.unit.iloc[0])
        for tail in TAILS:
            lo, hi, a, b = partition(score, tail)
            for name, mask in [('lower', lo), ('upper', hi)]:
                z = ss.iloc[np.flatnonzero(mask)]
                mae = z.raw.mean()
                base = z.baseline_raw.mean()
                rows.append(dict(**ident, tail=tail, group=name, n=len(z), fraction=len(z) / len(ss), cut_low=a, cut_high=b, mean_conditional=z.conditional.mean(), mae=mae, baseline_mae=base, relative_mae_gain=1 - mae / base))
            for aenc, benc in combinations(ENC, 2):
                al, ah, _, _ = partition(ss[aenc], tail)
                bl, bh, _, _ = partition(ss[benc], tail)
                for name, one, two in [('lower', al, bl), ('upper', ah, bh)]:
                    k = int((one & two).sum())
                    expect = one.sum() * two.sum() / len(ss)
                    den = min(one.sum(), two.sum())
                    agree.append(dict(**ident, tail=tail, group=name, encoder1=aenc, encoder2=benc, overlap=k / den, jaccard=k / (one | two).sum(), chance_corrected_overlap=(k - expect) / (den - expect)))
    return (pd.DataFrame(rows), pd.DataFrame(profiles), pd.DataFrame(agree))

def graph_stats(adj, selected):
    k = int(selected.sum())
    sub = adj[selected][:, selected]
    deg = np.asarray(adj.sum(axis=1)).ravel()
    inside = float(sub.sum())
    available = float(deg[selected].sum())
    same = inside / available if available else np.nan
    nc, labels = connected_components(sub, directed=False)
    return dict(n_selected=k, components=int(nc), largest_component_fraction=float(np.bincount(labels).max() / k), isolated_selected_fraction=float(np.mean(np.asarray(sub.sum(axis=1)).ravel() == 0)), same_tail_neighbor_fraction=same, join_enrichment=same / ((k - 1) / (len(selected) - 1)))

def spatial(loc):
    rows = []
    references = []
    for sample, ss in loc.groupby('sample', sort=False):
        xy = ss[['x_um', 'y_um']].to_numpy()
        score = ss.conditional.to_numpy()
        n = len(ss)
        for radius in [100, 150]:
            pairs = cKDTree(xy).query_pairs(radius, output_type='ndarray')
            adj = sparse.csr_matrix((np.ones(2 * len(pairs)), (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]])), shape=(n, n))
            ident = dict(family=ss.family.iloc[0], sample=sample, unit=ss.unit.iloc[0], radius_um=radius)
            for tail in TAILS:
                lo, hi, _, _ = partition(score, tail)
                for name, selected in [('lower', lo), ('upper', hi)]:
                    rows.append(dict(**ident, tail=tail, group=name, graph_isolated=int(np.sum(np.asarray(adj.sum(axis=1)).ravel() == 0)), **graph_stats(adj, selected)))
    return (pd.DataFrame(rows), pd.DataFrame(references))

def programs(loc, family):
    if family in FAMILIES[:2]:
        folder = AUDIT / 'programs/arrays' / family
        ck = 'outside_total_counts'
        dk = 'outside_detected_genes'
    else:
        folder = AUDIT / 'external' / family / 'arrays'
        ck = 'outside_counts'
        dk = 'outside_detected'
    rows = []
    profiles = []
    members = []
    for path in sorted(folder.glob('HALLMARK_*.npz')):
        a = np.load(path)
        assert len(a['observed']) == len(loc)
        name = path.stem
        assert np.max(np.abs(a['full_conditional'] - loc.conditional.to_numpy())) < 2e-06
        vals = np.stack([a[o].astype(float) for o in ['observed', 'signed', 'absolute']], axis=1)
        for sample, ss in loc.groupby('sample', sort=False):
            ix = ss.index.to_numpy()
            v = vals[ix]
            section_sd = v.std(axis=0, ddof=1)
            ident = dict(family=family, sample=sample, unit=ss.unit.iloc[0], pathway=name)
            for grouping, score in [('full', ss.conditional.to_numpy()), ('program_excluded', a['program_excluded_conditional'][ix])]:
                for tail in TAILS:
                    lo, hi, lower, upper = partition(score, tail)
                    w1, w4, ns, ret1, ret4 = overlap_weights(a[ck][ix], a[dk][ix], lo, hi)
                    sd = np.sqrt(((lo.sum() - 1) * v[lo].var(axis=0, ddof=1) + (hi.sum() - 1) * v[hi].var(axis=0, ddof=1)) / (lo.sum() + hi.sum() - 2))
                    for adjustment, p, q in [('unadjusted', lo.astype(float), hi.astype(float))] + ([('overlap_adjusted', w1, w4)] if grouping == 'program_excluded' else []):
                        estimable = min(p.sum(), q.sum()) > 0
                        d = np.average(v, weights=q, axis=0) - np.average(v, weights=p, axis=0) if estimable else np.full(3, np.nan)
                        for k, outcome in enumerate(['observed', 'signed', 'absolute']):
                            rows.append(dict(**ident, grouping=grouping, tail=tail, adjustment=adjustment, outcome=outcome, effect=d[k], standardized_effect=d[k] / sd[k] if sd[k] > 0 else np.nan, section_standardized_effect=d[k] / section_sd[k] if section_sd[k] > 0 else np.nan, lower_mean=np.average(v[:, k], weights=p) if estimable else np.nan, upper_mean=np.average(v[:, k], weights=q) if estimable else np.nan, n_lower=int(lo.sum()), n_upper=int(hi.sum()), lower_retained=ret1 if adjustment == 'overlap_adjusted' else 1.0, upper_retained=ret4 if adjustment == 'overlap_adjusted' else 1.0, n_overlap_strata=ns, estimable=estimable))
                dec = np.searchsorted(np.quantile(score, np.arange(1, 10) / 10), score, side='left') + 1
    return (pd.DataFrame(rows), pd.DataFrame(profiles))

def summarize(df, dest):
    keys = ['family', 'unit', 'pathway', 'grouping', 'tail', 'adjustment', 'outcome']
    cols = ['effect', 'standardized_effect', 'section_standardized_effect', 'lower_retained', 'upper_retained']
    unit = df.groupby(keys)[cols].mean().reset_index()
    unit.to_csv(dest / 'unit_program_effects.csv', index=False)
    rows = []
    for scope in ['all', 'excluding_P07'] if df.family.iloc[0] == 'idc_visium' else ['all']:
        s = unit[unit.unit.ne('NCBI776')] if scope == 'excluding_P07' else unit
        for key, z in s.groupby([k for k in keys if k != 'unit']):
            for metric in ['effect', 'standardized_effect', 'section_standardized_effect']:
                v = z[metric].dropna().to_numpy()
                n = len(v)
                if not n:
                    continue
                mean = v.mean()
                se = v.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
                rows.append(dict(zip([k for k in keys if k != 'unit'], key)) | dict(scope=scope, metric=metric, mean=mean, n_units=n, n_positive=int((v > 0).sum()), n_negative=int((v < 0).sum()), ci_low=mean - t.ppf(0.975, n - 1) * se, ci_high=mean + t.ppf(0.975, n - 1) * se, loo_min=float(((v.sum() - v) / (n - 1)).min()) if n > 1 else np.nan, loo_max=float(((v.sum() - v) / (n - 1)).max()) if n > 1 else np.nan))
    pd.DataFrame(rows).to_csv(dest / 'cohort_program_summary.csv', index=False)

def main(family):
    dest = OUT / family
    dest.mkdir(parents=True, exist_ok=True)
    loc = load_locations(family)
    loc.to_parquet(dest / 'locations.parquet', index=False)
    q, d, e = quality_and_stability(loc)
    q.to_csv(dest / 'tail_quality.csv', index=False)
    e.to_csv(dest / 'encoder_stability.csv', index=False)
    effects, profiles = programs(loc, family)
    effects.to_csv(dest / 'section_program_effects.csv', index=False)
    summarize(effects, dest)
    s, r = spatial(loc)
    s.to_csv(dest / 'spatial_coherence.csv', index=False)
    (dest / 'COMPLETE.json').write_text(json.dumps(dict(status='complete', family=family, n_sections=loc['sample'].nunique(), n_programs=effects.pathway.nunique(), n_spots=len(loc)), indent=2) + '\n')
    print(family, 'COMPLETE', flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--family', choices=FAMILIES, required=True)
    main(p.parse_args().family)

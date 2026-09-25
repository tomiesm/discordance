"""Fixed-stratum overlap effects and fixed-group spatial-block uncertainty."""
from src.paper.paths import PROJECT_ROOT, STAGE_SOURCE, CELL_SOURCE, ANALYSIS_ROOT, CELL_ROOT, stage_dir, cell_dir
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest, t
from statsmodels.stats.multitest import multipletests
OUT = stage_dir('matching')
BIO = OUT.parent / 'programs'

def stratify(counts, detected, n=5):
    a = np.searchsorted(np.quantile(counts, np.linspace(0, 1, n + 1))[1:-1], counts, side='right')
    b = np.searchsorted(np.quantile(detected, np.linspace(0, 1, n + 1))[1:-1], detected, side='right')
    return np.unique(a * n + b, return_inverse=True)[1]

def aggregate(y, lo, hi, strata, blocks):
    nblocks = blocks.max() + 1
    ns = strata.max() + 1
    counts = []
    sums = []
    for mask in [lo, hi]:
        key = blocks[mask] * ns + strata[mask]
        counts.append(np.bincount(key, minlength=nblocks * ns).reshape(nblocks, ns))
        sums.append(np.stack([np.bincount(key, weights=y[mask, k], minlength=nblocks * ns).reshape(nblocks, ns) for k in range(y.shape[1])], axis=-1))
    return (counts, sums)

def estimates(counts, sums, weights, min_per_group=10):
    eligible = (counts[0].sum(axis=0) >= min_per_group) & (counts[1].sum(axis=0) >= min_per_group)
    c = [weights @ x for x in counts]
    s = [(weights @ x.reshape(len(x), -1)).reshape(len(weights), x.shape[1], x.shape[2]) for x in sums]
    raw = s[1].sum(axis=1) / c[1].sum(axis=1)[:, None] - s[0].sum(axis=1) / c[0].sum(axis=1)[:, None]
    overlap = np.divide(2 * c[0] * c[1], c[0] + c[1], out=np.zeros_like(c[0], dtype=float), where=c[0] + c[1] > 0) * eligible[None, :]
    mean = [np.divide(a, b[:, :, None], out=np.zeros_like(a), where=b[:, :, None] > 0) for a, b in zip(s, c)]
    denom = overlap.sum(axis=1)
    adjusted = np.divide(np.sum(overlap[:, :, None] * (mean[1] - mean[0]), axis=1), denom[:, None], out=np.full((len(weights), s[0].shape[-1]), np.nan), where=denom[:, None] > 0)
    return (raw, adjusted, eligible)

def main():
    results = []
    bootstrap_registry = []
    coverage = pd.read_csv(BIO / 'coverage.csv')
    for cohort, cp in coverage.groupby('cohort', sort=False):
        dest = BIO / 'arrays' / cohort
        locations = pd.read_csv(dest / 'locations.csv')
        cached = {}
        for sid, g in locations.groupby('sample_id', sort=False):
            xy = g[['x_um', 'y_um']].to_numpy()
            xy = xy - xy.min(axis=0)
            for width in [800, 400]:
                _, block = np.unique(np.floor(xy / width).astype(int), axis=0, return_inverse=True)
                n = block.max() + 1
                rng = np.random.default_rng(20260920)
                weights = rng.multinomial(n, np.full(n, 1 / n), size=500).astype(float)
                cached[sid, width] = (block, weights)
                bootstrap_registry.append({'cohort': cohort, 'sample': sid, 'width_um': width, 'occupied_blocks': int(n), 'n_bootstrap': 500})
        for row in cp[cp.eligible].itertuples():
            d = np.load(dest / f'{row.pathway}.npz')
            outcomes = [k for k in ['observed', 'signed', 'absolute', 'common_observed', 'common_signed', 'common_absolute'] if k in d]
            for sid, g in locations.groupby('sample_id', sort=False):
                ix = g.index.to_numpy()
                score = d['program_excluded_conditional'][ix]
                lo = score <= np.quantile(score, 0.25)
                hi = score >= np.quantile(score, 0.75)
                y = np.stack([d[k][ix].astype(float) for k in outcomes], axis=-1)
                sd = np.sqrt(((lo.sum() - 1) * y[lo].var(axis=0, ddof=1) + (hi.sum() - 1) * y[hi].var(axis=0, ddof=1)) / (lo.sum() + hi.sum() - 2))
                for nbins in [5, 10]:
                    strata = stratify(d['outside_total_counts'][ix], d['outside_detected_genes'][ix], nbins)
                    for width in [800, 400] if nbins == 5 else [800]:
                        block, weights = cached[sid, width]
                        counts, sums = aggregate(y, lo, hi, strata, block)
                        point_raw, point_adj, eligible = estimates(counts, sums, np.ones((1, block.max() + 1)))
                        if nbins == 5:
                            draws_raw, draws_adj, _ = estimates(counts, sums, weights)
                        retained = [float(c[:, eligible].sum() / c.sum()) for c in counts]
                        for j, outcome in enumerate(outcomes):
                            for mode, point, draws in [('unadjusted', point_raw, None if nbins != 5 else draws_raw), ('overlap_adjusted', point_adj, None if nbins != 5 else draws_adj)]:
                                if mode == 'unadjusted' and nbins != 5:
                                    continue
                                estimate = float(point[0, j])
                                ci = [np.nan, np.nan]
                                valid = 0
                                if draws is not None:
                                    values = draws[:, j]
                                    valid = int(np.isfinite(values).sum())
                                    if valid >= 475:
                                        ci = np.nanquantile(values, [0.025, 0.975]).tolist()
                                results.append({'cohort': cohort, 'sample': sid, 'patient': g.patient.iloc[0], 'pathway': row.pathway, 'outcome': outcome, 'adjustment': mode, 'bins_per_covariate': nbins, 'block_width_um': width, 'n_blocks': int(block.max() + 1), 'estimate': estimate, 'pooled_group_sd': float(sd[j]), 'standardized_estimate': estimate / sd[j] if sd[j] else np.nan, 'ci_low': ci[0], 'ci_high': ci[1], 'n_valid_bootstrap': valid, 'n_eligible_strata': int(eligible.sum()), 'Q1_retained_fraction': retained[0], 'Q4_retained_fraction': retained[1]})
            print(cohort, row.pathway, 'spatial/overlap complete', flush=True)
    r = pd.DataFrame(results)
    r.to_csv(OUT / 'section_inference.csv', index=False)
    pd.DataFrame(bootstrap_registry).to_csv(OUT / 'bootstrap_blocks.csv', index=False)
    primary = r[(r.bins_per_covariate == 5) & (r.block_width_um == 800)]
    keys = ['cohort', 'patient', 'pathway', 'outcome', 'adjustment']
    patient = primary.groupby(keys)[['estimate', 'standardized_estimate', 'Q1_retained_fraction', 'Q4_retained_fraction']].mean().reset_index()
    patient.to_csv(OUT / 'patient_effects.csv', index=False)
    summaries = []
    for key, g in patient.groupby(['cohort', 'pathway', 'outcome', 'adjustment']):
        for metric in ['estimate', 'standardized_estimate']:
            values = g[metric].dropna().to_numpy()
            n = len(values)
            mean = values.mean()
            se = values.std(ddof=1) / np.sqrt(n)
            nonzero = values[values != 0]
            p = binomtest(int((nonzero > 0).sum()), len(nonzero), 0.5).pvalue if len(nonzero) else 1.0
            loo = (values.sum() - values) / (n - 1)
            summaries.append(dict(zip(['cohort', 'pathway', 'outcome', 'adjustment'], key)) | {'metric': metric, 'n_patients': n, 'mean': mean, 't_ci_low': mean - t.ppf(0.975, n - 1) * se, 't_ci_high': mean + t.ppf(0.975, n - 1) * se, 'n_positive': int((values > 0).sum()), 'n_negative': int((values < 0).sum()), 'sign_p': p, 'leave_one_patient_out_min': float(loo.min()), 'leave_one_patient_out_max': float(loo.max())})
    summary = pd.DataFrame(summaries)
    summary['sign_fdr'] = np.nan
    for _, g in summary.groupby(['cohort', 'outcome', 'adjustment', 'metric']):
        summary.loc[g.index, 'sign_fdr'] = multipletests(g.sign_p, method='fdr_bh')[1]
    summary.to_csv(OUT / 'cohort_patient_summary.csv', index=False)
if __name__ == '__main__':
    main()

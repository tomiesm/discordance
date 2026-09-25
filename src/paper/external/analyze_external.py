"""Propagate fixed external predictions into utility, scores and program effects."""
from src.paper.paths import PROJECT_ROOT, STAGE_SOURCE, CELL_SOURCE, ANALYSIS_ROOT, CELL_ROOT, stage_dir, cell_dir
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t
OUT = stage_dir('external')
AUDIT = ANALYSIS_ROOT
ROOT = PROJECT_ROOT
REPO = PROJECT_ROOT
from src.discordance import compute_conditional_discordance
from src.pathways import load_gene_sets
from src.paper.genes.gene_audit import overlap_weights, pearson_cols
ENC = ['uni', 'virchow2', 'hoptimus0']
SAMPLES = {'coad': ['TENX111', 'TENX147', 'TENX148', 'TENX149'], 'idc_visium': ['TENX13', 'TENX14', 'TENX39', 'TENX53', 'TENX68', 'NCBI776', 'NCBI681', 'NCBI682', 'NCBI683', 'NCBI684']}

def load(family):
    base = REPO / 'outputs' / family
    genes = json.loads((base / 'gene_panel.json').read_text())
    arrays = []
    ids = []
    sections = []
    groups = []
    preds = {e: [] for e in ENC}
    registry = []
    replacement = {}
    if family == 'idc_visium':
        assert (OUT / 'refit_complete.json').exists(), 'Joint Block A fits must finish first'
        for enc in ENC:
            fd = OUT / 'visium_replacement' / enc
            replacement[enc] = (json.loads((fd / 'test_spot_ids.json').read_text()), np.load(fd / 'test_predictions.npy', mmap_mode='r'))
    for fold, sample in enumerate(SAMPLES[family]):
        fd = base / 'predictions/uni' / f'fold{fold}'
        y = np.load(fd / 'test_targets.npy')
        ii = json.loads((fd / 'test_spot_ids.json').read_text())
        assert all((s.startswith(sample + '_') for s in ii))
        arrays.append(y)
        ids.extend(ii)
        sections.extend([sample] * len(ii))
        group = '10x_block_A' if family == 'idc_visium' and sample in ['TENX13', 'TENX14'] else sample
        groups.extend([group] * len(ii))
        for enc in ENC:
            old = base / 'predictions' / enc / f'fold{fold}'
            assert json.loads((old / 'test_spot_ids.json').read_text()) == ii
            assert np.array_equal(np.load(old / 'test_targets.npy', mmap_mode='r'), y)
            if group == '10x_block_A':
                ri, pp = replacement[enc]
                ix = pd.Index(ri).get_indexer(ii)
                assert (ix >= 0).all()
                p = np.array(pp[ix])
                source = str(OUT / 'visium_replacement' / enc)
            else:
                p = np.load(old / 'test_predictions.npy')
                source = str(old)
            if group != '10x_block_A':
                residual = np.load(old / 'test_residuals.npy')
                assert np.array_equal(residual, (y - p).astype(np.float32))
            assert p.shape == y.shape and np.isfinite(p).all()
            preds[enc].append(p)
            registry.append(dict(family=family, sample=sample, specimen_group=group, encoder=enc, prediction_source=source))
    loc = pd.DataFrame(dict(spot_id=ids, sample=sections, specimen_group=groups))
    return (np.concatenate(arrays), {e: np.concatenate(p) for e, p in preds.items()}, loc, genes, registry)

def tails(score, tail=0.25):
    a, b = np.quantile(score, [tail, 1 - tail])
    return (score <= a, score >= b)

def main(family):
    dest = OUT / family
    dest.mkdir(exist_ok=True)
    (dest / 'arrays').mkdir(exist_ok=True)
    y32, preds, loc, genes, registry = load(family)
    n, g = y32.shape
    pd.DataFrame(registry).to_csv(dest / 'prediction_registry.csv', index=False)
    y = y32.astype(float)
    sr = np.zeros_like(y)
    ae = np.zeros_like(y)
    raw = []
    checks = []
    gene_metrics = []
    quality = []
    baseline = np.empty_like(y)
    for group, gg in loc.groupby('specimen_group', sort=False):
        ix = gg.index.to_numpy()
        train = loc.specimen_group.ne(group).to_numpy()
        mean = y[train].mean(axis=0)
        median = np.empty(g)
        for start in range(0, g, 256):
            median[start:start + 256] = np.median(y32[train, start:start + 256].astype(float), axis=0)
        baseline[ix] = np.abs(y[ix] - median)
        for sample in gg['sample'].unique():
            oldfold = SAMPLES[family].index(sample)
            for enc in ENC:
                fd = OUT / 'visium_replacement' / enc if group == '10x_block_A' else REPO / 'outputs' / family / 'predictions' / enc / f'fold{oldfold}'
        print(family, group, 'training baseline ready', flush=True)
    for enc, p32 in preds.items():
        p = p32.astype(float)
        r = y - p
        sr += r / 3
        ae += np.abs(r) / 3
        raw.append(np.abs(r).mean(axis=1))
        for sample, ss in loc.groupby('sample', sort=False):
            ix = ss.index.to_numpy()
            cor = pearson_cols(y[ix], p[ix])
            mae = np.abs(r[ix]).mean(axis=0)
            bias = r[ix].mean(axis=0)
            bm = baseline[ix].mean(axis=0)
            for j, gene in enumerate(genes):
                gene_metrics.append(dict(family=family, sample=sample, specimen_group=ss.specimen_group.iloc[0], encoder=enc, gene=gene, pearson=cor[j], mae=mae[j], baseline_mae=bm[j], signed_bias=bias[j], relative_mae_gain=1 - mae[j] / bm[j] if bm[j] > 0 else np.nan))
            quality.append(dict(family=family, sample=sample, specimen_group=ss.specimen_group.iloc[0], encoder=enc, n_spots=len(ix), mae=float(mae.mean()), mse=float((r[ix] ** 2).mean()), baseline_mae=float(bm.mean()), relative_mae_gain=float(1 - mae.mean() / bm.mean()), mean_gene_pearson=float(np.nanmean(cor))))
        del p, r
    del preds
    total = y32.sum(axis=1)
    score = np.mean([compute_conditional_discordance(v, total) for v in raw], axis=0)
    raw = np.mean(raw, axis=0)
    base = REPO / 'outputs' / family
    old = pd.concat([pd.read_parquet(base / 'scores' / f'{s}_discordance.parquet') for s in SAMPLES[family]]).set_index('spot_id').loc[loc.spot_id]
    oldscore = old[[f'D_cond_{e}_ridge' for e in ENC]].mean(axis=1).to_numpy()
    oldraw = old[[f'D_raw_{e}_ridge' for e in ENC]].mean(axis=1).to_numpy()
    for c in ['x', 'y']:
        loc[c] = old[c].to_numpy()
    loc['raw'] = raw
    loc['conditional'] = score
    loc['B1_raw'] = oldraw
    loc['B1_conditional'] = oldscore
    loc['total_expression'] = total
    loc.to_parquet(dest / 'scores.parquet', index=False)
    maps = []
    quartiles = []
    for sample, ss in loc.groupby('sample', sort=False):
        ix = ss.index.to_numpy()
        a, b = tails(oldscore[ix])
        c, d = tails(score[ix])
        maps.append(dict(sample=sample, specimen_group=ss.specimen_group.iloc[0], raw_max_change=float(np.max(np.abs(raw[ix] - oldraw[ix]))), conditional_max_change=float(np.max(np.abs(score[ix] - oldscore[ix]))), spearman=float(spearmanr(score[ix], oldscore[ix]).statistic), Q1_overlap=float((a & c).sum() / a.sum()), Q4_overlap=float((b & d).sum() / b.sum())))
        edges = np.quantile(score[ix], [0.25, 0.5, 0.75])
        q = np.select([score[ix] <= edges[0], score[ix] <= edges[1], score[ix] < edges[2]], [1, 2, 3], default=4)
        for k in range(1, 5):
            quartiles.append(dict(sample=sample, specimen_group=ss.specimen_group.iloc[0], quartile=k, n=int((q == k).sum()), mae=float(raw[ix][q == k].mean()), baseline_mae=float(baseline[ix][q == k].mean())))
    pd.DataFrame(maps).to_csv(dest / 'score_changes.csv', index=False)
    pd.DataFrame(quartiles).to_csv(dest / 'quartile_quality.csv', index=False)
    pd.DataFrame(gene_metrics).to_csv(dest / 'gene_metrics.csv', index=False)
    pd.DataFrame(quality).to_csv(dest / 'prediction_quality.csv', index=False)
    counts = np.rint(np.expm1(y))
    rounderr = float(np.max(np.abs(counts - np.expm1(y))))
    assert rounderr < 0.5
    checks.append(dict(check='integer_count_recovery', max_abs=rounderr, passed=True))
    totalcounts = counts.sum(axis=1)
    totaldet = (y > 0).sum(axis=1)
    abs_sum = ae.sum(axis=1)
    total64 = y.sum(axis=1)
    sets = load_gene_sets(str(REPO / 'data/gene_sets/h.all.v2024.1.Hs.symbols.gmt'))
    panels = {cc: set(json.loads((REPO / 'data/v3' / f'gene_list_{cc}.json').read_text())) for cc in ['biomarkers', '10x_janesick']}
    records = []
    coverage = []
    for name, members in sets.items():
        measured = [v for v in members if v in genes]
        pi = np.array([genes.index(v) for v in measured])
        coverage.append(dict(pathway=name, n_genes=len(pi), eligible=len(pi) >= 5, genes=';'.join(measured)))
        if len(pi) < 5:
            continue
        oy = total64 - y[:, pi].sum(axis=1)
        oc = totalcounts - counts[:, pi].sum(axis=1)
        od = totaldet - (y[:, pi] > 0).sum(axis=1)
        raw_other = (abs_sum - ae[:, pi].sum(axis=1)) / (g - len(pi))
        excluded = compute_conditional_discordance(raw_other, oy)
        outcomes = {k: arr[:, pi].mean(axis=1) for k, arr in [('observed', y), ('signed', sr), ('absolute', ae)]}
        common_members = {}
        for cc, panel in panels.items():
            common = [v for v in measured if v in panel]
            common_members[cc] = common
            if len(common) >= 5:
                ji = [genes.index(v) for v in common]
                outcomes.update({f'common_{cc}_{k}': arr[:, ji].mean(axis=1) for k, arr in [('observed', y), ('signed', sr), ('absolute', ae)]})
        np.savez_compressed(dest / 'arrays' / f'{name}.npz', **outcomes, full_conditional=score, program_excluded_conditional=excluded, outside_counts=oc, outside_detected=od, outside_sum_log_expression=oy)
        (dest / 'arrays' / f'{name}_members.json').write_text(json.dumps(dict(measured=measured, common=common_members), indent=2) + '\n')
        for sample, ss in loc.groupby('sample', sort=False):
            ix = ss.index.to_numpy()
            ident = dict(family=family, sample=sample, specimen_group=ss.specimen_group.iloc[0], pathway=name)
            vals = np.stack(list(outcomes.values()), axis=1)[ix]
            names = list(outcomes)
            for grouping, sc in [('full', score[ix]), ('program_excluded', excluded[ix])]:
                for tail in [0.25] if grouping == 'full' else [0.2, 0.25, 0.3]:
                    lo, hi = tails(sc, tail)
                    w1, w4, ns, ret1, ret4 = overlap_weights(oc[ix], od[ix], lo, hi)
                    sd = np.sqrt(((lo.sum() - 1) * vals[lo].var(axis=0, ddof=1) + (hi.sum() - 1) * vals[hi].var(axis=0, ddof=1)) / (lo.sum() + hi.sum() - 2))
                    for adjustment, a, b in [('unadjusted', lo.astype(float), hi.astype(float)), ('overlap_adjusted', w1, w4)]:
                        d = np.average(vals, axis=0, weights=b) - np.average(vals, axis=0, weights=a) if min(a.sum(), b.sum()) > 0 else np.full(len(names), np.nan)
                        for j, outcome in enumerate(names):
                            records.append(dict(**ident, grouping=grouping, tail=tail, adjustment=adjustment, outcome=outcome, effect=float(d[j]), standardized_effect=float(d[j] / sd[j]) if sd[j] > 0 else np.nan, Q1_retained=1.0 if adjustment == 'unadjusted' else ret1, Q4_retained=1.0 if adjustment == 'unadjusted' else ret4, n_overlap_strata=ns))
        print(family, name, 'program completed', flush=True)
    pd.DataFrame(coverage).to_csv(dest / 'program_coverage.csv', index=False)
    eff = pd.DataFrame(records)
    eff.to_csv(dest / 'section_program_effects.csv', index=False)
    keys = ['family', 'specimen_group', 'pathway', 'grouping', 'tail', 'adjustment', 'outcome']
    unit = eff.groupby(keys)[['effect', 'standardized_effect', 'Q1_retained', 'Q4_retained']].mean().reset_index()
    unit.to_csv(dest / 'specimen_program_effects.csv', index=False)
    summaries = []
    for scope in ['all', 'excluding_P07'] if family == 'idc_visium' else ['all']:
        sub = unit[unit.specimen_group.ne('NCBI776')] if scope == 'excluding_P07' else unit
        for key, z in sub.groupby([k for k in keys if k != 'specimen_group']):
            v = z.standardized_effect.dropna().to_numpy()
            nn = len(v)
            if not nn:
                continue
            mean = v.mean()
            se = v.std(ddof=1) / np.sqrt(nn) if nn > 1 else np.nan
            summaries.append(dict(zip([k for k in keys if k != 'specimen_group'], key)) | dict(scope=scope, n_groups=nn, mean=mean, ci_low=mean - t.ppf(0.975, nn - 1) * se, ci_high=mean + t.ppf(0.975, nn - 1) * se, n_positive=int((v > 0).sum()), n_negative=int((v < 0).sum()), loo_min=float(((v.sum() - v) / (nn - 1)).min()) if nn > 1 else np.nan, loo_max=float(((v.sum() - v) / (nn - 1)).max()) if nn > 1 else np.nan))
    pd.DataFrame(summaries).to_csv(dest / 'cohort_program_summary.csv', index=False)
    gm = pd.DataFrame(gene_metrics).groupby(['specimen_group', 'gene'])[['pearson', 'mae', 'baseline_mae']].mean().reset_index()
    features = gm.groupby('gene').mean(numeric_only=True).reset_index().merge(pd.read_csv(base / 'gene_features.csv')[['gene', 'morans_i']], on='gene')
    features.to_csv(dest / 'revised_gene_features.csv', index=False)
    moran_rho = float(spearmanr(features.pearson, features.morans_i, nan_policy='omit').statistic)
    (dest / 'checks.json').write_text(json.dumps(dict(status='pass', checks=checks), indent=2) + '\n')
    (dest / 'SUMMARY.json').write_text(json.dumps(dict(status='complete', family=family, n_sections=len(SAMPLES[family]), n_specimen_groups=loc.specimen_group.nunique(), n_genes=g, n_spots=n, moran_predictability_spearman=moran_rho), indent=2) + '\n')
    print(family, 'ALL ANALYSES COMPLETE', flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', choices=list(SAMPLES), required=True)
    main(parser.parse_args().family)

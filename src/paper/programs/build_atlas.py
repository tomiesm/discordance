"""Complete generic-error program atlas; no spot-level significance tests."""
from src.paper.paths import PROJECT_ROOT, STAGE_SOURCE, CELL_SOURCE, ANALYSIS_ROOT, CELL_ROOT, stage_dir, cell_dir
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import yaml
ROOT = PROJECT_ROOT
REPO = PROJECT_ROOT
OUT = stage_dir('programs')
from src.pathways import load_gene_sets, compute_studentized_gene_residuals
from src.discordance import compute_conditional_discordance

def contrast(y, lo, hi):
    a = np.asarray(y[lo], float)
    b = np.asarray(y[hi], float)
    delta = b.mean() - a.mean()
    sd = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return {'Q1_mean': float(a.mean()), 'Q4_mean': float(b.mean()), 'delta': float(delta), 'cohens_d': float(delta / sd) if sd else np.nan, 'pooled_sd': float(sd), 'n_Q1': len(a), 'n_Q4': len(b)}

def tails(s, t=0.25):
    a, b = np.quantile(s, [t, 1 - t])
    return (s <= a, s >= b)

def main():
    cfg = yaml.safe_load((REPO / 'config.yaml').read_text())
    sets = load_gene_sets(str(REPO / 'data/gene_sets/h.all.v2024.1.Hs.symbols.gmt'))
    panels = {c['name']: json.loads((REPO / 'data/v3' / f"gene_list_{c['name']}.json").read_text()) for c in cfg['cohorts'].values()}
    common = set.intersection(*(set(g) for g in panels.values()))
    cov = pd.read_csv(OUT.parent / 'coverage/coverage_summary.csv').set_index('sample')
    diagnostics = pd.read_parquet(OUT.parent / 'scores/spot_score_diagnostics.parquet').set_index('spot_id')
    coverage, contrasts, quartiles, continuous, overlaps, gene_rows, scale_rows, checks, predictive = ([], [], [], [], [], [], [], [], [])
    for cc in cfg['cohorts'].values():
        cohort = cc['name']
        genes = panels[cohort]
        gmap = {g: i for i, g in enumerate(genes)}
        dest = OUT / 'arrays' / cohort
        dest.mkdir(parents=True, exist_ok=True)
        yfold = []
        ids = []
        sr = []
        ae = []
        for fold in range(4):
            fd = REPO / 'outputs/predictions' / cohort / 'uni/ridge' / f'fold{fold}'
            yfold.append(np.load(fd / 'test_targets.npy'))
            ids += json.loads((fd / 'test_spot_ids.json').read_text())
            r = np.stack([np.load(REPO / 'outputs/predictions' / cohort / ec['name'] / 'ridge' / f'fold{fold}' / 'test_residuals.npy') for ec in cfg['encoders']])
            sr.append(r.mean(axis=0))
            ae.append(np.abs(r).mean(axis=0))
            del r
        y = np.concatenate(yfold)
        signed = np.concatenate(sr)
        absolute = np.concatenate(ae)
        del sr, ae
        baseline = []
        for fold in range(4):
            median = np.median(np.concatenate([v for k, v in enumerate(yfold) if k != fold]).astype(float), axis=0)
            baseline.append(np.abs(yfold[fold] - median))
        baseline = np.concatenate(baseline)
        del yfold
        counts = np.rint(np.expm1(y.astype(float)))
        rounderror = float(np.max(np.abs(counts - np.expm1(y.astype(float)))))
        assert rounderror < 0.1
        checks.append({'check': f'{cohort}:integer_count_recovery', 'max_round_error': rounderror, 'pass': True})
        sc = diagnostics.loc[ids].copy()
        coords = pd.concat([pd.read_parquet(REPO / 'outputs/phase2/scores' / cohort / f'{sid}_discordance.parquet').set_index('spot_id')[['x', 'y']] for sid in cc['samples']]).loc[ids]
        sc['x_um'] = coords.x * sc.sample_id.map(cov.pixel_size_um)
        sc['y_um'] = coords.y * sc.sample_id.map(cov.pixel_size_um)
        sc.reset_index()[['spot_id', 'sample_id', 'patient', 'cohort', 'x_um', 'y_um']].to_csv(dest / 'locations.csv', index=False)
        sample_idx = {sid: np.flatnonzero(sc.sample_id.to_numpy() == sid) for sid in cc['samples']}
        legacy = np.zeros(signed.shape, dtype=float)
        for sid, ix in sample_idx.items():
            legacy[ix] = compute_studentized_gene_residuals(signed[ix], y[ix])
            for j, gene in enumerate(genes):
                r = signed[ix, j]
                z = legacy[ix, j]
                mask = r != 0
                scales = np.divide(r[mask], z[mask])
                b = np.searchsorted(np.percentile(y[ix, j], np.linspace(0, 100, 21)[1:-1]), y[ix, j], side='right')
                ns = np.bincount(b, minlength=20)
                occupied = ns[ns > 0]
                scale_rows.append({'cohort': cohort, 'sample': sid, 'patient': sc.iloc[ix[0]].patient, 'gene': gene, 'min_scale': float(scales.min()) if len(scales) else np.nan, 'median_scale': float(np.median(scales)) if len(scales) else np.nan, 'max_weight': float(np.max(1 / scales)) if len(scales) else np.nan, 'mean_abs_unscaled': float(np.abs(r).mean()), 'mean_abs_legacy_scaled': float(np.abs(z).mean()), 'occupied_bins': len(occupied), 'small_bins_lt3': int((occupied < 3).sum()), 'max_abs_legacy_residual': float(np.abs(z).max())})
            lo, hi = tails(sc.iloc[ix].conditional.to_numpy())
            for j, gene in enumerate(genes):
                row = {'cohort': cohort, 'sample': sid, 'patient': sc.iloc[ix[0]].patient, 'gene': gene}
                for name, values in [('observed', y[ix, j]), ('signed', signed[ix, j]), ('absolute', absolute[ix, j])]:
                    row.update({name + '_' + k: v for k, v in contrast(values, lo, hi).items() if k not in ['n_Q1', 'n_Q4']})
                a = counts[ix, j][lo].mean()
                b = counts[ix, j][hi].mean()
                row.update({'Q1_mean_counts': float(a), 'Q4_mean_counts': float(b), 'log2_count_mean_ratio_pseudocount1': float(np.log2((b + 1) / (a + 1))), 'legacy_log2_mean_logexpression_ratio': float(np.log2((y[ix, j][hi].mean() + 1e-06) / (y[ix, j][lo].mean() + 1e-06))), 'Q1_detected_fraction': float((y[ix, j][lo] > 0).mean()), 'Q4_detected_fraction': float((y[ix, j][hi] > 0).mean())})
                gene_rows.append(row)
        for pname, pgenes in sets.items():
            members = [g for g in pgenes if g in gmap]
            shared = [g for g in members if g in common]
            coverage.append({'cohort': cohort, 'pathway': pname, 'n_genes': len(members), 'n_total': len(pgenes), 'fraction_covered': len(members) / len(pgenes), 'eligible': len(members) >= 5, 'genes': ';'.join(members), 'n_common': len(shared), 'common_eligible': len(shared) >= 5, 'common_genes': ';'.join(shared)})
            if len(members) < 5:
                continue
            pi = np.array([gmap[g] for g in members])
            other = np.array([i for i in range(len(genes)) if i not in set(pi)])
            assert not set(pi) & set(other) and len(pi) + len(other) == len(genes)
            outside_sum = y[:, other].sum(axis=1)
            outside_count = counts[:, other].sum(axis=1)
            outside_detect = (y[:, other] > 0).sum(axis=1)
            raw = absolute[:, other].mean(axis=1)
            cond = compute_conditional_discordance(raw, outside_sum)
            section_cond = np.empty(len(y), float)
            for ix in sample_idx.values():
                section_cond[ix] = compute_conditional_discordance(raw[ix], outside_sum[ix])
            outcomes = {'observed': y[:, pi].mean(axis=1), 'signed': signed[:, pi].mean(axis=1), 'absolute': absolute[:, pi].mean(axis=1), 'legacy_scaled_signed': legacy[:, pi].mean(axis=1)}
            if len(shared) >= 5:
                ci = [gmap[g] for g in shared]
                outcomes.update({'common_observed': y[:, ci].mean(axis=1), 'common_signed': signed[:, ci].mean(axis=1), 'common_absolute': absolute[:, ci].mean(axis=1)})
            variants = {'full_conditional': sc.conditional.to_numpy(), 'program_excluded_conditional': cond, 'program_excluded_raw': raw, 'program_excluded_section_centered': section_cond}
            pbase = baseline[:, pi].mean(axis=1)
            np.savez_compressed(dest / f'{pname}.npz', **{k: np.asarray(v) for k, v in outcomes.items()}, **variants, outside_sum_log_expression=outside_sum, outside_total_counts=outside_count, outside_detected_genes=outside_detect, program_baseline_mae=pbase)
            for sid, ix in sample_idx.items():
                identity = {'cohort': cohort, 'sample': sid, 'patient': sc.iloc[ix[0]].patient, 'pathway': pname, 'n_genes': len(members), 'n_common': len(shared)}
                original_lo, original_hi = tails(variants['full_conditional'][ix])
                old = pd.read_csv(REPO / 'outputs/phase3/pathways' / cohort / 'per_sample' / f'{sid}_pathway_scores.csv').set_index('spot_id')
                ref = old.loc[np.array(ids)[ix], pname].to_numpy()
                diff = float(np.max(np.abs(ref - outcomes['legacy_scaled_signed'][ix])))
                checks.append({'check': f'{sid}/{pname}:legacy_score_reproduction', 'max_abs': diff, 'pass': diff < 2e-05})
                assert diff < 2e-05
                for method, values in variants.items():
                    vals = values[ix]
                    lo, hi = tails(vals)
                    overlaps.append({**identity, 'grouping': method, 'Q1_overlap_fraction': float(np.sum(lo & original_lo) / np.sum(original_lo)), 'Q4_overlap_fraction': float(np.sum(hi & original_hi) / np.sum(original_hi)), 'rank_correlation': float(spearmanr(vals, variants['full_conditional'][ix]).statistic)})
                    tailset = [0.2, 0.25, 0.3] if method == 'program_excluded_conditional' else [0.25]
                    for tail in tailset:
                        lower, upper = tails(vals, tail)
                        for outcome, ys in outcomes.items():
                            contrasts.append({**identity, 'grouping': method, 'tail_fraction': tail, 'outcome': outcome, **contrast(ys[ix], lower, upper)})
                    qedges = np.quantile(vals, [0.25, 0.5, 0.75])
                    q = np.select([vals <= qedges[0], vals <= qedges[1], vals < qedges[2]], [1, 2, 3], default=4)
                    for outcome, ys in outcomes.items():
                        continuous.append({**identity, 'grouping': method, 'outcome': outcome, 'spearman': float(spearmanr(vals, ys[ix]).statistic)})
                        for quartile in range(1, 5):
                            mask = q == quartile
                            quartiles.append({**identity, 'grouping': method, 'outcome': outcome, 'quartile': quartile, 'n': int(mask.sum()), 'mean': float(ys[ix][mask].mean())})
                    for qname, mask in [('Q1', lo), ('Q4', hi)]:
                        mae = outcomes['absolute'][ix][mask].mean()
                        bm = pbase[ix][mask].mean()
                        predictive.append({**identity, 'grouping': method, 'quartile': qname, 'mae': float(mae), 'baseline_mae': float(bm), 'relative_mae_improvement': float(1 - mae / bm)})
            print(cohort, pname, 'complete', flush=True)
        del y, signed, absolute, counts, baseline, legacy
    for name, rows in [('coverage', coverage), ('section_contrasts', contrasts), ('quartile_profiles', quartiles), ('continuous_associations', continuous), ('group_membership_sensitivity', overlaps), ('gene_characterization', gene_rows), ('legacy_scaling_diagnostics', scale_rows), ('program_prediction_quality', predictive)]:
        pd.DataFrame(rows).to_csv(OUT / f'{name}.csv', index=False)
    (OUT / 'checks.json').write_text(json.dumps({'status': 'pass', 'checks': checks}, indent=2) + '\n')
if __name__ == '__main__':
    main()

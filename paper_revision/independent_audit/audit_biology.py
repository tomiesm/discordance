"""Independent score/marker/pathway reconstruction; no project code imports.

Compare archived, revised, direct-solve, and archived-plus-training-mean models.
Keep the original analysis definitions; this is not new biological validation.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import yaml

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OLD = ROOT / 'outputs_v3'
NEW = ROOT / 'paper_revision/clean_repo/outputs'
ENCODERS = ['uni', 'virchow2', 'hoptimus0']
METHODS = ['archived', 'revision', 'direct', 'restored']
MARKERS = {
    'epithelial': ['EPCAM', 'KRT18', 'KRT19', 'KRT8', 'CDH1', 'MUC1'],
    'macrophage': ['CD68', 'CD163', 'CSF1R', 'CD14', 'ITGAM'],
}


def conditional(error, total):
    # All observed deciles have >30 spots: assert rather than importing bin merge.
    cutpoints = np.quantile(total, np.arange(1, 10) / 10)
    bins = np.searchsorted(cutpoints, total, side='right')
    sizes = np.bincount(bins, minlength=10)
    assert sizes.min() >= 30
    means = np.array([error[bins == b].mean() for b in range(10)])
    return error.astype(np.float64) - means[bins], sizes


def effect(values, upper, lower):
    a, b = values[upper], values[lower]
    variance = ((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2)
    return float((a.mean()-b.mean()) / (np.sqrt(variance) if variance > 0 else 1e-6))


def studentize(residual, target):
    # Independently implement the original per-section, per-gene definition.
    out = np.empty(residual.shape, dtype=np.float64)
    n_bins = min(20, max(2, len(target)//10))
    for j in range(target.shape[1]):
        y, r = target[:, j], residual[:, j]
        cuts = np.percentile(y, np.linspace(0, 100, n_bins+1)[1:-1])
        labels = np.searchsorted(cuts, y, side='right')
        present = np.unique(labels)
        global_sd = np.abs(r).std()
        sds = []
        for label in present:
            vals = np.abs(r[labels == label])
            sd = vals.std() if len(vals) >= 3 else np.nan
            sds.append(global_sd if not np.isfinite(sd) or sd == 0 else sd)
        sds = np.asarray(sds)
        floor = np.percentile(sds[sds > 0], 1) if (sds > 0).any() else 1.0
        for label, sd in zip(present, np.maximum(sds, floor)):
            mask = labels == label
            out[mask, j] = r[mask] / sd
    return out


def main():
    config = yaml.safe_load((ROOT / 'config_v3.yaml').read_text())
    pathways = {}
    for line in (ROOT / 'clean_repo/data/gene_sets/h.all.v2024.1.Hs.symbols.gmt').read_text().splitlines():
        fields = line.split('\t')
        pathways[fields[0]] = [g for g in fields[2:] if g]
    all_effects, rank_records, checks = [], [], []
    for cohort, cfg in config['cohorts'].items():
        family = cfg['name']
        genes = json.loads((ROOT / f'data/v3/gene_list_{family}.json').read_text())
        residuals = {method: [] for method in METHODS}
        scores = {method: [] for method in METHODS}
        ids, targets = [], []
        for encoder in ENCODERS:
            enc_residuals = {method: [] for method in METHODS}
            enc_ids, enc_targets = [], []
            for fold in range(4):
                original = OLD / f'predictions/{family}/{encoder}/ridge/fold{fold}'
                revised = NEW / f'predictions/{family}/{encoder}/ridge/fold{fold}'
                audit = HERE / f'models/{family}/{encoder}/fold{fold}'
                target = np.load(original / 'test_targets.npy')  # independently verified against raw HEST
                enc_targets.append(target)
                enc_ids.extend(json.loads((original / 'test_spot_ids.json').read_text()))
                sources = {'archived': original / 'test_predictions.npy',
                           'revision': revised / 'test_predictions.npy',
                           'direct': audit / 'direct_predictions.npy',
                           'restored': audit / 'baseline_restored_predictions.npy'}
                for method, path in sources.items():
                    enc_residuals[method].append(target - np.load(path))
            enc_targets = np.concatenate(enc_targets)
            if not ids:
                ids, targets = enc_ids, enc_targets
            assert ids == enc_ids and np.array_equal(targets, enc_targets)
            for method in METHODS:
                resid = np.concatenate(enc_residuals[method])
                d, bins = conditional(np.abs(resid).mean(axis=1), enc_targets.sum(axis=1))
                residuals[method].append(resid)
                scores[method].append(d)
            print(f'SCORES {cohort}/{encoder}; minimum decile size {bins.min()}', flush=True)
        sample_names = np.array([s.split('_', 1)[0] for s in ids])
        averaged_residuals = {m: np.mean(residuals[m], axis=0) for m in METHODS}
        averaged_scores = {m: np.mean(scores[m], axis=0) for m in METHODS}
        for sid in cfg['samples']:
            # Original pathway aggregation sorts spot IDs before studentization.
            idx = np.flatnonzero(sample_names == sid)
            idx = idx[np.argsort(np.array(ids)[idx])]
            sample_ids = np.array(ids)[idx].tolist()
            y = targets[idx]
            method_masks = {}
            for method in METHODS:
                d = averaged_scores[method][idx]
                lo, hi = np.quantile(d, [.25, .75])
                upper, lower = d >= hi, d <= lo
                method_masks[method] = (upper, lower)
                for cell_type, markers in MARKERS.items():
                    cols = [genes.index(g) for g in markers if g in genes]
                    assert len(cols) >= 2
                    values = y[:, cols].mean(axis=1)
                    all_effects.append(dict(cohort=cohort, sample=sid, method=method,
                                           kind='marker', name=cell_type, d=effect(values, upper, lower)))
                stud = studentize(averaged_residuals[method][idx], y)
                for name, members in pathways.items():
                    cols = [genes.index(g) for g in members if g in genes]
                    if len(cols) < 5:
                        continue
                    all_effects.append(dict(cohort=cohort, sample=sid, method=method,
                                           kind='pathway', name=name,
                                           d=effect(stud[:, cols].mean(axis=1), upper, lower)))
                if method in ['archived', 'revision']:
                    base = OLD if method == 'archived' else NEW
                    saved = pd.read_parquet(base / f'phase2/scores/{family}/{sid}_discordance.parquet').set_index('spot_id').loc[sample_ids]
                    cols = [f'D_cond_{e}_ridge' for e in ENCODERS]
                    diff = float(np.max(np.abs(d - saved[cols].mean(axis=1).values)))
                    assert diff < 2e-6, (sid, method, 'score mismatch', diff)
                    checks.append(dict(cohort=cohort, sample=sid, method=method, max_score_difference=diff))
            for method in ['archived', 'direct', 'restored']:
                upper, _ = method_masks[method]
                revised_upper, _ = method_masks['revision']
                rank_records.append(dict(cohort=cohort, sample=sid, method=method,
                                         rho_vs_revision=float(spearmanr(averaged_scores[method][idx], averaged_scores['revision'][idx]).statistic),
                                         q4_overlap_fraction=float((upper & revised_upper).sum()/upper.sum())))
            print(f'BIOLOGY {cohort}/{sid} complete', flush=True)
    effects = pd.DataFrame(all_effects)
    summary = effects.groupby(['cohort', 'kind', 'name', 'method']).d.mean().unstack('method')
    effects.to_csv(HERE / 'independent_effects_by_section.csv', index=False)
    summary.to_csv(HERE / 'independent_effect_summary.csv')
    ranks = pd.DataFrame(rank_records)
    ranks.to_csv(HERE / 'independent_score_stability.csv', index=False)
    for cohort, cfg in config['cohorts'].items():
        for method in ['archived', 'revision']:
            base = OLD if method == 'archived' else NEW
            cell = pd.read_csv(base / f'phase3/deconvolution/{cfg["name"]}/celltype_summary.csv').set_index('cell_type')
            pathway = pd.read_csv(base / f'phase3/pathways/{cfg["name"]}/pathway_de.csv').set_index(['sample_id', 'pathway'])
            for marker in MARKERS:
                delta = abs(summary.loc[(cohort, 'marker', marker), method] - cell.loc[marker, 'mean_cohens_d'])
                assert delta < 1e-5, (cohort, method, marker, delta)
                checks.append(dict(cohort=cohort, method=method, marker=marker, mean_effect_difference=float(delta)))
            p = effects[(effects.cohort==cohort) & (effects.method==method) & (effects.kind=='pathway')]
            delta = [abs(row.d - pathway.loc[(row['sample'], row['name']), 'cohens_d']) for _, row in p.iterrows()]
            assert max(delta) < 1e-5, (cohort, method, 'pathway difference', max(delta))
            checks.append(dict(cohort=cohort, method=method, max_pathway_effect_difference=float(max(delta))))
    (HERE / 'biology_checks.json').write_text(json.dumps(checks, indent=2) + '\n')
    print(summary.loc[(slice(None), slice(None), ['epithelial', 'macrophage', 'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION']), :].to_string(), flush=True)
    print(ranks.groupby(['cohort', 'method'])[['rho_vs_revision', 'q4_overlap_fraction']].agg(['min', 'median']).to_string(), flush=True)
    print('INDEPENDENT BIOLOGY CHECKS PASSED', flush=True)


if __name__ == '__main__':
    main()

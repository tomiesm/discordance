#!/usr/bin/env python3
"""Compare archived and corrected computations without editing manuscript assets."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[1]
OLD = REPO.parents[1] / 'outputs_v3'
NEW = REPO / 'outputs'
DEST = NEW / 'revision_01/comparison'
FAMILIES = ['biomarkers', '10x_janesick', 'coad', 'idc_visium']
ENCODERS = ['uni', 'virchow2', 'hoptimus0']


def read_json(path):
    return json.loads(path.read_text())


def score_dir(base, family):
    return base / 'phase2/scores' / family if family in FAMILIES[:2] else base / family / 'scores'


def pathway_path(base, family):
    return (base / 'phase3/pathways' / family if family in FAMILIES[:2] else base / family) / 'pathway_consistency.csv'


def feature_path(base, family):
    return (base / 'phase3/gene_predictability' / family if family in FAMILIES[:2] else base / family) / 'gene_features.csv'


def numeric(value):
    return isinstance(value, (float, int, np.number)) and not isinstance(value, (bool, np.bool_))


def flatten(value, prefix=''):
    """Keep named records aligned by identifier, not changing rank in top-N lists."""
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ['time_s', 'seconds']:
                continue
            result.update(flatten(item, f'{prefix}/{key}'))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            if isinstance(item, dict):
                label = next((str(item[k]) for k in ['gene', 'pathway', 'feature', 'sample_id',
                                                     'patient_id', 'cell_type'] if k in item), str(i))
                result.update(flatten(item, f'{prefix}/{label}'))
    elif value is None or isinstance(value, (str, bool, int, float)):
        result[prefix] = value
    return result


def calibration_summary():
    # Explicit paths avoid relying on one output layout for external cohorts.
    files = list((NEW / 'predictions').glob('*/*/ridge/fold*/calibration.json'))
    files += list((NEW / 'coad/predictions').glob('*/fold*/calibration.json'))
    files += list((NEW / 'idc_visium/predictions').glob('*/fold*/calibration.json'))
    records = [read_json(p) for p in files]
    rows = []
    for item in records:
        row = {k: item[k] for k in ['family', 'encoder', 'fold', 'n_train', 'n_test', 'n_genes',
                                   'alpha', 'old_mean_pearson', 'corrected_mean_pearson']}
        for version in ['old', 'corrected']:
            row.update({f'{version}_{k}': v for k, v in item[version].items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(DEST / 'calibration_per_fold.csv', index=False)
    result = {}
    for family in FAMILIES:
        subset = [r for r in records if r['family'] == family]
        if not subset:
            continue
        weights = np.array([r['n_test'] * r['n_genes'] for r in subset], dtype=float)
        item = {'n_models': len(subset), 'n_spots': int(sum(r['n_test'] for r in subset if r['encoder'] == 'uni'))}
        for version in ['old', 'corrected']:
            item[version] = {k: float(np.average([r[version][k] for r in subset], weights=weights))
                             for k in subset[0][version] if k != 'rmse'}
            item[version]['rmse'] = float(np.sqrt(np.average([r[version]['rmse'] ** 2 for r in subset], weights=weights)))
        item['max_absolute_fold_mean_pearson_change'] = max(abs(r['corrected_mean_pearson'] - r['old_mean_pearson']) for r in subset)
        result[family] = item
    return result


def scores():
    rows, transitions = [], []
    for family in FAMILIES:
        for p in sorted(score_dir(NEW, family).glob('*_discordance.parquet')):
            old = pd.read_parquet(score_dir(OLD, family) / p.name).set_index('spot_id')
            new = pd.read_parquet(p).set_index('spot_id')
            assert old.index.is_unique and new.index.is_unique
            assert set(old.index) == set(new.index), (family, p.name)
            new = new.loc[old.index]
            columns = [f'D_cond_{e}_ridge' for e in ENCODERS]
            a, b = old[columns].mean(axis=1), new[columns].mean(axis=1)
            assert np.isfinite(a).all() and np.isfinite(b).all()
            aq, bq = a >= a.quantile(.75), b >= b.quantile(.75)
            al, bl = a <= a.quantile(.25), b <= b.quantile(.25)
            rows.append({'family': family, 'sample_id': p.stem.replace('_discordance', ''),
                         'n_spots': len(a), 'spearman_old_vs_corrected': float(spearmanr(a, b).statistic),
                         'old_q4_retained': float((aq & bq).sum() / aq.sum()),
                         'q4_jaccard': float((aq & bq).sum() / (aq | bq).sum()),
                         'old_q1_retained': float((al & bl).sum() / al.sum())})
            # Match the analysis's inclusive Q1/Q4 thresholds, including ties.
            qa = np.select([a <= a.quantile(.25), a <= a.quantile(.5), a < a.quantile(.75)], [1, 2, 3], default=4)
            qb = np.select([b <= b.quantile(.25), b <= b.quantile(.5), b < b.quantile(.75)], [1, 2, 3], default=4)
            for i in range(1, 5):
                for j in range(1, 5):
                    transitions.append({'family': family, 'sample_id': rows[-1]['sample_id'],
                                        'old_quartile': i, 'corrected_quartile': j,
                                        'n_spots': int(((qa == i) & (qb == j)).sum())})
    df = pd.DataFrame(rows)
    df.to_csv(DEST / 'discordance_per_section.csv', index=False)
    pd.DataFrame(transitions).to_csv(DEST / 'quartile_transitions.csv', index=False)
    return {family: {'n_sections': len(group),
                     **{f'median_{k}': float(group[k].median()) for k in
                        ['spearman_old_vs_corrected', 'old_q4_retained', 'q4_jaccard', 'old_q1_retained']}}
            for family, group in df.groupby('family')}


def compare_tables():
    result = {'de': {}, 'pathways': {}, 'gene_predictability': {}, 'marker_scores': {}}
    pathways = {}
    for family in FAMILIES:
        path = pathway_path(NEW, family)
        if path.exists():
            old, new = pd.read_csv(pathway_path(OLD, family)), pd.read_csv(path)
            joined = old.merge(new, on='pathway', suffixes=('_old', '_corrected'), validate='one_to_one')
            joined.to_csv(DEST / f'{family}_pathway_effects.csv', index=False)
            pathways[family] = {'old': old, 'corrected': new}
            result['pathways'][family] = {
                'n_pathways': len(new),
                'n_effect_sign_reversals': int((joined.mean_cohens_d_old * joined.mean_cohens_d_corrected < 0).sum()),
                'effect_pearson_old_vs_corrected': float(pearsonr(joined.mean_cohens_d_old, joined.mean_cohens_d_corrected).statistic),
                'old_n_positive': int((old.mean_cohens_d > 0).sum()),
                'corrected_n_positive': int((new.mean_cohens_d > 0).sum()),
                'old_effect_range': [float(old.mean_cohens_d.min()), float(old.mean_cohens_d.max())],
                'corrected_effect_range': [float(new.mean_cohens_d.min()), float(new.mean_cohens_d.max())],
            }
            if family in FAMILIES[2:]:
                result['pathways'][family]['old_n_sig_all_samples'] = int((old.n_sig == (4 if family == 'coad' else 10)).sum())
                result['pathways'][family]['corrected_n_sig_all_samples'] = int((new.n_sig == (4 if family == 'coad' else 10)).sum())
                result['pathways'][family]['corrected_n_direction_consistency_100pct'] = int((new.consistency == 1).sum())

        path = feature_path(NEW, family)
        if path.exists():
            old, new = pd.read_csv(feature_path(OLD, family)), pd.read_csv(path)
            old.merge(new, on='gene', suffixes=('_old', '_corrected'), validate='one_to_one').to_csv(
                DEST / f'{family}_gene_features.csv', index=False)
            result['gene_predictability'][family] = {
                version: {'mean_pooled_gene_pearson': float(df.mean_pearson.mean()),
                          'median_pooled_gene_pearson': float(df.mean_pearson.median()),
                          'morans_vs_predictability_pearson': float(pearsonr(df['spatial_autocorrelation' if 'spatial_autocorrelation' in df else 'morans_i'], df.mean_pearson).statistic),
                          'morans_vs_predictability_spearman': float(spearmanr(df['spatial_autocorrelation' if 'spatial_autocorrelation' in df else 'morans_i'], df.mean_pearson).statistic)}
                for version, df in [('old', old), ('corrected', new)]}

    for family in FAMILIES[:2]:
        result['de'][family] = {}
        for kind in ['unmatched', 'matched']:
            rel = Path('phase3/de') / family / f'meta_de_{kind}.csv'
            old, new = pd.read_csv(OLD / rel), pd.read_csv(NEW / rel)
            joined = old.merge(new, on='gene', suffixes=('_old', '_corrected'), validate='one_to_one')
            joined.to_csv(DEST / f'{family}_de_{kind}.csv', index=False)
            result['de'][family][kind] = {
                'old_n_reproducible': int((old.reproducibility >= .5).sum()),
                'corrected_n_reproducible': int((new.reproducibility >= .5).sum()),
                'n_median_effect_sign_reversals': int((joined.median_log2fc_old * joined.median_log2fc_corrected < 0).sum()),
                'median_effect_pearson_old_vs_corrected': float(pearsonr(joined.median_log2fc_old, joined.median_log2fc_corrected).statistic)}
        rel = Path('phase3/deconvolution') / family / 'celltype_summary.csv'
        joined = pd.read_csv(OLD / rel).merge(pd.read_csv(NEW / rel), on='cell_type', suffixes=('_old', '_corrected'), validate='one_to_one')
        joined.to_csv(DEST / f'{family}_marker_scores.csv', index=False)
        result['marker_scores'][family] = joined.to_dict('records')

    result['pathway_cross_cohort'] = {}
    for other in FAMILIES[1:]:
        if other not in pathways:
            continue
        result['pathway_cross_cohort'][other] = {}
        for version in ['old', 'corrected']:
            joined = pathways['biomarkers'][version].merge(pathways[other][version], on='pathway', suffixes=('_discovery', '_other'))
            result['pathway_cross_cohort'][other][version] = {
                'n_shared': len(joined),
                'n_same_mean_effect_sign': int((joined.mean_cohens_d_discovery * joined.mean_cohens_d_other > 0).sum())}
            if other == 'coad':
                stable = pathways[other][version].query('consistency >= 0.75')
                stable_shared = pathways['biomarkers'][version].merge(stable, on='pathway', suffixes=('_discovery', '_other'))
                result['pathway_cross_cohort'][other][version].update({
                    'n_same_majority_direction_among_shared': int(((joined.direction == 'up') ==
                                                                  (joined.n_positive_delta > joined.n_negative_delta)).sum()),
                    'n_coad_consistent_at_least_75pct': len(stable),
                    'n_consistent_coad_shared_with_discovery': len(stable_shared),
                    'n_consistent_coad_shared_positive_both': int(((stable_shared.mean_cohens_d_discovery > 0) &
                                                                 (stable_shared.mean_cohens_d_other > 0)).sum())})
    return result


def json_comparisons():
    rows, snapshots = [], {}
    selected = [*NEW.glob('phase2/gate*.json'), *NEW.glob('phase3/gene_predictability/*/ols_results.json'),
                *NEW.glob('phase3/within_patient/*summary.json'), *NEW.glob('phase4/*/*/*.json'),
                *NEW.glob('phase4/bridge_genes/*.json'), *NEW.glob('figure_data/reproducibility/*.json'),
                *NEW.glob('coad/analysis_results.json'), *NEW.glob('idc_visium/analysis_results.json')]
    for path in sorted(set(selected)):
        rel = path.relative_to(NEW)
        if not (OLD / rel).exists():
            continue
        a, b = read_json(OLD / rel), read_json(path)
        snapshots[str(rel)] = {'old': a, 'corrected': b}
        fa, fb = flatten(a), flatten(b)
        for key in sorted(set(fa) | set(fb)):
            x, y = fa.get(key), fb.get(key)
            if x != y and (numeric(x) or numeric(y) or isinstance(x, bool)):
                rows.append({'file': str(rel), 'key': key, 'old': x, 'corrected': y,
                             'delta': y-x if numeric(x) and numeric(y) else None})
    pd.DataFrame(rows).to_csv(DEST / 'summary_scalar_changes.csv', index=False)
    (DEST / 'summary_snapshots.json').write_text(json.dumps(snapshots, indent=2) + '\n')


def gates_and_interior():
    result = {'gates': {}, 'interior': {}}
    for family, cohort in [('biomarkers', 'discovery'), ('10x_janesick', 'validation')]:
        result['gates'][cohort] = {}
        result['interior'][cohort] = {}
        for version, base in [('old', OLD), ('corrected', NEW)]:
            g1 = read_json(base / 'phase2/gate2_1_agreement.json')['cohorts'][cohort]
            g2 = read_json(base / 'phase2/gate2_2_spatial.json')['cohorts'][cohort]
            g3 = read_json(base / 'phase2/gate2_3_dual_track.json')['cohorts'][cohort]
            result['gates'][cohort][version] = {
                'agreement_n_pass': g1['n_pass'], 'agreement_n_total': g1['n_total'],
                'agreement_median_rho': float(np.median([v['median_rho'] for v in g1['samples'].values()])),
                'spatial_n_pass': g2['n_pass'],
                'spatial_median_morans_i': float(np.median([v['morans_i'] for v in g2['samples'].values()])),
                'spatial_range': [min(v['morans_i'] for v in g2['samples'].values()), max(v['morans_i'] for v in g2['samples'].values())],
                'raw_dual_track_overall_median_rho': g3['overall_median_rho'],
                'raw_dual_track_ridge_median_by_encoder': {e: g3['configs'][f'{e}_ridge']['median_rho'] for e in ENCODERS}}
            path = base / 'phase3/de_interior' / family / 'comparison_with_full.csv'
            if path.exists():
                d = pd.read_csv(path)
                result['interior'][cohort][version] = {
                    'full_vs_interior_effect_pearson': float(pearsonr(d.full_log2fc, d.interior_log2fc).statistic),
                    'n_reproducible_interior': int((d.interior_repro >= .5).sum())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partial', action='store_true', help='Allow comparisons before external analyses complete.')
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    if not args.partial:
        assert (NEW / 'revision_01/23_figure1_schematic.completed.json').exists(), 'Pipeline incomplete'
        assert (NEW / 'revision_01/21_idc_visium_generalization.completed.json').exists(), 'Visium analysis incomplete'
    summary = {'status': 'partial' if args.partial else 'complete', 'calibration': calibration_summary(),
               'discordance': scores(), **compare_tables(), **gates_and_interior()}
    if not args.partial:
        assert sum(v['n_models'] for v in summary['calibration'].values()) == 66
        assert {f: v['n_sections'] for f, v in summary['discordance'].items()} == dict(zip(FAMILIES, [11, 7, 4, 10]))
    json_comparisons()
    (DEST / 'comparison_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ['status', 'calibration', 'discordance', 'de', 'pathway_cross_cohort']}, indent=2))


if __name__ == '__main__':
    main()

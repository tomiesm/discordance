"""Compare exact members, same-name programs and gene predictability separately."""
from src.paper.paths import PROJECT_ROOT, STAGE_SOURCE, CELL_SOURCE, ANALYSIS_ROOT, CELL_ROOT, stage_dir, cell_dir
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
OUT = stage_dir('external')
AUDIT = ANALYSIS_ROOT

def main():
    common = pd.read_csv(OUT / 'idc_common_member_cohort_effects.csv')
    cp = pd.read_csv(OUT / 'idc_common_member_patient_effects.csv')
    coverage = pd.read_csv(OUT / 'common_member_coverage.csv')
    old = pd.read_csv(AUDIT / 'matching/cohort_patient_summary.csv')
    old = old[old.metric.eq('standardized_estimate')]
    rows = []
    checks = []
    tailrows = []
    utility = []
    generows = []
    genecomparisons = []
    quartile_checks = []
    iq = pd.read_csv(AUDIT / 'genes/patient_prediction_quality.csv')
    iq = iq[iq.grouping.eq('full') & iq.quartile.eq('Q1')]
    for family in ['coad', 'idc_visium']:
        dest = OUT / family
        es = pd.read_csv(dest / 'cohort_program_summary.csv')
        ep = pd.read_csv(dest / 'specimen_program_effects.csv')
        primary = es[es.grouping.eq('program_excluded') & es['tail'].eq(0.25)]
        for cc in ['biomarkers', '10x_janesick']:
            for mode in ['same_name', 'exact_common_members']:
                if mode == 'same_name':
                    ref = old[old.cohort.eq(cc)].rename(columns={'mean': 'idc_mean'})
                    ref = ref[['pathway', 'outcome', 'adjustment', 'idc_mean', 'n_positive', 'n_negative']]
                else:
                    ref = common[common.family.eq(family) & common.idc_cohort.eq(cc)].rename(columns={'standardized_effect': 'idc_mean'})
                    ref = ref[['pathway', 'outcome', 'adjustment', 'idc_mean']]
                    for x in coverage[coverage.family.eq(family) & coverage.idc_cohort.eq(cc) & coverage.eligible].itertuples():
                        ext = json.loads((dest / 'arrays' / f'{x.pathway}_members.json').read_text())['common'][cc]
                        assert set(ext) == set(x.genes.split(';')) and len(ext) == x.n_common
                        checks.append(dict(check=f'{family}/{cc}/{x.pathway}:identical_members', passed=True))
                for r in ref.itertuples():
                    ename = r.outcome if mode == 'same_name' else f'common_{cc}_{r.outcome}'
                    found = primary[primary.pathway.eq(r.pathway) & primary.outcome.eq(ename) & primary.adjustment.eq(r.adjustment)]
                    for e in found.itertuples():
                        members = json.loads((dest / 'arrays' / f'{r.pathway}_members.json').read_text())
                        idcp = cp[cp.family.eq(family) & cp.idc_cohort.eq(cc) & cp.pathway.eq(r.pathway) & cp.outcome.eq(r.outcome) & cp.adjustment.eq(r.adjustment)]
                        idcpos = int((idcp.standardized_effect > 0).sum()) if mode == 'exact_common_members' else int(r.n_positive)
                        idcneg = int((idcp.standardized_effect < 0).sum()) if mode == 'exact_common_members' else int(r.n_negative)
                        rows.append(dict(family=family, idc_cohort=cc, comparison=mode, pathway=r.pathway, outcome=r.outcome, adjustment=r.adjustment, scope=e.scope, n_common=len(members['common'][cc]), n_external_members=len(members['measured']), idc_standardized_effect=r.idc_mean, external_standardized_effect=e.mean, same_direction=bool(r.idc_mean * e.mean > 0), idc_positive=idcpos, idc_negative=idcneg, external_positive=e.n_positive, external_negative=e.n_negative, n_external_groups=e.n_groups, external_ci_low=e.ci_low, external_ci_high=e.ci_high, external_loo_min=e.loo_min, external_loo_max=e.loo_max))
        z = ep[ep.grouping.eq('program_excluded') & ep.outcome.isin(['observed', 'signed', 'absolute'])]
        for key, sub in z.groupby(['specimen_group', 'pathway', 'outcome', 'adjustment']):
            values = sub.set_index('tail').standardized_effect.reindex([0.2, 0.25, 0.3]).to_numpy()
            tailrows.append(dict(family=family, **dict(zip(['specimen_group', 'pathway', 'outcome', 'adjustment'], key)), effect_20=values[0], effect_25=values[1], effect_30=values[2], all_same_direction=bool(np.all(np.isfinite(values)) and (np.all(values > 0) or np.all(values < 0)))))
        q = pd.read_csv(dest / 'prediction_quality.csv')
        qq = pd.read_csv(dest / 'quartile_quality.csv')
        for sample, ss in qq.groupby('sample'):
            v = ss.sort_values('quartile').mae.to_numpy()
            quartile_checks.append(dict(family=family, sample=sample, specimen_group=ss.specimen_group.iloc[0], raw_MAE_increases_all_quartiles=bool(np.all(np.diff(v) > 0)), Q4_raw_MAE_greater_Q1=bool(v[-1] > v[0]), Q1_mae=float(v[0]), Q4_mae=float(v[-1])))
        for group, gg in q.groupby('specimen_group'):
            vals = gg.groupby('sample')[['mae', 'baseline_mae', 'mean_gene_pearson']].mean().mean()
            quart = qq[qq.specimen_group.eq(group)].groupby('quartile')[['mae', 'baseline_mae']].mean()
            utility.append(dict(family=family, specimen_group=group, mae=vals.mae, baseline_mae=vals.baseline_mae, relative_mae_gain=1 - vals.mae / vals.baseline_mae, mean_gene_pearson=vals.mean_gene_pearson, Q1_mae=quart.loc[1, 'mae'], Q1_baseline_mae=quart.loc[1, 'baseline_mae'], Q1_relative_gain=1 - quart.loc[1, 'mae'] / quart.loc[1, 'baseline_mae'], Q4_mae=quart.loc[4, 'mae'], Q4_baseline_mae=quart.loc[4, 'baseline_mae'], Q4_relative_gain=1 - quart.loc[4, 'mae'] / quart.loc[4, 'baseline_mae']))
        gm = pd.read_csv(dest / 'gene_metrics.csv', usecols=['specimen_group', 'gene', 'pearson'])
        for scope in ['all', 'excluding_P07'] if family == 'idc_visium' else ['all']:
            ge = gm[gm.specimen_group.ne('NCBI776')] if scope == 'excluding_P07' else gm
            ge = ge.groupby(['specimen_group', 'gene']).pearson.mean().groupby('gene').mean().rename('external_pearson')
            for cc in ['biomarkers', '10x_janesick']:
                gi = iq[iq.cohort.eq(cc)].groupby('gene').whole_section_gene_pearson.mean().rename('idc_pearson')
                merged = pd.concat([gi, ge], axis=1, join='inner').dropna().reset_index()
                merged['family'] = family
                merged['idc_cohort'] = cc
                merged['scope'] = scope
                genecomparisons.append(merged)
                generows.append(dict(family=family, idc_cohort=cc, scope=scope, n_genes=len(merged), spearman=float(spearmanr(merged.idc_pearson, merged.external_pearson).statistic), pearson=float(pearsonr(merged.idc_pearson, merged.external_pearson).statistic)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'program_comparisons.csv', index=False)
    summaries = []
    for key, z in df.groupby(['family', 'idc_cohort', 'comparison', 'outcome', 'adjustment', 'scope']):
        a = z.idc_standardized_effect
        b = z.external_standardized_effect
        summaries.append(dict(zip(['family', 'idc_cohort', 'comparison', 'outcome', 'adjustment', 'scope'], key)) | dict(n_programs=len(z), n_same_direction=int(z.same_direction.sum()), n_both_positive=int(((a > 0) & (b > 0)).sum()), n_both_negative=int(((a < 0) & (b < 0)).sum()), spearman=float(spearmanr(a, b).statistic) if len(z) > 2 else np.nan))
    pd.DataFrame(summaries).to_csv(OUT / 'program_comparison_summary.csv', index=False)
    pd.DataFrame(tailrows).to_csv(OUT / 'tail_sensitivity.csv', index=False)
    pd.DataFrame(utility).to_csv(OUT / 'specimen_prediction_quality.csv', index=False)
    pd.DataFrame(quartile_checks).to_csv(OUT / 'quartile_ordering.csv', index=False)
    pd.DataFrame(generows).to_csv(OUT / 'gene_predictability_comparisons.csv', index=False)
    pd.concat(genecomparisons).to_csv(OUT / 'gene_predictability_pairs.csv', index=False)
    (OUT / 'comparison_checks.json').write_text(json.dumps(dict(status='pass', checks=checks), indent=2) + '\n')
    print('Cross-panel comparisons complete; identical-member checks:', len(checks))
if __name__ == '__main__':
    main()

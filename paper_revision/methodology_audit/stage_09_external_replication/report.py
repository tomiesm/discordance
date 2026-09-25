"""External evidence report, with replication levels kept separate."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
LABEL={'biomarkers':'IDC discovery','10x_janesick':'IDC validation','coad':'COAD','idc_visium':'Visium'}


def main():
    quality=pd.read_csv(OUT/'specimen_prediction_quality.csv')
    ordering=pd.read_csv(OUT/'quartile_ordering.csv')
    comparisons=pd.read_csv(OUT/'program_comparison_summary.csv')
    gp=pd.read_csv(OUT/'gene_predictability_comparisons.csv')
    maps=pd.read_csv(OUT/'idc_visium/score_changes.csv')
    tails=pd.read_csv(OUT/'tail_sensitivity.csv')
    fits=json.loads((OUT/'refit_complete.json').read_text())
    summaries={f:pd.read_csv(OUT/f/'cohort_program_summary.csv') for f in ['coad','idc_visium']}
    primary={f:s[s.grouping.eq('program_excluded')&s['tail'].eq(.25)&s.adjustment.eq('overlap_adjusted')&s.outcome.isin(['observed','signed','absolute'])&s.scope.eq('all')] for f,s in summaries.items()}
    text=['# External replication after specimen-group correction','',
        '2026-09-20. Three joint Block A Visium fits and the downstream propagation are complete. '
        'Original code, B1 outputs and both manuscript copies remain unchanged. See the [frozen protocol](PROTOCOL.md).','',
        '## Model correction and prediction quality','',
        'TENX13 and TENX14 were excluded together during training for each of three encoders. '
        'The other 24 Visium fits retain the same training/test membership and are reused; COAD reuses its '
        '12 corrected fits. This fixes the known shared-specimen leakage in the two Block A test sections. '
        'It does not establish that every other Visium section is from an unrelated donor. NCBI776 is P07, '
        'already represented in Xenium, and is also excluded in the independent-external sensitivity.','',
        '| Dataset | Known specimen groups | Groups beating training-median MAE | Range of relative MAE gain |','|---|---:|---:|---:|']
    for f,z in quality.groupby('family',sort=False):
        text.append(f'| {LABEL[f]} | {len(z)} | {int(z.relative_mae_gain.gt(0).sum())}/{len(z)} | {100*z.relative_mae_gain.min():+.1f}% to {100*z.relative_mae_gain.max():+.1f}% |')
    text += ['', 'MAE gains average encoders and sections within known specimens before the group comparison. '
        'A positive gain means smaller absolute error than a training-only per-gene median prediction. '
        'Correlation, useful absolute prediction and Q1/Q4 ranking are distinct results.','',
        'COAD outperforms its training-median baseline in all four patients. Visium does so in only '
        'three of nine known specimen groups overall; global absolute prediction accuracy is therefore '
        'limited for this transcriptome-wide application. In its Q1 subsets, Visium improves on the '
        'baseline in eight of nine groups (TENX39 is the exception). Report both populations rather '
        'than treating relative concordance as uniformly accurate whole-section prediction.','',
        'Visium raw MAE rises monotonically from Q1 through Q4 in seven of ten sections and Q4 exceeds '
        'Q1 in nine. NCBI682 has Q1 MAE 0.344 versus Q4 0.322. Conditional groups rank error relative '
        'to pooled expression-bin means, so they cannot always be called the lowest/highest absolute-'
        'error locations across differing expression levels. All four COAD sections are monotone.','',
        'The two replacement test sections have the following score changes relative to archived B1:','',
        '| Section | Conditional-score rank correlation | Q1 overlap | Q4 overlap |','|---|---:|---:|---:|']
    for r in maps[maps['sample'].isin(['TENX13','TENX14'])].itertuples():text.append(f'| {r.sample} | {r.spearman:.3f} | {100*r.Q1_overlap:.1f}% | {100*r.Q4_overlap:.1f}% |')
    rest=maps[~maps['sample'].isin(['TENX13','TENX14'])]
    text += ['', f'Pooled bin means were recalculated across all 40,350 Visium spots. The other eight sections '
        f'retain their predictions (maximum raw-score arithmetic difference {rest.raw_max_change.max():.2g}), '
        f'but their conditional scores also change: Q4 overlap ranges from {100*rest.Q4_overlap.min():.1f}% '
        f'to {100*rest.Q4_overlap.max():.1f}%. These propagated groups are used throughout this stage.','',
        '## Broad error structure versus expression direction','',
        'Primary full-score descriptions and program-excluded support analyses are both saved. The following '
        'table uses program-excluded grouping and outside-program count/detection overlap adjustment. '
        'It counts positive cohort-average contrasts, without converting them into independent discoveries.','',
        '| Dataset | Testable programs | Positive observed expression | Positive signed residual | Positive absolute error |','|---|---:|---:|---:|---:|']
    for f,p in primary.items():
        counts={o:int(p[p.outcome.eq(o)]['mean'].gt(0).sum()) for o in ['observed','signed','absolute']};nn=int(p.pathway.nunique())
        text.append(f'| {LABEL[f]} | {nn} | {counts["observed"]} | {counts["signed"]} | {counts["absolute"]} |')
    text += ['', 'For comparison, the primary full-score, unadjusted descriptions give the following counts:','',
        '| Dataset | Positive observed expression | Positive signed residual | Positive absolute error |','|---|---:|---:|---:|']
    for f,s in summaries.items():
        p=s[s.grouping.eq('full')&s['tail'].eq(.25)&s.adjustment.eq('unadjusted')&s.scope.eq('all')]
        counts={o:int(p[p.outcome.eq(o)]['mean'].gt(0).sum()) for o in ['observed','signed','absolute']}
        text.append(f'| {LABEL[f]} | {counts["observed"]} | {counts["signed"]} | {counts["absolute"]} |')
    text += ['', 'In Visium, 49/50 observed program means are higher in the original generic Q4 comparison '
        'and remain so after program exclusion alone; none is higher after count/detection overlap '
        'adjustment. That substantial change must accompany the adjusted results. The adjusted analysis '
        'compares a different, overlap-weighted population at similar measured abundance/detection; those '
        'covariates can also represent biology. It does not prove that the unadjusted associations are '
        'artifacts or establish program-specific suppression. The unadjusted EMT mean itself is near '
        'zero (−0.024 SD, four of nine groups positive), so the broad raw abundance increase does not '
        'supply replicated EMT enrichment.','',
        'Across section–program comparisons, median retained Q1/Q4 fractions are 93.8%/87.2% in COAD '
        'and 90.0%/89.5% in Visium; minimum Q4 retention is 22.2% and 10.3%, respectively. '
        'These overlap limitations constrain the adjusted population, even when the median support is high.','',
        'Excluding a program from the score reduces its direct contribution to group selection, but '
        'correlated genes and shared tissue/count effects remain. Broad positive absolute-error contrasts '
        'support structured prediction difficulty; they do not imply that every program is activated in Q4.','',
        'Selected pre-existing biological claims, with standardized Q4−Q1 effects averaged over known groups:','',
        '| Dataset / program | Observed expression | Signed residual | Absolute error |','|---|---:|---:|---:|']
    selected=['HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION','HALLMARK_COMPLEMENT','HALLMARK_E2F_TARGETS','HALLMARK_G2M_CHECKPOINT']
    for f,p in primary.items():
        for name in selected:
            z=p[p.pathway.eq(name)].set_index('outcome')
            if z.empty:continue
            text.append(f'| {LABEL[f]} / {name.removeprefix("HALLMARK_")} | '+' | '.join(f'{z.loc[o,"mean"]:+.3f}' for o in ['observed','signed','absolute'])+' |')
    text += ['', 'These full-panel programs contain different measured genes across platforms. Their names '
        'cannot by themselves establish replication. The exact-member comparisons below address that '
        'separately. Per-specimen directions, approximate intervals and leave-group-out ranges are retained '
        'in each dataset\'s tables. For Visium, group-based intervals assume independence of the known groups '
        'and are descriptive because remaining donor links are unresolved.','',
        '## Exact-member program comparisons','',
        'Programs require at least five identical measured members in both panels. Each platform excludes '
        'all its measured members of that program from grouping and conditioning. Counts below are '
        'cohort-mean direction agreement for overlap-adjusted contrasts; they are not a pathway replication rate. '
        'Visium rows exclude P07 from the external mean.','',
        '| External / IDC comparator | Eligible programs | Observed direction agreement | Signed direction agreement | Absolute-error direction agreement |','|---|---:|---:|---:|---:|']
    c=comparisons[comparisons.comparison.eq('exact_common_members')&comparisons.adjustment.eq('overlap_adjusted')]
    for f in ['coad','idc_visium']:
        scope='excluding_P07' if f=='idc_visium' else 'all'
        for cc in ['biomarkers','10x_janesick']:
            z=c[c.family.eq(f)&c.idc_cohort.eq(cc)&c.scope.eq(scope)].set_index('outcome');nn=int(z.loc['observed','n_programs'])
            text.append(f'| {LABEL[f]} / {LABEL[cc]} | {nn} | '+' | '.join(f'{int(z.loc[o,"n_same_direction"])}/{nn}' for o in ['observed','signed','absolute'])+' |')
    text += ['', 'COAD has only three eligible common-member programs with IDC discovery and four with IDC '
        'validation. The measured EMT Hallmark does not reach the common-member threshold for either '
        'COAD comparison. Accordingly, COAD cannot provide exact-member EMT replication under this rule. '
        'All same-name comparisons, ineligible coverage and common-member effect sizes are also saved.','',
        '## Gene predictability and spatial expression','',
        'Gene correlations are averaged across encoders and sections within patient/specimen, then equally '
        'across patients/specimens. Cross-gene rank correlations below are descriptive; genes are dependent '
        'and different technologies measure different count distributions.','',
        '| External / IDC comparator | Shared genes | Spearman correlation of predictability |','|---|---:|---:|']
    for r in gp.itertuples():
        if r.family=='idc_visium' and r.scope!='excluding_P07':continue
        text.append(f'| {LABEL[r.family]} / {LABEL[r.idc_cohort]} | {r.n_genes} | {r.spearman:.3f} |')
    for family in ['coad','idc_visium']:
        summary=json.loads((OUT/family/'SUMMARY.json').read_text())
        text += ['',f'{LABEL[family]} gene predictability versus the archived expression-only Moran\'s I feature '
            f'has Spearman correlation {summary["moran_predictability_spearman"]:.3f}. Targets and coordinates '
            'are unchanged, so the target-only feature is reused. This is a spatial-expression association, '
            'not a causal explanation or certification of the original multivariable gene-CV analysis.']
    text += ['', '## Threshold and verification checks','',
        '| Dataset / outcome | Same sign across 20/80, 25/75 and 30/70 tails |','|---|---:|']
    for (f,o),z in tails[tails.adjustment.eq('overlap_adjusted')].groupby(['family','outcome']):
        text.append(f'| {LABEL[f]} / {o} | {int(z.all_same_direction.sum())}/{len(z)} specimen–program contrasts |')
    text += ['', 'Threshold agreement assesses sensitivity of these fixed predictions, not independent biological '
        'replication. No new spot-level significance counts or blanket pathway-conservation percentage is used.','',
        'Numerical verification includes source IDs/targets/residuals, training-only means and medians, '
        'reconstructed full and excluded scores, raw program means, independent stratified weighting, '
        'specimen aggregation, exact common members and saved-model reproduction. Refit diagnostics are '
        'reported in the [fit record](refit_complete.json); independent family checks are linked below.','',
        'The independent centered-ridge check, using the actual deterministic PCA training scores, '
        'matches all three models within 1e-6; coefficients and reconstructed intercepts match exactly. '
        'The preliminary PCA-design and discrete-quantile checker discrepancies are preserved and '
        'explained in [numerical verification notes](NUMERICAL_VERIFICATION.md).','',
        '- [COAD independent checks](coad/independent_checks.json), [Visium independent checks](idc_visium/independent_checks.json), [common-member checks](comparison_checks.json).',
        '- [Independent raw IDC common-member effects and patient aggregation](idc_comparator_independent_checks.json).',
        '- [Specimen prediction quality](specimen_prediction_quality.csv), [gene comparison summary](gene_predictability_comparisons.csv), [gene pairs](gene_predictability_pairs.csv).',
        '- [All program comparisons](program_comparisons.csv), [comparison summary](program_comparison_summary.csv), [coverage](common_member_coverage.csv), [threshold sensitivity](tail_sensitivity.csv).',
        '- [COAD specimen effects](coad/specimen_program_effects.csv), [Visium specimen effects](idc_visium/specimen_program_effects.csv), [Visium score changes](idc_visium/score_changes.csv).',
        '- [Overview figure](figures/external_overview.pdf), [program contrasts](figures/external_programs.pdf).','',
        '## Consequence for the revision','',
        'Retain application of the prediction-error framework to additional datasets and the supported '
        'predictability/error-structure associations. Reassess each expression program separately, with '
        'coverage and patient/specimen directions visible. Limited Visium overall MAE utility and '
        'conditional-versus-raw ordering must remain explicit. Generic Q4 is not established as a uniform '
        'EMT-zone class by a large EMT absolute-error contrast. This stage completes the authorized '
        'external sequence; figure-level manuscript integration and any stronger inferential claims remain '
        'separate tasks.','']
    (OUT/'RESULTS.md').write_text('\n'.join(text))
    (OUT/'figures').mkdir(exist_ok=True)
    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    fig.suptitle('Whole-section utility; Visium cross-gene comparisons exclude paired P07',fontsize=12)
    colors={'coad':'#d39140','idc_visium':'#3b899d'}
    for f,z in quality.groupby('family'):
        for r in z.itertuples():axs[0].barh(f'{LABEL[f]} {r.specimen_group}',100*r.relative_mae_gain,color=colors[f])
    axs[0].axvline(0,color='grey',lw=1);axs[0].set(xlabel='MAE improvement over training median (%)',title='Prediction utility by known specimen')
    z=gp[(gp.family!='idc_visium')|gp.scope.eq('excluding_P07')]
    for r in z.itertuples():axs[1].barh(f'{LABEL[r.family]} / {LABEL[r.idc_cohort]}',r.spearman,color=colors[r.family])
    axs[1].set(xlabel='Cross-gene Spearman correlation',title='Predictability across measured genes',xlim=(-.1,1))
    axs[1].axvline(0,color='grey',lw=1)
    for ext in ['pdf','png']:fig.savefig(OUT/'figures'/f'external_overview.{ext}',dpi=160)
    plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(14,5),layout='constrained',sharey=True)
    for k,o in enumerate(['observed','signed','absolute']):
        for j,f in enumerate(['coad','idc_visium']):
            p=primary[f][primary[f].outcome.eq(o)].set_index('pathway').reindex(selected)
            axs[k].scatter(p['mean'],np.arange(len(selected))+(j-.5)*.15,color=colors[f],label=LABEL[f])
        axs[k].axvline(0,color='grey',lw=1);axs[k].set(title=o.capitalize(),xlabel='Standardized Q4−Q1 contrast',yticks=np.arange(len(selected)),yticklabels=[n.removeprefix('HALLMARK_').replace('_',' ') for n in selected])
    axs[2].legend();fig.suptitle('Program-excluded, overlap-adjusted contrasts; platform-specific measured members')
    for ext in ['pdf','png']:fig.savefig(OUT/'figures'/f'external_programs.{ext}',dpi=160)
    plt.close(fig)
    (OUT/'summary.json').write_text(json.dumps(dict(status='complete',new_visium_fits=3,reused_visium_fits=24,reused_coad_fits=12,
        n_coad_patients=4,n_visium_known_specimen_groups=9,visium_unrelated_donors_confirmed=False,scope='bounded external replication sequence'),indent=2)+'\n')
    print('External report complete')


if __name__=='__main__':main()

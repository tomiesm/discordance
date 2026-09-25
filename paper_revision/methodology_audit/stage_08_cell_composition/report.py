"""Cell-composition checkpoint before external replication."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent


def main():
    mixture=pd.read_csv(OUT/'cell_mixture.csv');spot=pd.read_csv(OUT/'whole_spot_effects.csv');within=pd.read_csv(OUT/'within_lineage_effects.csv')
    m=mixture[(mixture.kind=='group')&(mixture.width_um==800)&mixture.label.isin(['Tumor','Stromal','Macrophage'])]
    (OUT/'figures').mkdir(exist_ok=True)
    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    samples=['NCBI785','NCBI784','NCBI783'];colors=['#2d7d9a','#64a7ba','#ce8b40']
    for i,sample in enumerate(samples):
        g=m[m['sample']==sample].set_index('label').loc[['Tumor','Stromal','Macrophage']]
        x=np.arange(3)+(i-1)*.23
        axs[0].errorbar(x,100*g.difference,yerr=100*np.stack([g.difference-g.ci_low,g.ci_high-g.difference]),fmt='o',color=colors[i],label=sample+' / '+('P07' if i<2 else 'P08'))
    axs[0].set(xticks=np.arange(3),xticklabels=['Tumor','Stromal','Macrophage'],ylabel='Q4−Q1 cell fraction (percentage points)',title='Source-cell mixtures in original B1 groups')
    axs[0].axhline(0,color='grey',lw=1);axs[0].legend(fontsize=8)
    e=spot[(spot.endpoint=='HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION')&(spot.outcome=='signed')&(spot.width_um==800)]
    for i,sample in enumerate(samples):
        g=e[e['sample']==sample].set_index('adjustment').loc[['unadjusted','technical','composition']]
        axs[1].plot(np.arange(3),g.estimate,'o-',color=colors[i],label=sample)
    axs[1].set(xticks=np.arange(3),xticklabels=['Unadjusted','Technical','Composition'],ylabel='Signed EMT residual Q4−Q1 (log1p units)',title='Program-excluded groups, same eligible spots')
    axs[1].axhline(0,color='grey',lw=1)
    fig.suptitle('Three sections from two patients; source annotations and conditional spatial intervals')
    for ext in ['png','pdf']:fig.savefig(OUT/'figures'/f'composition_effects.{ext}',dpi=160)
    plt.close(fig)
    text=['# Source-cell composition and lineage follow-up','',
        '2026-09-20. Completed second step of the authorized sequence. B1 and all original files remain unchanged. '
        'This analysis covers NCBI785/NCBI784 from P07 and NCBI783 from P08 only; source cell labels are RNA-derived '
        'supportive annotations, not independent lineage ground truth. See [protocol](PROTOCOL.md).','',
        '## Composition associations are patient-dependent','',
        'QC source-cell fractions among spots with >=20 assigned QC cells, using the original whole-section '
        'B1 quartile cutoffs. Numbers are percentage-point Q4−Q1 differences.','',
        '| Section / patient | Tumor | Stromal | Macrophage |','|---|---:|---:|---:|']
    for sample in samples:
        g=m[m['sample']==sample].set_index('label')
        text.append(f'| {sample} / {"P07" if sample!="NCBI783" else "P08"} | '+' | '.join(f'{100*g.loc[label,"difference"]:+.1f}' for label in ['Tumor','Stromal','Macrophage'])+' |')
    text += ['',
        'P07 Q4 therefore has fewer source tumor cells and more stromal/macrophage cells; P08 Q4 has more tumor '
        'cells and fewer macrophages. The two P07 sections are repeated evidence within one patient. This '
        'supports a composition association in particular specimens, not one uniform Q4 microenvironment.','',
        'Eligibility is uneven. NCBI785 contributes 885 Q1 and 255 Q4 spots, NCBI784 444/306, and NCBI783 '
        '460/464. These are subsets of the original groups; do not describe the fractions as measurements '
        'of every Q1/Q4 spot. All source labels, support sizes and 800/1600 µm block intervals are retained.','',
        '## What composition adjustment changes','',
        'All 27 eligible Hallmarks and 33 unique measured named markers were assessed. Whole-spot outcomes '
        'use the tested gene/program excluded from grouping and conditioning. The unadjusted, technical and '
        'composition regressions use the same eligible observations within each comparison.','',
        '- In P07, EPCAM observed-expression contrasts are −0.840/−0.611 log1p units before adjustment and '
        '−0.019/−0.079 after technical/composition adjustment. Much of the observed epithelial-marker '
        'difference is compatible with the measured mixture differences; this is not proof of a causal mechanism.',
        '- P07 CD163 observed contrasts attenuate from +0.871/+0.821 to +0.222/+0.362. P08 attenuates from '
        '−1.386 to −0.076. Macrophage abundance and CD163 expression within macrophages are different endpoints.',
        '- The measured EMT-Hallmark observed contrast after full adjustment is negative in all three '
        'sections (−0.112, −0.158, −0.161), whereas its signed residual contrast remains positive in P07 '
        '(+0.421/+0.352) and approaches zero in P08 (−0.034). A signed prediction error is not a direct '
        'measure of EMT activation.',
        '- Absolute-error contrasts remain positive for all 60 assessed endpoints in each P07 section, '
        'and for 46/60 in P08 after composition adjustment. In P08, the EMT absolute-error contrast '
        'attenuates to +0.005 (800 µm interval −0.054 to +0.091). Broad error associations and individual '
        'program associations need separate scopes.','',
        '## Direct within-lineage expression','',
        'QC cell expression is first averaged within assigned spots, then compared between gene/program-'
        'excluded Q1/Q4 locations. The primary rule requires >=5 cells of the tested lineage per spot; '
        '>=20 is a sensitivity. Adjustment includes other-gene cell counts/detection, cell area and, for '
        'the source Tumor group, its DCIS fraction. This estimates measured cell expression, not cell-level '
        'histology prediction residuals.','',
        'Within source Tumor cells, adjusted EMT-Hallmark mean log-expression contrasts are +0.024 '
        '(NCBI785), −0.016 (NCBI784) and −0.056 (NCBI783). At 1600 µm, intervals are +0.003 to +0.039, '
        '−0.046 to +0.007, and −0.093 to −0.024 respectively. These small heterogeneous effects do not '
        'supply replicated positive EMT enrichment in generic Q4. The >=20-cell estimates retain the '
        'same signs, but P08 then has only 30 Q1 spots. Individual TF results and all other programs are saved.','',
        'Within source macrophages, adjusted CD163 contrasts have intervals crossing zero in all three '
        'sections at 800 µm. The P07 increase in macrophage fraction is thus distinct from demonstrating '
        'greater CD163 expression within those macrophages. Small CD163 signal in other source lineages '
        'must not be interpreted as validated lineage switching; labeling, segmentation and transcript '
        'assignment remain possible explanations.','',
        '## Relationship to earlier EMT experiments','',
        'The verified tumor epithelial/TF coexpression neighborhood analyses already controlled stromal '
        'abundance and other composition variables. Their positive P08 association is with a signed, '
        'marker-disjoint EMT residual, not generic high-error Q4. Their existing scale/sampling sensitivities '
        'and heterogeneous P07 findings remain. They were reviewed and reused, not counted as a new '
        'independent replication or rerun with thresholds selected for improvement.','',
        'The available evidence can answer the reviewers with patient-specific mixture and within-lineage '
        'results. It does not identify generic discordance with tumor-cell EMT. Source annotation coverage '
        'does not extend to all eight patients, and measured cell fractions are restricted to successfully '
        'mapped QC cells.','',
        '## Inference and verification','',
        'Intervals use 499 resamples at 800 and 1600 µm, conditional on fixed fitted predictions, groups '
        'and covariates. No full-pipeline uncertainty, causal interpretation, multiplicity-adjusted discovery '
        'claim or population test across two patients is made. All primary fitted designs are full rank '
        'and all reported primary interval configurations meet the valid-draw threshold.','',
        '- [Whole-spot effects](whole_spot_effects.csv), [cell mixtures](cell_mixture.csv), [within-lineage effects](within_lineage_effects.csv), [support](support.csv).',
        '- [Figure](figures/composition_effects.pdf), [endpoint members](endpoint_register.json).',
        '- [Construction/solver checks](checks.json), [independent fraction aggregation and statsmodels fits](independent_checks.json).',
        '- [Independent dense cell-matrix aggregation and lineage fits](lineage_checks.json); '
        '[fixed spatial-grid origin correction](BLOCK_ORIGIN_CORRECTION.md).',
        '- Block-weighted fits were independently checked against explicitly repeated observations; source '
        'fractions were reconciled against the earlier verified mappings. The grid origin is the finite '
        'coordinate minimum of all modeled spots in the section, fixed across endpoints.','',
        '## Next step','',
        'Proceed to external replication using the same separation of observed expression, signed residual '
        'and absolute error, with specimen-grouped Visium replacement fits before recalculating its scores.','']
    (OUT/'RESULTS.md').write_text('\n'.join(text))
    (OUT/'summary.json').write_text(json.dumps(dict(status='complete',n_patients=2,n_sections=3,n_endpoints=60,next='external replication'),indent=2)+'\n')


if __name__=='__main__':main()

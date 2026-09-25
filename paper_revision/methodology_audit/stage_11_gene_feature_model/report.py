"""Replay cached annotation categories and report the bounded model audit."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent;REPO=OUT.parents[1]/'clean_repo'
sys.path.insert(0,str(REPO))
from src.gene_annotations import _classify_genes_by_name, _classify_go_terms


def main():
    checks=[]
    for cohort in ['biomarkers','10x_janesick']:
        a=pd.read_csv(REPO/'outputs/phase3/cache'/f'go_slim_{cohort}.csv').fillna('Unknown')
        if cohort=='10x_janesick':
            replay=_classify_genes_by_name(a.gene.tolist())
            assert (replay.primary_function.to_numpy()==a.primary_function.to_numpy()).all()
            assert (replay.all_functions.to_numpy()==a.all_functions.to_numpy()).all()
            checks.append(dict(check=cohort+':exact_name_fallback_replay',n_genes=len(a),passed=True))
        else:
            labels=[_classify_go_terms(v.split(';')) if v!='Unknown' else 'Unknown' for v in a.all_functions]
            assert labels==a.primary_function.tolist()
            checks.append(dict(check=cohort+':namespace_classifier_replay',n_genes=len(a),passed=True))
    (OUT/'annotation_replay_checks.json').write_text(json.dumps(dict(status='pass',n_checks=len(checks),checks=checks),indent=2)+'\n')
    m=pd.read_csv(OUT/'model_metrics.csv');c=pd.read_csv(OUT/'coefficients.csv');p=pd.read_csv(OUT/'cv_predictions.csv');loo=pd.read_csv(OUT/'patient_influence.csv')
    current=m[m.version.eq('revised_all_sections')]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(1,2,figsize=(9,4),layout='constrained')
    for axis,cohort,title in zip(ax,['biomarkers','10x_janesick'],['IDC discovery','IDC validation']):
        z=p[p.cohort.eq(cohort)&p.version.eq('revised_all_sections')&p.model.eq('numeric_with_spatial')]
        r=current[current.cohort.eq(cohort)&current.model.eq('numeric_with_spatial')].iloc[0]
        axis.scatter(z.actual,z.predicted,s=12,alpha=.55,c='#2879ad',edgecolors='none');lim=[min(z.actual.min(),z.predicted.min())-.03,max(z.actual.max(),z.predicted.max())+.03]
        axis.plot(lim,lim,color='.6',lw=1);axis.set(xlim=lim,ylim=lim,title=f'{title}: gene-CV r={r.pearson:.3f}',xlabel='Assessed gene predictability',ylabel='Held-out gene-feature model prediction');axis.set_aspect('equal')
    fig.suptitle('Predictability associated with expression and spatial features\n280 measured genes per cohort; patient-balanced all-section summaries',fontsize=11)
    fig.savefig(OUT/'numeric_gene_cv.png',dpi=180);fig.savefig(OUT/'numeric_gene_cv.pdf');plt.close(fig)
    coef=c[c.version.eq('revised_all_sections')&c.feature.eq('spatial_autocorrelation')]
    check=json.loads((OUT/'checks.json').read_text());scaling=[v for v in check['checks'] if 'training_only_vs_global' in v['check']]
    md=lambda x:x.to_markdown(index=False,floatfmt='.3f')
    lines=['# Gene-feature model audit: results','',
      '2026-09-20. **Retain the association between spatial expression organization and assessed gene predictability. Rebuild the figure with all-section, patient-balanced summaries and emphasize the numerical feature model.** The original OLS and gene-CV calculations reproduce, but their inputs and biological labels require correction. This is an explanatory analysis of measured genes; it does not improve the histology-to-expression predictor or validate RNA-free prediction of unseen genes.','',
      '## What was checked and changed','',
      '- Original response: per-encoder Pearson correlation pooled across all held-out spots/folds, then averaged across encoders. It was not an average of per-section correlations. The revised response averages the existing encoder-averaged section correlations within patient, then equally across patients.','- Original expression moments include all raw section spots rather than exactly the modeled spots; the variance averages within-section variances and omits between-section mean variation. Revised first/second moments use exactly modeled spots with equal-section/equal-patient weights and include mixture variance.','- Original Moran’s I is calculated only on the first three configured sections. The revised feature covers all 11 discovery and seven validation sections, using the same six-neighbor graph definition and patient-balanced aggregation.','- Five-fold gene splits remain seed 42. Training-fold scaling and categorical encoding are explicit; numerical OLS with an intercept is invariant to this scaling change here. Largest full-model held-out prediction change from global to fold-specific preprocessing: '+', '.join(f"{x['check'].split(':')[0]} {x['max_abs']:.2g}" for x in scaling)+'. No harmful scaling leakage was demonstrated.','',
      'These changes alter the response and feature definitions. Better correlations between versions are not measured gains on one unchanged prediction task. Within each revised cohort, the incremental comparison of models does use the same response, genes and folds.','',
      '## Retained numerical result','',md(current[['cohort','model','pearson','spearman','mae','relative_MSE_gain','predictive_R2']]),'',
      'Adding Moran’s I to expression mean/CV/pathway count increases revised gene-CV Pearson r from .634 to .801 in discovery and .717 to .808 in validation. Mean-only models give .635 and .701. The numerical model’s positive spatial coefficients are .110 and .116 correlation-response units per one feature SD, conditional on the other numerical features. This supports an association beyond those specified expression features; it does not identify a causal mechanism.','',md(coef[['cohort','model','coefficient']]),'',
      '[Numerical model cross-validation plot](numeric_gene_cv.png). Raw coefficients, fold predictions, all old/response-only/revised comparisons and rank diagnostics are saved as CSV files. All 24 model fits have full-rank gene-CV folds.','',
      'Patient omission recomputes cohort features and responses and repeats the full cached-annotation model. The spatial coefficient remains positive in all eight omissions; CV r ranges .823–.839 in discovery and .775–.840 in validation. This is a sensitivity of cohort summaries, not a fresh held-out-patient test of the gene-feature model.','',md(loo[['cohort','omitted_patient','spatial_coefficient','pearson','relative_MSE_gain']]),'',
      '## Annotation issue and figure consequence','',
      'The BioMart query in the original code requests `namespace_1003`, and then searches those namespace strings for detailed function keywords. In the discovery cache, 277 genes have namespace strings and become “Other”; three are “Unknown”. Validation instead exactly reproduces the gene-name-prefix fallback for all 280 genes (235 “Other”). These are not equivalent curated GO-Slim functional annotations. The classifier outputs were independently replayed from the caches, without a live annotation update.','',
      'The localization cache uses coarse, priority-ordered categories from the first search result, with 96 discovery and 78 validation genes “Unknown”. Its current biological correctness was not independently curated. Keep the full cached-annotation model as a labeled sensitivity, and avoid a main-text conclusion that a consistently curated functional ontology explains the cohort difference. The numerical feature model does not depend on these cached categories.','',
      '## Interpretation and verification','',
      'RNA-derived mean, variability and Moran features require measured transcription. Random folds of genes test prediction within these panels; genes share programs and are not independent biological replicates. The two cohorts also share 90 genes. Neither ordinary gene-wise OLS p-values nor the same gene CV establishes generalization to unrelated gene families, new patient populations or unmeasured transcripts. Use effect sizes and model comparisons, without causal “governs” or “intrinsically invisible” language.','',
      'Original OLS coefficients and archived gene-CV predictions reproduce to floating-point tolerance. A separate solver checks all new OLS/CV predictions. Scalar Moran calculations check five genes in each of 18 sections, and direct spot-weight calculations verify moments and CV. There are 100 recorded construction/reproduction checks plus two annotation replay checks, all passing.','',
      '[Protocol](PROTOCOL.md), [checks](checks.json), [annotation replay](annotation_replay_checks.json), [model comparisons](model_metrics.csv), [annotation inventory](annotation_audit.csv), [patient influence](patient_influence.csv).','']
    (OUT/'RESULTS.md').write_text('\n'.join(lines))
    assert check['status']=='pass'
    (OUT/'summary.json').write_text(json.dumps(dict(status='complete',n_cohorts=2,n_sections=18,n_genes_per_cohort=280,n_models=24,numerical_gene_cv_pearson=dict(zip(current[current.model.eq('numeric_with_spatial')].cohort,current[current.model.eq('numeric_with_spatial')].pearson)),annotation_status='Inconsistent provenance; numerical model primary, cached model sensitivity'),indent=2)+'\n')
    print('Stage 11 report and annotation replay complete')


if __name__=='__main__':main()

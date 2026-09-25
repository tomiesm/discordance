"""Original gene-feature reproduction and patient-balanced all-section reassessment."""
from src.paper.paths import PROJECT_ROOT, STAGE_SOURCE, CELL_SOURCE, ANALYSIS_ROOT, CELL_ROOT, stage_dir, cell_dir
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold
import statsmodels.api as sm
import yaml
OUT = stage_dir('features')
AUDIT = ANALYSIS_ROOT
ROOT = PROJECT_ROOT
REPO = PROJECT_ROOT
from src.spatial import build_spatial_weights, morans_i
from src.pathways import load_gene_sets
NUM = ['mean_expression', 'cv_expression', 'pathway_count', 'spatial_autocorrelation']
CAT = ['primary_localization', 'primary_function']

def encode(train, test, columns, categorical):
    means = train[columns].mean()
    sd = train[columns].std(ddof=0)
    sd = sd.where(sd > 0, 1)
    a = (train[columns].fillna(means) - means) / sd
    b = (test[columns].fillna(means) - means) / sd
    names = list(columns)
    xa = [a.to_numpy()]
    xb = [b.to_numpy()]
    unseen = []
    if categorical:
        for col in CAT:
            levels = sorted(train[col].unique())
            unseen += [f'{col}:{v}' for v in sorted(set(test[col]) - set(levels))]
            for level in levels[1:]:
                xa.append(train[col].eq(level).to_numpy()[:, None])
                xb.append(test[col].eq(level).to_numpy()[:, None])
                names.append(col + '=' + level)
    return (np.column_stack([np.ones(len(train)), *xa]).astype(float), np.column_stack([np.ones(len(test)), *xb]).astype(float), ['intercept'] + names, unseen)

def cv(features, columns, categorical, global_scaling=False):
    y = features.mean_pearson.to_numpy()
    out = np.empty(len(y))
    baseline = np.empty(len(y))
    folds = np.zeros(len(y), int)
    diagnostics = []
    if global_scaling:
        whole, _, _, _ = encode(features, features, columns, categorical)
    for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=42).split(features)):
        if global_scaling:
            a = whole[tr]
            b = whole[te]
            unseen = []
        else:
            a, b, _, unseen = encode(features.iloc[tr], features.iloc[te], columns, categorical)
        beta, _, rank, _ = np.linalg.lstsq(a, y[tr], rcond=None)
        out[te] = b @ beta
        baseline[te] = y[tr].mean()
        folds[te] = k
        independent = sm.OLS(y[tr], a).fit().predict(b)
        assert np.max(np.abs(out[te] - independent)) < 1e-09
        diagnostics.append(dict(fold=k, n_train=len(tr), n_test=len(te), rank=int(rank), n_columns=a.shape[1], unseen_categories=';'.join(unseen)))
    mse = np.mean((y - out) ** 2)
    bmse = np.mean((y - baseline) ** 2)
    metrics = dict(pearson=float(pearsonr(y, out).statistic), spearman=float(spearmanr(y, out).statistic), mae=float(np.mean(np.abs(y - out))), baseline_mae=float(np.mean(np.abs(y - baseline))), mse=float(mse), baseline_mse=float(bmse), relative_MSE_gain=float(1 - mse / bmse), predictive_R2=float(1 - np.sum((y - out) ** 2) / np.sum((y - y.mean()) ** 2)), n_rank_deficient_folds=sum((d['rank'] < d['n_columns'] for d in diagnostics)))
    return (out, baseline, folds, metrics, diagnostics)

def features_from_sections(section, static, exclude=None):
    s = section[section.patient.ne(exclude)] if exclude else section
    p = s.groupby(['patient', 'gene'])[['expression_mean', 'second_moment', 'morans_i', 'prediction_pearson']].mean().reset_index()
    a = p.groupby('gene')[['expression_mean', 'second_moment', 'morans_i', 'prediction_pearson']].mean().reindex(static.gene).reset_index()
    f = static.copy()
    f['mean_pearson'] = a.prediction_pearson.to_numpy()
    f['mean_expression'] = a.expression_mean.to_numpy()
    f['cv_expression'] = np.sqrt(np.maximum(a.second_moment.to_numpy() - a.expression_mean.to_numpy() ** 2, 0)) / (a.expression_mean.to_numpy() + 1e-06)
    f['spatial_autocorrelation'] = a.morans_i.to_numpy()
    return (f, p)

def main():
    checks = []
    sets = load_gene_sets(str(REPO/'data/gene_sets/h.all.v2024.1.Hs.symbols.gmt'))
    quality = pd.read_csv(AUDIT/'genes/prediction_quality.csv')
    quality = quality[quality.grouping.eq('full') & quality.quartile.eq('Q1')]
    for cohort in ['biomarkers', '10x_janesick']:
        dest = OUT / cohort
        dest.mkdir(exist_ok=True)
        genes = json.loads((REPO/'data/v3'/f'gene_list_{cohort}.json').read_text())
        old = pd.read_csv(REPO/'outputs/phase3/gene_predictability'/cohort/'gene_features.csv')
        assert old.gene.tolist() == genes
        loc=pd.read_csv(AUDIT/'programs/arrays'/cohort/'locations.csv');base=REPO/'outputs/predictions'/cohort/'uni/ridge'
        y=np.concatenate([np.load(base/f'fold{k}/test_targets.npy') for k in range(4)]).astype(float)
        ids=sum([json.loads((base/f'fold{k}/test_spot_ids.json').read_text()) for k in range(4)],[]);assert np.array_equal(ids,loc.spot_id)
        genes=json.loads((REPO/'data/v3'/f'gene_list_{cohort}.json').read_text());assert genes==old.gene.tolist()
        sections=[];spot_weights=np.empty(len(y))
        for sample,ss in loc.groupby('sample_id',sort=False):
            ix=ss.index.to_numpy();v=y[ix];mean=v.mean(axis=0);second=np.mean(v*v,axis=0);z=v-mean
            w=build_spatial_weights(ss[['x_um','y_um']].to_numpy(),n_neighbors=6)
            denom=(z*z).sum(axis=0);moran=np.divide(len(v)/w.sum()*np.sum(z*(w@z),axis=0),denom,out=np.zeros(len(genes)),where=denom>0)
            for j in [0,37,101,179,279]:
                expected=morans_i(v[:,j],w);err=abs(expected-moran[j]);assert err<1e-10
                checks.append(dict(check=f'{cohort}/{sample}/{genes[j]}:scalar_Moran',max_abs=float(err),passed=True))
            pred=quality[quality.cohort.eq(cohort)&quality['sample'].eq(sample)].set_index('gene').loc[genes,'whole_section_gene_pearson'].to_numpy()
            for j,g in enumerate(genes):sections.append(dict(cohort=cohort,sample=sample,patient=ss.patient.iloc[0],gene=g,expression_mean=mean[j],second_moment=second[j],morans_i=moran[j],prediction_pearson=pred[j]))
            patient=ss.patient.iloc[0];nsec=loc[loc.patient.eq(patient)].sample_id.nunique();spot_weights[ix]=1/loc.patient.nunique()/nsec/len(ix)
        section=pd.DataFrame(sections);section.to_csv(dest/'section_features.csv',index=False)
        revised,patient=features_from_sections(section,old);patient.to_csv(dest/'patient_features.csv',index=False);revised.to_csv(dest/'revised_features.csv',index=False)

if __name__ == '__main__':
    main()

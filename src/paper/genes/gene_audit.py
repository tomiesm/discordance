"""All-gene biological and prediction-quality audit with leave-one-gene-out groups."""
from src.paper.paths import PROJECT_ROOT, STAGE_SOURCE, CELL_SOURCE, ANALYSIS_ROOT, CELL_ROOT, stage_dir, cell_dir
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata, t, binomtest
from statsmodels.stats.multitest import multipletests
OUT = stage_dir('genes')
AUDIT = ANALYSIS_ROOT
from src.paper.cohort_data import load_cohort, REPO
from src.discordance import compute_conditional_discordance
from src.pathways import load_gene_sets
MARKERS = {'epithelial': 'EPCAM KRT8 KRT18 KRT19 KRT17 CDH1 MUC1 SCUBE2 ESR1 FOXA1 TP63 COL17A1 DSP CLDN4 GATA3'.split(), 'canonical_EMT': 'VIM CDH2 FN1 SNAI1 SNAI2 ZEB1 ZEB2'.split(), 'ECM_stromal': 'HSPG2 FBLN1 COL4A1 LAMB1 MMP2 PDGFRB TIMP1 COL1A1 COL3A1 DCN FAP ACTA2 LUM'.split(), 'macrophage': 'CD163 CD68 CSF1R CD14 ITGAM'.split(), 'proliferation': 'MKI67 CENPF PCLAF'.split()}

def tails(s):
    a, b = np.quantile(s, [0.25, 0.75])
    return (s <= a, s >= b)

def rho(x, y):
    return float(spearmanr(x, y).statistic) if np.std(x) > 0 and np.std(y) > 0 else np.nan

def overlap_weights(counts, detect, lo, hi):
    grid = np.linspace(0, 1, 6)[1:-1]
    strata = 5 * np.searchsorted(np.quantile(counts, grid), counts, side='right') + np.searchsorted(np.quantile(detect, grid), detect, side='right')
    n1 = np.bincount(strata[lo], minlength=25)
    n4 = np.bincount(strata[hi], minlength=25)
    ok = (n1 >= 10) & (n4 >= 10)
    h = np.divide(2 * n1 * n4, n1 + n4, out=np.zeros(25), where=n1 + n4 > 0) * ok
    a = np.divide(h, n1, out=np.zeros(25), where=n1 > 0)[strata] * lo
    b = np.divide(h, n4, out=np.zeros(25), where=n4 > 0)[strata] * hi
    return (a, b, int(ok.sum()), float((lo & ok[strata]).sum() / lo.sum()), float((hi & ok[strata]).sum() / hi.sum()))

def pearson_cols(y, p):
    a = y - y.mean(axis=0)
    b = p - p.mean(axis=0)
    den = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))
    return np.divide((a * b).sum(axis=0), den, out=np.full(y.shape[1], np.nan), where=den > 0)

def summarize_patients(df):
    keys = ['cohort', 'patient', 'gene', 'grouping', 'adjustment', 'outcome']
    columns = ['effect', 'standardized_effect', 'Q1_mean', 'Q4_mean', 'Q1_retained', 'Q4_retained', 'count_log2_ratio_pseudocount1']
    patient = df.groupby(keys)[columns].mean().reset_index()
    patient.to_csv(OUT / 'patient_effects.csv', index=False)
    rows = []
    for key, g in patient.groupby([k for k in keys if k != 'patient']):
        for metric in ['effect', 'standardized_effect']:
            v = g[metric].dropna().to_numpy()
            n = len(v)
            if not n:
                continue
            mean = v.mean()
            se = v.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
            nonzero = v[v != 0]
            p = binomtest(int((nonzero > 0).sum()), len(nonzero), 0.5).pvalue if len(nonzero) else 1.0
            rows.append(dict(zip([k for k in keys if k != 'patient'], key)) | dict(metric=metric, n_patients=n, mean=mean, ci_low=mean - t.ppf(0.975, n - 1) * se, ci_high=mean + t.ppf(0.975, n - 1) * se, n_positive=int((v > 0).sum()), n_negative=int((v < 0).sum()), sign_p=p, loo_min=float(((v.sum() - v) / (n - 1)).min()) if n > 1 else np.nan, loo_max=float(((v.sum() - v) / (n - 1)).max()) if n > 1 else np.nan))
    cohort = pd.DataFrame(rows)
    cohort['sign_fdr'] = np.nan
    for _, g in cohort.groupby(['cohort', 'grouping', 'adjustment', 'outcome', 'metric']):
        cohort.loc[g.index, 'sign_fdr'] = multipletests(g.sign_p, method='fdr_bh')[1]
    cohort.to_csv(OUT / 'cohort_effects.csv', index=False)
    return (patient, cohort)

def main():
    OUT.mkdir(exist_ok=True)
    contrasts = []
    predictive = []
    maps = []
    coverage = []
    checks = []
    profiles = []
    sets = load_gene_sets(str(REPO / 'data/gene_sets/h.all.v2024.1.Hs.symbols.gmt'))
    MARKERS['complement'] = sets['HALLMARK_COMPLEMENT']
    for cohort in ['biomarkers', '10x_janesick']:
        y32, r, loc, genes = load_cohort(cohort)
        y = y32.astype(float)
        signed = r.mean(axis=0)
        absolute = np.abs(r).mean(axis=0)
        pred = y - signed
        counts = np.rint(np.expm1(y))
        assert np.max(np.abs(counts - np.expm1(y))) < 0.1
        total_y = y.sum(axis=1)
        total_abs = absolute.sum(axis=1)
        total_counts = counts.sum(axis=1)
        det = (y > 0).sum(axis=1)
        baseline = np.empty_like(y)
        for patient, g in loc.groupby('patient', sort=False):
            ix = g.index.to_numpy()
            train = loc.patient.ne(patient).to_numpy()
            baseline[ix] = np.abs(y[ix] - np.median(y[train], axis=0))
        sec = {}
        for sample, g in loc.groupby('sample_id', sort=False):
            ix = g.index.to_numpy()
            ss = loc.iloc[ix].conditional.to_numpy()
            qedges = np.quantile(ss, [0.25, 0.5, 0.75])
            q = np.select([ss <= qedges[0], ss <= qedges[1], ss < qedges[2]], [1, 2, 3], default=4)
            cor = np.nanmean(np.stack([pearson_cols(y[ix], y[ix] - r[e, ix, :]) for e in range(3)]), axis=0)
            sec[sample] = (ix, ss, q, cor)
        del r
        for cat, members in MARKERS.items():
            for gene in members:
                coverage.append(dict(cohort=cohort, category=cat, gene=gene, measured=gene in genes))
        for j, gene in enumerate(genes):
            omitted = compute_conditional_discordance((total_abs - absolute[:, j]) / (len(genes) - 1), total_y - y[:, j])
            if j in [0, len(genes) // 2, len(genes) - 1]:
                others = np.arange(len(genes)) != j
                direct = compute_conditional_discordance(absolute[:, others].mean(axis=1), y[:, others].sum(axis=1))
                err = float(np.max(np.abs(omitted - direct)))
                assert err < 1e-10
                checks.append(dict(check=f'{cohort}/{gene}:excluded_score_direct', max_abs=err, passed=True))
            for sample, (ix, full, q, corr) in sec.items():
                patient = loc.iloc[ix[0]].patient
                ident = dict(cohort=cohort, patient=patient, sample=sample, gene=gene)
                low, high = tails(full)
                elo, ehi = tails(omitted[ix])
                maps.append(dict(**ident, rank_correlation=rho(full, omitted[ix]), Q1_overlap=float((low & elo).sum() / low.sum()), Q4_overlap=float((high & ehi).sum() / high.sum())))
                values = np.stack([y[ix, j], counts[ix, j], (y[ix, j] > 0).astype(float), pred[ix, j], signed[ix, j], absolute[ix, j]], axis=1)
                names = ['observed_log', 'observed_counts', 'detected', 'predicted_log', 'signed', 'absolute']
                for k in range(1, 5):
                    mask = q == k
                    profiles.append(dict(**ident, quartile=k, n_spots=int(mask.sum()), **{name: float(values[mask, c].mean()) for c, name in enumerate(names)}, baseline_mae=float(baseline[ix, j][mask].mean())))
                for grouping, lo, hi in [('full', low, high), ('gene_excluded', elo, ehi)]:
                    w1, w4, nstrata, ret1, ret4 = overlap_weights(total_counts[ix] - counts[ix, j], det[ix] - (y[ix, j] > 0), lo, hi)
                    sd = np.sqrt(((lo.sum() - 1) * values[lo].var(axis=0, ddof=1) + (hi.sum() - 1) * values[hi].var(axis=0, ddof=1)) / (lo.sum() + hi.sum() - 2))
                    for adjustment, a, b, retained1, retained4 in [('unadjusted', lo.astype(float), hi.astype(float), 1.0, 1.0), ('overlap_adjusted', w1, w4, ret1, ret4)]:
                        means1 = np.average(values, axis=0, weights=a) if a.sum() > 0 else np.full(6, np.nan)
                        means4 = np.average(values, axis=0, weights=b) if b.sum() > 0 else np.full(6, np.nan)
                        effect = means4 - means1
                        for c, name in enumerate(names):
                            contrasts.append(dict(**ident, grouping=grouping, adjustment=adjustment, outcome=name, Q1_mean=means1[c], Q4_mean=means4[c], effect=effect[c], standardized_effect=effect[c] / sd[c] if sd[c] > 0 else np.nan, Q1_retained=retained1, Q4_retained=retained4, n_overlap_strata=nstrata, count_log2_ratio_pseudocount1=float(np.log2((means4[1] + 1) / (means1[1] + 1))) if name == 'observed_counts' else np.nan))
                    for label, mask in [('Q1', lo), ('Q4', hi)]:
                        mae = float(absolute[ix, j][mask].mean())
                        bm = float(baseline[ix, j][mask].mean())
                        predictive.append(dict(**ident, grouping=grouping, quartile=label, n_spots=int(mask.sum()), mae=mae, baseline_mae=bm, relative_mae_gain=1 - mae / bm if bm > 0 else np.nan, signed_bias=float(signed[ix, j][mask].mean()), whole_section_gene_pearson=float(corr[j]), ensemble_quartile_spearman=rho(pred[ix, j][mask], y[ix, j][mask]), whole_section_mean_log=float(y[ix, j].mean()), whole_section_detection=float((y[ix, j] > 0).mean()), whole_section_sd=float(y[ix, j].std())))
            if (j + 1) % 40 == 0:
                print(cohort, j + 1, 'genes complete', flush=True)
        del y, y32, signed, absolute, counts, baseline, pred
    frames = {'section_effects': contrasts, 'prediction_quality': predictive, 'gene_exclusion_overlap': maps, 'marker_coverage': coverage, 'quartile_profiles': profiles}
    for name, rows in frames.items():
        pd.DataFrame(rows).to_csv(OUT / f'{name}.csv', index=False)
    patient, cohort = summarize_patients(pd.DataFrame(contrasts))
    (OUT / 'checks.json').write_text(json.dumps(dict(status='pass', checks=checks), indent=2) + '\n')
    print('Gene construction and patient summaries complete', flush=True)
if __name__ == '__main__':
    main()

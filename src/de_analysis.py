"""
Differential expression analysis for Phase 3.

Provides Wilcoxon rank-sum DE between discordant and concordant spot groups,
meta-DE aggregation across samples, and full expression loading utilities.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from scipy import stats
from statsmodels.stats.multitest import multipletests


def partition_spots_by_discordance(D_cond: np.ndarray,
                                   top_quantile: float = 0.75,
                                   bottom_quantile: float = 0.25
                                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Partition spots into discordant (top quartile) and concordant (bottom quartile).

    Args:
        D_cond: Conditional discordance scores (N_spots,)
        top_quantile: Upper quantile threshold
        bottom_quantile: Lower quantile threshold

    Returns:
        disc_mask: Boolean mask for discordant spots
        conc_mask: Boolean mask for concordant spots
    """
    q_top = np.quantile(D_cond, top_quantile)
    q_bot = np.quantile(D_cond, bottom_quantile)
    disc_mask = D_cond >= q_top
    conc_mask = D_cond <= q_bot
    return disc_mask, conc_mask


def wilcoxon_de(expression: np.ndarray, gene_names: List[str],
                disc_mask: np.ndarray, conc_mask: np.ndarray,
                min_frac_expressed: float = 0.10) -> pd.DataFrame:
    """Per-gene Wilcoxon rank-sum test between discordant and concordant groups.

    Args:
        expression: Dense expression matrix (N_spots, N_genes)
        gene_names: Gene names matching columns of expression
        disc_mask: Boolean mask for discordant spots
        conc_mask: Boolean mask for concordant spots
        min_frac_expressed: Minimum fraction of spots with nonzero expression
                           in at least one group to include gene

    Returns:
        DataFrame with columns: gene, pval, log2fc, cohens_d,
        frac_disc, frac_conc, mean_disc, mean_conc, fdr
    """
    expr_disc = expression[disc_mask]
    expr_conc = expression[conc_mask]
    n_disc = expr_disc.shape[0]
    n_conc = expr_conc.shape[0]

    records = []
    epsilon = 1e-6

    for g_idx, gene in enumerate(gene_names):
        vals_disc = expr_disc[:, g_idx]
        vals_conc = expr_conc[:, g_idx]

        frac_disc = np.mean(vals_disc > 0)
        frac_conc = np.mean(vals_conc > 0)

        if max(frac_disc, frac_conc) < min_frac_expressed:
            continue

        mean_disc = np.mean(vals_disc)
        mean_conc = np.mean(vals_conc)

        # Log2 fold change
        log2fc = np.log2((mean_disc + epsilon) / (mean_conc + epsilon))

        # Cohen's d
        pooled_var = ((np.var(vals_disc, ddof=1) * (n_disc - 1) +
                       np.var(vals_conc, ddof=1) * (n_conc - 1)) /
                      (n_disc + n_conc - 2))
        pooled_sd = np.sqrt(pooled_var) if pooled_var > 0 else epsilon
        cohens_d = (mean_disc - mean_conc) / pooled_sd

        # Wilcoxon rank-sum (Mann-Whitney U)
        try:
            stat, pval = stats.mannwhitneyu(vals_disc, vals_conc,
                                            alternative='two-sided')
        except ValueError:
            pval = 1.0
            stat = 0.0

        records.append({
            'gene': gene,
            'pval': pval,
            'log2fc': log2fc,
            'cohens_d': cohens_d,
            'frac_disc': frac_disc,
            'frac_conc': frac_conc,
            'mean_disc': mean_disc,
            'mean_conc': mean_conc,
            'n_disc': n_disc,
            'n_conc': n_conc,
        })

    if not records:
        return pd.DataFrame(columns=['gene', 'pval', 'log2fc', 'cohens_d',
                                     'frac_disc', 'frac_conc', 'mean_disc',
                                     'mean_conc', 'n_disc', 'n_conc', 'fdr'])

    df = pd.DataFrame(records)

    # BH FDR correction
    _, fdr, _, _ = multipletests(df['pval'].values, method='fdr_bh')
    df['fdr'] = fdr

    return df.sort_values('pval').reset_index(drop=True)


def meta_de(de_results: List[pd.DataFrame],
            fdr_threshold: float = 0.05,
            log2fc_threshold: float = 0.25) -> pd.DataFrame:
    """Aggregate DE results across samples, ranking by reproducibility.

    Args:
        de_results: List of per-sample DE DataFrames (from wilcoxon_de)
        fdr_threshold: FDR threshold for significance
        log2fc_threshold: Minimum absolute log2FC

    Returns:
        DataFrame with: gene, n_samples_sig, n_samples_tested,
        reproducibility, median_log2fc, mean_cohens_d
    """
    gene_stats: Dict[str, Dict] = {}

    for de_df in de_results:
        if de_df.empty:
            continue
        for _, row in de_df.iterrows():
            gene = row['gene']
            if gene not in gene_stats:
                gene_stats[gene] = {
                    'n_tested': 0,
                    'n_sig': 0,
                    'log2fcs': [],
                    'cohens_ds': [],
                }
            gene_stats[gene]['n_tested'] += 1
            is_sig = (row['fdr'] < fdr_threshold and
                      abs(row['log2fc']) > log2fc_threshold)
            if is_sig:
                gene_stats[gene]['n_sig'] += 1
            gene_stats[gene]['log2fcs'].append(row['log2fc'])
            gene_stats[gene]['cohens_ds'].append(row['cohens_d'])

    records = []
    for gene, st in gene_stats.items():
        records.append({
            'gene': gene,
            'n_samples_sig': st['n_sig'],
            'n_samples_tested': st['n_tested'],
            'reproducibility': st['n_sig'] / st['n_tested'] if st['n_tested'] > 0 else 0,
            'median_log2fc': np.median(st['log2fcs']),
            'mean_cohens_d': np.mean(st['cohens_ds']),
        })

    df = pd.DataFrame(records)
    return df.sort_values('reproducibility', ascending=False).reset_index(drop=True)


def cross_encoder_jaccard(de_results_by_encoder: Dict[str, pd.DataFrame],
                          top_n: int = 200,
                          fdr_threshold: float = 0.05) -> pd.DataFrame:
    """Compute Jaccard overlap of top DE genes between encoder pairs.

    Args:
        de_results_by_encoder: Dict mapping encoder name to meta-DE DataFrame
        top_n: Number of top genes to compare
        fdr_threshold: FDR threshold for filtering

    Returns:
        DataFrame with encoder1, encoder2, jaccard, n_overlap, n_union
    """
    encoders = sorted(de_results_by_encoder.keys())
    records = []

    for i, enc1 in enumerate(encoders):
        df1 = de_results_by_encoder[enc1]
        genes1 = set(df1.sort_values('reproducibility', ascending=False)
                     .head(top_n)['gene'])
        for enc2 in encoders[i + 1:]:
            df2 = de_results_by_encoder[enc2]
            genes2 = set(df2.sort_values('reproducibility', ascending=False)
                         .head(top_n)['gene'])
            intersection = genes1 & genes2
            union = genes1 | genes2
            jaccard = len(intersection) / len(union) if union else 0
            records.append({
                'encoder1': enc1,
                'encoder2': enc2,
                'jaccard': jaccard,
                'n_overlap': len(intersection),
                'n_union': len(union),
            })

    return pd.DataFrame(records)

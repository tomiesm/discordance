"""Independent aggregation checks and descriptive score-audit figures."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent


def main():
    spots=pd.read_parquet(OUT/'spot_score_diagnostics.parquet')
    cs=pd.read_csv(OUT/'score_covariate_correlations.csv')
    genes=pd.read_csv(OUT/'gene_error_contributions.csv')
    checks=[]
    for sid,g in spots.groupby('sample_id'):
        for score in ['raw','conditional']:
            expected=cs[(cs['sample']==sid)&(cs.score==score)&(cs.covariate=='total_expr')].spearman.iloc[0]
            got=g[score].rank().corr(g.total_expr.rank())
            checks.append({'check':f'{sid}/{score}:rank_correlation','max_abs':float(abs(got-expected)),'pass':bool(abs(got-expected)<1e-12)})
        got=genes.loc[genes['sample']==sid,'mean_abs_error'].mean()
        err=abs(got-g.raw.mean())
        checks.append({'check':f'{sid}:gene_to_spot_mean_error','max_abs':float(err),'pass':bool(err<2e-6)})
    means=spots.groupby(['cohort','pooled_expression_bin']).conditional.mean()
    checks.append({'check':'pooled_bin_mean_zero','max_abs':float(means.abs().max()),'pass':bool(means.abs().max()<2e-6)})
    assert all(c['pass'] for c in checks)
    (OUT/'independent_checks.json').write_text(json.dumps({'status':'pass','checks':checks},indent=2)+'\n')
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    d=pd.read_csv(OUT/'section_depth_profiles.csv')
    with PdfPages(figdir/'section_expression_profiles.pdf') as pdf:
        for cohort,df in d.groupby('cohort',sort=False):
            fig,ax=plt.subplots(4,3,figsize=(12,12),sharex=True)
            for a,(sid,g) in zip(ax.flat,df.groupby('sample',sort=False)):
                a.errorbar(g.section_bin+1,g.conditional_mean,yerr=g.conditional_sd,fmt='o-',label='Conditional error mean ± SD',capsize=2)
                a.axhline(0,c='grey',lw=.7);a.set_title(sid+' / '+g.patient.iloc[0]);a.set_xlabel('Within-section expression decile')
                a.set_ylabel('Conditional error')
                b=a.twinx();b.plot(g.section_bin+1,g.Q4_fraction,c='#bd5a24',ls='--',label='Q4 fraction');b.set_ylim(0,1);b.set_ylabel('Fraction in Q4',color='#bd5a24')
                b.axhline(.25,c='#bd5a24',lw=.7,alpha=.4)
            for a in list(ax.flat)[df['sample'].nunique():]:a.set_visible(False)
            fig.suptitle(cohort+': pooled centering does not ensure within-section depth balance\nOrange dashed line: Q4 frequency; reference 0.25',fontsize=13)
            fig.tight_layout(rect=(0,0,1,.94));pdf.savefig(fig);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4),gridspec_kw={'width_ratios':[1,1.1]})
    order=cs[cs.covariate=='total_expr']['sample'].drop_duplicates().tolist()
    for score,color in [('raw','#4d7399'),('conditional','#b45536')]:
        values=cs[(cs.covariate=='total_expr')&(cs.score==score)].set_index('sample').loc[order,'spearman']
        axs[0].plot(range(len(order)),values,'o-',label=score,color=color)
    axs[0].axhline(0,c='grey',lw=.7);axs[0].set_xticks(range(len(order)),order,rotation=90);axs[0].set_ylabel('Spearman correlation with sum log1p(counts)');axs[0].legend(frameon=False)
    comp=pd.read_csv(OUT/'score_comparisons.csv');comp=comp[comp.score_a=='conditional']
    labels={'raw':'Raw error','baseline_conditional':'Conditional baseline error','section_centered_diagnostic':'Within-section centering'}
    for i,(kind,label) in enumerate(labels.items()):
        v=comp[comp.score_b==kind].Q4_overlap_fraction
        axs[1].scatter(np.full(len(v),i),v,alpha=.7,s=22)
        axs[1].plot([i-.2,i+.2],[v.median()]*2,c='black',lw=2)
    axs[1].axhline(.25,c='grey',ls=':',label='25% reference, not a spatial null');axs[1].set_xticks(range(3),list(labels.values()),rotation=10);axs[1].set_ylim(0,1);axs[1].set_ylabel('Fraction of current Q4 retained');axs[1].legend(frameon=False,fontsize=8)
    fig.tight_layout();fig.savefig(figdir/'score_diagnostics.png',dpi=180);fig.savefig(figdir/'score_diagnostics.pdf');plt.close(fig)


if __name__=='__main__':main()

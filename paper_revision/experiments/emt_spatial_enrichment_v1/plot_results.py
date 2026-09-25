"""Observed spatial concentration versus conditional random-labeling nulls."""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name]='4'
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'
FIG=HERE/'figures'
LABELS={'global':'Within section','technical':'Identity + detection + size','context':'+ Local cell composition'}
SAMPLES=['NCBI785','NCBI784','NCBI783']


def main():
    FIG.mkdir(exist_ok=True)
    table=pd.read_csv(OUT/'statistics.csv')
    primary=table.loc[(table.radius_um==100)&table.mark.eq('coexpression')]
    configurations=[(s,n) for s in SAMPLES for n in LABELS]
    fig,axes=plt.subplots(1,2,figsize=(13,7),constrained_layout=True)
    colors=['#4c78a8','#e45756','#59a14f']
    for i,(sample,null) in enumerate(configurations):
        f=primary.loc[primary['sample'].eq(sample)&primary.null.eq(null)].set_index('statistic')
        row=f.loc['positive_pairs']
        expected=row.expected_pairs_analytic
        axes[0].plot([row.null_025/expected,row.null_975/expected],[i,i],color='#bcbcbc',lw=6,solid_capstyle='round')
        axes[0].scatter([row.observed/expected],[i],color=colors[SAMPLES.index(sample)],s=55,zorder=4)
        r=f.loc['qualifying_centers']
        axes[1].plot([r.null_025,r.null_975],[i,i],color='#bcbcbc',lw=6,solid_capstyle='round')
        axes[1].scatter([r.observed],[i],color=colors[SAMPLES.index(sample)],s=55,zorder=4)
    labels=[f'{s} | {LABELS[n]}' for s,n in configurations]
    for ax in axes:
        ax.set_yticks(range(len(labels)),labels=labels,fontsize=9)
        ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    axes[0].axvline(1,color='black',ls='--',lw=1)
    axes[0].set_xlabel('Positive-positive pairs / exact null expectation')
    axes[0].set_title('Coexpression spatial concentration at 100 µm')
    axes[1].set_xlabel('Number of qualifying neighborhood centers')
    axes[1].set_title('Neighborhood search repeated under the null',fontsize=11)
    fig.suptitle('Dots: observed results | Grey intervals: central 95% of 1,999 null realizations\nThree sections, two patients; intervals describe the null, not effect-size confidence intervals',fontsize=12)
    fig.savefig(FIG/'spatial_enrichment_overview.png',dpi=180)
    fig.savefig(FIG/'spatial_enrichment_overview.pdf');plt.close(fig)
    with PdfPages(FIG/'scan_adjusted_neighborhoods.pdf') as pdf:
        for sample in SAMPLES:
            obs=pd.read_parquet(OUT/f'{sample}_tumor_inputs.parquet')
            fig,axes=plt.subplots(1,3,figsize=(16,6),constrained_layout=True)
            for ax,null in zip(axes,LABELS):
                local=pd.read_parquet(OUT/f'{sample}_100um_{null}_coexpression_local.parquet')
                significant=local.p_scan_three_sections<=.05
                old=local.old_component>=0
                ax.scatter(local.x_um,local.y_um,s=.5,color='#dddddd',linewidths=0,rasterized=True)
                ax.scatter(local.loc[old,'x_um'],local.loc[old,'y_um'],s=3,color='#517cb2',linewidths=0,label='Original qualifying centers',rasterized=True)
                ax.scatter(local.loc[significant,'x_um'],local.loc[significant,'y_um'],s=3,color='#cf3b35',linewidths=0,label='Scan-adjusted p ≤ 0.05',rasterized=True)
                ax.set_title(f'{LABELS[null]}\n{significant.sum():,} centers pass full-search correction',fontsize=10)
                ax.set_aspect('equal');ax.invert_yaxis();ax.set_xlabel('Native x (µm)');ax.set_ylabel('Native y (µm)')
                ax.legend(fontsize=7,loc='upper right')
            fig.suptitle(f'{sample} | {obs.patient.iloc[0]} | EMT-associated TF/epithelial coexpression\nLocal tests account for every eligible center and three sections; this does not test discordance',fontsize=12)
            fig.savefig(FIG/f'{sample}_scan_adjusted.png',dpi=180);pdf.savefig(fig);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
    for j,sample in enumerate(SAMPLES):
        x=table.loc[table['sample'].eq(sample)&table.null.eq('technical')&table.statistic.eq('positive_pairs')&table.mark.eq('coexpression')].sort_values('radius_um')
        ax.plot(x.radius_um,x.observed/x.expected_pairs_analytic,'o-',label=sample,color=colors[j])
        nuclear=table.loc[table['sample'].eq(sample)&table.null.eq('technical')&table.statistic.eq('positive_pairs')&table.mark.eq('nuclear')].iloc[0]
        ax.scatter([100+j*2-2],[nuclear.observed/nuclear.expected_pairs_analytic],marker='x',s=75,color=colors[j])
    ax.axhline(1,color='black',lw=1,ls='--');ax.set(xlabel='Neighborhood radius (µm)',ylabel='Positive-pair count / null expectation',
        title='Technical-stratified null | Circles: total-cell coexpression; crosses: nuclear-only')
    ax.legend();ax.grid(alpha=.2)
    fig.savefig(FIG/'scale_and_nuclear_sensitivity.png',dpi=180)
    fig.savefig(FIG/'scale_and_nuclear_sensitivity.pdf');plt.close(fig)
    print('FIGURES COMPLETE',flush=True)


def extent_maps():
    evidence=pd.read_csv(OUT/'component_extent_evidence.csv')
    geometry=pd.read_csv(HERE.parent/'emt_cells_v1/results/candidate_neighborhood_components.csv')
    combined=evidence.merge(geometry,on=['sample','patient','component'],validate='many_to_one')
    combined.to_csv(OUT/'component_extent_with_coordinates.csv',index=False)
    with PdfPages(FIG/'component_extent_maps.pdf') as pdf:
        for sample in SAMPLES:
            obs=pd.read_parquet(OUT/f'{sample}_tumor_inputs.parquet')
            old=obs.candidate_component_100um>=0
            fig,axes=plt.subplots(1,3,figsize=(16,6),constrained_layout=True)
            for ax,null in zip(axes,LABELS):
                rows=evidence.loc[evidence['sample'].eq(sample)&evidence.null.eq(null)]
                selected=rows.loc[rows.p_extent_two_statistics_three_sections<=.05,'component']
                supported=obs.candidate_component_100um.isin(selected)
                ax.scatter(obs.x_um,obs.y_um,s=.5,color='#dddddd',linewidths=0,rasterized=True)
                ax.scatter(obs.loc[old,'x_um'],obs.loc[old,'y_um'],s=3,color='#517cb2',linewidths=0,label='Original candidate regions',rasterized=True)
                ax.scatter(obs.loc[supported,'x_um'],obs.loc[supported,'y_um'],s=3,color='#cf3b35',linewidths=0,label='Extent-adjusted p ≤ 0.05',rasterized=True)
                ax.set_title(f'{LABELS[null]}\n{len(selected)} of {len(rows)} regions pass extent comparison',fontsize=10)
                ax.set_aspect('equal');ax.invert_yaxis();ax.set_xlabel('Native x (µm)');ax.set_ylabel('Native y (µm)')
                ax.legend(fontsize=7)
            fig.suptitle(f'{sample} | Exploratory connected-neighborhood extent test\nCorrection covers the full spatial search, three sections and two localization statistics',fontsize=12)
            fig.savefig(FIG/f'{sample}_component_extent.png',dpi=180);pdf.savefig(fig);plt.close(fig)
    print('EXTENT MAPS COMPLETE',flush=True)


if __name__=='__main__':
    main()
    if (OUT/'component_extent_evidence.csv').exists():extent_maps()

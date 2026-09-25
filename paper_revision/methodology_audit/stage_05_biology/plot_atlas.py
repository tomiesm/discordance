"""Standalone figures for the broad, program-excluded biological atlas."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent


def main():
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    data=pd.read_csv(OUT.parent/'stage_06_inference/patient_effects.csv')
    coverage=pd.read_csv(OUT/'coverage.csv')
    programs=sorted(set.intersection(*(set(g.loc[g.eligible,'pathway']) for _,g in coverage.groupby('cohort'))))
    order=['P03','P04','P05','P06','P01','P02','P07','P08']
    data=data[(data.adjustment=='overlap_adjusted')&data.pathway.isin(programs)]
    fig,axs=plt.subplots(1,3,figsize=(14,11),sharey=True)
    for a,outcome,title in zip(axs,['observed','signed','absolute'],['Observed expression','Signed prediction residual','Absolute prediction error']):
        f=data[data.outcome==outcome].pivot(index='pathway',columns='patient',values='standardized_estimate').loc[programs,order]
        lo,hi=(0,3) if outcome=='absolute' else (-1.5,1.5)
        im=a.imshow(f.to_numpy(),aspect='auto',vmin=lo,vmax=hi,cmap='viridis' if outcome=='absolute' else 'RdBu_r')
        a.set_title(title);a.set_xticks(range(8),order);a.axvline(3.5,c='black',lw=1.5)
        a.set_xlabel('Discovery patients | Validation patients')
        a.set_yticks(range(len(programs)),[s.replace('HALLMARK_','').replace('_',' ').title() for s in programs],fontsize=8)
        fig.colorbar(im,ax=a,orientation='horizontal',fraction=.04,pad=.07,label='Q4 − Q1 / original pooled group SD',extend='max' if outcome=='absolute' else 'both')
    fig.suptitle('All 25 Hallmarks testable in both panels\nProgram genes excluded from grouping; count/detection overlap adjustment\nPanel-specific gene sets differ; each column averages sections within one patient',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.92));fig.savefig(figdir/'patient_program_atlas.pdf');fig.savefig(figdir/'patient_program_atlas.png',dpi=180);plt.close(fig)
    cells=pd.read_csv(OUT/'source_cell_program_context.csv')
    cells=cells[(cells.pathway=='HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION')&(cells.scope=='panel')]
    f=cells.pivot(index='sample',columns='source_group',values='fraction_program_transcripts').fillna(0).loc[['NCBI785','NCBI784','NCBI783']]
    fig,ax=plt.subplots(figsize=(9,5));bottom=np.zeros(3)
    colors={'Stromal':'#398d87','Tumor':'#ae395e','Myoepithelial':'#4e6fb5','Other non-tumor':'#c7ae77','Normal epithelial':'#7cb464','Published transitional':'#9c61b4','Uncertain/hybrid':'#aaa'}
    for group in f.columns:
        v=f[group].to_numpy()*100;ax.bar(range(3),v,bottom=bottom,label=group,color=colors[group]);bottom+=v
    ax.set_xticks(range(3),['NCBI785 (P07)','NCBI784 (P07)','NCBI783 (P08)']);ax.set_ylim(0,100)
    ax.set_ylabel('Fraction of measured EMT-Hallmark transcripts (%)')
    ax.set_title('Measured EMT-Hallmark signal spans source-annotated cell groups\nQC-passing assigned cells; whole-section context, not Q4 enrichment')
    ax.legend(bbox_to_anchor=(1.02,1),loc='upper left',frameon=False)
    fig.tight_layout();fig.savefig(figdir/'emt_cell_context.pdf');fig.savefig(figdir/'emt_cell_context.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()

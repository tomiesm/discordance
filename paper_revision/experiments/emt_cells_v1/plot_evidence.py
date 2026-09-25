"""All-section maps and every fixed-rule candidate component, including controls."""
from common import *
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
from PIL import Image

FIGURES=HERE/'figures'
COLORS={'Tumor':'#ad3354','Stromal':'#288f8b','Myoepithelial':'#376db0',
        'Normal epithelial':'#ddac39','Published transitional':'#9a51a1',
        'Uncertain/hybrid':'#9f9f9f','Other non-tumor':'#d0d0d0'}
GENE_COLORS={'EPCAM':'#aaaabb','CDH1':'#aaaabb','KRT8':'#aaaabb',
             'SNAI1':'#d32f2f','ZEB1':'#d48b00','ZEB2':'#167ac6'}


def map_axis(ax):
    ax.set_aspect('equal');ax.invert_yaxis()
    ax.set_xlabel('Native x (µm)',fontsize=8);ax.set_ylabel('Native y (µm)',fontsize=8)
    ax.tick_params(labelsize=7)


def continuous(ax,frame,col,title,vmax=None):
    valid=np.isfinite(frame[col])
    values=frame.loc[valid,col].to_numpy(float)
    high=float(np.quantile(values,.98)) if vmax is None else vmax
    high=max(high,1e-6)
    artist=ax.scatter(frame.loc[valid,'x_um'],frame.loc[valid,'y_um'],c=values,s=.5,linewidths=0,
                      vmin=0,vmax=high,cmap='viridis',rasterized=True)
    plt.colorbar(artist,ax=ax,shrink=.7,pad=.015,fraction=.035)
    ax.set_title(title,fontsize=9);map_axis(ax)


def whole_maps(sample,obs,pdf):
    q=obs.loc[obs.qc_pass];tumor=q.loc[q.source_group=='Tumor']
    fig,axes=plt.subplots(2,3,figsize=(15,10),constrained_layout=True)
    for group,color in COLORS.items():
        subset=q.loc[q.source_group==group]
        axes[0,0].scatter(subset.x_um,subset.y_um,s=.5,color=color,linewidths=0,rasterized=True)
    axes[0,0].legend(handles=[Patch(color=c,label=g) for g,c in COLORS.items()],fontsize=6,loc='upper right')
    axes[0,0].set_title('Published cell identity (QC-pass)',fontsize=9);map_axis(axes[0,0])
    axes[0,1].scatter(tumor.x_um,tumor.y_um,s=.5,color='#d0d0d0',linewidths=0,rasterized=True)
    for name,color,size in [('tumor_tf_candidate','#d34c18',2),('tumor_nuclear_tf_candidate','#285dae',4)]:
        subset=tumor.loc[tumor[name]]
        axes[0,1].scatter(subset.x_um,subset.y_um,s=size,color=color,linewidths=0,label=name,rasterized=True)
    axes[0,1].legend(fontsize=6);axes[0,1].set_title('Measured same-cell TF/epithelial coexpression',fontsize=9);map_axis(axes[0,1])
    continuous(axes[0,2],tumor,'tf_candidate_fraction_100um','Candidate fraction among local tumor cells (100 µm)',.3)
    continuous(axes[1,0],tumor,'measured_hallmark_emt','Measured Hallmark EMT-associated score | tumor labels')
    continuous(axes[1,1],tumor,'reference_projected_program','Reference-projected program | not observed missing genes')
    axes[1,2].scatter(tumor.x_um,tumor.y_um,s=.5,color='#d0d0d0',linewidths=0,rasterized=True)
    zone=tumor.loc[tumor.candidate_component_100um>=0]
    axes[1,2].scatter(zone.x_um,zone.y_um,s=1.5,color='#9c319c',linewidths=0,rasterized=True)
    axes[1,2].set_title('Fixed-rule candidate neighborhood centers | not EMT labels',fontsize=9);map_axis(axes[1,2])
    fig.suptitle(f'{sample} | {obs.patient.iloc[0]} | Measured evidence and candidate neighborhoods\nSource tumor identity and TF coexpression do not establish a validated EMT state.',fontsize=13)
    pdf.savefig(fig);fig.savefig(FIGURES/f'{sample}_cell_evidence.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,10),constrained_layout=True)
    for ax,gene in zip(axes.flat,GENE_COLORS):
        frame=q[['x_um','y_um',gene+'_counts']].copy()
        frame['signal']=np.log1p(frame[gene+'_counts'])
        positive=frame.signal.to_numpy();positive=positive[positive>0]
        continuous(ax,frame,'signal',gene+' | log1p measured count',float(np.quantile(positive,.99)) if len(positive) else 1)
    fig.suptitle(f'{sample}: directly measured epithelial and EMT-related TF transcripts\nAll QC-pass cell types shown; these marker maps are not cell-state assignments.',fontsize=13)
    pdf.savefig(fig);fig.savefig(FIGURES/f'{sample}_measured_genes.png',dpi=170);plt.close(fig)
    metadata=json.loads((PROJECT/f'data/hest/metadata/{sample}.json').read_text())
    thumb=Image.open(PROJECT/f'data/hest/thumbnails/{sample}_downscaled_fullres.jpeg')
    fig,axes=plt.subplots(1,2,figsize=(13,7),constrained_layout=True)
    extent=[0,metadata['fullres_px_width'],metadata['fullres_px_height'],0]
    for ax in axes:ax.imshow(thumb,extent=extent);ax.set_axis_off()
    axes[0].set_title('Original H&E thumbnail')
    candidate=obs.loc[obs.tumor_tf_candidate]
    axes[1].scatter(candidate.he_x,candidate.he_y,s=2,facecolors='none',edgecolors='#03ffcf',linewidths=.5)
    axes[1].set_title('Candidate cells at supplied transcript centroids')
    fig.suptitle(f'{sample}: histology context only; no inferred global coordinate transform applied')
    fig.savefig(FIGURES/f'{sample}_histology_context.png',dpi=180);pdf.savefig(fig);plt.close(fig)


def load_selected_transcripts(sample):
    path=RESULTS/sample/'selected_measured_transcripts.parquet'
    if path.exists():return pd.read_parquet(path)
    pieces=[]
    for batch in pq.ParquetFile(PROJECT/f'data/hest/transcripts/{sample}_transcripts.parquet').iter_batches(
        batch_size=1000000,columns=['cell_id','feature_name','qv','overlaps_nucleus','x_location','y_location'],use_threads=False):
        d=batch.to_pandas();use=(d.qv>=20)&d.feature_name.isin([g.encode() for g in GENE_COLORS])
        d=d.loc[use].copy();d['gene']=d.feature_name.str.decode('utf-8')
        if sample=='NCBI783':d['cell_id']=d.cell_id.str.decode('utf-8')
        else:d['cell_id']=d.cell_id.astype(str)
        pieces.append(d.drop(columns=['feature_name']))
    frame=pd.concat(pieces,ignore_index=True);frame.to_parquet(path,index=False)
    return frame


def boundary_segments(frame,cell_ids):
    subset=frame.loc[frame.cell_id.isin(cell_ids)]
    return [f[['vertex_x','vertex_y']].to_numpy(float) for _,f in subset.groupby('cell_id',sort=False)],subset.cell_id.unique()


def component_pages(sample,obs,pdf):
    transcripts=load_selected_transcripts(sample)
    bounds={}
    for kind in ['cell','nucleus']:
        frame=pd.read_parquet(SOURCES/'vendor'/sample/f'{kind}_boundaries.parquet')
        if sample=='NCBI783':frame['cell_id']=frame.cell_id.str.decode('utf-8')
        else:frame['cell_id']=frame.cell_id.astype(str)
        bounds[kind]=frame
    for component in sorted(set(obs.candidate_component_100um)-{-1}):
        center=obs.loc[obs.candidate_component_100um==component]
        lower=center[['x_um','y_um']].min().to_numpy()-100
        upper=center[['x_um','y_um']].max().to_numpy()+100
        neighborhood=obs.loc[(obs.x_um>=lower[0])&(obs.x_um<=upper[0])&(obs.y_um>=lower[1])&(obs.y_um<=upper[1])]
        candidates=neighborhood.loc[neighborhood.tumor_tf_candidate]
        midpoint=center[['x_um','y_um']].mean().to_numpy()
        focused=candidates.iloc[np.argmin(np.sum((candidates[['x_um','y_um']].to_numpy()-midpoint)**2,axis=1))]
        fig,axes=plt.subplots(1,2,figsize=(13,6),constrained_layout=True)
        for group,color in COLORS.items():
            ids=neighborhood.index[neighborhood.source_group==group]
            segments,_=boundary_segments(bounds['cell'],ids)
            axes[0].add_collection(LineCollection(segments,colors=color,linewidths=.4))
        for ax,lo,hi in [(axes[0],lower,upper),(axes[1],focused[['x_um','y_um']].to_numpy(float)-25,focused[['x_um','y_um']].to_numpy(float)+25)]:
            subset=transcripts.loc[(transcripts.x_location>=lo[0])&(transcripts.x_location<=hi[0])&(transcripts.y_location>=lo[1])&(transcripts.y_location<=hi[1])]
            for gene,color in GENE_COLORS.items():
                points=subset.loc[subset.gene==gene]
                ax.scatter(points.x_location,points.y_location,s=1.5 if ax is axes[0] else 8,color=color,linewidths=0,alpha=.7,rasterized=True)
            ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);map_axis(ax)
        near_ids=neighborhood.index[(np.abs(neighborhood.x_um-focused.x_um)<50)&(np.abs(neighborhood.y_um-focused.y_um)<50)]
        segments,_=boundary_segments(bounds['cell'],near_ids)
        axes[1].add_collection(LineCollection(segments,colors='#bbbbbb',linewidths=.6))
        for kind,style in [('cell','solid'),('nucleus','dashed')]:
            segments,_=boundary_segments(bounds[kind],[focused.name])
            axes[1].add_collection(LineCollection(segments,colors='black',linewidths=1.1,linestyles=style))
        axes[0].scatter(candidates.x_um,candidates.y_um,s=10,facecolors='none',edgecolors='black',linewidths=.5)
        axes[0].set_title('Measured transcripts and vendor cell boundaries\nTumor: red outlines; stroma: teal; myoepithelial: blue',fontsize=9)
        text='; '.join(f'{g}={int(focused[g+"_counts"])}' for g in ['SNAI1','ZEB1','ZEB2'])
        axes[1].set_title(f'Fixed representative: nearest candidate to component center\nCell {focused.name}; {text}; nuclear TF genes={int(focused.nuclear_tf_genes_detected)}',fontsize=9)
        fig.legend(handles=[Patch(color=GENE_COLORS[g],label=g) for g in ['SNAI1','ZEB1','ZEB2']]+[Patch(color='#aaaabb',label='EPCAM/CDH1/KRT8')],loc='outside lower center',ncol=4,fontsize=8)
        fig.suptitle(f'{sample} | component {component} | TF coexpression evidence, not a validated EMT zone')
        pdf.savefig(fig)
        if component==0:fig.savefig(FIGURES/f'{sample}_component0_transcripts.png',dpi=190)
        plt.close(fig)
    print('COMPONENT EVIDENCE',sample,flush=True)


def overview():
    reference=pd.read_csv(RESULTS/'reference_donor_performance.csv')
    depth=pd.read_csv(RESULTS/'reference_depth_sensitivity.csv')
    cells=pd.DataFrame(json.loads((RESULTS/'CELL_EVIDENCE_SUMMARY.json').read_text()))
    residual=pd.read_csv(RESULTS/'residual_associations.csv')
    fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    for panel,frame in reference.groupby('panel'):
        axes[0,0].plot(np.arange(len(frame)),frame.rho_learned,'o-',ms=3,label=panel)
    axes[0,0].axhline(.3,color='grey',ls='--',lw=1)
    axes[0,0].set(xlabel='Held-out reference donor',ylabel='Spearman rho',title='Gene-disjoint reference target (20 donors)');axes[0,0].legend()
    x=np.arange(len(cells));axes[0,1].bar(x-.18,100*cells.fraction_tumor_tf_candidates,.36,label='Measured coexpression')
    axes[0,1].bar(x+.18,100*cells.n_candidates_nuclear_supported/cells.n_source_tumor_qc,.36,label='Also nuclear-supported')
    axes[0,1].set(xticks=x,xticklabels=cells['sample'],ylabel='% of source tumor cells passing QC',title='Measured TF/epithelial coexpression');axes[0,1].legend(fontsize=8)
    bars=[]
    for panel in ['panel313','panel280']:
        bars.append([reference.loc[reference.panel==panel,'rho_learned'].median(),reference.loc[reference.panel==panel,'rho_depth'].median(),depth.loc[depth.panel==panel,'partial_rho_adjusting_depth_and_detection'].median()])
    for j,label in enumerate(['Learned program','Depth alone','Partial rho (depth/detection adjusted)']):
        axes[1,0].bar(np.arange(2)+(j-1)*.23,np.array(bars)[:,j],.23,label=label)
    axes[1,0].set(xticks=[0,1],xticklabels=['313-gene panel','280-gene panel'],ylabel='Median donor correlation',title='Reference depth sensitivity');axes[1,0].legend(fontsize=7)
    selected=residual.loc[(residual.minimum_tumor_cells==5)&(residual.score.isin(['D_cond','disjoint_D_cond']))]
    for j,name in enumerate(['D_cond','disjoint_D_cond']):
        frame=selected.loc[selected.score==name].set_index('sample').reindex(cells['sample'])
        axes[1,1].bar(np.arange(3)+(j-.5)*.32,frame.partial_spearman,.32,label=name)
    axes[1,1].axhline(0,color='grey',lw=.6)
    axes[1,1].set(xticks=np.arange(3),xticklabels=cells['sample'],ylabel='Partial Spearman rho',title='Secondary residual association | 2 patients');axes[1,1].legend(fontsize=8)
    fig.suptitle('Cell-resolved EMT feasibility: measured candidates, reference limits, and residual checks',fontsize=13)
    fig.savefig(FIGURES/'overview.png',dpi=180);fig.savefig(FIGURES/'overview.pdf');plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    with PdfPages(FIGURES/'all_section_evidence.pdf') as pdf, PdfPages(FIGURES/'all_candidate_transcript_evidence.pdf') as evidence:
        for sample in PROTOCOL['samples']:
            obs=pd.read_parquet(RESULTS/sample/'cell_evidence.parquet')
            whole_maps(sample,obs,pdf)
            component_pages(sample,obs,evidence)
            print('FIGURES COMPLETE',sample,flush=True)
    overview()
    print('ALL FIGURES COMPLETE',flush=True)


if __name__=='__main__':main()

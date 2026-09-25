from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr,spearmanr
from functools import lru_cache
import h5py

from src.paper.paths import PROJECT_ROOT, ANALYSIS_ROOT, CELL_ROOT
ROOT=R=PROJECT_ROOT
A=ANALYSIS_ROOT;T=A/'donors/tables'
OUT=ROOT/'outputs/paper_figures';SRC=OUT;BUILD=OUT/'logs'
for folder in [OUT,BUILD,OUT/'previews',OUT/'source_data']:
    folder.mkdir(parents=True,exist_ok=True)
C={'biomarkers':'#2CA6A4','10x_janesick':'#E8785E','coad':'#7B68EE','idc_visium':'#57904C'}
L={'biomarkers':'Discovery','10x_janesick':'Validation','coad':'COAD','idc_visium':'Visium'}
IDC=['biomarkers','10x_janesick'];FAMILIES=IDC+['coad','idc_visium'];LOW='#4A90D9';HIGH='#D94A4A';MID='#D2D2D2'
EMT='HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION';PROGRAMS=[EMT,'HALLMARK_COMPLEMENT','HALLMARK_E2F_TARGETS','HALLMARK_G2M_CHECKPOINT']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,'legend.fontsize':6,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'figure.dpi':130})
SOURCES={};ASSETS=[]
def track(p):
    p=Path(p).resolve()
    if str(p) not in SOURCES:SOURCES[str(p)]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    return p
def table(name):return pd.read_csv(track(T/(name+'.csv')))
def oldtable(name):
    if name=='external_gene_features':
        frames=[]
        for family in ['coad','idc_visium']:
            frame=pd.read_csv(A/'external'/family/'revised_gene_features.csv');frame['family']=family;frames.append(frame)
        return pd.concat(frames,ignore_index=True)
    if name=='figure2_CD8A_map':
        sample='TENX193';family='biomarkers';enc='uni'
        genes=json.loads((R/'data/v3'/f'gene_list_{family}.json').read_text());j=genes.index('CD8A')
        for k in range(4):
            fd=R/'outputs/predictions'/family/enc/'ridge'/f'fold{k}'
            ids=json.loads((fd/'test_spot_ids.json').read_text());mask=np.array([i.startswith(sample+'_') for i in ids])
            if not mask.any():continue
            y=np.load(fd/'test_targets.npy')[mask,j].astype(float)
            pred=np.mean([np.load(R/'outputs/predictions'/family/e/'ridge'/f'fold{k}'/'test_predictions.npy')[mask,j].astype(float) for e in ['uni','virchow2','hoptimus0']],axis=0)
            frame=locations().set_index('spot_id').loc[np.array(ids)[mask],['x_um','y_um']].reset_index()
            frame['observed']=y;frame['predicted']=pred;frame['signed_residual']=y-pred
            return frame
    raise KeyError(name)
def audit(name):return pd.read_csv(track(A/name))
def savedata(d,name):d.to_csv(OUT/'source_data'/(name+'.csv'),index=False)
def save(fig,rel,reason):
    path=SRC/rel;path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,bbox_inches='tight',pad_inches=.045)
    fig.savefig(OUT/'previews'/(path.stem+'.png'),dpi=160,bbox_inches='tight',pad_inches=.045)
    plt.close(fig);ASSETS.append(dict(path=rel,action='updated',reason=reason));print(path.stem,flush=True)
def flush(name):
    (BUILD/(name+'_sources.json')).write_text(json.dumps(SOURCES,indent=2)+'\n')
    (BUILD/(name+'_assets.json')).write_text(json.dumps(ASSETS,indent=2)+'\n')
def short(s):return s.replace('HALLMARK_','').replace('_',' ').title().replace('Epithelial Mesenchymal Transition','EMT-associated')
def zero(ax):ax.axvline(0,color='.65',lw=.6,zorder=0)
def hzero(ax):ax.axhline(0,color='.65',lw=.6,zorder=0)
def scatterfit(ax,x,y,color,label=None):
    ax.scatter(x,y,s=7,c=color,alpha=.55,edgecolors='none',rasterized=True)
    b=np.polyfit(x,y,1);xx=np.array([min(x),max(x)]);ax.plot(xx,np.polyval(b,xx),c=color,lw=1)
    ax.text(.04,.95,f'Pearson r = {pearsonr(x,y).statistic:.3f}',transform=ax.transAxes,va='top',fontsize=7)
    if label:ax.set_title(label)
def heat(ax,d,limit=None,cmap='RdBu_r',numbers=False):
    a=d.to_numpy(float);limit=limit or max(.1,np.nanmax(abs(a)))
    cm=plt.get_cmap(cmap).copy();cm.set_bad('#bbbbbb')
    im=ax.imshow(np.ma.masked_invalid(a),cmap=cm,vmin=-limit,vmax=limit,aspect='auto',interpolation='nearest')
    ax.set_xticks(range(len(d.columns)),d.columns,rotation=45,ha='right');ax.set_yticks(range(len(d)),[short(str(x)) for x in d.index])
    if numbers:
        for i in range(len(d)):
            for j in range(len(d.columns)):
                if np.isfinite(a[i,j]):ax.text(j,i,f'{a[i,j]:.3f}',ha='center',va='center',fontsize=7)
    return im
@lru_cache(None)
def locations():
    frames=[]
    for family in FAMILIES:
        frames.append(pd.read_parquet(A/'cutoffs'/family/'locations.parquet'))
    d=pd.concat(frames,ignore_index=True);reg=table('specimen_registry').set_index('sample').patient
    d['unit']=d['sample'].map(reg).fillna(d.unit);return d

@lru_cache(None)
def thumb(sample):
    with h5py.File(track(ROOT/'data/hest/st'/f'{sample}.h5ad'),'r') as f:
        g=f['uns/spatial'];g=g[list(g.keys())[0]];im=g['images/downscaled_fullres'][:];scale=float(g['scalefactors/tissue_downscaled_fullres_scalef'][()])
    pixel=audit('coverage/coverage_summary.csv').set_index('sample').loc[sample,'pixel_size_um'];return im,scale,pixel
def limits(ax,ss):
    ax.set_xlim(ss.x_um.min()/1000-.12,ss.x_um.max()/1000+.12);ax.set_ylim(ss.y_um.max()/1000+.12,ss.y_um.min()/1000-.12);ax.set_aspect('equal');ax.set_axis_off()
def he(ax,sample,ss=None):
    im,scale,pixel=thumb(sample);ax.imshow(im,extent=[0,im.shape[1]/scale*pixel/1000,im.shape[0]/scale*pixel/1000,0],rasterized=True)
    if ss is not None:limits(ax,ss)
    ax.set_axis_off()
def mapplot(ax,ss,values,cmap='viridis',vmin=None,vmax=None,title='',size=.8):
    art=ax.scatter(ss.x_um/1000,ss.y_um/1000,c=values,s=size,cmap=cmap,vmin=vmin,vmax=vmax,edgecolors='none',rasterized=True);limits(ax,ss);ax.set_title(title);return art
def quartile(ax,ss,title='Q1 / Q4'):
    q1,q4=np.quantile(ss.conditional,[.25,.75]);colors=np.where(ss.conditional<=q1,LOW,np.where(ss.conditional>=q4,HIGH,MID));ax.scatter(ss.x_um/1000,ss.y_um/1000,c=colors,s=.8,edgecolors='none',rasterized=True);limits(ax,ss);ax.set_title(title)
def primary_programs(d):return d[d['tail'].eq(.25)&d.grouping.eq('program_excluded')&d.adjustment.eq('overlap_adjusted')]

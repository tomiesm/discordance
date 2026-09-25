"""Shared plotting and explicit figure-source bookkeeping."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT=Path(__file__).resolve().parent;AUDIT=OUT.parent;ROOT=AUDIT.parents[1];REPO=AUDIT.parent/'clean_repo'
for folder in ['main','supplement','tables','previews']: (OUT/folder).mkdir(exist_ok=True)
FAMILIES=['biomarkers','10x_janesick','coad','idc_visium'];IDC=FAMILIES[:2]
LABEL={'biomarkers':'IDC discovery','10x_janesick':'IDC validation','coad':'COAD','idc_visium':'IDC Visium'}
COL={'biomarkers':'#197b80','10x_janesick':'#c96739','coad':'#7763a8','idc_visium':'#5c8a42'}
LOW='#2879ad';HIGH='#bd3935';MID='#dedede'
EMT='HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION'
PROGRAMS=[EMT,'HALLMARK_COMPLEMENT','HALLMARK_E2F_TARGETS','HALLMARK_G2M_CHECKPOINT']
PNAME={EMT:'EMT-associated','HALLMARK_COMPLEMENT':'Complement','HALLMARK_E2F_TARGETS':'E2F targets','HALLMARK_G2M_CHECKPOINT':'G2M checkpoint'}
MARKERS=['EPCAM','KRT19','KRT17','ESR1','FOXA1','TP63','FBLN1','COL4A1','PDGFRB','MMP2','CD163','CD68','MKI67','CENPF','PCLAF','SNAI1','ZEB1','ZEB2']
SAMPLES=['NCBI785','NCBI784','NCBI783'];SNAME={'NCBI785':'P07 / 785','NCBI784':'P07 / 784','NCBI783':'P08 / 783'}
SCOL={'NCBI785':'#277da8','NCBI784':'#67a8ba','NCBI783':'#b14b62'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
SOURCES={};FIGURES=[]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def track(path):
    path=Path(path).resolve()
    if str(path) not in SOURCES:SOURCES[str(path)]={'bytes':path.stat().st_size,'sha256':digest(path)}
    return path


def read(name):return pd.read_csv(track(OUT/'tables'/f'{name}.csv'))
def original(name):return pd.read_csv(track(AUDIT/name))
def table(df,name):df.to_csv(OUT/'tables'/f'{name}.csv',index=False);return df
def short(s):return s.replace('HALLMARK_','').replace('_',' ').title()
def panel(ax,letter,title):ax.set_title(f'{letter}  {title}',loc='left',fontweight='bold',pad=9)
def zero(ax):ax.axvline(0,color='.65',lw=.8,zorder=0)
def hzero(ax):ax.axhline(0,color='.65',lw=.8,zorder=0)


def save(fig,name,title,caption,folder='main',pages=None):
    target=OUT/folder/name
    fig.suptitle(title,fontsize=14,fontweight='bold')
    fig.savefig(target.with_suffix('.pdf'));fig.savefig(target.with_suffix('.png'),dpi=170)
    if pages is not None:pages.savefig(fig)
    plt.close(fig)
    FIGURES.append(dict(name=name,title=title,folder=folder,caption=caption,pdf=str(target.with_suffix('.pdf').relative_to(OUT)),png=str(target.with_suffix('.png').relative_to(OUT))))
    print(name,'saved',flush=True)


def flush(name):
    (OUT/f'{name}_sources.json').write_text(json.dumps(SOURCES,indent=2)+'\n')
    (OUT/f'{name}_figures.json').write_text(json.dumps(FIGURES,indent=2)+'\n')


def heat(ax,frame,title=None,limit=None,cmap='RdBu_r',fmt=None):
    a=frame.to_numpy(float);limit=limit or max(.1,float(np.nanmax(np.abs(a))))
    im=ax.imshow(np.ma.masked_invalid(a),aspect='auto',cmap=cmap,vmin=-limit,vmax=limit,interpolation='nearest')
    ax.set_xticks(range(len(frame.columns)),frame.columns,rotation=45,ha='right');ax.set_yticks(range(len(frame)),frame.index)
    if title:ax.set_title(title)
    if fmt:
        for y in range(a.shape[0]):
            for x in range(a.shape[1]):
                if np.isfinite(a[y,x]):ax.text(x,y,format(a[y,x],fmt),ha='center',va='center',fontsize=8,color='white' if abs(a[y,x])>.65*limit else '#222')
    return im


def effects(ax,df,items,item='pathway',value='standardized_effect',group='family',colors=COL,labels=LABEL,points=True):
    present=list(df[group].unique());groups=[g for g in FAMILIES if g in present]+[g for g in present if g not in FAMILIES];offsets=np.linspace(-.19,.19,len(groups)) if len(groups)>1 else [0]
    for g,offset in zip(groups,offsets):
        for j,label in enumerate(items):
            vals=df.loc[df[group].eq(g)&df[item].eq(label),value].dropna().to_numpy()
            if not len(vals):continue
            y=j+offset
            if points:ax.scatter(vals,y+np.linspace(-.04,.04,len(vals)),s=15,alpha=.48,color=colors[g],edgecolors='none')
            ax.scatter([vals.mean()],[y],s=48,marker='D',color=colors[g],edgecolors='white',linewidths=.4,zorder=4)
    zero(ax);ax.set_yticks(range(len(items)),[PNAME.get(x,x) for x in items]);ax.invert_yaxis()
    ax.set_xlabel('Q4 − Q1 / pooled within-tail SD')
    return [Line2D([0],[0],marker='D',lw=0,color=colors[g],label=labels.get(g,g),markersize=5) for g in groups]


def forest(ax,df,items,item='endpoint',value='estimate',group='sample',scale=1):
    for gi,g in enumerate(SAMPLES):
        d=df[df[group].eq(g)]
        for j,label in enumerate(items):
            rows=d[d[item].eq(label)]
            if len(rows)==0:continue
            assert len(rows)==1,(g,label,len(rows));r=rows.iloc[0];v=r[value]*scale;y=j+(gi-1)*.22
            if np.isfinite(r.ci_low) and np.isfinite(r.ci_high):ax.plot([r.ci_low*scale,r.ci_high*scale],[y,y],color=SCOL[g],lw=1)
            ax.scatter(v,y,s=25,color=SCOL[g],zorder=3)
    zero(ax);ax.set_yticks(range(len(items)),[PNAME.get(x,x) for x in items]);ax.invert_yaxis()


def section_legend():return [Line2D([0],[0],marker='o',lw=0,color=SCOL[s],label=SNAME[s]) for s in SAMPLES]

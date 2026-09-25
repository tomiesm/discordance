"""Repair figure presentation from fixed results; no analysis pipeline changes."""
from pathlib import Path
import hashlib, json, re, sys
import fitz
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from scipy.stats import pearsonr

B=Path(__file__).resolve().parent;O=B.parent;S=O/'source';REV=O.parent
A=O/'figure_review_archive_20260925';T=REV/'methodology_audit/stage_15_donor_correction/tables'
OUT=O/'visual_review';OUT.mkdir(exist_ok=True)
C={'biomarkers':'#2CA6A4','10x_janesick':'#E8785E','coad':'#7B68EE'}
L={'biomarkers':'Discovery','10x_janesick':'Validation'};IDC=list(L)
EN={'uni':'UNI2-h','virchow2':'Virchow2','hoptimus0':'H-Optimus-0'}
RN={'ridge':'Ridge','mlp':'MLP','xgboost':'XGBoost'}
INPUTS={};CHANGES={};DATA={}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'figure.dpi':160})
def load(path):
    path=Path(path);INPUTS[str(path)]=dict(sha256=sha(path),bytes=path.stat().st_size)
    return pd.read_csv(path)
def tab(name):return load(T/(name+'.csv'))
def save(fig,rel,reason,data=None):
    fig.savefig(S/rel,bbox_inches='tight',pad_inches=.045)
    fig.savefig(OUT/(Path(rel).stem+'_after.png'),dpi=160,bbox_inches='tight',pad_inches=.045)
    plt.close(fig);register(rel,reason)
    if data is not None:
        DATA[rel]=data.to_json(orient='split',double_precision=15)
def register(rel,reason):
    CHANGES[rel]=dict(before_sha256=sha(A/'source'/rel),after_sha256=sha(S/rel),reason=reason)
def pname(s):
    special={'EPITHELIAL_MESENCHYMAL_TRANSITION':'EMT Hallmark','TNFA_SIGNALING_VIA_NFKB':'TNFα via NF-κB','IL2_STAT5_SIGNALING':'IL2 / STAT5 signaling','IL6_JAK_STAT3_SIGNALING':'IL6 / JAK / STAT3 signaling','MTORC1_SIGNALING':'mTORC1 signaling','PI3K_AKT_MTOR_SIGNALING':'PI3K / AKT / mTOR signaling','TGF_BETA_SIGNALING':'TGFβ signaling','KRAS_SIGNALING_DN':'KRAS signaling down','KRAS_SIGNALING_UP':'KRAS signaling up','MYC_TARGETS_V1':'MYC targets V1','MYC_TARGETS_V2':'MYC targets V2','E2F_TARGETS':'E2F targets','G2M_CHECKPOINT':'G2M checkpoint','P53_PATHWAY':'p53 pathway','DNA_REPAIR':'DNA repair','UV_RESPONSE_DN':'UV response down','UV_RESPONSE_UP':'UV response up','WNT_BETA_CATENIN_SIGNALING':'WNT / β-catenin signaling'}
    s=s.removeprefix('HALLMARK_');return special.get(s,s.replace('_',' ').capitalize())
def heat(ax,d,limit=2,labels=None,fs=7):
    cm=plt.get_cmap('RdBu_r').copy();cm.set_bad('#bbbbbb')
    im=ax.imshow(np.ma.masked_invalid(d.to_numpy(float)),aspect='auto',cmap=cm,vmin=-limit,vmax=limit,interpolation='nearest')
    ax.set_xticks(range(len(d.columns)),d.columns,rotation=45,ha='right')
    ax.set_yticks(range(len(d)),labels if labels is not None else [pname(x) for x in d.index]);ax.tick_params(axis='y',labelsize=fs)
    finite=d.to_numpy(float);finite=finite[np.isfinite(finite)]
    ax.figure._heat_min=min(getattr(ax.figure,'_heat_min',np.inf),finite.min())
    ax.figure._heat_max=max(getattr(ax.figure,'_heat_max',-np.inf),finite.max())
    return im
def scale(fig,im,**kwargs):
    low=getattr(fig,'_heat_min',np.ma.min(im.get_array()))<im.norm.vmin
    high=getattr(fig,'_heat_max',np.ma.max(im.get_array()))>im.norm.vmax
    return fig.colorbar(im,extend='both' if low and high else 'min' if low else 'max' if high else 'neither',**kwargs)
def zero(ax):ax.axvline(0,c='.65',lw=.6,zorder=0)
def scatter(ax,x,y,color,title,identity=False):
    ax.scatter(x,y,s=7,c=color,alpha=.55,edgecolors='none',rasterized=True)
    b=np.polyfit(x,y,1);xx=np.array([min(x),max(x)]);ax.plot(xx,np.polyval(b,xx),c=color,lw=1)
    ax.text(.04,.95,f'Pearson r = {pearsonr(x,y).statistic:.3f}',transform=ax.transAxes,va='top',fontsize=8)
    ax.set_title(title)
    if identity:
        lo=min(np.min(x),np.min(y));hi=max(np.max(x),np.max(y))
        ax.plot([lo,hi],[lo,hi],c='.5',ls='--',lw=.8,zorder=0)
def programs(d):return d[d['tail'].eq(.25)&d.grouping.eq('program_excluded')&d.adjustment.eq('overlap_adjusted')]

def workflow():
    # Keep the original three phases and cohort overview; replace unreadable
    # miniature result plots with explicit analysis labels.
    fig,ax=plt.subplots(figsize=(7.2,4.2));fig.subplots_adjust(0,0,1,1);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    def box(x,y,w,h,title,body='',edge='#277e98',fill='#f0f7fa',fs=8):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008',lw=.8,edgecolor=edge,facecolor=fill))
        ax.text(x+w/2,y+h*.70 if body else y+h/2,title,ha='center',va='center',fontsize=fs,fontweight='bold',color='#18363e')
        if body:ax.text(x+w/2,y+h*.30,body,ha='center',va='center',fontsize=fs-0.5,linespacing=1.35)
    def arrow(x,y,u,v):ax.add_patch(FancyArrowPatch((x,y),(u,v),arrowstyle='-|>',mutation_scale=9,lw=1,color='#55646b'))
    box(.015,.91,.47,.075,'Discovery: 11 sections / 11 donors','280 genes; four holdout groups',edge=C[IDC[0]],fs=8)
    box(.515,.91,.47,.075,'Validation: 7 sections / 4 donor groups','280 genes; donor groups held out together',edge=C[IDC[1]],fs=8)
    ax.text(.015,.866,'Phase 1: Predict gene expression from histology',fontsize=9,fontweight='bold')
    box(.015,.65,.20,.17,'H&E patches','224 × 224 pixels\naligned to spatial bins')
    box(.265,.65,.23,.17,'Image embeddings','UNI2-h · Virchow2\nH-Optimus-0')
    box(.548,.65,.20,.17,'Expression prediction','PCA (256) + ridge\nintercept from training')
    box(.80,.65,.185,.17,'Observed RNA','Xenium\nlog1p counts')
    arrow(.219,.735,.256,.735);arrow(.502,.735,.539,.735)
    ax.text(.015,.599,'Phase 2: Score relative prediction error',fontsize=9,fontweight='bold')
    box(.015,.405,.25,.145,'Absolute prediction error','Mean across measured genes',edge='#ae7e22',fill='#fff9ec')
    box(.32,.405,.365,.145,'Conditional discordance','Subtract mean error among spots\nwith similar total expression',edge='#ae7e22',fill='#fff9ec')
    box(.74,.405,.245,.145,'Quartiles within each section','Q1: relatively well predicted\nQ4: relatively poorly predicted',edge='#ae7e22',fill='#fff9ec',fs=7.5)
    arrow(.648,.642,.648,.575);arrow(.89,.642,.89,.575)
    ax.plot([.14,.89],[.575,.575],c='#55646b',lw=.8);arrow(.14,.575,.14,.556)
    arrow(.27,.477,.312,.477);arrow(.69,.477,.731,.477)
    ax.text(.015,.355,'Checks: agreement across models · spatial structure · disjoint gene sets',fontsize=8)
    ax.text(.015,.294,'Phase 3: Characterize discordant and concordant locations',fontsize=9,fontweight='bold')
    xs=[.015,.267,.519,.771];titles=['Gene expression','Pathway comparisons','Cell context','Reproducibility']
    bodies=['Observed expression\nQ4 − Q1','Observed expression\nSigned and absolute errors','Marker scores\nSource cell annotations','Sections and cohorts\nOther tissues']
    for x,t,b in zip(xs,titles,bodies):box(x,.093,.214,.155,t,b,edge='#267b70',fill='#eff8f5',fs=8)
    ax.text(.5,.035,'Exclude tested genes from grouping; report donor effects and uncertainty',ha='center',fontsize=8)
    save(fig,'figures/main_figure.pdf','Readable workflow with the same three phases, explicit magnitude score and Q1/Q4; obsolete miniature result plots removed.')

def prediction():
    q=load(REV/'focused_revision/source_data/figure2_gene_correlations.csv')
    fig,axs=plt.subplots(1,2,figsize=(7.2,2.1),layout='constrained')
    for ax,cohort in zip(axs,IDC):
        z=q[q.cohort.eq(cohort)].sort_values('pearson');v=z.pearson.to_numpy();parts=ax.violinplot(v,positions=[0],showextrema=False)
        for p in parts['bodies']:p.set_facecolor(C[cohort]);p.set_alpha(.25)
        ax.scatter(np.random.RandomState(42).normal(0,.04,len(v)),v,s=4,color=C[cohort],alpha=.45);ax.plot([-.15,.15],[v.mean()]*2,c=C[cohort],lw=2)
        for j,yy in zip([0,1,2,-3,-2,-1],np.r_[np.linspace(v.min(),v.min()+.105,3),np.linspace(v.max()-.105,v.max(),3)]):
            r=z.iloc[j];ax.annotate(r.gene,xy=(.12,r.pearson),xytext=(.28,yy),fontsize=7,va='center',arrowprops=dict(arrowstyle='-',lw=.35,color='.6'))
        ax.set(xlim=(-.32,.65),xticks=[],ylabel='Mean gene Pearson r',title=f'{L[cohort]}: mean r = {v.mean():.3f}')
    save(fig,'figures/figure1/fig1c_pearson_distribution.pdf','Larger gene labels and clearer separation of annotation leaders.',q)
    m=tab('model_cohort_correlations');fig,axs=plt.subplots(1,2,figsize=(7.2,1.8),layout='constrained')
    for ax,cohort in zip(axs,IDC):
        z=m[m.cohort.eq(cohort)].pivot(index='encoder',columns='regressor',values='pearson').reindex(index=list(EN),columns=list(RN))
        ax.imshow(z,aspect='auto',cmap='Blues',vmin=0,vmax=1)
        ax.set(xticks=range(3),xticklabels=list(RN.values()),yticks=range(3),yticklabels=list(EN.values()),title=L[cohort])
        for i in range(3):
            for j in range(3):ax.text(j,i,f'{z.iloc[i,j]:.3f}',ha='center',va='center',fontsize=9,color='white' if z.iloc[i,j]>.55 else 'black')
    save(fig,'figures/figure1/fig1d_encoder_regressor_heatmap.pdf','High contrast numerical labels; correct encoder and regressor names; common 0–1 color scale.',m)
    q=tab('model_gene_cohort_quality');ridge=q[q.regressor.eq('ridge')].groupby(['cohort','gene']).pearson.mean()
    for cohort,suffix in zip(IDC,['disc','val']):
        z=q[q.cohort.eq(cohort)].copy();z['model']=z.encoder+' / '+z.regressor;d=z.pivot(index='gene',columns='model',values='pearson').loc[ridge.loc[cohort].sort_values(ascending=False).index]
        fig,ax=plt.subplots(figsize=(7.2,1.85),layout='constrained');im=ax.imshow(d.T,aspect='auto',cmap='viridis',vmin=0,vmax=1,interpolation='nearest')
        labels=[EN[s.split(' / ')[0]]+' / '+RN[s.split(' / ')[1]] for s in d.columns]
        ax.set(yticks=range(9),yticklabels=labels,xticks=[],xlabel='280 genes ordered by ridge predictability',title=L[cohort]);ax.tick_params(axis='y',labelsize=7)
        fig.colorbar(im,ax=ax,label='Mean Pearson r',pad=.015)
        save(fig,f'supplementary_figures/figS3a_gene_heatmap_{suffix}.pdf','Larger model labels in a compact heatmap displayed at full page width.',d)

def heatmaps():
    p=programs(tab('program_cohort_effects'));p=p[p.family.isin(IDC)&p.metric.eq('standardized_effect')]
    z=p.copy();z['column']=z.family.map({'biomarkers':'D','10x_janesick':'V'})+' '+z.outcome.map({'observed':'expression','signed':'signed','absolute':'error'})
    frame=z.pivot(index='pathway',columns='column',values='mean').reindex(columns=['D expression','V expression','D signed','V signed','D error','V error'])
    fig,ax=plt.subplots(figsize=(4.2,4),layout='constrained');im=heat(ax,frame,fs=7.2);ax.tick_params(axis='x',labelsize=7)
    scale(fig,im,ax=ax,shrink=.5,pad=.02,label='Q4 − Q1 / pooled SD').ax.tick_params(labelsize=7)
    save(fig,'figures/figure3/fig3c_pathway_heatmap.pdf','Larger pathway labels with correct acronym capitalization; effect values and color limits retained.',frame)
    fig,axs=plt.subplots(1,3,figsize=(7.2,4.8),layout='constrained',sharey=True)
    for i,(ax,outcome) in enumerate(zip(axs,['observed','signed','absolute'])):
        d=p[p.outcome.eq(outcome)].pivot(index='pathway',columns='family',values='mean').reindex(columns=IDC);d.columns=['Discovery','Validation']
        im=heat(ax,d,fs=7.5);ax.set_title({'observed':'Observed expression','signed':'Signed residual','absolute':'Absolute error'}[outcome],fontsize=8)
        if i:ax.tick_params(axis='y',left=False,labelleft=False)
    scale(fig,im,ax=axs,shrink=.46,pad=.015,label='Q4 − Q1 / pooled SD')
    save(fig,'supplementary_figures/figS9a_full_pathway_heatmap.pdf','One readable pathway label column and the missing common quantitative color scale.',p)
    cov=load(REV/'methodology_audit/stage_05_biology/coverage.csv');cov=cov[cov.eligible];d=cov.pivot(index='pathway',columns='cohort',values='n_genes')
    fig,ax=plt.subplots(figsize=(7.2,4.5),layout='constrained');d.rename(columns=L).plot.barh(ax=ax,color=[C[x] for x in d.columns],width=.8)
    ax.set(xlabel='Measured Hallmark members',ylabel='');ax.set_yticklabels([pname(x) for x in d.index],fontsize=7);ax.legend(title=None)
    save(fig,'supplementary_figures/figS9b_pathway_overlap.pdf','Readable pathway names and Discovery/Validation legend instead of internal dataset identifiers.',d)
    sp=programs(tab('program_section_effects'))
    for family,suffix in zip(IDC,['discovery','validation']):
        z=sp[sp.family.eq(family)&sp.outcome.eq('observed')];d=z.pivot(index='pathway',columns='sample',values='standardized_effect')
        fig,ax=plt.subplots(figsize=(7.2,4.3),layout='constrained');im=heat(ax,d,fs=7);scale(fig,im,ax=ax,shrink=.5,pad=.015,label='Observed expression contrast / SD')
        save(fig,f'supplementary_figures/figS17{ "a" if family==IDC[0] else "b"}_pathway_heatmap_{suffix}.pdf','Larger pathway labels with consistent biological abbreviations.',d)
    pu=programs(tab('program_unit_effects'))
    for family,rel,size in [('coad','supplementary_figures/figS18b_coad_pathway_heatmap.pdf',(6.2,4.3)),('idc_visium','visium_figures/fig_visium_pathway_heatmap.pdf',(7.2,6.5))]:
        z=pu[pu.family.eq(family)&pu.outcome.eq('observed')];d=z.pivot(index='pathway',columns='unit',values='standardized_effect')
        d=d.rename(columns={'10x_block_A':'TENX13 / TENX14'})
        fig,ax=plt.subplots(figsize=size,layout='constrained');im=heat(ax,d,fs=7.3);scale(fig,im,ax=ax,shrink=.5,label='Observed expression contrast / SD')
        save(fig,rel,'Readable pathway labels and biological capitalization; named paired Visium sections replace internal block code.',d)

def gene_and_cell_panels():
    s=tab('gene_section_effects');s=s[s.grouping.eq('gene_excluded')&s.adjustment.eq('overlap_adjusted')&s.outcome.eq('observed_log')]
    names=['EPCAM','KRT19','FOXA1','ESR1','KRT17','TP63','CD163','FBLN1','PDGFRB','MKI67','SNAI1','ZEB1','ZEB2']
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.2),layout='constrained')
    for ax,cohort in zip(axs,IDC):
        d=s[s.cohort.eq(cohort)].pivot(index='gene',columns='sample',values='standardized_effect').reindex(names);im=heat(ax,d,1.5,labels=names,fs=7);ax.set_title(L[cohort]);ax.tick_params(axis='x',labelsize=6.5)
    scale(fig,im,ax=axs,shrink=.5,label='Adjusted Q4 − Q1 / SD');save(fig,'supplementary_figures/figS8a_per_sample_volcanos.pdf','Uppercase gene symbols and larger labels.',s)
    d=load(REV/'focused_revision/source_data/marker_score_section_effects.csv');fig,axs=plt.subplots(1,2,figsize=(7.2,2.9),layout='constrained')
    for ax,cohort in zip(axs,IDC):
        z=d[d.cohort.eq(cohort)].pivot(index='cell_type',columns='sample',values='standardized_effect');labels=[x.replace('_',' ').capitalize().replace('Nk cell','NK cell') for x in z.index]
        im=heat(ax,z,1.5,labels=labels,fs=7);ax.set_title(L[cohort]);ax.tick_params(axis='x',labelsize=6.5)
    scale(fig,im,ax=axs,shrink=.6,label='Marker score contrast / SD');save(fig,'supplementary_figures/figS10_deconvolution_detail.pdf','Larger cell type labels with correct NK capitalization.',d)
    m=load(REV/'methodology_audit/stage_08_cell_composition/cell_mixture.csv');m=m[m.width_um.eq(1600)&m.kind.eq('group')]
    if m.empty:
        m=load(REV/'methodology_audit/stage_08_cell_composition/cell_mixture.csv');m=m[m.width_um.eq(1600)&m.label.isin(['Tumor','Stromal','Macrophage'])]
    fig,ax=plt.subplots(figsize=(6,2.8),layout='constrained');samples=['NCBI785','NCBI784','NCBI783'];cols=['#347da1','#78aabe','#bc6280']
    for k,(sample,col) in enumerate(zip(samples,cols)):
        z=m[m['sample'].eq(sample)].set_index('label')
        for j,label in enumerate(['Tumor','Stromal','Macrophage']):
            r=z.loc[label];y=j+(k-1)*.2;ax.plot([r.ci_low*100,r.ci_high*100],[y,y],c=col);ax.scatter(r.difference*100,y,c=col,s=17,label=sample if j==0 else None)
    zero(ax);ax.set(yticks=range(3),yticklabels=['Tumor','Stromal','Macrophage'],xlabel='Q4 − Q1 cell fraction (percentage points)');ax.invert_yaxis();ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=3,frameon=False)
    save(fig,'supplementary_figures/figS14b_celltype_per_sample.pdf','Move legend outside the plotting area so it cannot obscure the tumor fraction interval.',m)
    f=tab('gene_features');fig,axs=plt.subplots(1,2,figsize=(7.2,2.7),layout='constrained');rng=np.random.RandomState(42)
    for ax,cohort in zip(axs,IDC):
        z=f[f.cohort.eq(cohort)];groups=sorted(z.primary_localization.unique());v=[z[z.primary_localization.eq(g)].mean_pearson for g in groups]
        ax.boxplot(v,showfliers=False,patch_artist=True,boxprops={'facecolor':C[cohort],'alpha':.25})
        for i,values in enumerate(v,1):ax.scatter(i+rng.uniform(-.13,.13,len(values)),values,s=4,c=C[cohort],alpha=.5,edgecolors='none',rasterized=True)
        ax.set(xticks=range(1,len(groups)+1),xticklabels=groups,ylabel='Mean prediction r',title=L[cohort]);ax.tick_params(axis='x',rotation=65,labelsize=6.5)
    save(fig,'supplementary_figures/figS11b_localization_predictability.pdf','Show the gene points already described in the caption, including sparse categories.',f)

def identity_lines():
    d=tab('gene_cohort_effects');d=d[d.outcome.eq('observed_log')&d.metric.eq('standardized_effect')];fig,axs=plt.subplots(1,2,figsize=(7.2,2.6),layout='constrained')
    for ax,cohort in zip(axs,IDC):
        a=d[d.cohort.eq(cohort)&d.grouping.eq('full')&d.adjustment.eq('unadjusted')].set_index('gene')['mean'];b=d[d.cohort.eq(cohort)&d.grouping.eq('gene_excluded')&d.adjustment.eq('overlap_adjusted')].set_index('gene')['mean'].reindex(a.index)
        scatter(ax,a,b,C[cohort],L[cohort],True);ax.set(xlabel='Full score: unadjusted contrast',ylabel='After gene exclusion and adjustment');zero(ax);ax.axhline(0,c='.65',lw=.6,zorder=0)
    save(fig,'supplementary_figures/figS13_matched_vs_unmatched.pdf','Add the identity lines described in the caption; retain all points and descriptive fits.',d)
    d=load(REV/'focused_revision/source_data/section_gene_absolute_residuals.csv');fig,axs=plt.subplots(1,3,figsize=(7.2,2.3),layout='constrained')
    for ax,patient,a,b in zip(axs,['P01','P02','P07'],['TENX99','TENX97','NCBI785'],['TENX98','TENX95','NCBI784']):
        scatter(ax,d[a],d[b],C['10x_janesick'],patient,True);ax.set(xlabel=a+' gene MAE',ylabel=b+' gene MAE')
    save(fig,'supplementary_figures/figS12a_within_patient_gene_scatter.pdf','Add the identity lines described in the caption; retain points and descriptive fits.',d)

def reproducibility():
    old=load(REV/'focused_revision/source_data/section_gene_absolute_residuals.csv');fig,axs=plt.subplots(2,2,figsize=(7.2,5.1),layout='constrained')
    for ax,patient,a,b in zip(axs[0],['P02','P07'],['TENX97','NCBI785'],['TENX95','NCBI784']):
        scatter(ax,old[a],old[b],C['10x_janesick'],patient,True);ax.set(xlabel=a+' mean absolute error',ylabel=b+' mean absolute error')
    pairs=load(REV/'methodology_audit/stage_14_remaining_analyses/patient_profile_pairs.csv');pairs=pairs[pairs.family.eq('10x_janesick')&pairs.grouping.eq('program_excluded')&pairs.adjustment.eq('overlap_adjusted')&pairs.outcome.eq('observed')]
    for j,same in enumerate([True,False]):
        v=pairs[pairs.same_patient.eq(same)].pearson;axs[1,0].scatter(j+np.linspace(-.09,.09,len(v)),v,c=C['10x_janesick'],s=15,alpha=.65)
    axs[1,0].set(xticks=[0,1],xticklabels=['Within donor','Between donors'],ylabel='Pathway profile Pearson r',title='Validation section pairs');axs[1,0].text(.02,.04,'Permutation p = 1/9',transform=axs[1,0].transAxes,fontsize=7)
    q=load(REV/'focused_revision/source_data/bridge_predictability.csv');scatter(axs[1,1],q.biomarkers,q['10x_janesick'],'#597487','All 90 shared genes');axs[1,1].set(xlabel='Discovery prediction r',ylabel='Validation prediction r')
    for ax,letter in zip(axs.flat,'abcd'):ax.text(.5,-.32,f'({letter})',ha='center',transform=ax.transAxes,fontsize=8)
    save(fig,'figures/figure5/fig5_combined_acd.pdf','Consistent panel letters and clearer labels; same section pairs, shared genes, effects and permutation result.')
    d=tab('external_program_comparisons');d=d[d.family.eq('coad')&d.comparison.eq('exact_common_members')&d.outcome.eq('observed')&d.adjustment.eq('overlap_adjusted')&d.scope.eq('all')]
    fig,ax=plt.subplots(figsize=(3.5,4.5),layout='constrained')
    for i,r in enumerate(d.itertuples()):
        ax.plot([r.idc_standardized_effect,r.external_standardized_effect],[i,i],c='.7',lw=1);ax.scatter(r.idc_standardized_effect,i,c=C[r.idc_cohort],s=20);ax.scatter(r.external_standardized_effect,i,c=C['coad'],marker='s',s=18)
    zero(ax);ax.set(yticks=range(len(d)),yticklabels=[L[r.idc_cohort]+'\n'+pname(r.pathway) for r in d.itertuples()],xlabel='Observed Q4 − Q1 / SD\n(identical measured genes)');ax.invert_yaxis();ax.tick_params(axis='y',labelsize=7)
    ax.legend(handles=[Line2D([],[],marker='o',linestyle='',color=C[c],label=L[c],markersize=4) for c in IDC]+[Line2D([],[],marker='s',linestyle='',color=C['coad'],label='COAD',markersize=4)],loc='lower center',bbox_to_anchor=(.5,1.01),fontsize=6.5,ncol=1,frameon=False)
    save(fig,'figures/figure6/fig6c_cross_cancer_pathway.pdf','Explicit cohort legend and readable pathway labels; same seven matched comparisons.',d)

def pdf_local_changes():
    rel='supplementary_figures/figS5a_pairwise_correlation_matrices.pdf';source=A/'source'/rel;INPUTS[str(source)]={'sha256':sha(source)}
    old=fitz.open(source);new=fitz.open();p=new.new_page(width=518,height=284)
    for src,target,title in [(fitz.Rect(52.2244,14.901,227.7936,188.8087),fitz.Rect(28,24,221,217),'TENX193'),(fitz.Rect(282.4254,14.901,457.9946,188.8087),fitz.Rect(261,24,454,217),'NCBI785')]:
        p.show_pdf_page(target,old,0,clip=src)
        p.insert_text(((target.x0+target.x1-fitz.get_text_length(title,fontsize=9))/2,16),title,fontsize=9,fontname='helv')
        for j in range(9):
            y=target.y0+(j+.5)*target.height/9;x=target.x0+(j+.5)*target.width/9
            p.insert_text((target.x0-12,y+2.5),str(j+1),fontsize=8)
            p.insert_text((x-2.5,target.y1+12),str(j+1),fontsize=8)
    p.show_pdf_page(fitz.Rect(471,62,514,170),old,0,clip=fitz.Rect(477,54,516,151))
    for i,enc in enumerate(EN.values()):
        txt='   '.join(f'{i*3+j+1}: {enc} / {r}' for j,r in enumerate(RN.values()))
        p.insert_textbox(fitz.Rect(14,237+i*14,505,251+i*14),txt,fontsize=8,fontname='helv',align=1)
    new.save(S/rel);register(rel,'Use readable numbered model keys while preserving the original matrix and color scale graphics exactly.')
    rel='supplementary_figures/figS2_batch_effect_pca.pdf';old=fitz.open(A/'source'/rel);p=old[0]
    for w in p.get_text('words'):
        if re.fullmatch(r'(TENX|NCBI)\d+',w[4]):p.add_redact_annot(fitz.Rect(w[:4]),fill=None)
    p.apply_redactions(images=0,graphics=0);old.save(S/rel);register(rel,'Remove overlapping section name annotations; preserve all points, PCA coordinates, legends and variance/silhouette values.')
    rel='supplementary_figures/figS18_spatial_coexpression.pdf';old=fitz.open(A/'source'/rel);p=old[0]
    matches=p.search_for('Positive-positive pairs / exact null expectation');assert len(matches)==1
    r=matches[0]
    # Matplotlib's text bounding boxes overlap vertically even though glyphs
    # do not. Restrict deletion below the tick labels to preserve every tick.
    p.add_redact_annot(fitz.Rect(r.x0,r.y0+4,r.x1,r.y1),fill=None);p.apply_redactions(images=0,graphics=0)
    status=p.insert_textbox(fitz.Rect(r.x0-20,r.y0,r.x1+20,r.y1+8),'Pairs of coexpressing cells / expected pairs',fontsize=10,fontname='helv',align=1);assert status>=0
    old.save(S/rel);register(rel,'Replace the joined and ambiguous positive-positive axis label with plain wording; all plotted marks and null intervals preserved.')

if __name__=='__main__':
    functions=[workflow,prediction,heatmaps,gene_and_cell_panels,identity_lines,reproducibility,pdf_local_changes]
    if len(sys.argv)>1:
        previous=json.loads((B/'figure_review_manifest.json').read_text())
        INPUTS.update(previous['inputs']);CHANGES.update(previous['assets']);DATA.update(previous['plotted_data'])
        assert set(sys.argv[1:]) <= {f.__name__ for f in functions}
        functions=[f for f in functions if f.__name__ in sys.argv[1:]]
    for f in functions:
        f();print(f.__name__,'done',flush=True)
    for path,meta in INPUTS.items():assert sha(path)==meta['sha256']
    (B/'figure_review_manifest.json').write_text(json.dumps(dict(inputs=INPUTS,assets=CHANGES,plotted_data=DATA),indent=2)+'\n')
    print('Updated',len(CHANGES),'figure assets from fixed results.')

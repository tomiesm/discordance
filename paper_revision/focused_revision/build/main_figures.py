"""Refresh existing panel slots; do not redesign the seven-figure manuscript."""
from common import *
import shutil,fitz
from itertools import combinations

def preserve_assets():
    for folder in ['figures','supplementary_figures','visium_figures']:
        for source in (OLD/folder).rglob('*'):
            if source.is_file():
                dest=SRC/folder/source.relative_to(OLD/folder)
                if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    if not (SRC/'references.bib').exists():shutil.copy2(OLD/'references.bib',SRC/'references.bib')

def schematic():
    source=track(OLD/'figures/main_figure.pdf');doc=fitz.open(source);p=doc[0];blocks=p.get_text('blocks')
    changes={1:'11 sections / 11 donors | 280 genes | grouped CV',3:'7 sections / 4 donor groups | 280 genes | grouped CV',11:'Pretrained histology',13:'Pretrained histology',15:'Pretrained histology',18:'PCA(256) then ridge + intercept',19:'Grouped cross-validation',26:'measured RNA',27:'y - prediction',30:'Signed residual',32:'D_cond = expression-bin-centered MAE',35:"Moran's I > 0; ring-reference p = .001",36:'Expression Contrasts',38:'Donor effects and uncertainty',39:'Program Associations',41:'Observed / signed / absolute outcomes',42:'Cell Context',44:'Marker scores and source-cell fractions',47:'Reliability Checks',51:"Moran's I > 0; p = .001",53:'Conditional split-half agreement'}
    for i in changes:
        r=fitz.Rect(blocks[i][:4]);r.x0-=1;r.x1+=1;p.add_redact_annot(r,fill=(1,1,1))
    p.apply_redactions(images=0,graphics=0)
    for i,txt in changes.items():
        r=fitz.Rect(blocks[i][:4]);center=(r.x0+r.x1)/2
        width=max(r.width,90 if i in [1,3] else 75 if i in [32,35,38,41,44] else 50)
        fs=3.3 if i in [1,3,18,19,32,35,38,41,44,51,53] else min(5,r.height*.7)
        r=fitz.Rect(center-width/2,r.y0-.5,center+width/2,r.y1+2)
        status=p.insert_textbox(r,txt,fontsize=fs,fontname='helv',align=1,color=(.15,.15,.15));assert status>=0,(i,status)
    doc.save(SRC/'figures/main_figure.pdf');doc.close();ASSETS.append(dict(path='figures/main_figure.pdf',action='local vector text correction',reason='Correct donor groups, score interpretation and unsupported biological labels.'))

def prediction():
    reg=table('specimen_registry');fig,axs=plt.subplots(2,11,figsize=(7.2,2.2));fig.subplots_adjust(hspace=.45,wspace=.08)
    for i,cohort in enumerate(IDC):
        z=reg[reg.cohort.eq(cohort)]
        for j,(_,r) in enumerate(z.iterrows()):
            he(axs[i,j],r['sample']);axs[i,j].set_title(r['sample'],fontsize=5,color=C[cohort]);axs[i,j].text(.5,-.06,r.patient,ha='center',transform=axs[i,j].transAxes,fontsize=5)
        for j in range(len(z),11):axs[i,j].axis('off')
        axs[i,0].text(-.16,.5,L[cohort],rotation=90,va='center',transform=axs[i,0].transAxes,color=C[cohort],fontsize=6)
    save(fig,'figures/figure1/fig1b_he_grid.pdf','Preserve images/order; correct discovery donor labels.')
    ss=oldtable('figure2_CD8A_map');ss=ss.merge(locations()[['spot_id','conditional']],on='spot_id',validate='one_to_one')
    fig,axs=plt.subplots(1,4,figsize=(7.2,2.3),layout='constrained');he(axs[0],'TENX193',ss);axs[0].set_title('H&E')
    lo,hi=np.quantile(np.r_[ss.observed,ss.predicted],[.02,.98]);lim=np.quantile(abs(ss.signed_residual),.98)
    for ax,col,title in zip(axs[1:],['observed','predicted','signed_residual'],['Observed CD8A','Predicted CD8A','Observed − predicted']):
        im=mapplot(ax,ss,ss[col],cmap='RdBu_r' if col=='signed_residual' else 'viridis',vmin=-lim if col=='signed_residual' else lo,vmax=lim if col=='signed_residual' else hi,title=title);fig.colorbar(im,ax=ax,shrink=.65,pad=.01)
    save(fig,'figures/figure1/fig1e_spatial_prediction.pdf','Corrected predictions; consistent observed-minus-predicted sign.')
    q=table('model_gene_cohort_quality');q=q[q.regressor.eq('ridge')].groupby(['cohort','gene']).pearson.mean().reset_index();savedata(q,'figure2_gene_correlations')
    fig,axs=plt.subplots(1,2,figsize=(7.2,2.1),layout='constrained')
    for ax,cohort in zip(axs,IDC):
        z=q[q.cohort.eq(cohort)].sort_values('pearson');v=z.pearson.to_numpy();parts=ax.violinplot(v,positions=[0],showextrema=False)
        for part in parts['bodies']:part.set_facecolor(C[cohort]);part.set_alpha(.25)
        ax.scatter(np.random.RandomState(42).normal(0,.04,len(v)),v,s=4,color=C[cohort],alpha=.45);ax.plot([-.15,.15],[v.mean()]*2,c=C[cohort],lw=2)
        for j,yy in zip([0,1,2,-3,-2,-1],np.r_[np.linspace(v.min(),v.min()+.08,3),np.linspace(v.max()-.08,v.max(),3)]):
            r=z.iloc[j];ax.annotate(r.gene,xy=(.12,r.pearson),xytext=(.28,yy),fontsize=5,va='center',arrowprops=dict(arrowstyle='-',lw=.35,color='.6'))
        ax.set(xlim=(-.4,.6),xticks=[],ylabel='Donor-mean gene Pearson r',title=f'{L[cohort]}: mean r = {v.mean():.3f}')
    save(fig,'figures/figure1/fig1c_pearson_distribution.pdf','Per-section correlations aggregated over donor and ridge encoder.')
    fig,axs=plt.subplots(1,2,figsize=(7.2,1.7),layout='constrained');m=table('model_cohort_correlations')
    for ax,cohort in zip(axs,IDC):
        z=m[m.cohort.eq(cohort)].pivot(index='encoder',columns='regressor',values='pearson').reindex(index=['uni','virchow2','hoptimus0'],columns=['ridge','mlp','xgboost']);im=heat(ax,z,limit=1,cmap='Blues',numbers=True);ax.set_title(L[cohort]);ax.tick_params(axis='x',rotation=0)
    save(fig,'figures/figure1/fig1d_encoder_regressor_heatmap.pdf','Same nine configurations with corrected donor-level averaging.')

def half_vectors():
    sys.path.insert(0,str(R));from src.discordance import compute_conditional_discordance
    result={}
    for family,sample in zip(IDC,['TENX193','NCBI785']):
        ys=[];errors=[];ids=[]
        for k in range(4):
            fd=R/'outputs/predictions'/family/'uni/ridge'/f'fold{k}';ys.append(np.load(track(fd/'test_targets.npy')));ids+=json.loads(track(fd/'test_spot_ids.json').read_text())
        y=np.concatenate(ys)
        for enc in ['uni','virchow2','hoptimus0']:
            errors.append(np.concatenate([abs(np.load(track(R/'outputs/predictions'/family/enc/'ridge'/f'fold{k}'/'test_residuals.npy'))).astype(float) for k in range(4)]))
        e=np.mean(errors,axis=0);ix=np.random.RandomState(42).permutation(280);a,b=ix[:140],ix[140:]
        va=compute_conditional_discordance(e[:,a].mean(axis=1),y[:,a].sum(axis=1));vb=compute_conditional_discordance(e[:,b].mean(axis=1),y[:,b].sum(axis=1));mask=np.array([x.startswith(sample+'_') for x in ids]);result[sample]=(va[mask],vb[mask])
        ref=audit('stage_04_reliability/gene_half_stability.csv');ref=ref[ref['sample'].eq(sample)&ref.seed.eq(42)&ref.source.eq('ridge_encoder_mean')&ref['mode'].eq('conditional_own_half')].spearman.item();assert abs(spearmanr(va[mask],vb[mask]).statistic-ref)<1e-10
    return result

def spatial():
    for sample in ['TENX193','NCBI785']:
        ss=locations().query('sample==@sample');fig,axs=plt.subplots(1,3,figsize=(7.2,2.4),layout='constrained');he(axs[0],sample,ss);axs[0].set_title(sample+' / H&E')
        lim=np.quantile(abs(ss.conditional),.98);im=mapplot(axs[1],ss,ss.conditional,'RdBu_r',-lim,lim,'Conditional discordance');fig.colorbar(im,ax=axs[1],shrink=.65,pad=.01)
        null=np.load(track(R/'outputs/figure_data'/f'morans_null_{sample}.npy'));obs=json.loads(track(R/'outputs/figure_data'/f'morans_obs_{sample}.json').read_text())
        ref=audit('stage_04_reliability/spatial_structure.csv').query('sample==@sample and graph=="original_k6" and score=="conditional"').morans_i.item();assert abs(ref-obs['I_obs'])<1e-10
        axs[2].hist(null,bins=25,color='.75');axs[2].axvline(ref,color=HIGH,lw=1.3);axs[2].set(xlabel="Moran's I",ylabel='Permutation count',title=f'Within-ring reference\nI = {ref:.3f}; p = {obs["p_value"]:.3f}')
        save(fig,f'figures/figure2/fig2b_{sample}.pdf','Fixed exemplars; corrected score and archived ring permutation.')
    hv=half_vectors();g=json.loads(track(R/'outputs/phase2/gate2_1_agreement.json').read_text());fig,axs=plt.subplots(1,3,figsize=(7.2,2.8),layout='constrained')
    for i,(key,family) in enumerate(zip(['discovery','validation'],IDC)):
        vals=list(g['cohorts'][key]['samples'].values());v=np.array([x['median_rho'] for x in vals]);axs[0].scatter(np.full(len(v),i)+np.linspace(-.08,.08,len(v)),v,c=C[family],s=13)
    axs[0].axhline(.4,c='.5',ls='--',lw=.8);axs[0].set(xticks=[0,1],xticklabels=['Discovery','Validation'],ylabel='Median pairwise Spearman ρ',title='Nine-model score agreement')
    for ax,sample in zip(axs[1:],['TENX193','NCBI785']):
        a,b=hv[sample];ax.scatter(a,b,s=.45,color='#507e98',alpha=.25,rasterized=True);ax.set(xlabel='Conditional score: half A',ylabel='Conditional score: half B',title=f'{sample}\nSpearman ρ = {spearmanr(a,b).statistic:.3f}')
    save(fig,'figures/figure2/fig2ac_agreement_dual_track.pdf','Preserve agreement and half-gene slots; show actual conditional scores.')

def marker_scores():
    reg=table('specimen_registry').set_index('sample');rows=[]
    for sample,r in reg.iterrows():
        p=track(R/'outputs/phase3/deconvolution'/r.cohort/'per_sample'/f'{sample}_celltype_scores.csv');d=pd.read_csv(p);ss=locations().query('sample==@sample');q1,q4=np.quantile(ss.conditional,[.25,.75]);assert len(d)==len(ss)
        # Reconstruct IDs explicitly: never assume saved row correspondence.
        ids=sample+'_'+d.barcode.astype(str);sc=ss.set_index('spot_id').loc[ids,'conditional'].to_numpy();assert np.max(abs(sc-d.D_cond))<1e-6
        for name in d.columns.difference(['barcode','D_cond']):
            lo=d.loc[sc<=q1,name].to_numpy();hi=d.loc[sc>=q4,name].to_numpy();sd=np.sqrt(((len(lo)-1)*lo.var(ddof=1)+(len(hi)-1)*hi.var(ddof=1))/(len(lo)+len(hi)-2))
            rows.append(dict(cohort=r.cohort,sample=sample,patient=r.patient,cell_type=name,standardized_effect=(hi.mean()-lo.mean())/sd,Q1_mean=lo.mean(),Q4_mean=hi.mean()))
    d=pd.DataFrame(rows);savedata(d,'marker_score_section_effects');return d

def biology():
    sample='TENX193';ss=locations().query('sample==@sample');d=pd.read_csv(track(R/'outputs/phase3/deconvolution/biomarkers/per_sample/TENX193_celltype_scores.csv'));d.index=sample+'_'+d.barcode.astype(str);d=d.loc[ss.spot_id]
    fig,axs=plt.subplots(1,4,figsize=(7.2,2.0),layout='constrained');he(axs[0],sample,ss);axs[0].set_title('H&E / '+sample);quartile(axs[1],ss)
    for ax,key in zip(axs[2:],['epithelial','macrophage']):
        im=mapplot(ax,ss,d[key],title=key.capitalize()+' markers');fig.colorbar(im,ax=ax,shrink=.65,pad=.01)
    save(fig,'figures/figure3/fig3a_spatial_biology.pdf','Preserve biological map exemplar; correct quartiles and marker labels.')
    g=table('gene_cohort_effects');g=g[g.grouping.eq('gene_excluded')&g.adjustment.eq('overlap_adjusted')&g.outcome.eq('observed_log')&g.metric.eq('standardized_effect')]
    names=['EPCAM','KRT19','FOXA1','ESR1','KRT17','TP63','CD163','FBLN1','PDGFRB','MKI67'];fig,axs=plt.subplots(1,2,figsize=(7.2,2.2),layout='constrained')
    for ax,cohort in zip(axs,IDC):
        z=g[g.cohort.eq(cohort)].set_index('gene').reindex(names)
        for j,(_,r) in enumerate(z.iterrows()):
            if pd.notna(r['mean']):ax.plot([r.ci_low,r.ci_high],[j,j],c=C[cohort],lw=1);ax.scatter(r['mean'],j,c=C[cohort],s=12)
        zero(ax);ax.set(yticks=range(len(names)),yticklabels=names,xlabel='Adjusted Q4 − Q1 / within-tail SD',title=L[cohort]);ax.invert_yaxis()
    save(fig,'figures/figure3/fig3d_volcano.pdf','Replace spot-significance axis with donor mean effects and approximate intervals in the same slot.')
    d=marker_scores().groupby(['cohort','patient','cell_type']).standardized_effect.mean().groupby(['cohort','cell_type']).mean().unstack(0)
    fig,ax=plt.subplots(figsize=(3.2,3.2),layout='constrained')
    for i,name in enumerate(d.index):
        v=d.loc[name];ax.plot(v.dropna().values,[i]*len(v.dropna()),color='.75',lw=1)
        for cohort in IDC:
            if pd.notna(v.get(cohort)):ax.scatter(v[cohort],i,c=C[cohort],s=22,label=L[cohort] if i==0 else None)
    from matplotlib.lines import Line2D
    zero(ax);ax.set(yticks=range(len(d)),yticklabels=[x.replace('_',' ') for x in d.index],xlabel='Unadjusted marker-score Q4 − Q1 / SD');ax.legend(handles=[Line2D([],[],marker='o',linestyle='',color=C[c],label=L[c],markersize=4) for c in IDC]);ax.invert_yaxis();save(fig,'figures/figure3/fig3b_deconvolution_dumbbell.pdf','Retain marker-score comparison; no deconvolution or significance-star claim.')
    p=primary_programs(table('program_cohort_effects'));p=p[p.family.isin(IDC)&p.metric.eq('standardized_effect')];p['column']=p.family.map({'biomarkers':'D','10x_janesick':'V'})+' '+p.outcome.map({'observed':'expression','signed':'signed','absolute':'error'})
    frame=p.pivot(index='pathway',columns='column',values='mean');frame=frame.reindex(columns=['D expression','V expression','D signed','V signed','D error','V error'])
    fig,ax=plt.subplots(figsize=(4.2,4.0),layout='constrained');im=heat(ax,frame,limit=2);ax.tick_params(axis='y',labelsize=5);fig.colorbar(im,ax=ax,shrink=.5,label='Adjusted Q4 − Q1 / within-tail SD');save(fig,'figures/figure3/fig3c_pathway_heatmap.pdf','Preserve pathway heatmap; distinguish observed, signed and absolute outcomes, without unsupported stars.')

def gene_features():
    f=table('gene_features');fig,axs=plt.subplots(1,2,figsize=(7.2,2.4),layout='constrained')
    for ax,cohort in zip(axs,IDC):
        z=f[f.cohort.eq(cohort)];scatterfit(ax,z.spatial_autocorrelation,z.mean_pearson,C[cohort],L[cohort]);ax.set(xlabel="Donor-mean gene Moran's I",ylabel='Donor-mean prediction Pearson r')
    save(fig,'figures/figure4/fig4a_morans_vs_pearson.pdf','All sections and correct donor-weighted expression/prediction features.')
    c=table('gene_feature_coefficients');c=c[c.model.eq('numeric_with_spatial')&c.feature.ne('intercept')];fig,axs=plt.subplots(1,2,figsize=(5.3,2.2),layout='constrained')
    for ax,cohort in zip(axs,IDC):
        z=c[c.cohort.eq(cohort)];ax.scatter(z.coefficient,np.arange(len(z)),c=C[cohort],s=20);zero(ax);ax.set(yticks=range(len(z)),yticklabels=['Mean expression','Expression CV','Pathway count','Spatial autocorrelation'],xlabel='Coefficient per feature SD',title=L[cohort]);ax.invert_yaxis()
    save(fig,'figures/figure4/fig4b_ols_coefficients.pdf','Numerical model; remove unsupported annotation inference and independent-gene p values.')
    cv=table('gene_feature_cv_predictions');cv=cv[cv.model.eq('numeric_with_spatial')];fig,axs=plt.subplots(1,2,figsize=(7.2,2.4),layout='constrained')
    for ax,cohort in zip(axs,IDC):
        z=cv[cv.cohort.eq(cohort)];scatterfit(ax,z.actual,z.predicted,C[cohort],L[cohort]);ax.plot([0,1],[0,1],ls='--',c='.6',lw=.7);ax.set(xlabel='Assessed gene predictability',ylabel='Held-out gene prediction')
    save(fig,'figures/figure4/fig4d_heldout_validation.pdf','Within-panel five-fold gene CV with corrected donor features.')

def reproducibility():
    q=table('gene_section_quality');q=q[q.grouping.eq('full')];sec=table('gene_section_effects');sec=sec[sec.grouping.eq('full')&sec.adjustment.eq('unadjusted')&sec.outcome.eq('absolute')]
    # Full-section gene MAE is obtained from both quartiles plus the middle via saved per-gene features.
    old=read_gene_residuals();fig,axs=plt.subplots(2,2,figsize=(7.2,5.1),layout='constrained')
    for ax,patient,ss in zip(axs[0],['P02','P07'],[['TENX97','TENX95'],['NCBI785','NCBI784']]):
        x=old[ss[0]];y=old[ss[1]];scatterfit(ax,x,y,C['10x_janesick'],patient);hi=max(x.max(),y.max());ax.plot([0,hi],[0,hi],ls='--',color='.6',lw=.7);ax.set(xlabel=ss[0]+' mean absolute error',ylabel=ss[1]+' mean absolute error')
    pairs=audit('stage_14_remaining_analyses/patient_profile_pairs.csv');pairs=pairs[pairs.family.eq('10x_janesick')&pairs.grouping.eq('program_excluded')&pairs.adjustment.eq('overlap_adjusted')&pairs.outcome.eq('observed')]
    for j,same in enumerate([True,False]):
        v=pairs[pairs.same_patient.eq(same)].pearson;axs[1,0].scatter(j+np.linspace(-.09,.09,len(v)),v,c=C['10x_janesick'],s=15,alpha=.65)
    axs[1,0].set(xticks=[0,1],xticklabels=['Within donor','Between donors'],ylabel='Program-profile Pearson r',title='Validation section pairs');axs[1,0].text(.02,.04,'Source-blocked grouping reference: p = 1/9',transform=axs[1,0].transAxes,fontsize=6)
    q=table('model_gene_cohort_quality');q=q[q.regressor.eq('ridge')].groupby(['cohort','gene']).pearson.mean().unstack('cohort').dropna();scatterfit(axs[1,1],q.biomarkers,q['10x_janesick'],'#597487','All 90 shared genes');axs[1,1].set(xlabel='Discovery prediction r',ylabel='Validation prediction r');savedata(q.reset_index(),'bridge_predictability')
    save(fig,'figures/figure5/fig5_combined_acd.pdf','Keep reproducibility layout; replace false discovery repeat pair with verified P07 and remove pair-independence inference.')

def read_gene_residuals():
    result={}
    for family in IDC:
        for k in range(4):
            fd=R/'outputs/predictions'/family/'uni/ridge'/f'fold{k}';ids=json.loads(track(fd/'test_spot_ids.json').read_text());a=[]
            for enc in ['uni','virchow2','hoptimus0']:a.append(abs(np.load(track(R/'outputs/predictions'/family/enc/'ridge'/f'fold{k}'/'test_residuals.npy'))).astype(float))
            a=np.mean(a,axis=0)
            for sample in table('specimen_registry').query('cohort==@family')['sample']:
                ix=np.array([s.startswith(sample+'_') for s in ids])
                if ix.any():result[sample]=a[ix].mean(axis=0)
    frame=pd.DataFrame(result);savedata(frame,'section_gene_absolute_residuals');return frame

def external():
    d=table('external_program_comparisons');d=d[d.family.eq('coad')&d.comparison.eq('exact_common_members')&d.outcome.eq('observed')&d.adjustment.eq('overlap_adjusted')&d.scope.eq('all')]
    fig,ax=plt.subplots(figsize=(3.5,4.5),layout='constrained')
    for i,r in enumerate(d.itertuples()):
        ax.plot([r.idc_standardized_effect,r.external_standardized_effect],[i,i],color='.7',lw=1);ax.scatter(r.idc_standardized_effect,i,c=C[r.idc_cohort],s=20);ax.scatter(r.external_standardized_effect,i,c=C['coad'],marker='s',s=18)
    zero(ax);ax.set(yticks=range(len(d)),yticklabels=[L[r.idc_cohort]+': '+short(r.pathway) for r in d.itertuples()],xlabel='Observed Q4 − Q1 / SD\n(identical members)');ax.invert_yaxis();save(fig,'figures/figure6/fig6c_cross_cancer_pathway.pdf','Exact measured-member comparisons; retain COAD panel position.')
    f=oldtable('external_gene_features');f=f[f.family.eq('coad')];fig,ax=plt.subplots(figsize=(3.3,2.7),layout='constrained');scatterfit(ax,f.morans_i,f.pearson,C['coad']);ax.set(xlabel="Gene Moran's I",ylabel='Gene prediction Pearson r');save(fig,'figures/figure6/fig6a_coad_morans_scatter.pdf','Audited four-section COAD features.')
    q=table('external_gene_pairs');q=q[q.family.eq('coad')&q.scope.eq('all')];q=q.groupby('gene')[['idc_pearson','external_pearson']].mean();fig,ax=plt.subplots(figsize=(3.3,2.7),layout='constrained');scatterfit(ax,q.idc_pearson,q.external_pearson,C['coad']);ax.set(xlabel='Breast-panel mean prediction r',ylabel='COAD prediction r',title=f'All {len(q)} shared union-panel genes');savedata(q.reset_index(),'COAD_union_gene_comparison');save(fig,'figures/figure6/fig6b_cross_tissue_scatter.pdf','Retain all shared union-panel genes; use corrected donor-weighted breast values.')

if __name__=='__main__':
    preserve_assets()
    for f in [schematic,prediction,spatial,biology,gene_features,reproducibility,external]:f();flush('main')

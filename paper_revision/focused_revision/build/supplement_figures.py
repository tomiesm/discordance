from common import *
import fitz
from matplotlib.backends.backend_pdf import PdfPages

def save_s(fig,name,reason):save(fig,'supplementary_figures/'+name+'.pdf',reason)
def qc_label():
    rel='supplementary_figures/figS1a_umi_distribution.pdf'
    doc=fitz.open(track(OLD/rel));page=doc[0]
    blocks=[b for b in page.get_text('blocks') if 'Total expression (log1p UMI)' in b[4]]
    assert len(blocks)==1
    rect=fitz.Rect(blocks[0][:4]);page.add_redact_annot(rect,fill=(1,1,1));page.apply_redactions(images=0,graphics=0)
    page.insert_text((rect.x0+6,rect.y1),'Sum of gene log1p counts',fontsize=6,rotate=90,fontname='helv')
    doc.save(SRC/rel);doc.close();ASSETS.append(dict(path=rel,action='local vector label correction',reason='Molecule-count log sum, not UMI library normalization; data unchanged.'))

def prediction():
    q=table('model_gene_cohort_quality');ridge=q[q.regressor.eq('ridge')].groupby(['cohort','gene']).pearson.mean()
    for cohort,suffix in zip(IDC,['disc','val']):
        z=q[q.cohort.eq(cohort)].copy();z['model']=z.encoder+' / '+z.regressor;d=z.pivot(index='gene',columns='model',values='pearson');order=ridge.loc[cohort].sort_values(ascending=False).index;d=d.loc[order]
        fig,ax=plt.subplots(figsize=(7.2,3.5),layout='constrained');im=ax.imshow(d.T,aspect='auto',cmap='viridis',vmin=0,vmax=1,interpolation='nearest');ax.set(yticks=range(len(d.columns)),yticklabels=d.columns,xticks=[],xlabel='280 genes ordered by ridge predictability',title=L[cohort]);fig.colorbar(im,ax=ax,label='Donor-mean Pearson r');save_s(fig,'figS3a_gene_heatmap_'+suffix,'Same model-performance heatmap role; corrected donor weights.')
    fig,axs=plt.subplots(1,2,figsize=(7.2,2.5),layout='constrained')
    for ax,cohort in zip(axs,IDC):
        d=q[q.cohort.eq(cohort)].groupby(['gene','regressor']).pearson.mean().unstack('regressor');scatterfit(ax,d.ridge,d.mlp,C[cohort],L[cohort]);ax.plot([0,1],[0,1],c='.5',ls='--',lw=.7);ax.set(xlabel='Ridge Pearson r',ylabel='MLP Pearson r')
    save_s(fig,'figS3c_ridge_vs_mlp','Correct section/donor correlation definition.')
    d=pd.read_parquet(track(A/'stage_13_figure_package/tables/score_definition.parquet'));fig,axs=plt.subplots(2,2,figsize=(7.2,4.2),layout='constrained')
    for i,cohort in enumerate(IDC):
        z=d[d.cohort.eq(cohort)].iloc[::20]
        for ax,key in zip(axs[i],['raw','conditional']):
            ax.scatter(z.total_expr,z[key],s=.8,c=C[cohort],alpha=.2,rasterized=True);ax.set(xlabel='Sum of modeled log1p counts',ylabel='Mean absolute error' if key=='raw' else 'Centered error',title=L[cohort]+' / '+key)
    save_s(fig,'figS4_conditional_discordance','Illustrate actual pooled expression-bin centering; no independence claim.')

def agreement():
    gate=json.loads(track(R/'outputs/phase2/gate2_1_agreement.json').read_text());rows=[]
    for key,family in zip(['discovery','validation'],IDC):
        for sample,r in gate['cohorts'][key]['samples'].items():rows.append(dict(family=family,sample=sample,**r))
    g=pd.DataFrame(rows);savedata(g,'model_agreement')
    # Compute all-nine model score correlations directly from fixed predictions and original centering.
    sys.path.insert(0,str(R));from src.discordance import compute_conditional_discordance
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.2),layout='constrained')
    for ax,family,sample in zip(axs,IDC,['TENX193','NCBI785']):
        ys=[];ids=[]
        for k in range(4):
            p=R/'outputs/predictions'/family/'uni/ridge'/f'fold{k}';ys.append(np.load(track(p/'test_targets.npy')));ids+=json.loads(track(p/'test_spot_ids.json').read_text())
        total=np.concatenate(ys).sum(axis=1);mask=np.array([s.startswith(sample+'_') for s in ids]);columns={}
        for enc in ['uni','virchow2','hoptimus0']:
            for reg in ['ridge','mlp','xgboost']:
                err=np.concatenate([abs(np.load(track(R/'outputs/predictions'/family/enc/reg/f'fold{k}'/'test_residuals.npy'))).mean(axis=1) for k in range(4)])
                columns[enc+' / '+reg]=compute_conditional_discordance(err,total)[mask]
        frame=pd.DataFrame(columns).corr(method='spearman');im=heat(ax,frame,limit=1);ax.set_title(sample);ax.tick_params(labelsize=4)
    fig.colorbar(im,ax=axs,shrink=.5,label='Spearman ρ');save_s(fig,'figS5a_pairwise_correlation_matrices','Actual nine-model conditional score correlation matrices for fixed exemplars.')
    fig,ax=plt.subplots(figsize=(5.5,2.7),layout='constrained')
    for family,z in g.groupby('family'):ax.scatter(z.n_spots,z.median_rho,c=C[family],label=L[family],s=18)
    ax.axhline(.4,c='.5',ls='--',lw=.7);ax.set(xlabel='Modeled locations',ylabel='Median score agreement');ax.legend();save_s(fig,'figS5b_spotcount_vs_agreement','Updated scores and analyzed counts.')

def spatial():
    reg=table('specimen_registry');fig,axs=plt.subplots(3,6,figsize=(7.2,5.5),layout='constrained')
    for ax,(_,r) in zip(axs.ravel(),reg.iterrows()):
        ss=locations().query('sample==@r["sample"]') if False else locations()[locations()['sample'].eq(r['sample'])]
        quartile(ax,ss,r['sample']+' / '+r.patient)
    save_s(fig,'figS6a_spatial_maps_all','Preserve all-section atlas; original Q1/Q4 and corrected donor identities.')
    sp=audit('stage_04_reliability/spatial_structure.csv').query('graph=="original_k6" and score=="conditional"');sp=sp.merge(reg[['sample','scored_spots']],on='sample')
    fig,ax=plt.subplots(figsize=(5.5,2.7),layout='constrained')
    for family,z in sp.groupby('cohort'):ax.scatter(z.scored_spots,z.morans_i,c=C[family],s=18,label=L[family])
    ax.set(xlabel='Modeled locations',ylabel="Moran's I");ax.legend();save_s(fig,'figS6b_morans_vs_samplesize','Revised spatial scores; same sample-size diagnostic.')
    halves=audit('stage_04_reliability/gene_half_stability.csv');z=halves.query('source=="ridge_encoder_mean" and mode=="conditional_own_half"');fig,ax=plt.subplots(figsize=(7.2,2.5),layout='constrained')
    for i,s in enumerate(reg['sample']):
        a=z[z['sample'].eq(s)].spearman.to_numpy();color=C[reg.set_index('sample').loc[s,'cohort']];ax.plot([i,i],np.quantile(a,[.025,.975]),c=color,lw=1);ax.scatter(i,np.median(a),c=color,s=16)
    ax.set(xticks=range(len(reg)),xticklabels=reg['sample'],ylabel='Conditional half-gene Spearman ρ');ax.tick_params(axis='x',rotation=90);save_s(fig,'figS7a_dual_track_all_samples','Twenty fixed disjoint partitions, median and 2.5–97.5 percentiles.')
    fig,ax=plt.subplots(figsize=(5.5,2.5),layout='constrained')
    for cohort in IDC:
        d=z[z.cohort.eq(cohort)].groupby('seed').spearman.median();ax.plot(d.index,d.values,'o-',ms=3,c=C[cohort],label=L[cohort])
    ax.set(xlabel='Fixed partition seed (140 genes per half)',ylabel='Median section Spearman ρ');ax.legend();save_s(fig,'figS7b_subsampling_curve','Replace mislabeled raw-score size curve with actual conditional-score partition sensitivity.')

def genes():
    d=table('gene_cohort_effects');d=d[d.outcome.eq('observed_log')&d.metric.eq('standardized_effect')]
    fig,axs=plt.subplots(1,2,figsize=(7.2,2.6),layout='constrained')
    for ax,cohort in zip(axs,IDC):
        a=d[d.cohort.eq(cohort)&d.grouping.eq('full')&d.adjustment.eq('unadjusted')].set_index('gene')['mean'];b=d[d.cohort.eq(cohort)&d.grouping.eq('gene_excluded')&d.adjustment.eq('overlap_adjusted')].set_index('gene')['mean'];scatterfit(ax,a,b,C[cohort],L[cohort]);ax.set(xlabel='Full-score unadjusted contrast',ylabel='Gene-excluded adjusted contrast');zero(ax);hzero(ax)
    save_s(fig,'figS13_matched_vs_unmatched','Compare full-score and independent-characterization sensitivity in the existing contrast slot.')
    s=table('gene_section_effects');s=s[s.grouping.eq('gene_excluded')&s.adjustment.eq('overlap_adjusted')&s.outcome.eq('observed_log')];names=['EPCAM','KRT19','FOXA1','ESR1','KRT17','TP63','CD163','FBLN1','PDGFRB','MKI67','SNAI1','ZEB1','ZEB2'];fig,axs=plt.subplots(1,2,figsize=(7.2,3.2),layout='constrained')
    for ax,cohort in zip(axs,IDC):
        z=s[s.cohort.eq(cohort)].pivot(index='gene',columns='sample',values='standardized_effect').reindex(names);im=heat(ax,z,limit=1.5);ax.set_title(L[cohort]);ax.tick_params(labelsize=5)
    fig.colorbar(im,ax=axs,shrink=.5,label='Adjusted Q4 − Q1 / SD');save_s(fig,'figS8a_per_sample_volcanos','Section-level effects replace invalid spot-significance volcanoes.')
    m=audit('stage_06_inference/matching_diagnostics.csv');m=m[m.k.eq(5)];fig,axs=plt.subplots(1,2,figsize=(7.2,2.6),layout='constrained')
    for cohort,z in m.groupby('cohort'):
        axs[0].scatter(z.retained_Q4_fraction,z.max_control_reuse,c=C[cohort],label=L[cohort],s=18);axs[1].scatter(z.embedding_mean_difference_norm_before,z.embedding_mean_difference_norm_after,c=C[cohort],s=18)
    axs[0].set(xlabel='Fraction of Q4 retained',ylabel='Maximum reuse of one Q1 control');axs[0].legend();axs[1].set(xlabel='Embedding mean difference before',ylabel='Embedding mean difference after');save_s(fig,'figS8b_matching_quality','Explicit morphology-overlap and reuse limitations.')

def programs():
    p=primary_programs(table('program_cohort_effects'));p=p[p.family.isin(IDC)&p.metric.eq('standardized_effect')];fig,axs=plt.subplots(1,3,figsize=(7.2,5),layout='constrained')
    for ax,outcome in zip(axs,['observed','signed','absolute']):
        z=p[p.outcome.eq(outcome)].pivot(index='pathway',columns='family',values='mean').reindex(columns=IDC);z.columns=['Discovery','Validation'];im=heat(ax,z,limit=2);ax.set_title(outcome.capitalize());ax.tick_params(axis='y',labelsize=4)
    save_s(fig,'figS9a_full_pathway_heatmap','Complete program means, with outcome types separated.')
    cov=audit('stage_05_biology/coverage.csv');cov=cov[cov.eligible];fig,ax=plt.subplots(figsize=(7.2,4.5),layout='constrained');d=cov.pivot(index='pathway',columns='cohort',values='n_genes');d.plot.barh(ax=ax,color=[C[x] for x in d.columns],width=.8);ax.set(xlabel='Measured Hallmark members',ylabel='');ax.set_yticklabels([short(x) for x in d.index],fontsize=5);save_s(fig,'figS9b_pathway_overlap','Measured-member coverage, including missing programs.')
    for family,name in zip(IDC,['figS17a_pathway_heatmap_discovery','figS17b_pathway_heatmap_validation']):
        z=primary_programs(table('program_section_effects'));z=z[z.family.eq(family)&z.outcome.eq('observed')];frame=z.pivot(index='pathway',columns='sample',values='standardized_effect');fig,ax=plt.subplots(figsize=(7.2,4.5),layout='constrained');im=heat(ax,frame,limit=2);ax.tick_params(axis='y',labelsize=5);fig.colorbar(im,ax=ax,shrink=.5,label='Observed-expression contrast / SD');save_s(fig,name,'Section profiles; discovery columns are separate donors.')

def cells():
    d=pd.read_csv(track(OUT/'source_data/marker_score_section_effects.csv'));fig,axs=plt.subplots(1,2,figsize=(7.2,2.9),layout='constrained')
    for ax,cohort in zip(axs,IDC):
        z=d[d.cohort.eq(cohort)].pivot(index='cell_type',columns='sample',values='standardized_effect');im=heat(ax,z,limit=1.5);ax.set_title(L[cohort]);ax.tick_params(labelsize=5)
    fig.colorbar(im,ax=axs,shrink=.6,label='Marker-score contrast / SD');save_s(fig,'figS10_deconvolution_detail','Supportive marker scores; no cell proportions inferred.')
    mixtures=audit('stage_08_cell_composition/cell_mixture.csv');mixtures=mixtures[mixtures.width_um.eq(1600)&mixtures.kind.eq('group')]
    if not len(mixtures):
        allm=audit('stage_08_cell_composition/cell_mixture.csv');mixtures=allm[allm.width_um.eq(1600)&allm.label.isin(['Tumor','Stromal','Macrophage'])]
    fig,ax=plt.subplots(figsize=(6,2.8),layout='constrained');names=['Tumor','Stromal','Macrophage'];samples=['NCBI785','NCBI784','NCBI783'];cols=['#347da1','#78aabe','#bc6280']
    for k,(sample,col) in enumerate(zip(samples,cols)):
        z=mixtures[mixtures['sample'].eq(sample)].set_index('label')
        for j,label in enumerate(names):
            r=z.loc[label];y=j+(k-1)*.2;ax.plot([r.ci_low*100,r.ci_high*100],[y,y],c=col);ax.scatter(r.difference*100,y,c=col,s=17,label=sample if j==0 else None)
    zero(ax);ax.set(yticks=range(3),yticklabels=names,xlabel='Q4 − Q1 source-cell fraction (percentage points)');ax.invert_yaxis();ax.legend();save_s(fig,'figS14b_celltype_per_sample','Measured source-cell fractions in supported subsets, spatial intervals.')
    w=audit('stage_08_cell_composition/whole_spot_effects.csv');w=w[w.width_um.eq(1600)&w.endpoint.eq('CD163')&w.outcome.eq('observed')];fig,ax=plt.subplots(figsize=(6,2.6),layout='constrained')
    for k,(sample,col) in enumerate(zip(samples,cols)):
        z=w[w['sample'].eq(sample)]
        for j,adj in enumerate(['unadjusted','technical','composition']):
            a=z[z.adjustment.eq(adj)]
            if len(a):r=a.iloc[0];ax.plot([r.ci_low,r.ci_high],[j+(k-1)*.2]*2,c=col);ax.scatter(r.estimate,j+(k-1)*.2,c=col,s=17,label=sample if j==0 else None)
    zero(ax);ax.set(yticks=range(3),yticklabels=['Unadjusted','Count/detection','Plus composition'],xlabel='CD163 observed-expression Q4 − Q1 (log1p units)');ax.legend();save_s(fig,'figS14a_macrophage_per_sample','Separate CD163 marker expression from measured cell mixture.')
    e=audit('stage_14_remaining_analyses/tumor_rich_emt_effects.csv');e=e[e.minimum_tumor_fraction.eq(.5)&e.width_um.eq(1600)&e.adjustment.eq('composition')];fig,axs=plt.subplots(1,3,figsize=(7.2,2.2),layout='constrained')
    for ax,outcome in zip(axs,['observed','signed','absolute']):
        for j,sample in enumerate(samples[:2]):
            r=e[e['sample'].eq(sample)&e.outcome.eq(outcome)].iloc[0];ax.plot([r.ci_low,r.ci_high],[j,j],c=cols[j]);ax.scatter(r.estimate,j,c=cols[j],s=20)
        zero(ax);ax.set(yticks=[0,1],yticklabels=['P07 / 785','P07 / 784'],xlabel='Q4 − Q1 (log1p units)',title=outcome.capitalize());ax.invert_yaxis()
    save_s(fig,'figS14c_tumor_rich_emt','New reviewer-requested tumor-rich EMT residual comparison; P08 insufficient support.')

def remaining():
    f=table('gene_features');fig,axs=plt.subplots(1,2,figsize=(7.2,2.7),layout='constrained')
    for ax,cohort in zip(axs,IDC):
        z=f[f.cohort.eq(cohort)];groups=sorted(z.primary_localization.unique());v=[z[z.primary_localization.eq(g)].mean_pearson for g in groups];ax.boxplot(v,showfliers=False,patch_artist=True,boxprops={'facecolor':C[cohort],'alpha':.35});ax.set(xticks=range(1,len(groups)+1),xticklabels=groups,ylabel='Donor-mean prediction r',title=L[cohort]);ax.tick_params(axis='x',rotation=65,labelsize=5)
    save_s(fig,'figS11b_localization_predictability','Cached categories shown descriptively; remove unsupported significance stars.')
    d=pd.read_csv(track(OUT/'source_data/section_gene_absolute_residuals.csv'));fig,axs=plt.subplots(1,3,figsize=(7.2,2.3),layout='constrained')
    for ax,patient,a,b in zip(axs,['P01','P02','P07'],['TENX99','TENX97','NCBI785'],['TENX98','TENX95','NCBI784']):scatterfit(ax,d[a],d[b],C['10x_janesick'],patient);ax.set(xlabel=a+' gene MAE',ylabel=b+' gene MAE')
    save_s(fig,'figS12a_within_patient_gene_scatter','Only three documented validation repeat-section pairs.')
    p=primary_programs(table('program_unit_effects'));p=p[p.family.eq('coad')&p.outcome.eq('observed')];fig,ax=plt.subplots(figsize=(6.2,4.3),layout='constrained');im=heat(ax,p.pivot(index='pathway',columns='unit',values='standardized_effect'),limit=2);fig.colorbar(im,ax=ax,shrink=.6,label='Observed-expression contrast / SD');save_s(fig,'figS18b_coad_pathway_heatmap','All COAD specimens and programs, effect sizes.')
    f=oldtable('external_gene_features');g=table('gene_features');fig,ax=plt.subplots(figsize=(5.5,2.4),layout='constrained');vals=[g[g.cohort.eq(x)].mean_pearson for x in IDC]+[f[f.family.eq('coad')].pearson];ax.boxplot(vals,showfliers=False);ax.set(xticks=[1,2,3],xticklabels=['Discovery','Validation','COAD'],ylabel='Gene prediction Pearson r');save_s(fig,'figS18c_coad_predictability_comparison','Audited gene-predictability distributions.')
    b=table('boundary_cohort_effects');b=b[b.kind.eq('program')&b.grouping.eq('excluded')&b.adjustment.eq('overlap_adjusted')&b.metric.eq('fixed_standardized_effect')];fig,axs=plt.subplots(1,2,figsize=(7.2,2.9),layout='constrained')
    for ax,outcome in zip(axs,['observed','absolute']):
        for family in IDC:
            z=b[b.family.eq(family)&b.outcome.eq(outcome)].pivot(index='endpoint',columns='band_um',values='mean');ax.scatter(z[0],z[400],c=C[family],s=16,label=L[family]);lo=min(z[0].min(),z[400].min());hi=max(z[0].max(),z[400].max());ax.plot([lo,hi],[lo,hi],c='.7',ls='--',lw=.7)
        ax.set(xlabel='All locations: Q4 − Q1 / reference SD',ylabel='After 400 µm exclusion',title=outcome.capitalize());ax.legend()
    save_s(fig,'figS22_interior_only_de','Explicit physical-boundary sensitivity, not the earlier embedding-defined interior estimand.')
    # Preserve the original PCA and change only the unsupported patient grouping labels.
    p=track(OLD/'supplementary_figures/figS2_batch_effect_pca.pdf');doc=fitz.open(p);page=doc[0]
    substitutions={'P03':'B01','P04':'B02','P05':'B03','P06':'B04','Pseudo-bulk PCA by patient':'Pseudo-bulk PCA by holdout group'};edits=[]
    for old,new in substitutions.items():
        for rect in page.search_for(old):edits.append((rect,new));page.add_redact_annot(rect,fill=(1,1,1))
    page.apply_redactions(images=0,graphics=0)
    for rect,new in edits:
        page.insert_text((rect.x0,rect.y1-1),new,fontsize=5.4 if 'PCA' in new else 6,fontname='helv')
    doc.save(SRC/'supplementary_figures/figS2_batch_effect_pca.pdf');doc.close();ASSETS.append(dict(path='supplementary_figures/figS2_batch_effect_pca.pdf',action='local labels',reason='Existing PCA; historical groups are not discovery patients.'))

def visium():
    f=oldtable('external_gene_features');f=f[f.family.eq('idc_visium')];fig,ax=plt.subplots(figsize=(4.6,3),layout='constrained');scatterfit(ax,f.morans_i,f.pearson,C['idc_visium']);ax.set(xlabel="Gene Moran's I",ylabel='Known-group mean prediction r');save(fig,'visium_figures/fig_visium_morans_scatter.pdf','Grouped Block A fits and all-section gene features.')
    p=primary_programs(table('program_unit_effects'));p=p[p.family.eq('idc_visium')&p.outcome.eq('observed')];fig,ax=plt.subplots(figsize=(7.2,6.5),layout='constrained');im=heat(ax,p.pivot(index='pathway',columns='unit',values='standardized_effect'),limit=2);ax.tick_params(axis='y',labelsize=5);fig.colorbar(im,ax=ax,shrink=.5,label='Adjusted expression Q4 − Q1 / SD');save(fig,'visium_figures/fig_visium_pathway_heatmap.pdf','All 50 programs and nine known groups; labeled outcome.')
    g=table('gene_features');fig,ax=plt.subplots(figsize=(4.6,2.8),layout='constrained');ax.boxplot([g[g.cohort.eq(x)].mean_pearson for x in IDC]+[f.pearson],showfliers=False);ax.set(xticks=[1,2,3],xticklabels=['Discovery','Validation','Visium'],ylabel='Mean gene prediction r');save(fig,'visium_figures/fig_visium_predictability_comparison.pdf','Updated predictability distributions, different panels acknowledged.')
    q=table('external_gene_pairs');q=q[q.family.eq('idc_visium')&q.scope.eq('excluding_P07')];q=q.groupby('gene')[['idc_pearson','external_pearson']].mean();fig,ax=plt.subplots(figsize=(4.6,3),layout='constrained');scatterfit(ax,q.idc_pearson,q.external_pearson,C['idc_visium']);ax.set(xlabel='Breast Xenium mean prediction r',ylabel='Visium mean prediction r',title=f'{len(q)} union-panel genes; paired P07 excluded');savedata(q.reset_index(),'Visium_union_gene_comparison');save(fig,'visium_figures/fig_visium_cross_technology_scatter.pdf','Complete shared genes; independent comparison excludes paired P07 specimen.')

if __name__=='__main__':
    functions=[qc_label,prediction,agreement,spatial,genes,programs,cells,remaining,visium]
    if len(sys.argv)>1:
        SOURCES.update(json.loads((BUILD/'supplement_sources.json').read_text()))
        ASSETS.extend(json.loads((BUILD/'supplement_assets.json').read_text()))
        functions=[f for f in functions if f.__name__ in sys.argv[1:]]
    for f in functions:f();flush('supplement')

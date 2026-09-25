"""Secondary residual associations and exact source matching for broader data."""
from common import *
import importlib.util
import pandas as pd
import anndata as ad
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from analyze_cells import partial_rho


def disjoint_scores():
    root=PROJECT/'paper_revision/clean_repo/outputs'
    genes=np.array(json.loads((PROJECT/'data/v3/gene_list_10x_janesick.json').read_text()))
    excluded=set(PROTOCOL['epithelial_genes']+PROTOCOL['canonical_emt_tfs']+hallmark_genes())
    keep=np.array([g not in excluded for g in genes])
    assert keep.sum()>0 and not set(genes[keep])&excluded
    spec=importlib.util.spec_from_file_location('original_discordance',PROJECT/'clean_repo/src/discordance.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    per_encoder=[]
    for encoder in ['uni','virchow2','hoptimus0']:
        frames=[]
        for fold in range(4):
            base=root/f'predictions/10x_janesick/{encoder}/ridge/fold{fold}'
            ids=np.array(json.loads((base/'test_spot_ids.json').read_text()))
            y=np.asarray(np.load(base/'test_targets.npy',mmap_mode='r'),float)
            p=np.asarray(np.load(base/'test_predictions.npy',mmap_mode='r'),float)
            residual=y-p
            tf=np.isin(genes,PROTOCOL['canonical_emt_tfs'])
            epi=np.isin(genes,PROTOCOL['epithelial_genes'])
            frames.append(pd.DataFrame({'spot_id':ids,'raw':np.abs(residual[:,keep]).mean(axis=1),
                'remaining_expression':y[:,keep].sum(axis=1),
                'signed_tf_residual':residual[:,tf].mean(axis=1),
                'observed_tf':y[:,tf].mean(axis=1),'predicted_tf':p[:,tf].mean(axis=1),
                'signed_epithelial_residual':residual[:,epi].mean(axis=1)}))
        frame=pd.concat(frames,ignore_index=True).set_index('spot_id')
        frame['disjoint_D_cond']=module.compute_conditional_discordance(frame.raw.to_numpy(),frame.remaining_expression.to_numpy(),10,30)
        per_encoder.append(frame.sort_index())
    for f in per_encoder[1:]:np.testing.assert_array_equal(f.index,per_encoder[0].index)
    result=sum(per_encoder)/3
    result.to_parquet(RESULTS/'all_validation_disjoint_residual_scores.parquet')
    dump(RESULTS/'disjoint_residual_definition.json',{'n_remaining_genes':int(keep.sum()),'remaining_genes':genes[keep].tolist(),
         'excluded_genes':genes[~keep].tolist(),'conditioning':'10 pooled validation-cohort expression bins per encoder; total expression uses remaining genes; average three encoder scores',
         'independence_limit':'Gene-disjoint scoring reduces direct arithmetic overlap but is not independent biological validation'})
    return result


def secondary_associations():
    disjoint=disjoint_scores();rows=[];assignments=[]
    for sample,patient in PROTOCOL['samples'].items():
        cells=pd.read_parquet(RESULTS/sample/'cell_evidence.parquet')
        spots=pd.read_parquet(PROJECT/f'paper_revision/clean_repo/outputs/phase2/scores/10x_janesick/{sample}_discordance.parquet').set_index('spot_id')
        coords=spots[['x','y']].to_numpy(float);tree=cKDTree(coords)
        step=np.median(tree.query(coords,k=2,workers=4)[0][:,1]);radius=step/2
        eligible=cells.qc_pass.to_numpy()&np.isfinite(cells[['he_x','he_y']]).all(axis=1).to_numpy()
        usecells=cells.loc[eligible];distance,index=tree.query(usecells[['he_x','he_y']].to_numpy(float),workers=4)
        assigned=distance<=radius
        usecells=usecells.iloc[np.flatnonzero(assigned)].copy();index=index[assigned]
        def count(mask):return np.bincount(index[mask],minlength=len(spots))
        tumor=usecells.source_group.to_numpy()=='Tumor'
        candidate=usecells.tumor_tf_candidate.to_numpy(bool)
        spots['n_cells']=count(np.ones(len(usecells),bool))
        spots['n_tumor_cells']=count(tumor);spots['n_tf_candidates']=count(candidate)
        spots['n_nuclear_tf_candidates']=count(usecells.tumor_nuclear_tf_candidate.to_numpy(bool))
        spots['tf_fraction']=np.divide(spots.n_tf_candidates,spots.n_tumor_cells,out=np.full(len(spots),np.nan),where=spots.n_tumor_cells>0)
        for group in ['Stromal','Myoepithelial','Tumor']:
            totals=count(usecells.source_group.to_numpy()==group)
            spots[group.lower()+'_fraction']=np.divide(totals,spots.n_cells,out=np.full(len(spots),np.nan),where=spots.n_cells>0)
        sums=np.bincount(index[tumor],weights=usecells.total_biological_transcripts.to_numpy()[tumor],minlength=len(spots))
        spots['mean_tumor_transcripts']=np.divide(sums,spots.n_tumor_cells,out=np.full(len(spots),np.nan),where=spots.n_tumor_cells>0)
        spots['D_cond']=spots[[f'D_cond_{e}_ridge' for e in ['uni','virchow2','hoptimus0']]].mean(axis=1)
        spots=spots.join(disjoint[['disjoint_D_cond','signed_tf_residual','observed_tf','predicted_tf','signed_epithelial_residual']])
        # Cells close to the disk edge are less certain in centroid assignment.
        # This statistic is retained, not used to select a favorable subset.
        spots['n_centroids_near_assignment_edge']=count(distance[assigned]>.9*radius)
        for minimum in [5,20]:
            valid=(spots.n_tumor_cells>=minimum)&np.isfinite(spots.tf_fraction)
            frame=spots.loc[valid]
            controls=np.column_stack([frame.stromal_fraction,frame.myoepithelial_fraction,frame.tumor_fraction,
                                       np.log1p(frame.mean_tumor_transcripts),np.log1p(frame.n_tumor_cells),np.log1p(frame.total_expr)])
            for name in ['D_cond','disjoint_D_cond','signed_tf_residual','observed_tf','predicted_tf','signed_epithelial_residual']:
                value=float(spearmanr(frame.tf_fraction,frame[name]).statistic) if len(frame)>=30 and np.ptp(frame.tf_fraction)>0 and np.ptp(frame[name])>0 else np.nan
                rows.append(dict(sample=sample,patient=patient,minimum_tumor_cells=minimum,n_spots=len(frame),score=name,
                                 spearman=value,partial_spearman=partial_rho(frame.tf_fraction,frame[name],controls),
                                 interpretation='Descriptive within-section TF coexpression association, not independent EMT validation'))
        spots.to_parquet(RESULTS/sample/'spot_residual_evidence.parquet')
        assignments.append(dict(sample=sample,patient=patient,n_qc_cells=int(eligible.sum()),n_assigned=len(usecells),
                                n_assigned_tumor=int(tumor.sum()),n_assigned_candidates=int(candidate.sum()),
                                assignment_radius_he_pixels=float(radius),coordinates='Measured transcript centroids, not fitted vendor-cell-center transform'))
    pd.DataFrame(rows).to_csv(RESULTS/'residual_associations.csv',index=False)
    dump(RESULTS/'spot_assignment.json',assignments)


def visium_compatibility():
    source,source_genes,source_ids=read_10x(SOURCES/'GSM7782699_filtered_feature_bc_matrix.h5')
    local=ad.read_h5ad(PROJECT/'data/hest/st/NCBI776.h5ad')
    with h5py.File(SOURCES/'GSM7782699_filtered_feature_bc_matrix.h5') as handle:
        source_gene_ids=np.array([x.decode() for x in handle['matrix/features/id'][:]])
    gene_index=pd.Index(source_gene_ids).get_indexer(local.var['gene_ids'].astype(str))
    cell_index=pd.Index(source_ids).get_indexer(local.obs_names)
    assert (gene_index>=0).all() and (cell_index>=0).all()
    difference=sparse.csr_matrix(local.X)-source[cell_index][:,gene_index]
    difference.eliminate_zeros()
    assert difference.nnz==0,'Local Visium/source mismatch'
    normalized=normalize(local.X);genes=np.array(local.var_names)
    frame=local.obs.copy()
    frame['observed_hallmark_emt']=gene_score(normalized,genes,hallmark_genes())
    frame['observed_epithelial']=gene_score(normalized,genes,PROTOCOL['epithelial_genes'])
    frame['observed_tf_program']=gene_score(normalized,genes,PROTOCOL['canonical_emt_tfs'])
    markers=[]
    for gene in sorted(set(PROTOCOL['epithelial_genes']+PROTOCOL['canonical_emt_tfs']+['VIM','FN1','CDH2'])):
        if gene in genes:
            values=local.X[:,int(np.flatnonzero(genes==gene)[0])].toarray().ravel()
            markers.append(dict(gene=gene,measured=True,n_spots_detected=int((values>0).sum()),total_transcripts=int(values.sum())))
            frame[gene+'_counts']=values
        else:markers.append(dict(gene=gene,measured=False,n_spots_detected=0,total_transcripts=0))
    frame.to_parquet(RESULTS/'NCBI776_broader_expression_evidence.parquet')
    pd.DataFrame(markers).to_csv(RESULTS/'NCBI776_marker_coverage.csv',index=False)
    # Source study sample correspondence is a linked block, not independent patient validation.
    dump(RESULTS/'VISIUM_COMPATIBILITY.json',dict(local_sample='NCBI776',source='GSM7782699',
        local_spots=local.n_obs,local_genes=local.n_vars,exact_source_count_match=True,
        available_hallmark_genes=sum(g in genes for g in hallmark_genes()),
        source_study='GSE243280, Sample 1; paired/serial multi-assay evidence, not an additional independent patient',
        interpretation='Broader measured-expression compatibility established; no tumor-specific EMT-zone call from mixed Visium spots'))
    print('VISIUM SOURCE MATCH VERIFIED',flush=True)


if __name__=='__main__':
    secondary_associations()
    visium_compatibility()
    print('RESIDUAL AND VISIUM CHECKS COMPLETE',flush=True)

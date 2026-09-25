"""Measured cell programs and fixed-rule descriptive candidate neighborhoods."""
from common import *
from datetime import datetime,timezone
import shutil
import anndata as ad
import pandas as pd
import joblib
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr,rankdata


def partial_rho(a,b,controls):
    valid=np.isfinite(a)&np.isfinite(b)&np.isfinite(controls).all(axis=1)
    a,b,controls=np.asarray(a)[valid],np.asarray(b)[valid],controls[valid]
    if len(a)<30 or np.ptp(a)==0 or np.ptp(b)==0:return np.nan
    x=np.column_stack([np.ones(len(a)),*[rankdata(controls[:,j]) for j in range(controls.shape[1])]])
    ra=rankdata(a);rb=rankdata(b)
    ra=ra-x@np.linalg.lstsq(x,ra,rcond=None)[0];rb=rb-x@np.linalg.lstsq(x,rb,rcond=None)[0]
    return float(np.corrcoef(ra,rb)[0,1])


def components(coords,eligible,radius):
    index=np.flatnonzero(eligible);labels=np.full(len(coords),-1,int)
    if not len(index):return labels
    pairs=cKDTree(coords[index]).query_pairs(radius,output_type='ndarray')
    graph=sparse.coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(len(index),len(index))).tocsr()
    _,ids=connected_components(graph,directed=False)
    labels[index]=ids
    return labels


def main():
    frozen=HERE/'cell_analysis_frozen';frozen.mkdir(exist_ok=True)
    for name in ['protocol.json','analyze_cells.py','common.py','prepare_cells.py','IMPLEMENTATION_LOG.md']:
        shutil.copy2(HERE/name,frozen/name)
    dump(frozen/'manifest.json',{'utc':datetime.now(timezone.utc).isoformat(),'sha256':{p.name:sha(p) for p in frozen.iterdir() if p.is_file() and p.name!='manifest.json'}})
    gates={x['panel']:x for x in json.loads((RESULTS/'REFERENCE_OBSERVABILITY.json').read_text())}
    ref=pd.read_parquet(RESULTS/'reference_cell_programs.parquet')
    ref_meta=ad.read_h5ad(RESULTS/'wu_reference.h5ad',backed='r').obs
    reference_partial=[]
    for panel in ['panel313','panel280']:
        for donor,frame in ref.loc[(ref.celltype_major=='Cancer Epithelial')&ref[panel+'_heldout_prediction'].notna()].groupby('orig.ident',observed=True):
            controls=ref_meta.loc[frame.index,['nCount_RNA','nFeature_RNA']].to_numpy(float)
            reference_partial.append(dict(panel=panel,donor=donor,n_cells=len(frame),
                partial_rho_adjusting_depth_and_detection=partial_rho(frame[panel+'_heldout_prediction'],frame[panel+'_disjoint_target'],controls)))
    pd.DataFrame(reference_partial).to_csv(RESULTS/'reference_depth_sensitivity.csv',index=False)
    cohort_summary=[];label_tables=[];zone_rows=[];sensitivity=[]
    for sample,patient in PROTOCOL['samples'].items():
        out=RESULTS/sample;data=ad.read_h5ad(out/'measured_cells.h5ad')
        genes=np.array(data.var_names);obs=data.obs.copy();coords=obs[['x_um','y_um']].to_numpy(float)
        normalized=normalize(data.X)
        flags,ntf,nepi=coexpression_flags(data.X,genes)
        nuclear_flags,nuclear_tfs,nuclear_epi=coexpression_flags(data.layers['nuclear_counts'],genes)
        tumor=obs.qc_pass.to_numpy()&(obs.source_group.to_numpy()=='Tumor')
        candidate=tumor&flags
        obs['tf_epithelial_coexpression']=flags
        obs['measured_tf_genes_detected']=ntf;obs['measured_epithelial_genes_detected']=nepi
        obs['nuclear_coexpression']=nuclear_flags
        obs['nuclear_tf_genes_detected']=nuclear_tfs
        obs['tumor_tf_candidate']=candidate
        obs['tumor_nuclear_tf_candidate']=tumor&nuclear_flags
        obs['epithelial_program']=gene_score(normalized,genes,PROTOCOL['epithelial_genes'])
        obs['measured_hallmark_emt']=gene_score(normalized,genes,hallmark_genes())
        obs['canonical_tf_program']=gene_score(normalized,genes,PROTOCOL['canonical_emt_tfs'])
        for gene in ['EPCAM','CDH1','KRT8','SNAI1','ZEB1','ZEB2','MYLK','LUM','ERBB2']:
            if gene in genes:
                j=int(np.flatnonzero(genes==gene)[0])
                obs[gene+'_counts']=data.X[:,j].toarray().ravel()
                obs[gene+'_nuclear_counts']=data.layers['nuclear_counts'][:,j].toarray().ravel()
        panel='panel313' if sample!='NCBI783' else 'panel280'
        model=joblib.load(RESULTS/f'{panel}_reference_model.joblib')
        index=pd.Index(genes).get_indexer(model['genes']);assert (index>=0).all()
        # Match the predictor's restricted-panel normalization exactly.
        reference_input=normalize(data.X[:,index]).toarray()
        standardized=model['scaler'].transform(reference_input)
        obs['reference_projected_program']=model['model'].predict(standardized)
        obs['fraction_features_beyond_4_reference_sd']=(np.abs(standardized)>4).mean(axis=1)
        reference_target=ref.loc[ref.celltype_major=='Cancer Epithelial',panel+'_disjoint_target'].to_numpy()
        low,high=np.quantile(reference_target,[.01,.99])
        obs['projection_outside_reference_target_1_99']=(obs.reference_projected_program<low)|(obs.reference_projected_program>high)
        for radius in [50,100,150]:
            nt,nc,fraction=neighborhood_counts(coords,tumor,candidate,radius)
            obs[f'tumor_neighbors_{radius}um']=nt
            obs[f'tf_candidate_neighbors_{radius}um']=nc
            obs[f'tf_candidate_fraction_{radius}um']=fraction
            qualifying=tumor&(nt>=20)&(nc>=5)&(fraction>=.1)
            labels=components(coords,qualifying,radius)
            obs[f'candidate_component_{radius}um']=labels
            n_components=len(set(labels)-{-1})
            sensitivity.append(dict(sample=sample,patient=patient,radius_um=radius,
                                     n_qualifying_tumor_centers=int(qualifying.sum()),n_components=n_components))
            if radius==100:
                for label in sorted(set(labels)-{-1}):
                    center=labels==label
                    distances=cKDTree(coords[center]).query(coords,workers=4)[0]
                    surrounding=distances<=radius
                    supported=surrounding&candidate
                    q=surrounding&obs.qc_pass.to_numpy()
                    zone_rows.append(dict(sample=sample,patient=patient,component=label,
                        n_qualifying_centers=int(center.sum()),n_tumor_in_union=int((surrounding&tumor).sum()),
                        n_tf_candidates_in_union=int(supported.sum()),
                        n_nuclear_supported_candidates=int((supported&nuclear_flags).sum()),
                        center_x_um=float(coords[center,0].mean()),center_y_um=float(coords[center,1].mean()),
                        mean_candidate_fraction_at_centers=float(np.mean(fraction[center])),
                        stromal_fraction_in_union=float(np.mean(obs.source_group.to_numpy()[q]=='Stromal')),
                        myoepithelial_fraction_in_union=float(np.mean(obs.source_group.to_numpy()[q]=='Myoepithelial')),
                        median_candidate_transcripts=float(np.median(obs.total_biological_transcripts.to_numpy()[supported]))))
        for group in ['Stromal','Myoepithelial']:
            use=obs.qc_pass.to_numpy()&(obs.source_group.to_numpy()==group)
            obs[f'{group.lower()}_neighbors_100um']=cKDTree(coords[use]).query_ball_point(coords,100,return_length=True,workers=4) if use.any() else 0
        for label,frame in obs.loc[obs.qc_pass].groupby('source_label',observed=True):
            label_tables.append(dict(sample=sample,patient=patient,label=label,source_group=str(frame.source_group.iloc[0]),
                n_cells=len(frame),n_tf_epithelial_coexpression=int(frame.tf_epithelial_coexpression.sum()),
                fraction_tf_epithelial_coexpression=float(frame.tf_epithelial_coexpression.mean()),
                n_nuclear_coexpression=int(frame.nuclear_coexpression.sum()),
                median_transcripts=float(frame.total_biological_transcripts.median()),
                mean_measured_hallmark=float(frame.measured_hallmark_emt.mean()),
                mean_reference_projection=float(frame.reference_projected_program.mean())))
        summary=dict(sample=sample,patient=patient,n_cells=len(obs),n_qc=int(obs.qc_pass.sum()),
            n_source_tumor_qc=int(tumor.sum()),n_tf_candidates=int(candidate.sum()),
            fraction_tumor_tf_candidates=float(candidate.sum()/tumor.sum()),
            n_candidates_nuclear_supported=int((candidate&nuclear_flags).sum()),
            n_100um_components=len(set(obs.candidate_component_100um)-{-1}),
            n_qualifying_tumor_centers=int((obs.candidate_component_100um>=0).sum()),
            reference_operational_gate_passed=gates[panel]['passed'],
            fraction_tumor_projection_outside_reference_range=float(obs.loc[tumor,'projection_outside_reference_target_1_99'].mean()),
            canonical_tf_genes=[str(g) for g in genes if g in PROTOCOL['canonical_emt_tfs']],
            epithelial_genes=[str(g) for g in genes if g in PROTOCOL['epithelial_genes']],
            conclusion_label='Measured TF/epithelial coexpression candidates; no validated EMT-zone call')
        obs.to_parquet(out/'cell_evidence.parquet')
        dump(out/'CELL_ANALYSIS_COMPLETE.json',summary);cohort_summary.append(summary)
        print('CELL ANALYSIS',summary,flush=True)
    pd.DataFrame(label_tables).to_csv(RESULTS/'source_label_evidence.csv',index=False)
    pd.DataFrame(zone_rows).to_csv(RESULTS/'candidate_neighborhood_components.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(RESULTS/'spatial_scale_sensitivity.csv',index=False)
    dump(RESULTS/'CELL_EVIDENCE_SUMMARY.json',cohort_summary)
    print('CELL ANALYSIS COMPLETE',flush=True)


if __name__=='__main__':main()

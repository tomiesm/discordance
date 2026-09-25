"""Measured coexpression and fixed spatial neighborhoods used in the paper."""
from .common import *
import anndata as ad
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components

def components(coords,eligible,radius):
    index=np.flatnonzero(eligible);labels=np.full(len(coords),-1,int)
    if not len(index):return labels
    pairs=cKDTree(coords[index]).query_pairs(radius,output_type='ndarray')
    graph=sparse.coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(len(index),len(index))).tocsr()
    _,ids=connected_components(graph,directed=False)
    labels[index]=ids
    return labels

def main():
    labels=[]
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
        for radius in [50,100,150]:
            nt,nc,fraction=neighborhood_counts(coords,tumor,candidate,radius)
            obs[f'tumor_neighbors_{radius}um']=nt
            obs[f'tf_candidate_neighbors_{radius}um']=nc
            obs[f'tf_candidate_fraction_{radius}um']=fraction
            qualifying=tumor&(nt>=20)&(nc>=5)&(fraction>=.1)
            labels=components(coords,qualifying,radius)
            obs[f'candidate_component_{radius}um']=labels
        for group in ['Stromal','Myoepithelial']:
            use=obs.qc_pass.to_numpy()&(obs.source_group.to_numpy()==group)
            obs[f'{group.lower()}_neighbors_100um']=cKDTree(coords[use]).query_ball_point(coords,100,return_length=True,workers=4) if use.any() else 0
        for label,frame in obs.loc[obs.qc_pass].groupby('source_label',observed=True):
            labels.append(dict(sample=sample,patient=patient,label=label,source_group=str(frame.source_group.iloc[0]),n_cells=len(frame),n_tf_epithelial_coexpression=int(frame.tf_epithelial_coexpression.sum()),fraction_tf_epithelial_coexpression=float(frame.tf_epithelial_coexpression.mean()),n_nuclear_coexpression=int(frame.nuclear_coexpression.sum()),median_transcripts=float(frame.total_biological_transcripts.median())))
        obs.to_parquet(out/'cell_evidence.parquet')
        print(sample, int(candidate.sum()), 'coexpressing tumor cells', flush=True)

    import pandas as pd
    pd.DataFrame(labels).to_csv(RESULTS/'source_label_evidence.csv',index=False)

if __name__ == '__main__':
    main()

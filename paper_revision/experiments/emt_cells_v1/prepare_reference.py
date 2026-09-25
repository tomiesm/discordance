"""Read the unchanged public reference with explicit gene/barcode alignment."""
from common import *
import anndata as ad
import pandas as pd
from scipy.io import mmread
from threadpoolctl import threadpool_limits


def main():
    path=RESULTS/'wu_reference.h5ad'
    if path.exists():print('REFERENCE EXISTS',flush=True);return
    base=SOURCES/'wu'
    genes=pd.read_csv(base/'count_matrix_genes.tsv',sep='\t',header=None).iloc[:,0].astype(str).to_numpy()
    ids=pd.read_csv(base/'count_matrix_barcodes.tsv',sep='\t',header=None).iloc[:,0].astype(str).to_numpy()
    metadata=pd.read_csv(base/'metadata.csv',index_col=0)
    print('READ MATRIX',len(ids),'cells',len(genes),'genes',flush=True)
    with threadpool_limits(limits=4):matrix=mmread(base/'count_matrix_sparse.mtx').tocsr()
    if matrix.shape==(len(genes),len(ids)):matrix=matrix.T.tocsr()
    assert matrix.shape==(len(ids),len(genes)) and len(set(genes))==len(genes)
    assert len(set(ids))==len(ids)
    metadata=metadata.loc[ids]
    np.testing.assert_array_equal(ids,metadata.index)
    totals=np.asarray(matrix.sum(axis=1)).ravel()
    np.testing.assert_allclose(totals,metadata.nCount_RNA,rtol=0,atol=0)
    data=ad.AnnData(matrix.astype(np.int32),obs=metadata,var=pd.DataFrame(index=genes))
    data.write_h5ad(path,compression='gzip')
    dump(RESULTS/'reference_metadata.json',dict(n_cells=len(ids),n_genes=len(genes),
         n_donors=int(metadata['orig.ident'].nunique()),major_types=metadata.celltype_major.value_counts().to_dict(),
         minor_types=metadata.celltype_minor.value_counts().to_dict(),counts_match_published_total=True))
    print('REFERENCE PREPARED',data.shape,flush=True)


if __name__=='__main__':main()

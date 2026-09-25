import os
for _name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_name] = '4'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import hashlib
import json
import h5py
import numpy as np
from scipy import sparse

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
SOURCES = HERE / 'sources'
RESULTS = HERE / 'results'
PROTOCOL = json.loads((HERE / 'protocol.json').read_text())

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024), b''): digest.update(chunk)
    return digest.hexdigest()

def read_10x(path):
    with h5py.File(path) as f:
        m=f['matrix']
        matrix=sparse.csc_matrix((m['data'][:],m['indices'][:],m['indptr'][:]),shape=tuple(m['shape'][:])).T.tocsr()
        genes=np.array([x.decode() for x in m['features/name'][:]])
        ids=np.array([x.decode() for x in m['barcodes'][:]])
    return matrix,genes,ids

def biological_mask(genes):
    return np.array([not any(g.startswith(p) for p in PROTOCOL['control_prefixes']) for g in genes])

def normalize(counts):
    counts=sparse.csr_matrix(counts,dtype=np.float64)
    totals=np.asarray(counts.sum(axis=1)).ravel()
    result=counts.multiply(np.divide(10000,totals,out=np.zeros_like(totals),where=totals>0)[:,None]).tocsr()
    result.data=np.log1p(result.data)
    return result

def hallmark_genes():
    for line in (PROJECT/'clean_repo/data/gene_sets/h.all.v2024.1.Hs.symbols.gmt').read_text().splitlines():
        fields=line.split('\t')
        if fields[0]=='HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION':return fields[2:]
    raise RuntimeError('Missing Hallmark EMT gene set')

def gene_score(matrix, genes, signature):
    index=np.flatnonzero(np.isin(genes,signature))
    if len(index)==0:return np.full(matrix.shape[0],np.nan)
    return np.asarray(matrix[:,index].mean(axis=1)).ravel()

def coexpression_flags(counts, genes):
    tf=np.flatnonzero(np.isin(genes,PROTOCOL['canonical_emt_tfs']))
    epithelial=np.flatnonzero(np.isin(genes,PROTOCOL['epithelial_genes']))
    n_tf=np.asarray((counts[:,tf]>0).sum(axis=1)).ravel()
    n_epithelial=np.asarray((counts[:,epithelial]>0).sum(axis=1)).ravel()
    return (n_tf>=2)&(n_epithelial>=1),n_tf,n_epithelial

def neighborhood_counts(coords,tumor,candidate,radius):
    from scipy.spatial import cKDTree
    tumor=np.asarray(tumor,bool);candidate=np.asarray(candidate,bool)&tumor
    totals=cKDTree(coords[tumor]).query_ball_point(coords,radius,return_length=True,workers=4) if tumor.any() else np.zeros(len(coords),int)
    positives=cKDTree(coords[candidate]).query_ball_point(coords,radius,return_length=True,workers=4) if candidate.any() else np.zeros(len(coords),int)
    fractions=np.divide(positives,totals,out=np.full(len(coords),np.nan),where=totals>0)
    return totals,positives,fractions

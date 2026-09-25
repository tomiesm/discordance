"""Verify NCBI776 counts against the primary publication's Visium download."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import requests
from scipy import sparse

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
URL = ("https://cf.10xgenomics.com/samples/spatial-exp/2.0.0/"
       "CytAssist_FFPE_Human_Breast_Cancer/"
       "CytAssist_FFPE_Human_Breast_Cancer_filtered_feature_bc_matrix.h5")


def main():
    path = OUT / "sources/CytAssist_FFPE_Human_Breast_Cancer_filtered_feature_bc_matrix.h5"
    if not path.exists():
        r = requests.get(URL, timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)
    with h5py.File(path, "r") as f:
        m = f["matrix"]
        x = sparse.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=tuple(m["shape"][:])).T.tocsr()
        genes = [a.decode() for a in m["features/name"][:]]
        barcodes = [a.decode() for a in m["barcodes"][:]]
    a = ad.read_h5ad(ROOT / "data/hest/st/NCBI776.h5ad", backed="r")
    # Resolve duplicate names exactly as AnnData/Scanpy loaders conventionally do.
    gene_index = ad.utils.make_index_unique(__import__("pandas").Index(genes))
    gi = {g: i for i, g in enumerate(gene_index)}
    bi = {b: i for i, b in enumerate(barcodes)}
    missing_genes = sorted(set(a.var_names) - set(gi))
    missing_barcodes = sorted(set(a.obs_names) - set(bi))
    assert not missing_genes and not missing_barcodes
    src = x[np.array([bi[b] for b in a.obs_names])][:, np.array([gi[g] for g in a.var_names])].tocsr()
    max_abs, differing, compared = 0.0, 0, 0
    for start in range(0, a.n_obs, 128):
        local = a.X[start:start + 128]
        if sparse.issparse(local):
            local = local.toarray()
        official = src[start:start + 128].toarray()
        delta = local.astype(np.float64) - official.astype(np.float64)
        max_abs = max(max_abs, float(np.max(np.abs(delta))))
        differing += int(np.count_nonzero(delta)); compared += delta.size
    result = {"verified_utc": datetime.now(timezone.utc).isoformat(), "sample": "NCBI776",
              "source_url": URL, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "source_bytes": path.stat().st_size, "source_shape_spots_genes": list(x.shape),
              "hest_shape_spots_genes": list(a.shape), "entries_compared": compared,
              "differing_entries": differing, "max_abs_count_difference": max_abs,
              "missing_genes": missing_genes, "missing_barcodes": missing_barcodes,
              "same_full_gene_set": set(a.var_names) == set(gene_index),
              "same_full_barcode_set": set(a.obs_names) == set(barcodes)}
    a.file.close()
    (OUT / "janesick_source_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

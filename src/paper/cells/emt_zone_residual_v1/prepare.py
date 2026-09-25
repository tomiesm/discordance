"""Validate square-bin counts and build marker-independent cell-to-bin mapping."""
from src.paper.paths import PROJECT_ROOT, STAGE_SOURCE, CELL_SOURCE, ANALYSIS_ROOT, CELL_ROOT, stage_dir, cell_dir
from src.paper.cells.emt_zone_residual_v1.common import *
import anndata as ad
import pyarrow.parquet as pq
from scipy.spatial import cKDTree
from datetime import datetime, timezone
import shutil

def run(sample):
    target = OUT / sample
    target.mkdir(exist_ok=True)
    a = ad.read_h5ad(PROJECT / f'data/hest/st/{sample}.h5ad')
    cells = pd.read_parquet(CELLS / sample / 'cell_evidence.parquet')
    cell_lookup = pd.Index(cells.index.astype(np.int64) if sample != 'NCBI783' else np.asarray(cells.index).astype('S'))
    genes = np.asarray(a.var_names)
    gl = pd.Index(genes.astype('S'))
    bio = biological(genes)
    nonmarker = bio & ~np.isin(genes, TF + EPI)
    origin, step, lookup = grid(a)
    historical = CELL_SOURCE / 'emt_zone_residual_v1' / f'{sample}_mapping.json'
    if historical.exists():
        transform = json.loads(historical.read_text())['details']
        matrix = np.array(transform['matrix'])
        native_origin = np.array(transform['origin'])
        mapping_method = 'Recovered historical native-coordinate rotation; validated by complete spot-by-gene count reconstruction'
    else:
        matrix = None
        mapping_method = 'Supplied HEST transcript HE coordinates; validated by complete spot-by-gene count reconstruction'
    edge_path = CELL_SOURCE / 'emt_zone_residual_v1' / f'{sample}_bin_edges.json'
    edge_repairs = json.loads(edge_path.read_text()) if edge_path.exists() else []
    assignment_repairs = []
    total = np.zeros(a.shape, np.int64)
    cell_counts = sparse.csr_matrix((len(cells), len(a)), dtype=np.int64)
    native_sum = np.zeros((len(a), 2))
    native_n = np.zeros(len(a), np.int64)
    total_rows = 0
    columns = ['cell_id', 'feature_name', 'qv', 'he_x', 'he_y', 'x_location', 'y_location']
    p = PROJECT / f'data/hest/transcripts/{sample}_transcripts.parquet'
    for k, b in enumerate(pq.ParquetFile(p).iter_batches(batch_size=1000000, columns=columns, use_threads=False)):
        d = b.to_pandas()
        g = gl.get_indexer(d.feature_name)
        assert (g >= 0).all()
        if matrix is None:
            s = assign_grid(d[['he_x', 'he_y']].to_numpy(float), origin, step, lookup)
        else:
            s = assign_grid(d[['x_location', 'y_location']].to_numpy(float) @ matrix.T, native_origin, 100.0, lookup)
        for repair in edge_repairs:
            if total_rows <= repair['transcript_row'] < total_rows + len(d):
                j = repair['transcript_row'] - total_rows
                assert s[j] == repair['from_spot_index']
                assert str(genes[g[j]]) == repair['gene']
                s[j] = repair['to_spot_index']
        assert (s >= 0).all(), 'Transcript outside reconstructed full grid'
        total += np.bincount(s * len(genes) + g, minlength=total.size).reshape(total.shape)
        ci = cell_lookup.get_indexer(d.cell_id.to_numpy())
        q = d.qv.to_numpy() >= 20
        use = q & (ci >= 0) & nonmarker[g]
        for repair in edge_repairs:
            if total_rows <= repair['transcript_row'] < total_rows + len(d):
                j = repair['transcript_row'] - total_rows
                if use[j]:
                    assignment_repairs.append((int(ci[j]), repair['from_spot_index'], repair['to_spot_index']))
        cell_counts += sparse.coo_matrix((np.ones(use.sum(), np.int64), (ci[use], s[use])), shape=cell_counts.shape).tocsr()
        use_native = q & nonmarker[g]
        native_n += np.bincount(s[use_native], minlength=len(a))
        for j, col in enumerate(['x_location', 'y_location']):
            native_sum[:, j] += np.bincount(s[use_native], weights=d[col].to_numpy()[use_native], minlength=len(a))
        total_rows += len(d)
        if k % 10 == 0:
            print(sample, 'transcript rows', total_rows, flush=True)
    raw = a.X.toarray() if sparse.issparse(a.X) else np.asarray(a.X)
    difference = total - raw
    if np.any(difference):
        dump(target / 'COUNT_RECONSTRUCTION_DISCREPANCY.json', {'different_entries': int(np.count_nonzero(difference)), 'absolute_difference': float(np.abs(difference).sum()), 'maximum_difference': float(np.abs(difference).max()), 'genes_with_difference': genes[np.any(difference, axis=0)].tolist()})
    np.testing.assert_array_equal(total, raw)
    majority = np.asarray(cell_counts.argmax(axis=1)).ravel()
    totals = np.asarray(cell_counts.sum(axis=1)).ravel()
    largest = np.asarray(cell_counts.max(axis=1).toarray()).ravel()
    share = np.divide(largest, totals, out=np.zeros(len(cells)), where=totals > 0)
    majority[(share < 0.5) | (totals == 0)] = -1
    changed_assignments = []
    for i in sorted(set((t[0] for t in assignment_repairs))):
        without = cell_counts.getrow(i).toarray().ravel()
        for cell, source, dest in assignment_repairs:
            if cell == i:
                without[source] += 1
                without[dest] -= 1
        old = int(without.argmax()) if without.sum() > 0 and without.max() / without.sum() >= 0.5 else -1
        if old != majority[i]:
            changed_assignments.append(str(cells.index[i]))
    assert not changed_assignments, 'Sub-nanometer edge reconciliation changed cell bin; investigate before inference'
    if matrix is None:
        centroid = assign_grid(cells[['he_x', 'he_y']].to_numpy(float), origin, step, lookup)
    else:
        centroid = assign_grid(cells[['transcript_native_x_um', 'transcript_native_y_um']].to_numpy(float) @ matrix.T, native_origin, 100.0, lookup)
    cells['full_spot_index'] = majority
    cells['dominant_nonmarker_share'] = share
    cells['centroid_full_spot_index'] = centroid
    cells['assignment_centroid_agrees'] = centroid == majority
    cdata = ad.read_h5ad(CELLS / sample / 'measured_cells.h5ad')
    np.testing.assert_array_equal(cdata.obs_names, cells.index)
    keep = ~np.isin(cdata.var_names, TF + EPI + HALLMARK)
    cx = cdata.X[:, keep]
    cells['nonmarker_counts'] = np.asarray(cx.sum(axis=1)).ravel()
    cells['nonmarker_detected'] = np.asarray((cx > 0).sum(axis=1)).ravel()
    cells.to_parquet(target / 'mapped_cells.parquet')
    sparse.save_npz(target / 'cell_nonmarker_bin_counts.npz', cell_counts)
    native = np.divide(native_sum, native_n[:, None], out=np.full_like(native_sum, np.nan), where=native_n[:, None] > 0)
    spots = pd.DataFrame({'barcode': a.obs_names, 'he_x': a.obsm['spatial'][:, 0], 'he_y': a.obsm['spatial'][:, 1], 'native_x': native[:, 0], 'native_y': native[:, 1], 'nonmarker_spot_counts': total[:, bio & ~np.isin(genes, TF + EPI + HALLMARK)].sum(axis=1)})
    spots.index.name = 'full_spot_index'
    spots.to_parquet(target / 'full_spots.parquet')
    np.savez_compressed(target / 'reconstructed_counts.npz', counts=total, genes=genes.astype(str), barcodes=np.asarray(a.obs_names).astype(str))
    qc = cells.qc_pass.to_numpy()
    tumor = qc & (cells.source_group.to_numpy() == 'Tumor')
    summary = {'sample': sample, 'all_transcript_rows': total_rows, 'exact_full_spot_gene_count_match': True, 'full_spots': len(a), 'full_genes': len(genes), 'grid_step_he_pixels': step, 'qc_cells': int(qc.sum()), 'assigned_qc_cells': int((qc & (majority >= 0)).sum()), 'qc_tumor_cells': int(tumor.sum()), 'assigned_qc_tumor_cells': int((tumor & (majority >= 0)).sum()), 'qc_assignment_centroid_agreement': float(np.mean((majority == centroid)[qc & (majority >= 0)])), 'qc_dominant_transcript_share_quantiles': np.quantile(share[qc], [0, 0.01, 0.05, 0.5, 1]).tolist(), 'mapping_method': mapping_method, 'sub_nanometer_bin_edge_reconciliations': edge_repairs, 'cell_assignments_changed_by_edge_reconciliations': changed_assignments, 'coordinate_scope': 'Internal consistency with observed expression bins; not independent validation of H&E anatomical registration', 'transcript_sha256': sha(p), 'hest_h5ad_sha256': sha(PROJECT / f'data/hest/st/{sample}.h5ad')}
    dump(target / 'MAPPING_VERIFIED.json', summary)
    print('VERIFIED', summary, flush=True)
if __name__ == '__main__':
    for s in PROTOCOL['samples']:
        if not (OUT / s / 'MAPPING_VERIFIED.json').exists():
            run(s)

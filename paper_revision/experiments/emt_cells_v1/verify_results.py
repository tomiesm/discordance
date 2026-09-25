"""Independent numerical checks; does not call the analysis flag/spatial helpers."""
from common import HERE, PROJECT, SOURCES, RESULTS, PROTOCOL, dump, sha
from datetime import datetime, timezone
import json
from decimal import Decimal
import numpy as np
import pandas as pd
import anndata as ad
import h5py
from scipy import sparse
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from scipy.stats import rankdata
import pyarrow.parquet as pq


def strings(values):
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in values])


def main():
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'sections': []}
    diagnostics = []
    rng = np.random.default_rng(20260920)
    associations = pd.read_csv(RESULTS/'residual_associations.csv')
    for sample in PROTOCOL['samples']:
        data = ad.read_h5ad(RESULTS/sample/'measured_cells.h5ad')
        obs = pd.read_parquet(RESULTS/sample/'cell_evidence.parquet')
        np.testing.assert_array_equal(obs.index, data.obs_names)
        with h5py.File(SOURCES/'vendor'/sample/'cell_feature_matrix.h5') as handle:
            m = handle['matrix']
            matrix = sparse.csc_matrix((m['data'][:], m['indices'][:], m['indptr'][:]), shape=tuple(m['shape'][:])).T.tocsr()
            np.testing.assert_array_equal(strings(m['barcodes'][:]), obs.index)
            j = pd.Index(strings(m['features/name'][:])).get_indexer(data.var_names)
            assert (j >= 0).all()
            difference = matrix[:, j] - data.X
            difference.eliminate_zeros()
            assert difference.nnz == 0
        raw_labels = pd.read_csv(SOURCES/PROTOCOL['source_annotations'][sample], dtype=str)
        # Vendor numeric IDs can be serialized as scientific notation (1e+05).
        # Decimal parsing preserves exact integer semantics independently of pandas.
        annotation_ids = raw_labels.iloc[:, 0].to_numpy()
        if sample != 'NCBI783':
            assert all(Decimal(v) == int(Decimal(v)) for v in annotation_ids)
            annotation_ids = [str(int(Decimal(v))) for v in annotation_ids]
        labels = dict(zip(annotation_ids, raw_labels.iloc[:, 1]))
        expected_labels = np.array([labels.get(v, 'Missing annotation') for v in obs.index])
        np.testing.assert_array_equal(expected_labels, obs.source_label)
        vendor_cells = pd.read_parquet(SOURCES/'vendor'/sample/'cells.parquet')
        vendor_cells.index = strings(vendor_cells.cell_id)
        native = vendor_cells.loc[obs.index, ['x_centroid', 'y_centroid']].to_numpy(float)
        np.testing.assert_array_equal(native, obs[['x_um', 'y_um']])
        counts = data.X.toarray()
        nuclear = data.layers['nuclear_counts'].toarray()
        assert (nuclear <= counts).all()
        total = counts.sum(axis=1)
        qc = (total >= 20) & ((counts > 0).sum(axis=1) >= 10)
        np.testing.assert_array_equal(qc, obs.qc_pass)
        tumor = qc & np.isin(expected_labels, ['DCIS_1', 'DCIS_2', 'Invasive_Tumor', 'Prolif_Invasive_Tumor', 'Tumor'])
        np.testing.assert_array_equal(tumor, qc & (obs.source_group == 'Tumor'))
        genes = list(data.var_names)
        tf = [genes.index(g) for g in ['SNAI1', 'SNAI2', 'ZEB1', 'ZEB2', 'TWIST1', 'TWIST2'] if g in genes]
        epithelial = [genes.index(g) for g in ['EPCAM', 'CDH1', 'KRT8', 'KRT18', 'KRT19', 'KRT7', 'CLDN3', 'CLDN4', 'OCLN', 'TJP1', 'DSP', 'MUC1'] if g in genes]
        flag = tumor & ((counts[:, tf] > 0).sum(axis=1) >= 2) & (counts[:, epithelial].sum(axis=1) > 0)
        nuclear_flag = tumor & ((nuclear[:, tf] > 0).sum(axis=1) >= 2) & (nuclear[:, epithelial].sum(axis=1) > 0)
        np.testing.assert_array_equal(flag, obs.tumor_tf_candidate)
        np.testing.assert_array_equal(nuclear_flag, obs.tumor_nuclear_tf_candidate)
        assert not (nuclear_flag & ~flag).any()
        fixed = rng.choice(len(obs), 64, replace=False)
        for i in fixed:
            squared = np.sum((native - native[i])**2, axis=1)
            for radius in [50, 100, 150]:
                near = squared <= radius**2
                n = int((near & tumor).sum())
                k = int((near & flag).sum())
                assert n == obs.iloc[i][f'tumor_neighbors_{radius}um']
                assert k == obs.iloc[i][f'tf_candidate_neighbors_{radius}um']
        for radius in [50, 100, 150]:
            eligible = tumor & (obs[f'tumor_neighbors_{radius}um'].to_numpy() >= 20) & (obs[f'tf_candidate_neighbors_{radius}um'].to_numpy() >= 5) & (obs[f'tf_candidate_fraction_{radius}um'].to_numpy() >= .1)
            ids = obs[f'candidate_component_{radius}um'].to_numpy()
            np.testing.assert_array_equal(eligible, ids >= 0)
            if eligible.sum() > 1:
                # Independent single-linkage construction must give identical components.
                groups = fcluster(linkage(pdist(native[eligible]), method='single'), radius, criterion='distance')
                pairs = pd.DataFrame({'computed': ids[eligible], 'checked': groups})
                assert pairs.groupby('computed').checked.nunique().max() == 1
                assert pairs.groupby('checked').computed.nunique().max() == 1
        # Independently stream selected barcodes from original transcript records:
        # checks nuclear assignment and the exact HE/native centroid definitions.
        chosen = np.unique(np.r_[fixed[:32], rng.choice(np.flatnonzero(flag), 32, replace=False)])
        selected_ids = obs.index[chosen]
        raw_ids = selected_ids.to_numpy(dtype='S') if sample == 'NCBI783' else selected_ids.to_numpy(dtype=np.int64)
        pieces = []
        for batch in pq.ParquetFile(PROJECT/f'data/hest/transcripts/{sample}_transcripts.parquet').iter_batches(
                batch_size=1000000, columns=['cell_id', 'feature_name', 'qv', 'overlaps_nucleus', 'x_location', 'y_location', 'he_x', 'he_y'], use_threads=False):
            d = batch.to_pandas()
            use = d.cell_id.isin(raw_ids) & (d.qv >= 20) & d.feature_name.isin([g.encode() for g in genes])
            if use.any():
                pieces.append(d.loc[use].copy())
        raw = pd.concat(pieces, ignore_index=True)
        raw['cell_id'] = strings(raw.cell_id)
        raw['gene'] = strings(raw.feature_name)
        direct = pd.crosstab(raw.cell_id, raw.gene).reindex(index=selected_ids, columns=genes, fill_value=0)
        np.testing.assert_array_equal(direct, counts[chosen])
        direct_nuclear = pd.crosstab(raw.loc[raw.overlaps_nucleus == 1, 'cell_id'], raw.loc[raw.overlaps_nucleus == 1, 'gene']).reindex(index=selected_ids, columns=genes, fill_value=0)
        np.testing.assert_array_equal(direct_nuclear, nuclear[chosen])
        centers = raw.groupby('cell_id')[['x_location', 'y_location', 'he_x', 'he_y']].mean().reindex(selected_ids)
        expected = obs.loc[selected_ids, ['transcript_native_x_um', 'transcript_native_y_um', 'he_x', 'he_y']]
        np.testing.assert_allclose(centers, expected, rtol=2e-7, atol=.005, equal_nan=True)
        del counts, nuclear, matrix, data, raw, pieces
        # QR projection independently checks the rank-adjusted association.
        spots = pd.read_parquet(RESULTS/sample/'spot_residual_evidence.parquet')
        frame = spots.loc[spots.n_tumor_cells >= 5]
        controls = np.column_stack([frame.stromal_fraction, frame.myoepithelial_fraction, frame.tumor_fraction,
                                    np.log1p(frame.mean_tumor_transcripts), np.log1p(frame.n_tumor_cells), np.log1p(frame.total_expr)])
        x = np.column_stack([np.ones(len(frame)), np.apply_along_axis(rankdata, 0, controls)])
        u, singular, _ = np.linalg.svd(x, full_matrices=False)
        q = u[:, singular > singular.max()*1e-12]
        y = rankdata(frame.tf_fraction)
        y -= q@(q.T@y)
        for score in ['D_cond', 'disjoint_D_cond']:
            z = rankdata(frame[score]);z -= q@(q.T@z)
            checked = np.corrcoef(y, z)[0, 1]
            saved = associations.loc[(associations['sample'] == sample) & (associations.minimum_tumor_cells == 5) & (associations.score == score), 'partial_spearman'].item()
            np.testing.assert_allclose(checked, saved, atol=1e-10)
        for label, mask in [('TF candidates', flag), ('Other QC tumor', tumor & ~flag), ('Nuclear-supported candidates', nuclear_flag)]:
            f = obs.loc[mask]
            diagnostics.append(dict(sample=sample, group=label, n_cells=len(f),
                median_transcripts=float(f.total_biological_transcripts.median()),
                median_cell_area_um2=float(f.cell_area.median()),
                median_nucleus_area_um2=float(f.nucleus_area.median()),
                median_stromal_neighbors_100um=float(f.stromal_neighbors_100um.median()),
                median_myoepithelial_neighbors_100um=float(f.myoepithelial_neighbors_100um.median())))
        report['sections'].append(dict(sample=sample, n_cells=len(obs), counts_and_annotations='exact',
            vendor_centroids='exact', candidate_flags='exact', sampled_raw_barcode_recounts=len(chosen),
            sampled_native_neighborhood_centers=len(fixed), neighborhood_radii_um=[50, 100, 150],
            all_components_independently_reconstructed=True, partial_correlations='independent SVD check passed'))
        print('INDEPENDENT CHECK PASS', sample, flush=True)
    for panel in ['panel313', 'panel280']:
        folds = json.loads((RESULTS/f'{panel}_reference_folds.json').read_text())
        assert len(folds) == 20
        for fold in folds:
            assert fold['donor'] not in fold['train_donors'] and len(fold['train_donors']) == 19
    reference = json.loads((RESULTS/'REFERENCE_OBSERVABILITY.json').read_text())
    for spec in reference:
        sample = 'NCBI785' if spec['panel'] == 'panel313' else 'NCBI783'
        with h5py.File(SOURCES/'vendor'/sample/'cell_feature_matrix.h5') as h:
            panel_genes = strings(h['matrix/features/name'][:])
        assert not set(spec['target_genes']) & set(panel_genes)
    report['reference'] = {'held_out_donor_checks': 40, 'gene_disjoint_targets': 'pass',
        'independent_weighted_ridge_normal_equation_checks': 'performed during reference analysis, first fold per panel'}
    originals = json.loads((PROJECT/'paper_revision/original_snapshot_manifest.json').read_text())
    for root, entries in originals.items():
        for name, info in entries.items():
            assert sha(PROJECT/root/name) == info['sha256'], str(PROJECT/root/name)
    report['unchanged_original_files'] = {root: len(entries) for root, entries in originals.items()}
    manifest = json.loads((HERE/'input_manifest.json').read_text())
    assert sha(HERE/'protocol.json') == manifest['protocol_sha256']
    for name, info in manifest['inputs'].items():
        assert sha(name) == info['sha256'], name
    downloads = json.loads((SOURCES/'Janesick_downloads.json').read_text())
    for item in downloads:
        assert sha(SOURCES/item['file']) == item['sha256'], item['file']
    report['raw_input_hashes_verified'] = len(manifest['inputs'])
    report['additional_download_hashes_verified'] = len(downloads)
    for snapshot in ['reference_frozen', 'cell_analysis_frozen']:
        spec = json.loads((HERE/snapshot/'manifest.json').read_text())
        for name, digest in spec['sha256'].items():
            assert sha(HERE/snapshot/name) == digest
    pd.DataFrame(diagnostics).to_csv(RESULTS/'posthoc_cell_context_diagnostics.csv', index=False)
    programs = pd.read_parquet(RESULTS/'reference_cell_programs.parquet')
    columns = ['full_hallmark_emt', 'full_epithelial', 'panel313_measured_hallmark',
               'panel280_measured_hallmark', 'panel313_tf_candidate', 'panel280_tf_candidate',
               'panel313_full_fit_prediction', 'panel280_full_fit_prediction']
    grouped = programs.groupby(['orig.ident', 'celltype_major'], observed=True)
    grouped[columns].mean().join(grouped.size().rename('n_cells')).reset_index().to_csv(
        RESULTS/'posthoc_reference_lineage_diagnostics.csv', index=False)
    report['status'] = 'pass'
    dump(RESULTS/'VERIFICATION.json', report)
    print('ALL INDEPENDENT CHECKS PASS', flush=True)


if __name__ == '__main__':
    main()

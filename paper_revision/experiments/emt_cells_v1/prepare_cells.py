"""Reconstruct measured per-cell expression and verify against vendor counts."""
from common import *
from datetime import datetime, timezone
import shutil
import pandas as pd
import pyarrow.parquet as pq
import anndata as ad


def source_group(label):
    if label in ['DCIS_1','DCIS_2','Invasive_Tumor','Prolif_Invasive_Tumor','Tumor']:
        return 'Tumor'
    if label in ['Stromal','Stromal Normal','Tumor Associated Stromal']:return 'Stromal'
    if label.startswith('Myoepi') or label == 'DST+ Myoepithelial':return 'Myoepithelial'
    if label in ['ESR1+ Epithelial','PIGR+ Epithelial','OPRPN+ Epithelial']:return 'Normal epithelial'
    if label == 'Transitional Cells':return 'Published transitional'
    if label in ['Unlabeled','Not Plotted','Missing annotation'] or 'Hybrid' in label:return 'Uncertain/hybrid'
    return 'Other non-tumor'


def fit_coordinate_transform(native, histology):
    """Fit on alternate control points and validate on unused points."""
    from skimage.transform import ProjectiveTransform
    from scipy.interpolate import RBFInterpolator
    _, unique=np.unique(native,axis=0,return_index=True)
    unique=np.sort(unique);native=native[unique];histology=histology[unique]
    model = ProjectiveTransform()
    assert model.estimate(native[::2], histology[::2])
    error=np.linalg.norm(model(native[1::2])-histology[1::2],axis=1)
    projective_error=float(error.max())
    if projective_error<=1:
        model.estimate(native,histology)
        info={'method':'projective','matrix':model.params.tolist()}
    else:
        model=RBFInterpolator(native[::2],histology[::2],neighbors=32,kernel='thin_plate_spline',smoothing=1e-8)
        error=np.linalg.norm(model(native[1::2])-histology[1::2],axis=1)
        if np.max(error)>1:
            raise RuntimeError(f'Local coordinate registration discrepancy: {np.max(error)} pixels')
        model=RBFInterpolator(native,histology,neighbors=32,kernel='thin_plate_spline',smoothing=1e-8)
        info={'method':'local thin-plate interpolation of supplied transcript coordinates','neighbors':32,'smoothing':1e-8}
    return model,dict(**info,n_control_points=len(native),global_projective_max_error_pixels=projective_error,
                     test_max_error_pixels=float(error.max()),test_p99_error_pixels=float(np.quantile(error,.99)),
                     test_median_error_pixels=float(np.median(error)))


def main():
    snapshot=HERE/'frozen'
    if not snapshot.exists():
        snapshot.mkdir()
        for p in HERE.glob('*.py'):shutil.copy2(p,snapshot/p.name)
        shutil.copy2(HERE/'protocol.json',snapshot/'protocol.json')
        originals=json.loads((PROJECT/'paper_revision/original_snapshot_manifest.json').read_text())
        for root,entries in originals.items():
            for name,info in entries.items():assert sha(PROJECT/root/name)==info['sha256'],name
        manifest={}
        for sample in PROTOCOL['samples']:
            paths=[PROJECT/f'data/hest/transcripts/{sample}_transcripts.parquet',
                   SOURCES/PROTOCOL['source_annotations'][sample]]
            paths+=list((SOURCES/'vendor'/sample).glob('*'))
            for p in paths:
                if p.is_file():manifest[str(p)]={'bytes':p.stat().st_size,'sha256':sha(p)}
        manifest[str(SOURCES/'Wu_2021_BRCA_scRNASeq.tar.gz')]={'sha256':sha(SOURCES/'Wu_2021_BRCA_scRNASeq.tar.gz')}
        dump(HERE/'input_manifest.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'inputs':manifest,
             'original_snapshot_files_verified':sum(len(x) for x in originals.values()),
             'protocol_sha256':sha(HERE/'protocol.json')})
    summaries=[]
    for sample,patient in PROTOCOL['samples'].items():
        out=RESULTS/sample;out.mkdir(exist_ok=True)
        if (out/'COUNTS_VERIFIED.json').exists():
            print('SKIP VERIFIED',sample,flush=True)
            summaries.append(json.loads((out/'COUNTS_VERIFIED.json').read_text()));continue
        print('RECONSTRUCT',sample,flush=True)
        vendor,genes,ids=read_10x(SOURCES/'vendor'/sample/'cell_feature_matrix.h5')
        vendor=vendor.astype(np.int32)
        cells=pd.read_parquet(SOURCES/'vendor'/sample/'cells.parquet')
        numeric=sample!='NCBI783'
        lookup=pd.Index(ids.astype(np.int64) if numeric else ids.astype('S'))
        cell_ids=cells.cell_id.to_numpy()
        if numeric:cell_ids=cell_ids.astype(np.int64)
        else:cell_ids=np.array([x.decode() if isinstance(x,bytes) else str(x) for x in cell_ids])
        cells=cells.set_index(cell_ids).loc[ids.astype(np.int64) if numeric else ids].reset_index(drop=True)
        gene_lookup=pd.Index(genes.astype('S'))
        bio=biological_mask(genes)
        reconstructed=np.zeros(vendor.shape,np.int32)
        nuclear=np.zeros_like(reconstructed)
        coordinate_sums=np.zeros((len(ids),4),float)
        coordinate_counts=np.zeros(len(ids),np.int64)
        total_rows=0;high_q_rows=0;unassigned=0;unknown_values={};control_points=[]
        parquet=pq.ParquetFile(PROJECT/f'data/hest/transcripts/{sample}_transcripts.parquet')
        for batch in parquet.iter_batches(batch_size=1000000,columns=['cell_id','feature_name','qv','overlaps_nucleus','x_location','y_location','he_x','he_y'],use_threads=False):
            d=batch.to_pandas()
            row=lookup.get_indexer(d.cell_id.to_numpy())
            col=gene_lookup.get_indexer(d.feature_name.to_numpy())
            assert (col>=0).all(),'Transcript gene not found in vendor matrix'
            high=d.qv.to_numpy()>=PROTOCOL['transcript_qv_minimum']
            high_q_rows+=int(high.sum())
            unassigned+=int((high & (row<0)).sum())
            for value,count in d.loc[high & (row<0),'cell_id'].value_counts().items():
                key=value.decode() if isinstance(value,bytes) else str(value)
                unknown_values[key]=unknown_values.get(key,0)+int(count)
            use=high & (row>=0)
            np.add.at(reconstructed,(row[use],col[use]),1)
            use_nuclear=use & (d.overlaps_nucleus.to_numpy()==1)
            np.add.at(nuclear,(row[use_nuclear],col[use_nuclear]),1)
            use_coordinates=use & bio[col]
            coordinate_counts+=np.bincount(row[use_coordinates],minlength=len(ids))
            for j,column in enumerate(['x_location','y_location','he_x','he_y']):
                coordinate_sums[:,j]+=np.bincount(row[use_coordinates],weights=d[column].to_numpy()[use_coordinates],minlength=len(ids))
            control_points.append(d[['x_location','y_location','he_x','he_y']].iloc[::1000].to_numpy(float))
            total_rows+=len(d)
        difference=sparse.csr_matrix(reconstructed)-vendor
        max_error=int(np.abs(difference.data).max()) if difference.nnz else 0
        if max_error:
            dump(out/'COUNT_DISCREPANCY.json',dict(nnz=difference.nnz,max_error=max_error,raw_total=int(reconstructed.sum()),vendor_total=int(vendor.sum()),unknown_cell_ids=unknown_values))
            raise RuntimeError(f'{sample}: reconstructed counts do not match vendor matrix')
        assert (nuclear<=reconstructed).all()
        counts=sparse.csr_matrix(reconstructed[:,bio]);nuc=sparse.csr_matrix(nuclear[:,bio])
        del reconstructed,nuclear
        biological_genes=genes[bio]
        total=np.asarray(counts.sum(axis=1)).ravel();detected=np.diff(counts.indptr)
        annotations=pd.read_csv(SOURCES/PROTOCOL['source_annotations'][sample])
        annotation_ids=annotations.iloc[:,0].astype(np.int64).astype(str) if numeric else annotations.iloc[:,0].astype(str)
        assert not annotation_ids.duplicated().any()
        labels=pd.Series(annotations.iloc[:,1].to_numpy(),index=annotation_ids).reindex(ids).fillna('Missing annotation')
        controls=np.vstack(control_points)
        np.savez_compressed(out/'coordinate_controls.npz',native=controls[:,:2],histology=controls[:,2:])
        native=cells[['x_centroid','y_centroid']].to_numpy(float)
        measured_centroid=np.divide(coordinate_sums,coordinate_counts[:,None],out=np.full_like(coordinate_sums,np.nan),where=coordinate_counts[:,None]>0)
        he=measured_centroid[:,2:]
        displacement=np.linalg.norm(measured_centroid[:,:2]-native,axis=1)
        registration=dict(method='Direct centroid of measured assigned Q>=20 biological transcripts in the supplied H&E frame; no fitted transform applied',
                          primary_spatial_analysis='Vendor cell centroids in native micrometers',
                          he_usage='Visualization and secondary spot assignment only; transcript centroid is not the geometric cell centroid',
                          native_transcript_vs_cell_centroid_um_quantiles=np.nanquantile(displacement,[.5,.95,.99,1]).tolist())
        np.testing.assert_array_equal(coordinate_counts,total)
        obs=pd.DataFrame({'sample':sample,'patient':patient,'source_label':labels.to_numpy(),
                         'source_group':[source_group(x) for x in labels],
                         'x_um':native[:,0],'y_um':native[:,1],'he_x':he[:,0],'he_y':he[:,1],
                         'total_biological_transcripts':total,'detected_biological_genes':detected,
                         'cell_area':cells.cell_area.to_numpy(),'nucleus_area':cells.nucleus_area.to_numpy()},index=ids)
        obs['transcript_native_x_um']=measured_centroid[:,0];obs['transcript_native_y_um']=measured_centroid[:,1]
        obs['transcript_centroid_displacement_um']=displacement
        obs.index.name='cell_id'
        obs['qc_pass']=(total>=PROTOCOL['cell_minimum_biological_transcripts']) & (detected>=PROTOCOL['cell_minimum_detected_biological_genes'])
        data=ad.AnnData(counts,obs=obs,var=pd.DataFrame(index=biological_genes))
        data.layers['nuclear_counts']=nuc
        data.obsm['spatial_um']=native;data.obsm['spatial_he']=he
        data.write_h5ad(out/'measured_cells.h5ad',compression='gzip')
        summary=dict(sample=sample,patient=patient,n_cells=len(ids),n_biological_genes=int(bio.sum()),
                     transcript_rows=total_rows,high_q_rows=high_q_rows,unassigned_high_q=unassigned,
                     unassigned_identifiers=unknown_values,vendor_matrix_exact_match=True,
                     n_qc_pass=int(obs.qc_pass.sum()),n_source_tumor_qc=int((obs.qc_pass & (obs.source_group=='Tumor')).sum()),
                     n_annotated=int((obs.source_label!='Missing annotation').sum()),
                     labels=obs.source_label.value_counts().to_dict(),coordinate_transform=registration)
        dump(out/'COUNTS_VERIFIED.json',summary);summaries.append(summary)
        print('VERIFIED',sample,len(ids),'cells',int(total.sum()),'biological transcripts',flush=True)
    dump(RESULTS/'count_verification.json',summaries)


if __name__=='__main__':main()

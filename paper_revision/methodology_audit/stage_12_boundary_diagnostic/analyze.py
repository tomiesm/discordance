"""Boundary geometry and fixed-score sensitivity; no model/score changes."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree,ConvexHull
from scipy.stats import spearmanr
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT=Path(__file__).resolve().parent;AUDIT=OUT.parent;ROOT=AUDIT.parents[1]
FAMILIES=['biomarkers','10x_janesick','coad','idc_visium']


def signed_box(xy,x0,y0,x1,y1):
    inside=np.minimum.reduce([xy[:,0]-x0,xy[:,1]-y0,x1-xy[:,0],y1-xy[:,1]])
    dx=np.maximum.reduce([x0-xy[:,0],xy[:,0]-x1,np.zeros(len(xy))]);dy=np.maximum.reduce([y0-xy[:,1],xy[:,1]-y1,np.zeros(len(xy))])
    return np.where(inside>=0,inside,-np.hypot(dx,dy))


def polygons(geometry):
    if geometry.geom_type=='Polygon':yield geometry
    elif hasattr(geometry,'geoms'):
        for part in geometry.geoms:yield from polygons(part)


def moran(xy,v):
    pairs=cKDTree(xy).query_pairs(150,output_type='ndarray');z=v-v.mean();den=np.sum(z*z)
    if not len(pairs) or den==0:return np.nan,0
    val=len(v)/len(pairs)*np.sum(z[pairs[:,0]]*z[pairs[:,1]])/den
    # Independent directed adjacency multiplication.
    from scipy.sparse import coo_matrix
    w=coo_matrix((np.ones(2*len(pairs)),(np.r_[pairs[:,0],pairs[:,1]],np.r_[pairs[:,1],pairs[:,0]])),shape=(len(v),len(v))).tocsr()
    direct=len(v)/w.sum()*(z@(w@z))/(z@z);assert abs(val-direct)<1e-10
    return val,int(len(pairs))


def main():
    cov=pd.read_parquet(AUDIT/'stage_01_provenance/spot_coverage.parquet');meta=pd.read_csv(AUDIT/'stage_01_provenance/coverage_summary.csv').set_index('sample')
    sources={r['relative_path']:Path(r['path']) for r in json.loads((AUDIT/'stage_01_provenance/sources/hest_source_manifest.json').read_text())['files'] if r['status']==200}
    diag=pd.read_parquet(AUDIT/'stage_03_score/spot_score_diagnostics.parquet').set_index('spot_id')
    allspots=[];effects=[];tails=[];interior=[];correlations=[];checks=[];figdata={}
    for family in FAMILIES:
        loc=pd.read_parquet(AUDIT/'stage_10_decile_sensitivity'/family/'locations.parquet')
        for sample,ss in loc.groupby('sample',sort=False):
            ss=ss.copy().reset_index(drop=True);c=cov[cov['sample'].eq(sample)].copy();c['spot_id']=sample+'_'+c.barcode;c=c.set_index('spot_id')
            aligned=c.loc[ss.spot_id];assert aligned.has_patch.all();pixel=meta.loc[sample,'pixel_size_um'];xy=ss[['x_um','y_um']].to_numpy()
            err=np.max(abs(xy-aligned[['x','y']].to_numpy()*pixel));assert err<1e-7
            checks.append(dict(check=sample+':coordinate_alignment',max_abs=float(err),passed=True))
            gj=json.loads(sources[f'tissue_seg/{sample}_contours.geojson'].read_text());geoms=[shapely.geometry.shape(f['geometry']) for f in gj['features']]
            repaired=shapely.union_all([shapely.make_valid(g) if not g.is_valid else g for g in geoms])
            # Tissue area excludes zero-area line remnants from make_valid.
            union=shapely.union_all(list(polygons(repaired)))
            area_difference=union.symmetric_difference(repaired).area;assert area_difference==0
            checks.append(dict(check=sample+':polygonal_area_preserved',symmetric_difference_area=area_difference,passed=True))
            native=aligned[['x','y']].to_numpy();pts=shapely.points(native)
            dist=shapely.distance(pts,union.boundary)*pixel;inside=shapely.covers(union,pts);ss['tissue_distance_um']=np.where(inside,dist,-dist)
            md=json.loads(sources[f'metadata/{sample}.json'].read_text())
            with h5py.File(ROOT/'data/hest/st'/f'{sample}.h5ad','r') as f:
                st=f['uns/spatial'];g=st[list(st.keys())[0]];thumb=g['images/downscaled_fullres'][:];scale=float(g['scalefactors/tissue_downscaled_fullres_scalef'][()])
            width=md.get('fullres_px_width',md.get('fullres_width',thumb.shape[1]/scale));height=md.get('fullres_px_height',md.get('fullres_height',thumb.shape[0]/scale))
            ss['image_distance_um']=signed_box(native,0,0,width,height)*pixel
            bounds=[c.x.min(),c.y.min(),c.x.max(),c.y.max()];ss['expression_extent_distance_um']=signed_box(native,*bounds)*pixel
            hull=shapely.Polygon(xy[ConvexHull(xy).vertices]);ss['sample_hull_distance_um']=shapely.distance(shapely.points(xy),hull.boundary)
            assert np.isfinite(ss[['tissue_distance_um','image_distance_um','expression_extent_distance_um','sample_hull_distance_um']].to_numpy()).all()
            ss['patch_tissue_fraction']=aligned.tissue_fraction.to_numpy();ss['fully_on_slide']=aligned.fully_on_slide.to_numpy()
            ss['baseline_conditional']=diag.loc[ss.spot_id,'baseline_conditional'].to_numpy() if family in FAMILIES[:2] else np.nan
            ss['panel_counts']=aligned.panel_counts.to_numpy();ss['detected_genes']=aligned.panel_genes_detected.to_numpy()
            selected=np.linspace(0,len(ss)-1,min(25,len(ss))).astype(int)
            for i in selected:
                p=shapely.Point(native[i]);expected=p.distance(union.boundary)*pixel*(1 if union.covers(p) else -1)
                assert abs(expected-ss.tissue_distance_um.iloc[i])<1e-8
                box=shapely.box(0,0,width,height);expected=p.distance(box.boundary)*pixel*(1 if box.covers(p) else -1)
                assert abs(expected-ss.image_distance_um.iloc[i])<1e-8
            checks.append(dict(check=sample+':scalar_geometry',n_points=len(selected),passed=True))
            score=ss.conditional.to_numpy();ident=dict(family=family,sample=sample,unit=ss.unit.iloc[0],n_total=len(ss))
            metrics=['conditional','raw','baseline_raw','baseline_conditional','patch_tissue_fraction','panel_counts','detected_genes']
            for distance in ['tissue_distance_um','image_distance_um','expression_extent_distance_um','sample_hull_distance_um']:
                for metric in metrics:
                    x=ss[metric].to_numpy();ok=np.isfinite(x)&np.isfinite(ss[distance]);rho=spearmanr(x[ok],ss.loc[ok,distance]).statistic if ok.sum()>2 and np.std(x[ok])>0 else np.nan
                    correlations.append(dict(**ident,distance=distance,metric=metric,spearman=rho))
                for band in [200,400]:
                    edge=ss[distance].to_numpy()<=band
                    for metric in metrics:
                        x=ss[metric].to_numpy();a=x[edge];b=x[~edge];sd=np.nanstd(x,ddof=1) if np.isfinite(x).any() else np.nan
                        if not np.isfinite(x).any():continue
                        effects.append(dict(**ident,distance=distance,band_um=band,metric=metric,n_edge=int(edge.sum()),n_interior=int((~edge).sum()),mean_edge=float(np.mean(a)) if len(a) else np.nan,mean_interior=float(np.mean(b)) if len(b) else np.nan,standardized_difference=float((np.mean(a)-np.mean(b))/sd) if min(len(a),len(b)) and sd>0 else np.nan))
                    for tail in [.1,.25]:
                        for group,use in [('upper',score>=np.quantile(score,1-tail)),('lower',score<=np.quantile(score,tail))]:
                            pe=use[edge].mean() if edge.any() else np.nan;pi=use[~edge].mean() if (~edge).any() else np.nan
                            tails.append(dict(**ident,distance=distance,band_um=band,tail=tail,group=group,n_edge=int(edge.sum()),n_interior=int((~edge).sum()),n_tail=int(use.sum()),edge_tail_rate=pe,interior_tail_rate=pi,relative_risk=pe/pi if pi>0 else np.nan,tail_fraction_in_edge=float(edge[use].mean())))
            full_moran,n_edges=moran(xy,score)
            for band in [200,400]:
                keep=(ss[['tissue_distance_um','image_distance_um','expression_extent_distance_um']]>band).all(axis=1).to_numpy();n=int(keep.sum())
                m,ne=moran(xy[keep],score[keep]) if n>=30 else (np.nan,0)
                record=dict(**ident,band_um=band,n_interior=n,fraction_retained=n/len(ss),moran_all=full_moran,moran_interior=m,n_edges_all=n_edges,n_edges_interior=ne,partial_image_patches=int((~ss.fully_on_slide).sum()),patch_fraction_below_095=int((ss.patch_tissue_fraction<.95).sum()))
                for tail,label in [(.1,'10'),(.25,'25')]:
                    for group,use in [('upper',score>=np.quantile(score,1-tail)),('lower',score<=np.quantile(score,tail))]:record[group+'_'+label+'_retained']=float(keep[use].mean())
                interior.append(record)
            checks.append(dict(check=sample+':edge_sum_vs_adjacency_Moran',passed=True))
            if sample in ['NCBI785','NCBI784','NCBI783']:figdata[sample]=(ss,thumb,scale,pixel,union,bounds,width,height)
            allspots.append(ss);print(sample,'boundary diagnostic complete',flush=True)
    pd.concat(allspots,ignore_index=True).to_parquet(OUT/'spot_boundary_diagnostics.parquet',index=False)
    for name,rows in [('edge_effects',effects),('tail_boundary_enrichment',tails),('interior_sensitivity',interior),('distance_correlations',correlations)]:pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    with PdfPages(OUT/'cell_linked_boundary_context.pdf') as pdf:
        for sample,(ss,thumb,scale,pixel,union,bounds,width,height) in figdata.items():
            fig,ax=plt.subplots(2,2,figsize=(11,9),layout='constrained');x=ss.x_um;y=ss.y_um
            ax[0,0].imshow(thumb);ax[0,0].set_title('H&E, saved tissue contour and input-grid extent')
            def contours(geom):
                if geom.geom_type=='Polygon':
                    yield np.asarray(geom.exterior.coords)
                    for ring in geom.interiors:yield np.asarray(ring.coords)
                elif hasattr(geom,'geoms'):
                    for g in geom.geoms:yield from contours(g)
            for ring in contours(union):ax[0,0].plot(ring[:,0]*scale,ring[:,1]*scale,color='#e69f00',lw=.7)
            x0,y0,x1,y1=bounds;ax[0,0].plot(np.array([x0,x1,x1,x0,x0])*scale,np.array([y0,y0,y1,y1,y0])*scale,color='#a846bd',lw=1)
            ax[0,0].set_xlim(0,thumb.shape[1]);ax[0,0].set_ylim(thumb.shape[0],0);ax[0,0].axis('off')
            score=ss.conditional.to_numpy();v=np.full(len(ss),'#dedede',dtype=object);v[score<=np.quantile(score,.1)]='#2879ad';v[score>=np.quantile(score,.9)]='#bd3935'
            ax[0,1].scatter(x/1000,y/1000,c=v,s=4,rasterized=True);ax[0,1].set_title('Fixed deciles: high red, low blue')
            for axis,key,title in [(ax[1,0],'tissue_distance_um','Signed distance to tissue-mask edge'),(ax[1,1],'expression_extent_distance_um','Distance to input-grid rectangle')]:
                art=axis.scatter(x/1000,y/1000,c=ss[key],s=4,cmap='viridis',vmin=0,vmax=1000,rasterized=True);axis.set_title(title);fig.colorbar(art,ax=axis,label='µm (clipped at 0 and 1000)')
            for axis in [ax[0,1],ax[1,0],ax[1,1]]:
                axis.set_aspect('equal');axis.invert_yaxis();axis.set_xlabel('Native x (mm)');axis.set_ylabel('Native y (mm)')
            fig.suptitle(f'{sample} ({ss.unit.iloc[0]}): tissue and acquisition boundaries differ\nGeometry uses saved HEST alignment; this is not independent anatomical validation')
            pdf.savefig(fig);fig.savefig(OUT/f'{sample}_boundary_context.png',dpi=150);plt.close(fig)
    (OUT/'checks.json').write_text(json.dumps(dict(status='pass',n_checks=len(checks),checks=checks),indent=2)+'\n')
    print('All boundary diagnostics complete',flush=True)


if __name__=='__main__':main()

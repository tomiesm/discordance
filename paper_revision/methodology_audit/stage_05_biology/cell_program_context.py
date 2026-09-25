"""Exploratory source-cell context for all eligible validation Hallmarks."""
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent


def main():
    cover=pd.read_csv(OUT/'coverage.csv');cover=cover[(cover.cohort=='10x_janesick')&cover.eligible]
    rows=[];checks=[]
    for sid in ['NCBI785','NCBI784','NCBI783']:
        a=ad.read_h5ad(ROOT/'paper_revision/experiments/emt_cells_v1/results'/sid/'measured_cells.h5ad')
        good=a.obs.qc_pass.to_numpy(dtype=bool);x=sparse.csr_matrix(a.X[good]);nuc=sparse.csr_matrix(a.layers['nuclear_counts'][good])
        labels=a.obs.loc[good,'source_group'].astype(str).to_numpy()
        for p in cover.itertuples():
            for scope,text in [('panel',p.genes),('common',p.common_genes if p.common_eligible else None)]:
                if text is None:continue
                genes=text.split(';');ix=a.var_names.get_indexer(genes);assert (ix>=0).all()
                total=np.asarray(x[:,ix].sum(axis=1)).ravel();nuclear=np.asarray(nuc[:,ix].sum(axis=1)).ravel()
                allcounts=total.sum();allnuclear=nuclear.sum()
                assigned=0
                for group in np.unique(labels):
                    use=labels==group;amount=total[use].sum();assigned+=amount
                    rows.append({'sample':sid,'patient':'P08' if sid=='NCBI783' else 'P07','pathway':p.pathway,'scope':scope,'n_genes':len(genes),
                        'source_group':group,'n_qc_cells':int(use.sum()),'fraction_of_qc_cells':float(use.mean()),
                        'program_transcripts':int(amount),'fraction_program_transcripts':float(amount/allcounts),
                        'mean_program_transcripts_per_cell':float(total[use].mean()),'fraction_cells_detecting_program':float((total[use]>0).mean()),
                        'nuclear_program_transcripts':int(nuclear[use].sum()),
                        'fraction_nuclear_program_transcripts':float(nuclear[use].sum()/allnuclear) if allnuclear else np.nan})
                assert assigned==allcounts
        checks.append({'sample':sid,'n_qc_cells':int(good.sum()),'all_program_genes_available':True,'source_labels_used_without_reclassification':True})
    pd.DataFrame(rows).to_csv(OUT/'source_cell_program_context.csv',index=False)
    effect=pd.read_csv(OUT.parent/'stage_06_inference/patient_effects.csv')
    shared=set.intersection(*(set(g.loc[g.eligible,'pathway']) for _,g in pd.read_csv(OUT/'coverage.csv').groupby('cohort')))
    variance=[]
    for outcome in ['observed','signed','absolute']:
        for adjustment in ['unadjusted','overlap_adjusted']:
            f=effect[(effect.outcome==outcome)&(effect.adjustment==adjustment)&effect.pathway.isin(shared)]
            matrix=f.pivot(index='pathway',columns='patient',values='standardized_estimate')
            assert matrix.shape==(len(shared),8) and matrix.notna().all().all()
            z=matrix.to_numpy();grand=z.mean();byprogram=z.mean(axis=1,keepdims=True)-grand;bypatient=z.mean(axis=0,keepdims=True)-grand
            remainder=z-grand-byprogram-bypatient;ss=np.square(z-grand).sum()
            pieces=[z.shape[1]*np.square(byprogram).sum(),z.shape[0]*np.square(bypatient).sum(),np.square(remainder).sum()]
            assert np.isclose(sum(pieces),ss)
            variance.append({'outcome':outcome,'adjustment':adjustment,'n_programs':len(shared),'n_patients':8,
                'program_fraction':float(pieces[0]/ss),'patient_fraction':float(pieces[1]/ss),'remainder_fraction':float(pieces[2]/ss),
                'scope':'descriptive additive decomposition, no causal interpretation or ANOVA test'})
    pd.DataFrame(variance).to_csv(OUT/'patient_program_variance.csv',index=False)
    (OUT/'cell_context_checks.json').write_text(json.dumps({'status':'pass','checks':checks},indent=2)+'\n')


if __name__=='__main__':main()

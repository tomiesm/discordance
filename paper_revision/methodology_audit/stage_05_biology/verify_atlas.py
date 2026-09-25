"""Independent spot-aligned arithmetic checks and inclusive-tail quartile audit."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
REPO=ROOT/'paper_revision/clean_repo'
OUT=Path(__file__).resolve().parent


def main():
    cov=pd.read_csv(OUT/'coverage.csv');contrasts=pd.read_csv(OUT/'section_contrasts.csv')
    checks=[];profiles=[]
    for cohort,cc in cov.groupby('cohort',sort=False):
        dest=OUT/'arrays'/cohort;loc=pd.read_csv(dest/'locations.csv')
        genes=json.loads((REPO/'data/v3'/f'gene_list_{cohort}.json').read_text())
        # Independent calculation from predictions rather than saved residuals.
        pred=[];y=[];ids=[]
        for fold in range(4):
            base=REPO/'outputs/predictions'/cohort
            ids+=json.loads((base/'uni/ridge'/f'fold{fold}'/'test_spot_ids.json').read_text())
            y.append(np.load(base/'uni/ridge'/f'fold{fold}'/'test_targets.npy'))
            pred.append(np.mean([np.load(base/e/'ridge'/f'fold{fold}'/'test_predictions.npy').astype(float) for e in ['uni','virchow2','hoptimus0']],axis=0))
        y=np.concatenate(y).astype(float);pred=np.concatenate(pred)
        assert loc.spot_id.tolist()==ids
        for row in cc[cc.eligible].itertuples():
            d=np.load(dest/f'{row.pathway}.npz')
            outcomes=[k for k in ['observed','signed','absolute','legacy_scaled_signed','common_observed','common_signed','common_absolute'] if k in d]
            members=row.genes.split(';');index=[genes.index(g) for g in members]
            if row.pathway=='HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION':
                w=np.zeros(len(genes));w[index]=1/len(index)
                for name,expected in [('observed',y@w),('signed',(y-pred)@w)]:
                    error=float(np.max(np.abs(expected-d[name])))
                    checks.append({'check':f'{cohort}/{name}:independent_target_prediction_arithmetic','max_abs':error,'pass':error<3e-6})
                other=[g for g in range(len(genes)) if g not in index]
                count=np.rint(np.expm1(y[:,other])).sum(axis=1)
                error=float(np.max(np.abs(count-d['outside_total_counts'])))
                checks.append({'check':f'{cohort}:outside_program_counts','max_abs':error,'pass':error==0})
            for sid,g in loc.groupby('sample_id',sort=False):
                ix=g.index.to_numpy()
                identity={'cohort':cohort,'sample':sid,'patient':g.patient.iloc[0],'pathway':row.pathway,'n_genes':row.n_genes,'n_common':row.n_common}
                for method in ['full_conditional','program_excluded_conditional','program_excluded_raw','program_excluded_section_centered']:
                    score=d[method][ix];a,b,c=np.quantile(score,[.25,.5,.75])
                    # Explicit sets agree with inclusive Q1/Q4 in the contrast tables.
                    masks=[score<=a,(score>a)&(score<=b),(score>b)&(score<c),score>=c]
                    assert np.all(np.stack(masks).sum(axis=0)==1)
                    ref=contrasts[(contrasts.cohort==cohort)&(contrasts['sample']==sid)&(contrasts.pathway==row.pathway)&(contrasts.grouping==method)&(contrasts.tail_fraction==.25)]
                    for outcome in outcomes:
                        ys=d[outcome][ix];rr=ref[ref.outcome==outcome].iloc[0]
                        delta=float(ys[masks[3]].astype(float).mean()-ys[masks[0]].astype(float).mean())
                        error=abs(delta-rr.delta)
                        checks.append({'check':f'{sid}/{row.pathway}/{method}/{outcome}:group_difference','max_abs':float(error),'pass':bool(error<1e-12)})
                        for q,mask in enumerate(masks,1):
                            profiles.append({**identity,'grouping':method,'outcome':outcome,'quartile':q,'n':int(mask.sum()),'mean':float(ys[mask].mean())})
        del y,pred
    assert all(c['pass'] for c in checks)
    # Correct only descriptive quartile profiles from the first run's boundary
    # convention. Q1/Q4 contrasts and group assignments were already inclusive.
    pd.DataFrame(profiles).to_csv(OUT/'quartile_profiles.csv',index=False)
    (OUT/'independent_checks.json').write_text(json.dumps({'status':'pass','checks':checks,
        'quartile_profile_correction':'Use inclusive Q1/Q4 endpoints matching contrasts; initial profiles assigned exact Q1 boundary to Q2. No contrast/model/grouping change.'},indent=2)+'\n')


if __name__=='__main__':main()

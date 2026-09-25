"""Exact common-member IDC effects for each external panel, with fixed IDC groups."""
from src.paper.paths import PROJECT_ROOT, STAGE_SOURCE, CELL_SOURCE, ANALYSIS_ROOT, CELL_ROOT, stage_dir, cell_dir
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
OUT = stage_dir('external')
AUDIT = ANALYSIS_ROOT
ROOT = PROJECT_ROOT
REPO = PROJECT_ROOT
from src.paper.cohort_data import load_cohort
from src.paper.genes.gene_audit import overlap_weights

def main():
    coverage = pd.read_csv(AUDIT / 'programs/coverage.csv')
    rows = []
    members = []
    checks = []
    extpanels = {f: set(json.loads((REPO / 'outputs' / f / 'gene_panel.json').read_text())) for f in ['coad', 'idc_visium']}
    for cohort in ['biomarkers', '10x_janesick']:
        y, r, loc, genes = load_cohort(cohort)
        sr = r.mean(axis=0)
        ae = np.abs(r).mean(axis=0)
        del r
        stored_locations = pd.read_csv(AUDIT / 'programs/arrays' / cohort / 'locations.csv')
        assert np.array_equal(loc.spot_id.to_numpy(), stored_locations.spot_id.to_numpy())
        checks.append(dict(check=cohort + ':archived_program_row_identity', passed=True))
        for row in coverage[(coverage.cohort == cohort) & coverage.eligible].itertuples():
            path = AUDIT / 'programs/arrays' / cohort / f'{row.pathway}.npz'
            d = np.load(path)
            assert len(d['observed']) == len(y)
            for family, panel in extpanels.items():
                common = [g for g in row.genes.split(';') if g in panel]
                members.append(dict(family=family, idc_cohort=cohort, pathway=row.pathway, n_common=len(common), genes=';'.join(common), eligible=len(common) >= 5))
                if len(common) < 5:
                    continue
                ji = [genes.index(g) for g in common]
                outcomes = np.stack([y[:, ji].mean(axis=1, dtype=float), sr[:, ji].mean(axis=1), ae[:, ji].mean(axis=1)], axis=1)
                for sample, ss in loc.groupby('sample_id', sort=False):
                    ix = ss.index.to_numpy()
                    score = d['program_excluded_conditional'][ix]
                    lo = score <= np.quantile(score, 0.25)
                    hi = score >= np.quantile(score, 0.75)
                    w1, w4, _, ret1, ret4 = overlap_weights(d['outside_total_counts'][ix], d['outside_detected_genes'][ix], lo, hi)
                    vals = outcomes[ix]
                    sd = np.sqrt(((lo.sum() - 1) * vals[lo].var(axis=0, ddof=1) + (hi.sum() - 1) * vals[hi].var(axis=0, ddof=1)) / (lo.sum() + hi.sum() - 2))
                    for adjustment, a, b in [('unadjusted', lo.astype(float), hi.astype(float)), ('overlap_adjusted', w1, w4)]:
                        effect = np.average(vals, weights=b, axis=0) - np.average(vals, weights=a, axis=0)
                        for k, outcome in enumerate(['observed', 'signed', 'absolute']):
                            rows.append(dict(family=family, idc_cohort=cohort, sample=sample, patient=ss.patient.iloc[0], pathway=row.pathway, n_common=len(common), outcome=outcome, adjustment=adjustment, effect=float(effect[k]), standardized_effect=float(effect[k] / sd[k]) if sd[k] > 0 else np.nan, Q1_retained=ret1, Q4_retained=ret4))
        print(cohort, 'common-member IDC comparisons complete', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'idc_common_member_section_effects.csv', index=False)
    pd.DataFrame(members).to_csv(OUT / 'common_member_coverage.csv', index=False)
    pat = df.groupby(['family', 'idc_cohort', 'patient', 'pathway', 'outcome', 'adjustment'])[['effect', 'standardized_effect']].mean().reset_index()
    pat.to_csv(OUT / 'idc_common_member_patient_effects.csv', index=False)
    pat.groupby(['family', 'idc_cohort', 'pathway', 'outcome', 'adjustment'])[['effect', 'standardized_effect']].mean().reset_index().to_csv(OUT / 'idc_common_member_cohort_effects.csv', index=False)
    (OUT / 'idc_comparator_checks.json').write_text(json.dumps(dict(status='pass', checks=checks), indent=2) + '\n')
if __name__ == '__main__':
    main()

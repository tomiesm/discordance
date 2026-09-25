"""Fixed-quartile biological contrasts after physical-boundary exclusion."""
from src.paper.paths import PROJECT_ROOT, STAGE_SOURCE, CELL_SOURCE, ANALYSIS_ROOT, CELL_ROOT, stage_dir, cell_dir
import sys
from src.paper.contrasts.common import *

def analyze_endpoint(loc, bound, values, score, count, det, endpoint, kind):
    rows = []
    for sample, ss in loc.groupby('sample', sort=False):
        ix = ss.index.to_numpy()
        v = values[ix]
        sd = v.std(axis=0, ddof=1)
        s = strata(count[ix], det[ix])
        b = bound.iloc[ix]
        for grouping, signal in [('full', ss.conditional.to_numpy()), ('excluded', score[ix])]:
            q1, q4 = tails(signal)
            for band in [0, 200, 400]:
                keep = np.ones(len(ss), bool) if band == 0 else (b[['tissue_distance_um', 'image_distance_um', 'expression_extent_distance_um']].to_numpy() > band).all(axis=1)
                lo = q1 & keep
                hi = q4 & keep
                n1 = int(lo.sum())
                n4 = int(hi.sum())
                w1, w4, ok = weights(s, lo, hi)
                modes = [('unadjusted', lo.astype(float), hi.astype(float))] + ([('overlap_adjusted', w1, w4)] if grouping == 'excluded' else [])
                for adjustment, a, c in modes:
                    nr1 = int(np.sum(a > 0))
                    nr4 = int(np.sum(c > 0))
                    estimable = min(n1, n4, nr1, nr4) >= 30 and min(a.sum(), c.sum()) > 0
                    low = np.average(v, weights=a, axis=0) if estimable else np.full(3, np.nan)
                    high = np.average(v, weights=c, axis=0) if estimable else np.full(3, np.nan)
                    d = high - low
                    if band == 200 and grouping == 'excluded' and (adjustment == 'overlap_adjusted') and estimable:
                        dif = []
                        hw = []
                        for j in np.flatnonzero(ok):
                            one = lo & (s == j)
                            four = hi & (s == j)
                            dif.append(v[four].mean(axis=0) - v[one].mean(axis=0))
                            hw.append(2 * one.sum() * four.sum() / (one.sum() + four.sum()))
                        check(f'{sample}/{endpoint}:direct_overlap', np.max(abs(d - np.average(dif, weights=hw, axis=0))))
                    for k, outcome in enumerate(OUTCOMES):
                        rows.append(dict(family=ss.family.iloc[0], sample=sample, unit=ss.unit.iloc[0], kind=kind, endpoint=endpoint, grouping=grouping, adjustment=adjustment, band_um=band, outcome=outcome, effect=d[k], fixed_section_sd=sd[k], fixed_standardized_effect=d[k] / sd[k] if sd[k] > 0 else np.nan, lower_mean=low[k], upper_mean=high[k], n_Q1=n1, n_Q4=n4, original_Q1=int(q1.sum()), original_Q4=int(q4.sum()), interior_fraction=keep.mean(), Q1_retained=n1 / q1.sum(), Q4_retained=n4 / q4.sum(), Q1_overlap_n=nr1, Q4_overlap_n=nr4, n_overlap_strata=int(ok.sum()), estimable=estimable))
    return rows

def summarize(d):
    keys = ['family', 'kind', 'endpoint', 'grouping', 'adjustment', 'outcome']
    pairkeys = keys + ['sample', 'unit']
    base = d[d.band_um.eq(0)][pairkeys + ['effect', 'fixed_standardized_effect', 'estimable']].rename(columns={'effect': 'full_effect', 'fixed_standardized_effect': 'full_standardized', 'estimable': 'full_estimable'})
    pairs = d[d.band_um.gt(0)].merge(base, on=pairkeys, validate='many_to_one')
    pairs['paired_estimable'] = pairs.estimable & pairs.full_estimable
    pairs.to_csv(OUT / 'boundary_paired_sections.csv', index=False)
    units = []
    summary = []
    for key, g in pairs.groupby(keys + ['band_um']):
        ident = dict(zip(keys + ['band_um'], key))
        ur = []
        for unit, z in g.groupby('unit'):
            valid = z[z.paired_estimable]
            row = ident | dict(unit=unit, n_sections_total=len(z), n_sections_paired=len(valid), n_Q1=int(valid.Q1_overlap_n.sum()), n_Q4=int(valid.Q4_overlap_n.sum()))
            for outcol, incol in [('full_effect', 'full_effect'), ('interior_effect', 'effect'), ('full_standardized', 'full_standardized'), ('interior_standardized', 'fixed_standardized_effect')]:
                row[outcol] = valid[incol].mean()
            units.append(row)
            ur.append(row)
        u = pd.DataFrame(ur)
        for scope in ['all', 'excluding_P07'] if ident['family'] == 'idc_visium' else ['all']:
            target = g[g.unit.ne('NCBI776')] if scope == 'excluding_P07' else g
            v = u[u.unit.ne('NCBI776')] if scope == 'excluding_P07' else u
            v = v.dropna(subset=['full_standardized', 'interior_standardized'])
            a = v.full_standardized.to_numpy()
            b = v.interior_standardized.to_numpy()
            delta = b - a
            n = len(b)
            summary.append(ident | dict(scope=scope, n_units_total=target.unit.nunique(), n_units_paired=n, n_sections_total=len(target), n_sections_paired=int(target.paired_estimable.sum()), full_effect=v.full_effect.mean(), interior_effect=v.interior_effect.mean(), full_standardized=np.mean(a) if n else np.nan, interior_standardized=np.mean(b) if n else np.nan, change_standardized=np.mean(delta) if n else np.nan, n_units_positive=int(np.sum(b > 0)), n_units_negative=int(np.sum(b < 0)), same_mean_direction=bool(np.mean(a) * np.mean(b) > 0) if n else False, loo_min=float(np.min((b.sum() - b) / (n - 1))) if n > 1 else np.nan, loo_max=float(np.max((b.sum() - b) / (n - 1))) if n > 1 else np.nan, Q1_median_retained=target.Q1_retained.median(), Q4_median_retained=target.Q4_retained.median()))
    pd.DataFrame(units).to_csv(OUT / 'boundary_unit_effects.csv', index=False)
    pd.DataFrame(summary).to_csv(OUT / 'boundary_cohort_summary.csv', index=False)

def main():
    boundary = parquet(AUDIT / 'boundary/spot_boundary_diagnostics.parquet').set_index('spot_id')
    rows = []
    for family in FAMILIES:
        loc = parquet(AUDIT / 'cutoffs' / family / 'locations.parquet').reset_index(drop=True)
        b = boundary.loc[loc.spot_id].reset_index(drop=True)
        check(f'{family}:score_alignment', np.max(abs(loc.conditional - b.conditional)))
        folder = AUDIT / ('programs/arrays/' + family if family in IDC else 'external/' + family + '/arrays')
        ck, dk = ('outside_total_counts', 'outside_detected_genes') if family in IDC else ('outside_counts', 'outside_detected')
        current = []
        for path in sorted(folder.glob('HALLMARK_*.npz')):
            with np.load(track(path)) as a:
                check(f'{family}/{path.stem}:row_alignment', np.max(abs(a['full_conditional'] - loc.conditional)))
                current.extend(analyze_endpoint(loc, b, np.stack([a[o].astype(float) for o in OUTCOMES], axis=1), a['program_excluded_conditional'], a[ck], a[dk], path.stem, 'program'))
        d = pd.DataFrame(current)
        ref = csv(AUDIT / 'cutoffs' / family / 'section_program_effects.csv')
        ref = ref[ref['tail'].eq(0.25)].rename(columns={'pathway': 'endpoint', 'effect': 'reference_effect', 'section_standardized_effect': 'reference_standardized'})
        ref['grouping'] = ref.grouping.replace({'program_excluded': 'excluded'})
        v = d[d.band_um.eq(0)].merge(ref, on=['family', 'sample', 'unit', 'endpoint', 'grouping', 'adjustment', 'outcome'], validate='one_to_one')
        check(f'{family}:full_effect_reproduction', np.nanmax(abs(v.effect - v.reference_effect)), 1e-09)
        check(f'{family}:full_scale_reproduction', np.nanmax(abs(v.fixed_standardized_effect - v.reference_standardized)), 1e-09)
        assert len(v) == len(d[d.band_um.eq(0)])
        rows.extend(current)
        pd.DataFrame(rows).to_csv(OUT / 'boundary_section_effects.csv', index=False)
        print(family, 'all programs complete', flush=True)
    from src.paper.cohort_data import load_cohort
    from src.discordance import compute_conditional_discordance
    coverage = csv(AUDIT / 'genes/marker_coverage.csv')
    for family in IDC:
        for path in (REPO / 'outputs/predictions' / family).glob('*/ridge/fold*/test_*'):
            if path.name in ['test_targets.npy', 'test_predictions.npy', 'test_spot_ids.json']:
                track(path)
        track(REPO / 'data/v3' / f'gene_list_{family}.json')
        track(AUDIT / 'scores/spot_score_diagnostics.parquet')
        y32, r, mloc, genes = load_cohort(family)
        y = y32.astype(float)
        sr = r.mean(axis=0)
        ae = np.abs(r).mean(axis=0)
        del r, y32
        loc = parquet(AUDIT / 'cutoffs' / family / 'locations.parquet')
        assert np.array_equal(loc.spot_id, mloc.spot_id)
        b = boundary.loc[loc.spot_id].reset_index(drop=True)
        count = np.rint(np.expm1(y))
        tc = count.sum(axis=1)
        td = (y > 0).sum(axis=1)
        ty = y.sum(axis=1)
        ta = ae.sum(axis=1)
        markers = sorted(coverage[coverage.cohort.eq(family) & coverage.measured].gene.unique())
        current = []
        for gene in markers:
            j = genes.index(gene)
            score = compute_conditional_discordance((ta - ae[:, j]) / (len(genes) - 1), ty - y[:, j])
            v = np.stack([y[:, j], sr[:, j], ae[:, j]], axis=1)
            current.extend(analyze_endpoint(loc, b, v, score, tc - count[:, j], td - (y[:, j] > 0), gene, 'marker'))
        d = pd.DataFrame(current)
        ref = csv(AUDIT / 'genes/section_effects.csv')
        ref = ref[ref.cohort.eq(family) & ref.gene.isin(markers) & ref.outcome.isin(['observed_log', 'signed', 'absolute'])].rename(columns={'gene': 'endpoint', 'effect': 'reference_effect'})
        ref['outcome'] = ref.outcome.replace({'observed_log': 'observed'})
        ref['grouping'] = ref.grouping.replace({'gene_excluded': 'excluded'})
        v = d[d.band_um.eq(0)].merge(ref, on=['sample', 'endpoint', 'grouping', 'adjustment', 'outcome'], validate='one_to_one')
        check(f'{family}:marker_full_reproduction', np.nanmax(abs(v.effect - v.reference_effect)), 1e-09)
        assert len(v) == len(d[d.band_um.eq(0)])
        rows.extend(current)
        pd.DataFrame(rows).to_csv(OUT / 'boundary_section_effects.csv', index=False)
        print(family, len(markers), 'registered markers complete', flush=True)
        del y, sr, ae, count
    summarize(pd.DataFrame(rows))
    finish('boundary')
    print('Boundary biology complete', flush=True)
if __name__ == '__main__':
    main()

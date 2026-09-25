"""Trace the existing boundary-ring control and quantify its score component."""
from src.paper.paths import PROJECT_ROOT, STAGE_SOURCE, CELL_SOURCE, ANALYSIS_ROOT, CELL_ROOT, stage_dir, cell_dir
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
OUT = stage_dir('boundary')
AUDIT = ANALYSIS_ROOT
REPO = PROJECT_ROOT
from src.spatial import assign_boundary_rings, _point_to_hull_distances, build_spatial_weights, morans_i
from src.paper.boundary.analyze import moran

def main():
    locations = pd.read_parquet(OUT / 'spot_boundary_diagnostics.parquet')
    old = json.loads((REPO / 'outputs/phase2/gate2_2_spatial.json').read_text())
    archived = {s: r for cohort in old['cohorts'].values() for s, r in cohort['samples'].items()}
    summary = []
    bands = []
    checks = []
    for sample, ss in locations.groupby('sample', sort=False):
        xy = ss[['x_um', 'y_um']].to_numpy()
        score = ss.conditional.to_numpy()
        family = ss.family.iloc[0]
        if family in ['biomarkers', '10x_janesick']:
            source = pd.read_parquet(REPO / 'outputs/phase2/scores' / family / f'{sample}_discordance.parquet').set_index('spot_id').loc[ss.spot_id]
            native = source[['x', 'y']].to_numpy()
            rings = assign_boundary_rings(native)
            observed = morans_i(score, build_spatial_weights(native, 6))
            err = abs(observed - archived[sample]['morans_i'])
            assert err < 1e-08
            checks.append(dict(check=sample + ':saved_ring_test_Moran_identity', max_abs=err, passed=True))
            oldp = archived[sample]['p_value']
        else:
            rings = assign_boundary_rings(xy)
            oldp = np.nan
        hull = ConvexHull(xy)
        distance = _point_to_hull_distances(xy, xy[hull.vertices])
        err = np.max(abs(distance - ss.sample_hull_distance_um))
        assert err < 1e-07
        checks.append(dict(check=sample + ':independent_hull_segment_distances', max_abs=float(err), passed=True))
        means = pd.Series(score).groupby(rings).transform('mean').to_numpy()
        residual = score - means
        total = np.sum((score - score.mean()) ** 2)
        explained = np.sum((means - score.mean()) ** 2) / total
        assert abs(explained - (1 - np.sum(residual ** 2) / total)) < 1e-10
        observed, _ = moran(xy, score)
        remaining, _ = moran(xy, residual)
        ident = dict(family=family, sample=sample, unit=ss.unit.iloc[0])
        summary.append(dict(**ident, n_rings=len(np.unique(rings)), score_variance_fraction_between_ring_means=explained, moran_radius150_original=observed, moran_radius150_after_ring_mean_removal=remaining, archived_k6_ring_permutation_p=oldp))
        for r in np.unique(rings):
            hit = rings == r
            bands.append(dict(**ident, ring=int(r), n=int(hit.sum()), mean_score=float(score[hit].mean()), mean_hull_distance_um=float(distance[hit].mean()), upper_decile_rate=float((score[hit] >= np.quantile(score, 0.9)).mean()), lower_decile_rate=float((score[hit] <= np.quantile(score, 0.1)).mean())))
    pd.DataFrame(summary).to_csv(OUT / 'ring_diagnostic.csv', index=False)
    pd.DataFrame(bands).to_csv(OUT / 'ring_profiles.csv', index=False)
    (OUT / 'ring_checks.json').write_text(json.dumps(dict(status='pass', n_checks=len(checks), checks=checks), indent=2) + '\n')
    print('Existing ring control traced and geometry verified')
if __name__ == '__main__':
    main()

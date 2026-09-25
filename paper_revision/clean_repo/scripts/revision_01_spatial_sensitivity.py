#!/usr/bin/env python3
"""Regenerate the previously reported spatial-neighbor sensitivity calculation."""

import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.spatial import build_spatial_weights, morans_i_permutation, assign_boundary_rings


def main():
    config = yaml.safe_load((REPO / 'config.yaml').read_text())
    output = REPO / config['output_dir']
    archive = REPO.parents[1] / 'outputs_v3/phase2/morans_i_k_sensitivity.csv'
    original = pd.read_csv(archive)
    settings = sorted(original.k.unique())
    gate = json.loads((output / 'phase2/gate2_2_spatial.json').read_text())
    columns = [f"D_cond_{encoder['name']}_ridge" for encoder in config['encoders']]
    records = []
    for cohort, cohort_config in config['cohorts'].items():
        for sid in cohort_config['samples']:
            df = pd.read_parquet(output / 'phase2/scores' / cohort_config['name'] / f'{sid}_discordance.parquet')
            coords = df[['x', 'y']].to_numpy()
            values = df[columns].mean(axis=1).to_numpy()
            valid = ~np.isnan(coords).any(axis=1)
            coords, values = coords[valid], values[valid]
            rings = assign_boundary_rings(coords)
            for k in settings:
                started = time.monotonic()
                if k == gate['n_neighbors']:
                    cached = gate['cohorts'][cohort]['samples'][sid]
                    observed, p = cached['morans_i'], cached['p_value']
                else:
                    weights = build_spatial_weights(coords, n_neighbors=int(k))
                    observed, p = morans_i_permutation(
                        values, weights, n_permutations=config['phase2']['spatial_n_permutations'],
                        permutation_groups=rings, seed=config['seed'])
                records.append({'cohort': cohort, 'sample': sid, 'k': int(k),
                                'morans_i': float(observed), 'p_value': float(p),
                                'n_spots': len(values), 'time_s': time.monotonic() - started})
            print(f'Completed {sid}: original k settings {settings}', flush=True)
    corrected = pd.DataFrame(records)
    assert len(corrected) == len(original) == 108
    joined = original.merge(corrected, on=['cohort', 'sample', 'k'], suffixes=('_old', '_corrected'), validate='one_to_one')
    assert len(joined) == 108
    corrected.to_csv(output / 'phase2/morans_i_k_sensitivity.csv', index=False)
    comparison = output / 'revision_01/comparison'
    comparison.mkdir(parents=True, exist_ok=True)
    joined.to_csv(comparison / 'spatial_neighbor_sensitivity.csv', index=False)
    summary = {str(k): {'n_sections': len(group), 'n_pass_p01': int((group.p_value < .01).sum()),
                        'min_morans_i': float(group.morans_i.min()), 'max_morans_i': float(group.morans_i.max())}
               for k, group in corrected.groupby('k')}
    (comparison / 'spatial_neighbor_sensitivity_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

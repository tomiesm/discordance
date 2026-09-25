#!/usr/bin/env python3
"""Regenerate the existing analyses after the ridge-only correction.

This preserves the original analytical definitions, including known limitations
deferred to later review points. It never writes into the submitted manuscript.
"""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / 'outputs'
AUDIT = OUTPUT / 'revision_01'
LOGS = AUDIT / 'logs'
LOGS.mkdir(parents=True, exist_ok=True)


def wait_for_refits(family, n_folds):
    while True:
        if family in ('biomarkers', '10x_janesick'):
            markers = [OUTPUT / 'predictions' / family / e / 'ridge' / f'fold{i}' / 'calibration.json'
                       for e in ['uni', 'virchow2', 'hoptimus0'] for i in range(n_folds)]
        else:
            markers = [OUTPUT / family / 'predictions' / e / f'fold{i}' / 'calibration.json'
                       for e in ['uni', 'virchow2', 'hoptimus0'] for i in range(n_folds)]
        if all(p.exists() for p in markers):
            return
        refit_log = LOGS / 'refit.log'
        if refit_log.exists() and 'Traceback (most recent call last)' in refit_log.read_text():
            raise RuntimeError('Refitting failed; see refit.log')
        time.sleep(10)


def run(script, *arguments):
    name = Path(script).stem
    marker = AUDIT / f'{name}.completed.json'
    if marker.exists():
        print(f'SKIP completed: {name}', flush=True)
        return
    command = [sys.executable, '-B', '-u', str(REPO / 'scripts' / script), *arguments]
    env = os.environ.copy()
    env.update({'OPENBLAS_NUM_THREADS': '4', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
                'MPLBACKEND': 'Agg', 'PYTHONDONTWRITEBYTECODE': '1'})
    started = time.monotonic()
    log = LOGS / f'{name}.stdout.log'
    print(f'START {name}', flush=True)
    with log.open('w') as handle:
        result = subprocess.run(command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
    text = log.read_text(errors='replace')
    if result.returncode or 'Traceback (most recent call last)' in text:
        raise RuntimeError(f'{name} failed: exit={result.returncode}; inspect {log}')
    marker.write_text(json.dumps({'command': command, 'seconds': time.monotonic() - started,
                                  'returncode': result.returncode, 'log': str(log)}, indent=2) + '\n')
    print(f'DONE {name}: {time.monotonic() - started:.1f}s', flush=True)


def batch(tasks):
    with ThreadPoolExecutor(max_workers=min(len(tasks), 4)) as pool:
        futures = [pool.submit(run, script, *args) for script, args in tasks]
        for future in futures:
            future.result()


def main():
    wait_for_refits('biomarkers', 4)
    wait_for_refits('10x_janesick', 4)
    run('04_discordance_scores.py')
    batch([(s, []) for s in ['05_multimodel_agreement.py', '06_spatial_structure.py', '07_dual_track.py']])
    run('revision_01_spatial_sensitivity.py')
    batch([(s, []) for s in ['08_de_analysis.py', '09_pathway_enrichment.py', '10_deconvolution.py',
                             '11_gene_predictability.py', '12_within_patient.py']])
    batch([(s, []) for s in ['13_heldout_validation.py', '14_encoder_consistency.py',
                             '15_bridge_gene_replication.py', '17_compute_figure_data.py']])
    batch([('17b_compute_reproducibility.py', []), ('22_interior_only_de.py', [])])
    wait_for_refits('coad', 4)
    wait_for_refits('idc_visium', 10)
    batch([('20_coad_generalization.py', ['--skip-embeddings', '--skip-predictions']),
           ('21_idc_visium_generalization.py', ['--skip-embeddings', '--skip-predictions'])])
    batch([(s, []) for s in ['18_main_figures.py', '19_supplementary_figures.py', '20_tables.py',
                             '16_summary_report.py']])
    run('23_figure1_schematic.py')
    print('All existing dependent analyses and figure/table generators completed.', flush=True)


if __name__ == '__main__':
    main()

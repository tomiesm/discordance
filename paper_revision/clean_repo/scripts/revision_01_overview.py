#!/usr/bin/env python3
"""Create a compact, exportable overview of the baseline correction."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'outputs/revision_01/comparison'
FAMILIES = ['biomarkers', '10x_janesick', 'coad', 'idc_visium']
LABELS = ['IDC discovery', 'IDC validation', 'COAD', 'IDC Visium']


def main():
    data = json.loads((ROOT / 'comparison_summary.json').read_text())
    assert data['status'] == 'complete'
    assert sum(v['n_models'] for v in data['calibration'].values()) == 66
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'savefig.dpi': 220})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    y = np.arange(4)
    a = [data['calibration'][f]['old']['mae'] for f in FAMILIES]
    b = [data['calibration'][f]['corrected']['mae'] for f in FAMILIES]
    axes[0].barh(y - .18, a, height=.32, color='#999999', label='Original')
    axes[0].barh(y + .18, b, height=.32, color='#0072B2', label='Corrected')
    axes[0].set(yticks=y, yticklabels=LABELS, xlabel='Held-out MAE (log1p expression)',
                title='A  Expression baseline restored')
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False)
    retention = [100 * data['discordance'][f]['median_old_q4_retained'] for f in FAMILIES]
    axes[1].barh(y, retention, height=.55, color='#0072B2')
    for i, v in enumerate(retention):
        axes[1].text(v + 1, i, f'{v:.1f}%', va='center', fontsize=8)
    axes[1].set(yticks=y, yticklabels=LABELS, xlim=(0, 100),
                xlabel='Original Q4 spots retained (%)', title='B  Discordant regions change')
    axes[1].invert_yaxis()
    axes[1].axvline(25, color='#888888', ls=':', lw=1)
    labels = []
    for i, (family, cell) in enumerate([(f, c) for f in FAMILIES[:2] for c in ['epithelial', 'macrophage']]):
        row = next(r for r in data['marker_scores'][family] if r['cell_type'] == cell)
        a, b = row['mean_cohens_d_old'], row['mean_cohens_d_corrected']
        axes[2].plot([a, b], [i, i], color='#bbbbbb', lw=2)
        axes[2].scatter(a, i, color='#999999', s=40, zorder=3)
        axes[2].scatter(b, i, color='#0072B2', s=40, zorder=3)
        labels.append(('Discovery' if family == 'biomarkers' else 'Validation') + '\n' + cell)
    axes[2].axvline(0, color='#888888', lw=1)
    axes[2].set(yticks=y, yticklabels=labels, xlabel="Mean Cohen's d (Q4 minus Q1)",
                title='C  Marker-score effects reverse')
    axes[2].invert_yaxis()
    fig.suptitle('Correction 01: ridge regression with a training-only intercept', fontsize=12)
    for extension in ['pdf', 'png']:
        fig.savefig(ROOT / f'correction_overview.{extension}', bbox_inches='tight')
    plt.close(fig)

    lines = ['# Correction 01: complete numerical overview', '',
             'These values preserve the existing analysis definitions. They do not resolve the deferred review points.', '',
             '![Baseline correction overview](clean_repo/outputs/revision_01/comparison/correction_overview.png)', '',
             'Q4 retention and rank correlations are medians across sections, using the average conditional-discordance score of the three ridge encoders. The dotted line in panel B marks 25% retention under independent quartile assignments.', '',
             '| Dataset | Models | Analyzed spots | MAE old → corrected | Positive residuals old → corrected | Median score rho | Median Q4 retained |',
             '|---|---:|---:|---|---|---:|---:|']
    for family, label in zip(FAMILIES, LABELS):
        cal, score = data['calibration'][family], data['discordance'][family]
        lines.append(f"| {label} | {cal['n_models']} | {cal['n_spots']:,} | "
                     f"{cal['old']['mae']:.4f} → {cal['corrected']['mae']:.4f} | "
                     f"{100*cal['old']['fraction_positive_residuals']:.2f}% → {100*cal['corrected']['fraction_positive_residuals']:.2f}% | "
                     f"{score['median_spearman_old_vs_corrected']:.4f} | {100*score['median_old_q4_retained']:.1f}% |")
    lines += ['', 'Complete per-fold and per-section data are in `clean_repo/outputs/revision_01/comparison/`. '
              'See `COMPUTATION_CORRECTION_01.md` for interpretation and manuscript impact.']
    (REPO.parent / 'COMPUTATION_CORRECTION_01_RESULTS.md').write_text('\n'.join(lines) + '\n')
    print('Saved correction overview PDF/PNG and numerical Markdown annex.')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Script 18: Generate main figures (Figures 1-6).

Usage:
    python scripts/18_main_figures.py                  # all figures
    python scripts/18_main_figures.py --figure 2       # just Figure 2
    python scripts/18_main_figures.py --figure 3 5     # Figures 3 and 5

Output:
    outputs/figures/main/fig{N}.pdf / .png
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.plotting import (
    COLORS, COHORT_LABELS, ENCODER_LABELS, LOC_COLORS,
    setup_style, save_figure, load_config, load_json,
    load_discordance_spatial, load_he_thumbnail, get_scalefactors,
    load_expression_for_gene, spatial_scatter, annotate_r,
    add_panel_label, cohort_color,
    FULL_WIDTH, HALF_WIDTH, COLUMN_1_5, MM_PER_INCH,
)
from src.utils import setup_logging, format_time

setup_style()


# ============================================================
# Figure 1: Study Design & Prediction Performance
# ============================================================

def figure1(config, out_dir, logger):
    """Generate Figure 1: Study design and prediction performance.

    Panels: 1b (H&E grid), 1c (Pearson distribution), 1d (heatmap), 1e (spatial map).
    Panel 1a is a manual schematic (skipped).
    """
    logger.info("Figure 1: Study Design & Prediction Performance")
    base = Path(config["output_dir"])

    # -- Panel 1b: H&E thumbnail grid --
    fig_b = _fig1b_he_grid(config, logger)
    save_figure(fig_b, out_dir / "fig1b_he_grid")

    # -- Panel 1c: Per-gene Pearson raincloud --
    fig_c = _fig1c_pearson_raincloud(config, logger)
    save_figure(fig_c, out_dir / "fig1c_pearson_distribution")

    # -- Panel 1d: Encoder x regressor heatmap --
    fig_d = _fig1d_heatmap(config, logger)
    save_figure(fig_d, out_dir / "fig1d_encoder_regressor_heatmap")

    # -- Panel 1e: Spatial prediction map --
    fig_e = _fig1e_spatial_prediction(config, logger)
    save_figure(fig_e, out_dir / "fig1e_spatial_prediction")

    logger.info("  Figure 1 complete.")


def _fig1b_he_grid(config, logger):
    """Grid of H&E thumbnails grouped by patient/cohort with patient brackets."""
    # Collect patient groups per cohort
    patient_groups = []
    for cohort_key in ["discovery", "validation"]:
        cc = config["cohorts"][cohort_key]
        for patient_id, patient_samples in cc["patient_mapping"].items():
            patient_groups.append({
                "patient_id": patient_id,
                "cohort_key": cohort_key,
                "samples": patient_samples,
            })

    disc_groups = [g for g in patient_groups if g["cohort_key"] == "discovery"]
    val_groups = [g for g in patient_groups if g["cohort_key"] == "validation"]
    ncols = max(sum(len(g["samples"]) for g in disc_groups),
                sum(len(g["samples"]) for g in val_groups))

    fig, axes = plt.subplots(2, ncols, figsize=(FULL_WIDTH, FULL_WIDTH * 0.32))

    bracket_info = []  # collect (row_idx, first_col, last_col, mid_col, patient_id)

    for row_idx, (groups, cohort_key) in enumerate(
            [(disc_groups, "discovery"), (val_groups, "validation")]):
        col = 0
        color = cohort_color(cohort_key)
        for group in groups:
            n_in_group = len(group["samples"])
            for si, sid in enumerate(group["samples"]):
                ax = axes[row_idx, col]
                img = load_he_thumbnail(sid, config)
                if img is not None:
                    # Crop black border rows/cols
                    gray = img.mean(axis=2)
                    row_mask = gray.mean(axis=1) > 10
                    col_mask = gray.mean(axis=0) > 10
                    if row_mask.any() and col_mask.any():
                        r0, r1 = np.where(row_mask)[0][[0, -1]]
                        c0, c1 = np.where(col_mask)[0][[0, -1]]
                        img = img[r0:r1+1, c0:c1+1]
                    ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color(color)
                    spine.set_linewidth(2)
                ax.set_title(sid, fontsize=5, color=color, fontweight='bold')
                col += 1

            mid_idx = col - n_in_group + n_in_group // 2
            bracket_info.append((row_idx, col - n_in_group, col - 1,
                                 mid_idx, group["patient_id"], n_in_group))

        # Hide unused columns
        for c in range(col, ncols):
            axes[row_idx, c].set_visible(False)

        # Cohort label
        axes[row_idx, 0].set_ylabel(
            "Discovery" if cohort_key == "discovery" else "Validation",
            fontsize=7, color=color, fontweight='bold', rotation=90, labelpad=8)

    fig.subplots_adjust(hspace=0.55, wspace=0.08)

    # Draw patient brackets and labels after layout is finalised so all
    # brackets in the same row share the exact same y position.
    from matplotlib.lines import Line2D as _Line2D
    tick_h = 0.008  # height of the small vertical ticks at bracket ends
    for row_idx in range(2):
        row_brackets = [b for b in bracket_info if b[0] == row_idx]
        if not row_brackets:
            continue
        # Consistent y for this row: lowest axes y0 minus offset
        row_y = axes[row_idx, 0].get_position().y0 - 0.04
        for _, first_col, last_col, mid_col, patient_id, n_in_group in row_brackets:
            # Patient label
            mid_ax = axes[row_idx, mid_col]
            mid_x = (mid_ax.get_position().x0 + mid_ax.get_position().x1) / 2
            fig.text(mid_x, row_y - 0.025, patient_id, ha='center', va='top',
                     fontsize=5, color='dimgray', fontstyle='italic',
                     transform=fig.transFigure)
            # Bracket line (horizontal bar + vertical ticks at ends)
            if n_in_group > 1:
                x0 = axes[row_idx, first_col].get_position().x0 + 0.01
                x1 = axes[row_idx, last_col].get_position().x1 - 0.01
                fig.add_artist(_Line2D(
                    [x0, x1], [row_y, row_y],
                    transform=fig.transFigure, color='dimgray',
                    linewidth=0.8, alpha=0.6))
                # Small vertical ticks at each end
                for xp in (x0, x1):
                    fig.add_artist(_Line2D(
                        [xp, xp], [row_y + tick_h, row_y],
                        transform=fig.transFigure, color='dimgray',
                        linewidth=0.8, alpha=0.6))

    return fig


def _fig1c_pearson_raincloud(config, logger):
    """Per-gene Pearson distribution as violin + strip plot."""
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, FULL_WIDTH * 0.35))

    for idx, cohort_key in enumerate(["discovery", "validation"]):
        ax = axes[idx]
        cohort_name = config["cohorts"][cohort_key]["name"]
        color = cohort_color(cohort_key)

        # Load per-gene Pearson from encoder consistency
        enc_path = (Path(config["output_dir"]) / "phase4" / "encoder_consistency" /
                    cohort_name / "per_gene_pearson_by_encoder.csv")
        df = pd.read_csv(enc_path)
        values = df["mean_pearson"].values

        # Violin
        parts = ax.violinplot(values, positions=[0], vert=True, showmedians=False,
                              showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.3)

        # Strip (jittered)
        jitter = np.random.RandomState(42).normal(0, 0.04, len(values))
        ax.scatter(jitter, values, s=2, alpha=0.4, color=color, zorder=3)

        # Boxplot
        bp = ax.boxplot(values, positions=[0], widths=0.15, patch_artist=True,
                        showfliers=False, zorder=4)
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.6)
        bp['medians'][0].set_color('white')
        bp['medians'][0].set_linewidth(1.5)

        # Label extremes — space labels to avoid overlap
        top3 = df.nlargest(3, "mean_pearson")
        bot3 = df.nsmallest(3, "mean_pearson")

        y_range = values.max() - values.min()
        min_gap = y_range * 0.04  # minimum vertical gap between labels

        def space_top(y_vals, min_gap):
            """Space labels downward from the highest value."""
            ys = sorted(y_vals, reverse=True)  # highest first
            out = [ys[0]]
            for y in ys[1:]:
                prev = out[-1]
                out.append(min(y, prev - min_gap))
            return dict(zip(ys, out))

        def space_bot(y_vals, min_gap):
            """Space labels upward from the lowest value."""
            ys = sorted(y_vals)  # lowest first
            out = [ys[0]]
            for y in ys[1:]:
                prev = out[-1]
                out.append(max(y, prev + min_gap))
            return dict(zip(ys, out))

        top_map = space_top(top3["mean_pearson"].tolist(), min_gap)
        for _, row in top3.iterrows():
            y_text = top_map[row["mean_pearson"]]
            ax.annotate(row["gene"], xy=(0.08, row["mean_pearson"]),
                        xytext=(0.18, y_text),
                        fontsize=5, fontweight='bold', color='darkgreen',
                        va='center',
                        arrowprops=dict(arrowstyle='->', color='darkgreen',
                                        lw=0.5))

        bot_map = space_bot(bot3["mean_pearson"].tolist(), min_gap)
        for _, row in bot3.iterrows():
            y_text = bot_map[row["mean_pearson"]]
            ax.annotate(row["gene"], xy=(0.08, row["mean_pearson"]),
                        xytext=(0.18, y_text),
                        fontsize=5, fontweight='bold', color='darkred',
                        va='center',
                        arrowprops=dict(arrowstyle='->', color='darkred',
                                        lw=0.5))

        mean_val = np.mean(values)
        ax.axhline(mean_val, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.text(0.3, mean_val, f'mean={mean_val:.3f}', fontsize=5, va='bottom', color='gray')

        ax.set_title(COHORT_LABELS[cohort_key], fontsize=7, color=color)
        ax.set_ylabel("Per-gene Pearson r" if idx == 0 else "")
        ax.set_xlim(-0.4, 0.55)
        ax.set_ylim(0, 1)
        ax.set_xticks([])

    fig.tight_layout()
    return fig


def _fig1d_heatmap(config, logger):
    """3x3 heatmap of mean Pearson per encoder-regressor combination."""
    base = Path(config["output_dir"])
    encoders = [e["name"] for e in config["encoders"]]
    regressors = [r["name"] for r in config["regressors"]]

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, FULL_WIDTH * 0.3))
    fig.subplots_adjust(right=0.88, wspace=0.35)

    for idx, cohort_key in enumerate(["discovery", "validation"]):
        ax = axes[idx]
        cohort_name = config["cohorts"][cohort_key]["name"]
        n_folds = config["cohorts"][cohort_key]["n_lopo_folds"]

        matrix = np.zeros((len(regressors), len(encoders)))
        for ri, reg in enumerate(regressors):
            for ei, enc in enumerate(encoders):
                fold_pearsons = []
                for fold in range(n_folds):
                    m_path = base / "predictions" / cohort_name / enc / reg / f"fold{fold}" / "metrics.json"
                    m = load_json(m_path)
                    if m:
                        fold_pearsons.append(m.get("__mean_pearson__", 0))
                matrix[ri, ei] = np.mean(fold_pearsons) if fold_pearsons else 0

        im = ax.imshow(matrix, cmap='Blues', aspect='auto',
                       vmin=0.58, vmax=0.78)
        # Annotate cells
        for ri in range(len(regressors)):
            for ei in range(len(encoders)):
                val = matrix[ri, ei]
                text_color = 'white' if val > 0.7 else 'black'
                ax.text(ei, ri, f'{val:.3f}', ha='center', va='center',
                        fontsize=6, color=text_color, fontweight='bold')

        ax.set_xticks(range(len(encoders)))
        ax.set_xticklabels([ENCODER_LABELS[e] for e in encoders], fontsize=5, rotation=30, ha='right')
        ax.set_yticks(range(len(regressors)))
        ax.set_yticklabels([r.title() for r in regressors], fontsize=6)
        ax.set_title(COHORT_LABELS[cohort_key], fontsize=7,
                     color=cohort_color(cohort_key))

    # Place colorbar in its own axis on the right
    cbar_ax = fig.add_axes([0.91, 0.2, 0.015, 0.6])
    fig.colorbar(im, cax=cbar_ax, label='Mean Pearson r')
    return fig


def _fig1e_spatial_prediction(config, logger):
    """Spatial prediction map: H&E, observed, predicted, residual for one sample/gene."""
    sample_id = "TENX193"
    gene_name = "CD8A"
    cohort_name = config["cohorts"]["discovery"]["name"]
    base = Path(config["output_dir"])

    fig, axes = plt.subplots(1, 4, figsize=(FULL_WIDTH, FULL_WIDTH * 0.22))
    fig.subplots_adjust(wspace=0.08)

    # H&E
    img = load_he_thumbnail(sample_id, config)
    if img is not None:
        axes[0].imshow(img)
    axes[0].set_title("H&E", fontsize=7)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    for sp in axes[0].spines.values():
        sp.set_visible(False)

    # Load spatial coords from discordance parquet
    disc_df = load_discordance_spatial(sample_id, cohort_name, config)
    x, y = disc_df["x"].values, disc_df["y"].values

    # Load expression (raw counts from h5ad) and apply log1p to match prediction space
    barcodes_expr, expr_vals = load_expression_for_gene(sample_id, gene_name, config)
    expr_map = dict(zip(barcodes_expr, expr_vals))
    observed = np.log1p(np.array([expr_map.get(b, 0) for b in disc_df["barcode"].values]))

    # Load predictions and residuals for this gene
    gene_list_path = Path(config["v3_data_dir"]) / f"gene_list_{cohort_name}.json"
    with open(gene_list_path) as f:
        gene_names = json.load(f)
    if gene_name in gene_names:
        gene_idx = gene_names.index(gene_name)
    else:
        gene_idx = 0

    # Aggregate predictions across folds (ridge, uni encoder)
    n_folds = config["cohorts"]["discovery"]["n_lopo_folds"]
    pred_map = {}
    resid_map = {}
    for fold in range(n_folds):
        fold_dir = base / "predictions" / cohort_name / "uni" / "ridge" / f"fold{fold}"
        preds = np.load(fold_dir / "test_predictions.npy")
        resids = np.load(fold_dir / "test_residuals.npy")
        with open(fold_dir / "test_spot_ids.json") as f:
            spots = json.load(f)
        prefix = f"{sample_id}_"
        for i, sid in enumerate(spots):
            if sid.startswith(prefix):
                barcode = sid[len(prefix):]
                pred_map[barcode] = preds[i, gene_idx]
                resid_map[barcode] = resids[i, gene_idx]

    predicted = np.array([pred_map.get(b, 0) for b in disc_df["barcode"].values])
    residual = np.array([resid_map.get(b, 0) for b in disc_df["barcode"].values])

    # Shared color scale for observed and predicted
    shared_vmin = np.percentile(np.concatenate([observed, predicted]), 2)
    shared_vmax = np.percentile(np.concatenate([observed, predicted]), 98)

    # Observed expression
    spatial_scatter(axes[1], x, y, observed, cmap='viridis', s=0.5,
                    title=f'{gene_name} (observed)',
                    vmin=shared_vmin, vmax=shared_vmax,
                    cbar_label='Expression (log1p counts)')

    # Predicted (same scale)
    r_val = np.corrcoef(observed, predicted)[0, 1]
    spatial_scatter(axes[2], x, y, predicted, cmap='viridis', s=0.5,
                    title=f'{gene_name} (predicted)',
                    vmin=shared_vmin, vmax=shared_vmax,
                    cbar_label='Expression (log1p counts)')
    axes[2].text(0.03, 0.03, f'r = {r_val:.2f}', transform=axes[2].transAxes,
                 fontsize=6, fontweight='bold', color='white',
                 bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.5, lw=0))

    # Residual: center around 0 to show spatial structure
    # (LOPO-CV causes systematic shift; centering reveals spatial patterns)
    residual_centered = residual - np.mean(residual)
    resid_abs_max = np.percentile(np.abs(residual_centered), 98)
    spatial_scatter(axes[3], x, y, residual_centered, cmap='RdBu_r', s=0.5,
                    vmin=-resid_abs_max, vmax=resid_abs_max,
                    title='Residual (centered)',
                    cbar_label='Over-/under-prediction')

    return fig


# ============================================================
# Figure 2: Discordance Validation
# ============================================================

def figure2(config, out_dir, logger):
    """Generate Figure 2: Discordance validation (Phase 2 gates).

    3 panels: a (agreement), b (spatial+Moran's), c (dual-track scatter).
    Former panel c (Moran's I bar chart) moved to supplementary.
    """
    logger.info("Figure 2: Discordance Validation")

    fig_ac = _fig2ac_agreement_and_dual_track(config, logger)
    save_figure(fig_ac, out_dir / "fig2ac_agreement_dual_track")

    fig2b_parts = _fig2b_spatial_morans(config, logger)
    for name, fig_part in fig2b_parts.items():
        save_figure(fig_part, out_dir / f"fig2b_{name}")

    logger.info("  Figure 2 complete.")


def _fig2ac_agreement_and_dual_track(config, logger):
    """Combined 1×3 panel: agreement dotplot + two dual-track scatters."""
    from matplotlib.lines import Line2D
    import matplotlib.gridspec as gridspec

    base = Path(config["output_dir"])
    gate1 = load_json(base / "phase2" / "gate2_1_agreement.json")
    threshold = gate1["threshold"]

    fig = plt.figure(figsize=(FULL_WIDTH, FULL_WIDTH * 0.38))
    gs = gridspec.GridSpec(1, 3, figure=fig,
                           width_ratios=[0.8, 1, 1], wspace=0.28)

    # --- Col 0: Agreement dotplot ---
    ax0 = fig.add_subplot(gs[0])

    cohort_samples = {}
    for cohort_key in ["discovery", "validation"]:
        cohort_data = gate1["cohorts"][cohort_key]
        samples = []
        for sid, sdata in cohort_data["samples"].items():
            samples.append({
                "sample_id": sid,
                "median_rho": sdata["median_rho"],
                "pass": sdata["pass"],
                "cohort": cohort_key,
            })
        samples.sort(key=lambda x: x["median_rho"])
        cohort_samples[cohort_key] = samples

    y_pos = 0
    yticks, ylabels = [], []
    group_ranges = {}

    for cohort_key in ["discovery", "validation"]:
        if cohort_key == "validation":
            y_pos += 1.5
        start_y = y_pos
        color = cohort_color(cohort_key)

        for s in cohort_samples[cohort_key]:
            marker = 'o' if s["pass"] else 'X'
            mfc = color if s["pass"] else 'white'
            mec = color
            ax0.plot(s["median_rho"], y_pos, marker=marker, color=mec,
                     markerfacecolor=mfc, markersize=4, markeredgewidth=1.0,
                     zorder=3)
            yticks.append(y_pos)
            ylabels.append(s["sample_id"])
            y_pos += 1

        group_ranges[cohort_key] = (start_y, y_pos - 1)

    for cohort_key, (y0, y1) in group_ranges.items():
        label = "Discovery" if cohort_key == "discovery" else "Validation"
        ax0.text(1.02, (y0 + y1) / 2, label, transform=ax0.get_yaxis_transform(),
                 fontsize=6, fontweight='bold', va='center', rotation=90,
                 color=cohort_color(cohort_key))

    ax0.axvline(threshold, color='gray', linestyle='--', linewidth=0.8, alpha=0.7,
                zorder=1)
    ax0.text(threshold - 0.01, y_pos - 0.3, f'threshold\n= {threshold}',
             fontsize=5, color='gray', va='top', ha='right')

    ax0.set_yticks(yticks)
    ax0.set_yticklabels(ylabels, fontsize=6)
    ax0.set_xlabel("Median pairwise Spearman ρ", fontsize=7)
    ax0.set_title("Multi model agreement gate", fontsize=7)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markeredgecolor='gray', label='Pass', markersize=4),
        Line2D([0], [0], marker='X', color='gray', markerfacecolor='white',
               markeredgecolor='gray', label='Fail', markersize=4, linestyle='None'),
    ]
    ax0.legend(handles=legend_elements, loc='lower right', fontsize=5, framealpha=0.8)

    # --- Cols 1-2: Dual-track scatters ---
    scatter_samples = {"discovery": "TENX193", "validation": "NCBI785"}
    scatter_axes = []

    for col_idx, (cohort_key, sample_id) in enumerate(scatter_samples.items()):
        ax = fig.add_subplot(gs[1 + col_idx])
        scatter_axes.append(ax)

        dt_path = base / "figure_data" / f"dual_track_spots_{sample_id}.csv"
        if not dt_path.exists():
            ax.text(0.5, 0.5, 'Pre-computation needed',
                    transform=ax.transAxes, ha='center', fontsize=7)
            continue

        dt_df = pd.read_csv(dt_path)

        quartiles = pd.qcut(dt_df["D_full"], 4, labels=False)
        colors_arr = np.array([COLORS['concordant']] * len(dt_df))
        colors_arr[quartiles == 1] = '#7BB3D9'
        colors_arr[quartiles == 2] = '#D9A07B'
        colors_arr[quartiles == 3] = COLORS['discordant']

        n_pts = len(dt_df)
        if n_pts > 8000:
            idx_sub = np.random.RandomState(42).choice(n_pts, 8000, replace=False)
        else:
            idx_sub = np.arange(n_pts)

        ax.scatter(dt_df["D_track_A"].values[idx_sub],
                   dt_df["D_track_B"].values[idx_sub],
                   c=colors_arr[idx_sub], s=0.3, alpha=0.3, rasterized=True)

        lims = [min(dt_df["D_track_A"].min(), dt_df["D_track_B"].min()),
                max(dt_df["D_track_A"].max(), dt_df["D_track_B"].max())]
        ax.plot(lims, lims, '--', color='gray', linewidth=0.6, alpha=0.6, zorder=1)

        rho, p = stats.spearmanr(dt_df["D_track_A"], dt_df["D_track_B"])
        annotate_r(ax, rho, pos='upper left', fontsize=6)

        ax.set_xlabel("Discordance (gene set A)", fontsize=6)
        ax.set_ylabel("Discordance (gene set B)", fontsize=6)
        ax.set_title(f'{sample_id} — {COHORT_LABELS[cohort_key]}', fontsize=7,
                     color=cohort_color(cohort_key))
        ax.set_aspect('equal')

    # Standardize scatter axis ranges
    all_lims = []
    for ax in scatter_axes:
        all_lims.extend([ax.get_xlim(), ax.get_ylim()])
    shared_min = min(l[0] for l in all_lims)
    shared_max = max(l[1] for l in all_lims)
    for ax in scatter_axes:
        ax.set_xlim(shared_min, shared_max)
        ax.set_ylim(shared_min, shared_max)
        ax.plot([shared_min, shared_max], [shared_min, shared_max],
                '--', color='gray', linewidth=0.6, alpha=0.6, zorder=1)

    # Quartile legend on last scatter
    q_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['concordant'],
               label='Q1 (most concordant)', markersize=4),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#7BB3D9',
               label='Q2', markersize=4),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#D9A07B',
               label='Q3', markersize=4),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['discordant'],
               label='Q4 (most discordant)', markersize=4),
    ]
    scatter_axes[1].legend(handles=q_legend, loc='lower right', fontsize=4.5,
                           framealpha=0.9, title='D quartile', title_fontsize=5)

    fig.subplots_adjust(left=0.06, right=0.98, top=0.87, bottom=0.12)

    # "Dual track gate" title spanning the two scatter panels
    pos1 = scatter_axes[0].get_position()
    pos2 = scatter_axes[1].get_position()
    mid_x = (pos1.x0 + pos2.x1) / 2
    fig.text(mid_x, 0.95, 'Dual track gate', ha='center', fontsize=8)

    return fig


def _fig2b_spatial_morans(config, logger):
    """Spatial maps + Moran's I null — returns dict of 2 figures (1×3 each)."""
    from matplotlib.colors import Normalize
    base = Path(config["output_dir"])
    representative = {"discovery": "TENX193", "validation": "NCBI785"}

    # Collect data first
    dcond_all = []
    disc_dfs = {}
    hist_data = {}
    for cohort_key, sample_id in representative.items():
        cohort_name = config["cohorts"][cohort_key]["name"]
        disc_df = load_discordance_spatial(sample_id, cohort_name, config)
        disc_dfs[(cohort_key, sample_id)] = disc_df
        dcond_all.append(disc_df["D_cond"].values)
        null_path = base / "figure_data" / f"morans_null_{sample_id}.npy"
        obs_path = base / "figure_data" / f"morans_obs_{sample_id}.json"
        if null_path.exists() and obs_path.exists():
            hist_data[sample_id] = {
                "null": np.load(null_path),
                "obs": load_json(obs_path),
            }
    dcond_all = np.concatenate(dcond_all)

    # Shared D_cond color scale using p2/p98 of combined data
    dcond_vmin = np.percentile(dcond_all, 2)
    dcond_vmax = np.percentile(dcond_all, 98)
    dcond_norm = Normalize(vmin=dcond_vmin, vmax=dcond_vmax)

    # Shared histogram x-axis — include full null range below 0
    hist_xmin, hist_xmax = np.inf, -np.inf
    for sd in hist_data.values():
        hist_xmin = min(hist_xmin, sd["null"].min())
        hist_xmax = max(hist_xmax, sd["obs"]["I_obs"])
    hist_xpad = (hist_xmax - hist_xmin) * 0.1
    hist_xlim = (hist_xmin - hist_xpad, hist_xmax + hist_xpad)

    cmap_dcond = plt.cm.RdBu_r

    figs = {}
    for cohort_key, sample_id in representative.items():
        color = cohort_color(cohort_key)
        disc_df = disc_dfs[(cohort_key, sample_id)]
        img_orig = load_he_thumbnail(sample_id, config)
        sf = get_scalefactors(sample_id, config)
        scalef = sf.get('tissue_downscaled_fullres_scalef', 0.02)

        # Rotate landscape images 90° CCW to portrait
        rotate = (img_orig is not None and img_orig.shape[1] > img_orig.shape[0])
        img = np.rot90(img_orig, k=1) if rotate else img_orig

        # Build D_cond canvas in ORIGINAL image space, then rotate
        orig_h, orig_w = (img_orig.shape[:2] if img_orig is not None
                          else (1000, 700))
        target_h = 2000
        upscale = target_h / max(orig_h, orig_w)
        canvas_h = int(round(orig_h * upscale))
        canvas_w = int(round(orig_w * upscale))
        px_x = (disc_df["x"].values * scalef * upscale).astype(int)
        px_y = (disc_df["y"].values * scalef * upscale).astype(int)
        canvas = np.ones((canvas_h, canvas_w, 4), dtype=np.float32)
        rgba_vals = cmap_dcond(dcond_norm(disc_df["D_cond"].values))
        spot_r = max(3, int(round(max(canvas_h, canvas_w) / 150)))
        for i in range(len(px_x)):
            yc = int(np.clip(px_y[i], 0, canvas_h - 1))
            xc = int(np.clip(px_x[i], 0, canvas_w - 1))
            canvas[max(0, yc-spot_r):min(canvas_h, yc+spot_r+1),
                   max(0, xc-spot_r):min(canvas_w, xc+spot_r+1)] = rgba_vals[i]
        if rotate:
            canvas = np.rot90(canvas, k=1)

        # Crop canvas to spot bounding box
        non_white = np.where(np.any(canvas[:, :, :3] != 1.0, axis=2))
        if len(non_white[0]) > 0:
            r_min, r_max = non_white[0].min(), non_white[0].max()
            c_min, c_max = non_white[1].min(), non_white[1].max()
            pad_px = max(spot_r * 2, 10)
            r_min = max(0, r_min - pad_px)
            r_max = min(canvas.shape[0], r_max + pad_px)
            c_min = max(0, c_min - pad_px)
            c_max = min(canvas.shape[1], c_max + pad_px)
            canvas = canvas[r_min:r_max, c_min:c_max]

        # --- 1×3 panel: [H&E | D_cond | Histogram] ---
        fig_w = FULL_WIDTH
        fig_h = FULL_WIDTH * 0.38
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = gridspec.GridSpec(1, 3, figure=fig,
                               width_ratios=[1, 1, 1.1], wspace=0.19)

        # Col 0: H&E
        ax_he = fig.add_subplot(gs[0])
        ax_he.imshow(img, aspect='auto')
        ax_he.set_title(f'{sample_id} — H&E', fontsize=7, color=color)
        ax_he.set_xticks([]); ax_he.set_yticks([])
        for sp in ax_he.spines.values():
            sp.set_visible(False)

        # Col 1: D_cond
        ax_dc = fig.add_subplot(gs[1])
        ax_dc.imshow(canvas, aspect='auto')
        ax_dc.set_title(f'{sample_id} — D_cond', fontsize=7, fontweight='bold')
        ax_dc.set_xticks([]); ax_dc.set_yticks([])
        for sp in ax_dc.spines.values():
            sp.set_visible(False)

        # Col 2: Histogram
        ax_hist = fig.add_subplot(gs[2])
        if sample_id in hist_data:
            sd = hist_data[sample_id]
            null_values = sd["null"]
            I_obs = sd["obs"]["I_obs"]
            p_val = sd["obs"]["p_value"]
            ax_hist.hist(null_values, bins=30, color=color, alpha=1.0,
                         edgecolor=color, linewidth=0.5, density=True)
            ax_hist.axvline(I_obs, color='red', linewidth=1.5, zorder=3)
            ax_hist.set_xlim(hist_xlim)
            nudge = (hist_xlim[1] - hist_xlim[0]) * 0.03
            ax_hist.text(I_obs - nudge, ax_hist.get_ylim()[1] * 0.9,
                         f'I = {I_obs:.3f}\np < {max(p_val, 0.001):.3f}',
                         fontsize=6, color='red', fontweight='bold',
                         va='top', ha='right')
            ax_hist.set_xlabel("Moran's I", fontsize=7)
            ax_hist.set_ylabel("Density", fontsize=7)
            ax_hist.set_title(f'{sample_id} — Permutation null',
                              fontsize=7, color=color)

        fig.subplots_adjust(left=0.01, right=0.98, top=0.90, bottom=0.08)

        # Vertical D_cond colorbar on the left side of D_cond axes
        pos_dc = ax_dc.get_position()
        cbar_h = pos_dc.height * 0.7
        cbar_y = pos_dc.y0 + (pos_dc.height - cbar_h) / 2
        cax = fig.add_axes([pos_dc.x0 - 0.02, cbar_y, 0.008, cbar_h])
        sc_map = plt.cm.ScalarMappable(norm=dcond_norm, cmap=cmap_dcond)
        fig.colorbar(sc_map, cax=cax, orientation='vertical')
        cax.yaxis.set_ticks_position('left')
        cax.yaxis.set_label_position('left')
        cax.set_ylabel('D_cond', fontsize=5, labelpad=2)
        cax.tick_params(labelsize=4)

        figs[sample_id] = fig

    return figs


# ============================================================
# Figure 3: Biological Characterization
# ============================================================

def figure3(config, out_dir, logger):
    """Generate Figure 3: Biological characterization of discordant spots.

    4 panels: a (spatial maps), b (cell type dumbbell), c (pathway heatmap),
    d (DE volcano). Former panels e/f moved to supplementary / merged into c.
    """
    logger.info("Figure 3: Biological Characterization")

    fig_a = _fig3a_spatial_biology(config, logger)
    save_figure(fig_a, out_dir / "fig3a_spatial_biology")

    fig_b = _fig3b_deconvolution_dumbbell(config, logger)
    save_figure(fig_b, out_dir / "fig3b_deconvolution_dumbbell")

    fig_c = _fig3c_pathway_heatmap(config, logger)
    save_figure(fig_c, out_dir / "fig3c_pathway_heatmap")

    fig_d = _fig3d_volcano(config, logger)
    save_figure(fig_d, out_dir / "fig3d_volcano")

    logger.info("  Figure 3 complete.")


def _fig3a_spatial_biology(config, logger):
    """Spatial maps: H&E, D_cond quartiles, epithelial score, macrophage score."""
    sample_id = "TENX193"
    cohort_name = config["cohorts"]["discovery"]["name"]
    base = Path(config["output_dir"])

    fig, axes = plt.subplots(1, 4, figsize=(FULL_WIDTH, FULL_WIDTH * 0.25))
    fig.subplots_adjust(wspace=0.08)

    # H&E — match Xenium coordinate orientation
    img = load_he_thumbnail(sample_id, config)
    if img is not None:
        axes[0].imshow(img)
        # Scale bar: 500 µm. Xenium pixel size = 0.2125 µm/pixel.
        # thumbnail scalef ≈ 0.0222 → 500 µm = 500/0.2125 * 0.0222 ≈ 52 px in thumbnail
        sf = get_scalefactors(sample_id, config)
        scalef = sf.get('tissue_downscaled_fullres_scalef', 0.0222)
        px_size_um = 0.2125  # Xenium default
        bar_um = 500
        bar_px = (bar_um / px_size_um) * scalef
        img_h, img_w = img.shape[:2]
        # Place in lower-right corner with white bar on dark background
        bar_y = img_h - 0.06 * img_h
        bar_x_end = img_w - 0.05 * img_w
        bar_x_start = bar_x_end - bar_px
        axes[0].plot([bar_x_start, bar_x_end], [bar_y, bar_y],
                     color='white', linewidth=2.5, solid_capstyle='butt')
        axes[0].text((bar_x_start + bar_x_end) / 2, bar_y - 0.015 * img_h,
                     f'{bar_um} \u00b5m', fontsize=4, color='white',
                     ha='center', va='bottom', fontweight='bold')
    axes[0].set_title("H&E", fontsize=8, fontweight='bold')
    axes[0].set_xticks([]); axes[0].set_yticks([])
    for sp in axes[0].spines.values():
        sp.set_visible(False)
    # Sample ID annotation
    axes[0].text(0.02, 0.02, f'{sample_id}\n(Patient P03, Discovery)',
                 transform=axes[0].transAxes, fontsize=4, va='bottom',
                 color='white', bbox=dict(boxstyle='round,pad=0.2',
                 facecolor='black', alpha=0.6))

    # D_cond quartiles with proper 4-color scheme
    disc_df = load_discordance_spatial(sample_id, cohort_name, config)
    quartiles = pd.qcut(disc_df["D_cond"], 4, labels=False)
    q_palette = {
        0: COLORS['concordant'],  # Q1: blue (most concordant)
        1: '#7BB3D9',             # Q2: light blue
        2: '#D9A07B',             # Q3: light red
        3: COLORS['discordant'],  # Q4: red (most discordant)
    }
    qcolors = np.array([q_palette[q] for q in quartiles])

    axes[1].scatter(disc_df["x"], disc_df["y"], c=qcolors, s=0.5,
                    alpha=0.6, edgecolors='none', rasterized=True)
    axes[1].set_aspect('equal')
    axes[1].invert_yaxis()
    axes[1].set_xticks([]); axes[1].set_yticks([])
    axes[1].set_title("Discordance quartiles", fontsize=8, fontweight='bold')
    for sp in axes[1].spines.values():
        sp.set_visible(False)

    # Quartile legend below the panel
    from matplotlib.lines import Line2D as _L2D
    q_legend = [
        _L2D([0], [0], marker='o', color='w', markerfacecolor=q_palette[0],
             label='Q1 (concordant)', markersize=4, markeredgewidth=0),
        _L2D([0], [0], marker='o', color='w', markerfacecolor=q_palette[1],
             label='Q2', markersize=4, markeredgewidth=0),
        _L2D([0], [0], marker='o', color='w', markerfacecolor=q_palette[2],
             label='Q3', markersize=4, markeredgewidth=0),
        _L2D([0], [0], marker='o', color='w', markerfacecolor=q_palette[3],
             label='Q4 (discordant)', markersize=4, markeredgewidth=0),
    ]
    axes[1].legend(handles=q_legend, loc='lower center', fontsize=4,
                   ncol=2, framealpha=0.9, handletextpad=0.3,
                   columnspacing=0.5, borderpad=0.3,
                   bbox_to_anchor=(0.5, -0.02))

    # Cell type scores — standardized colorbar height via explicit fraction/pad
    ct_path = base / "phase3" / "deconvolution" / cohort_name / "per_sample" / f"{sample_id}_celltype_scores.csv"
    if ct_path.exists():
        ct_df = pd.read_csv(ct_path)
        merged = disc_df.merge(ct_df, on="barcode", how="inner")

        for col_idx, (ct_name, cmap_name) in enumerate([("epithelial", "Blues"), ("macrophage", "Reds")]):
            if ct_name in merged.columns:
                spatial_scatter(axes[col_idx + 2], merged["x"], merged["y"],
                                merged[ct_name], cmap=cmap_name, s=0.5,
                                title=f'{ct_name.title()} score',
                                cbar_label='Signature score')
                # Make title consistent (bold)
                axes[col_idx + 2].set_title(f'{ct_name.title()} score',
                                            fontsize=8, fontweight='bold')
            else:
                axes[col_idx + 2].text(0.5, 0.5, f'{ct_name} N/A',
                                       transform=axes[col_idx + 2].transAxes, ha='center')
    else:
        for ax in axes[2:]:
            ax.text(0.5, 0.5, 'Data N/A', transform=ax.transAxes, ha='center')

    return fig


def _fig3b_deconvolution_dumbbell(config, logger):
    """Dumbbell plot: Cohen's d per cell type, discovery vs validation.

    Ordered by absolute effect size (epithelial at top). Missing cell types
    shown as 'n.a.' annotation. Significance indicated by filled vs open markers.
    """
    base = Path(config["output_dir"])

    fig, ax = plt.subplots(figsize=(HALF_WIDTH, HALF_WIDTH * 0.85))

    disc_path = base / "phase3" / "deconvolution" / config["cohorts"]["discovery"]["name"] / "celltype_summary.csv"
    val_path = base / "phase3" / "deconvolution" / config["cohorts"]["validation"]["name"] / "celltype_summary.csv"

    if not disc_path.exists() or not val_path.exists():
        ax.text(0.5, 0.5, 'Data N/A', transform=ax.transAxes, ha='center')
        return fig

    disc_ct = pd.read_csv(disc_path).set_index("cell_type")
    val_ct = pd.read_csv(val_path).set_index("cell_type")

    # Union of all cell types, ordered by max absolute d (largest at top)
    all_types = sorted(set(disc_ct.index) | set(val_ct.index))
    type_max_abs_d = {}
    for ct in all_types:
        d1 = abs(disc_ct.loc[ct, "mean_cohens_d"]) if ct in disc_ct.index else 0
        d2 = abs(val_ct.loc[ct, "mean_cohens_d"]) if ct in val_ct.index else 0
        type_max_abs_d[ct] = max(d1, d2)
    # Bottom of plot = smallest effect, top = largest effect
    all_types = sorted(all_types, key=lambda ct: type_max_abs_d[ct])

    for i, ct in enumerate(all_types):
        has_disc = ct in disc_ct.index
        has_val = ct in val_ct.index
        d_disc = disc_ct.loc[ct, "mean_cohens_d"] if has_disc else None
        d_val = val_ct.loc[ct, "mean_cohens_d"] if has_val else None

        # Connecting line (only if both available)
        if d_disc is not None and d_val is not None:
            ax.plot([d_disc, d_val], [i, i], color='gray', linewidth=0.8, zorder=1)

        # Discovery point — filled if >50% samples significant, open otherwise
        if d_disc is not None:
            n_sig = disc_ct.loc[ct, "n_sig_fdr05"]
            n_total = disc_ct.loc[ct, "n_samples"]
            frac_sig = n_sig / n_total if n_total > 0 else 0
            mfc = COLORS['discovery'] if frac_sig > 0.5 else 'white'
            ax.scatter(d_disc, i, color=COLORS['discovery'], s=35, zorder=3,
                       edgecolors=COLORS['discovery'], linewidth=1.0,
                       facecolors=mfc)
        else:
            # Mark as not available in this panel
            ax.text(-0.05, i, 'n.a.', fontsize=4, ha='right', va='center',
                    color=COLORS['discovery'], fontstyle='italic', alpha=0.7)

        # Validation point (diamond)
        if d_val is not None:
            n_sig = val_ct.loc[ct, "n_sig_fdr05"]
            n_total = val_ct.loc[ct, "n_samples"]
            frac_sig = n_sig / n_total if n_total > 0 else 0
            mfc = COLORS['validation'] if frac_sig > 0.5 else 'white'
            ax.scatter(d_val, i, marker='D', s=30, zorder=3,
                       edgecolors=COLORS['validation'], linewidth=1.0,
                       facecolors=mfc)
        else:
            ax.text(-0.05, i, 'n.a.', fontsize=4, ha='right', va='center',
                    color=COLORS['validation'], fontstyle='italic', alpha=0.7)

    ax.axvline(0, color='gray', linewidth=0.5, linestyle='-', alpha=0.5)
    ax.set_yticks(range(len(all_types)))
    ax.set_yticklabels([ct.replace('_', ' ').title() for ct in all_types], fontsize=6)
    ax.set_xlabel("Cohen's d (discordant vs concordant)", fontsize=7)
    ax.set_title("Cell type composition", fontsize=8, fontweight='bold')

    # Add direction labels
    ax.text(0.98, 0.02, 'Enriched in\ndiscordant \u2192', transform=ax.transAxes,
            fontsize=4.5, ha='right', va='bottom', color='gray', fontstyle='italic')
    ax.text(0.02, 0.02, '\u2190 Depleted in\ndiscordant', transform=ax.transAxes,
            fontsize=4.5, ha='left', va='bottom', color='gray', fontstyle='italic')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['discovery'],
               markeredgecolor=COLORS['discovery'], label='Discovery (sig.)', markersize=5),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor=COLORS['discovery'], label='Discovery (n.s.)', markersize=5),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=COLORS['validation'],
               markeredgecolor=COLORS['validation'], label='Validation (sig.)', markersize=5),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='white',
               markeredgecolor=COLORS['validation'], label='Validation (n.s.)', markersize=5),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=4.5,
              framealpha=0.9, handletextpad=0.3)

    fig.tight_layout()
    return fig


def _fig3c_pathway_heatmap(config, logger):
    """Heatmap: top pathways × cohorts, colored by signed Cohen's d.

    Shows top ~18 pathways by max absolute effect across cohorts.
    Significance stars for each cohort, reproducibility % as right annotation.
    Proper pathway name formatting.
    """
    base = Path(config["output_dir"])

    # Pathway name cleanup: MSigDB → readable format
    _pw_names = {
        'ADIPOGENESIS': 'Adipogenesis',
        'ALLOGRAFT_REJECTION': 'Allograft rejection',
        'ANDROGEN_RESPONSE': 'Androgen response',
        'ANGIOGENESIS': 'Angiogenesis',
        'APICAL_JUNCTION': 'Apical junction',
        'APICAL_SURFACE': 'Apical surface',
        'APOPTOSIS': 'Apoptosis',
        'BILE_ACID_METABOLISM': 'Bile acid metabolism',
        'CHOLESTEROL_HOMEOSTASIS': 'Cholesterol homeostasis',
        'COAGULATION': 'Coagulation',
        'COMPLEMENT': 'Complement',
        'DNA_REPAIR': 'DNA repair',
        'E2F_TARGETS': 'E2F targets',
        'EPITHELIAL_MESENCHYMAL_TRANSITION': 'EMT',
        'ESTROGEN_RESPONSE_EARLY': 'Estrogen response (early)',
        'ESTROGEN_RESPONSE_LATE': 'Estrogen response (late)',
        'FATTY_ACID_METABOLISM': 'Fatty acid metabolism',
        'G2M_CHECKPOINT': 'G2M checkpoint',
        'GLYCOLYSIS': 'Glycolysis',
        'HEDGEHOG_SIGNALING': 'Hedgehog signaling',
        'HEME_METABOLISM': 'Heme metabolism',
        'HYPOXIA': 'Hypoxia',
        'IL2_STAT5_SIGNALING': 'IL-2/STAT5 signaling',
        'IL6_JAK_STAT3_SIGNALING': 'IL-6/JAK/STAT3 signaling',
        'INFLAMMATORY_RESPONSE': 'Inflammatory response',
        'INTERFERON_ALPHA_RESPONSE': 'IFN-\u03b1 response',
        'INTERFERON_GAMMA_RESPONSE': 'IFN-\u03b3 response',
        'KRAS_SIGNALING_DN': 'KRAS signaling (down)',
        'KRAS_SIGNALING_UP': 'KRAS signaling (up)',
        'MITOTIC_SPINDLE': 'Mitotic spindle',
        'MTORC1_SIGNALING': 'mTORC1 signaling',
        'MYC_TARGETS_V1': 'MYC targets (v1)',
        'MYC_TARGETS_V2': 'MYC targets (v2)',
        'NOTCH_SIGNALING': 'Notch signaling',
        'OXIDATIVE_PHOSPHORYLATION': 'Oxidative phosphorylation',
        'P53_PATHWAY': 'p53 pathway',
        'PANCREAS_BETA_CELLS': 'Pancreas beta cells',
        'PEROXISOME': 'Peroxisome',
        'PI3K_AKT_MTOR_SIGNALING': 'PI3K/AKT/mTOR signaling',
        'PROTEIN_SECRETION': 'Protein secretion',
        'REACTIVE_OXYGEN_SPECIES_PATHWAY': 'ROS pathway',
        'SPERMATOGENESIS': 'Spermatogenesis',
        'TGF_BETA_SIGNALING': 'TGF-\u03b2 signaling',
        'TNFA_SIGNALING_VIA_NFKB': 'TNF-\u03b1/NF-\u03baB signaling',
        'UNFOLDED_PROTEIN_RESPONSE': 'Unfolded protein response',
        'UV_RESPONSE_DN': 'UV response (down)',
        'UV_RESPONSE_UP': 'UV response (up)',
        'WNT_BETA_CATENIN_SIGNALING': 'Wnt/\u03b2-catenin signaling',
        'XENOBIOTIC_METABOLISM': 'Xenobiotic metabolism',
    }

    def clean_pathway(name):
        key = name.replace('HALLMARK_', '')
        return _pw_names.get(key, key.replace('_', ' ').title())

    disc_pw = pd.read_csv(base / "phase3" / "pathways" / config["cohorts"]["discovery"]["name"] / "pathway_consistency.csv")
    val_pw = pd.read_csv(base / "phase3" / "pathways" / config["cohorts"]["validation"]["name"] / "pathway_consistency.csv")

    # Merge on pathway
    merged = disc_pw[["pathway", "mean_cohens_d", "reproducibility", "direction_consistency"]].rename(
        columns={"mean_cohens_d": "d_disc", "reproducibility": "repro_disc",
                 "direction_consistency": "dir_disc"}
    ).merge(
        val_pw[["pathway", "mean_cohens_d", "reproducibility", "direction_consistency"]].rename(
            columns={"mean_cohens_d": "d_val", "reproducibility": "repro_val",
                     "direction_consistency": "dir_val"}
        ),
        on="pathway", how="outer"
    )

    # Track which pathways exist in each cohort (NaN = not testable)
    merged["has_disc"] = merged["d_disc"].notna()
    merged["has_val"] = merged["d_val"].notna()

    # Compute max |d| for ranking, fill NaN for display
    merged["max_abs_d"] = merged[["d_disc", "d_val"]].abs().max(axis=1)
    merged = merged.fillna({"d_disc": 0, "d_val": 0, "repro_disc": 0,
                            "repro_val": 0, "dir_disc": 0, "dir_val": 0})

    # Top 18 pathways by max absolute effect size
    top_n = 18
    merged = merged.nlargest(top_n, "max_abs_d")
    # Sort by discovery d for visual coherence (ascending = blue at top, red at bottom)
    merged = merged.sort_values("d_disc", ascending=True).reset_index(drop=True)

    merged["pathway_clean"] = merged["pathway"].apply(clean_pathway)

    # Build heatmap matrix
    matrix = merged[["d_disc", "d_val"]].values

    # Color scale: all values are positive (enriched in discordant), use sequential
    vmax = np.ceil(matrix.max() * 10) / 10  # round up to nearest 0.1
    vmax = max(vmax, 0.5)  # minimum range

    fig, ax = plt.subplots(figsize=(HALF_WIDTH * 1.2, FULL_WIDTH * 0.42))

    im = ax.imshow(matrix, cmap='Reds', aspect='auto', vmin=0, vmax=vmax)

    ax.set_yticks(range(len(merged)))
    ax.set_yticklabels(merged["pathway_clean"].values, fontsize=5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Discovery", "Validation"], fontsize=6, fontweight='bold')

    # Significance stars per cell: reproducibility > 0.7 = *, > 0.9 = **
    for i, (_, row) in enumerate(merged.iterrows()):
        for j, (d_col, repro_col, has_col) in enumerate([
            ("d_disc", "repro_disc", "has_disc"),
            ("d_val", "repro_val", "has_val"),
        ]):
            if not row[has_col]:
                # Pathway not testable in this cohort — hatch the cell
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                             fill=False, hatch='///', edgecolor='gray',
                             linewidth=0.3, alpha=0.5))
            elif row[repro_col] > 0.9:
                ax.text(j, i, '**', ha='center', va='center', fontsize=6,
                        fontweight='bold', color='white' if row[d_col] > vmax * 0.5 else 'black')
            elif row[repro_col] > 0.7:
                ax.text(j, i, '*', ha='center', va='center', fontsize=7,
                        fontweight='bold', color='white' if row[d_col] > vmax * 0.5 else 'black')

    # Highlight pathways replicated in BOTH cohorts (repro > 0.7 in both, same sign)
    for i, (_, row) in enumerate(merged.iterrows()):
        both_sig = row["repro_disc"] > 0.7 and row["repro_val"] > 0.7
        same_dir = (np.sign(row["d_disc"]) == np.sign(row["d_val"])
                    if row["d_disc"] != 0 and row["d_val"] != 0 else False)
        if both_sig and same_dir:
            ax.plot(-0.7, i, marker='o', markersize=4, color='darkgreen',
                    clip_on=False, zorder=5)
    # Place "Replic." label at same y as column headers (aligned with xtick labels)
    ax.text(-0.7, -0.7, 'Replic.', ha='center', va='bottom', fontsize=4.5,
            fontweight='bold', color='darkgreen', clip_on=False)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.08, label="Cohen's d")
    cbar.ax.tick_params(labelsize=5)

    ax.set_title("Pathway enrichment in discordant spots", fontsize=8,
                 fontweight='bold', pad=10)

    # Legend for significance markers
    ax.text(1.0, -0.08, '** repro > 90%   * repro > 70%   /// not testable',
            transform=ax.transAxes, fontsize=4, ha='right', va='top', color='gray')

    fig.tight_layout()
    return fig


def _fig3d_volcano(config, logger):
    """Volcano plots of DE genes, one per cohort.

    Uses plain ASCII for axis labels to avoid font encoding issues.
    Standardized x-axis range across cohorts. Improved label spacing.
    """
    base = Path(config["output_dir"])

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, FULL_WIDTH * 0.38))

    # First pass: determine shared x-axis range AND shared y-cap
    all_fc = []
    all_neg_log = []
    cohort_dfs = {}
    for cohort_key in ["discovery", "validation"]:
        cohort_name = config["cohorts"][cohort_key]["name"]
        meta_path = base / "phase3" / "de" / cohort_name / "meta_de_unmatched.csv"
        if meta_path.exists():
            df = pd.read_csv(meta_path)
            df["neg_log_repro"] = -np.log10(
                1 - df["reproducibility"].clip(upper=0.9999) + 1e-10)
            all_fc.extend(df["median_log2fc"].values)
            all_neg_log.extend(df["neg_log_repro"].values)
            cohort_dfs[cohort_key] = df
    if all_fc:
        fc_lim = max(abs(np.percentile(all_fc, 1)), abs(np.percentile(all_fc, 99))) * 1.15
    else:
        fc_lim = 1.5
    # Shared y-cap across both panels for direct comparison
    y_cap = np.percentile(all_neg_log, 98) * 1.15 if all_neg_log else 5.0

    for idx, cohort_key in enumerate(["discovery", "validation"]):
        ax = axes[idx]

        if cohort_key not in cohort_dfs:
            ax.text(0.5, 0.5, 'Data N/A', transform=ax.transAxes, ha='center')
            continue
        df = cohort_dfs[cohort_key]

        # Add small jitter to genes at the y-ceiling to prevent dense band
        at_ceiling = df["neg_log_repro"] >= y_cap * 0.95
        n_at_ceil = at_ceiling.sum()
        if n_at_ceil > 0:
            jitter = np.random.RandomState(42).uniform(-0.08, 0.08, n_at_ceil) * y_cap
            df.loc[at_ceiling, "neg_log_repro"] = (
                df.loc[at_ceiling, "neg_log_repro"] + jitter)

        # Color categories
        sig_mask = df["reproducibility"] >= 0.5
        up_mask = df["median_log2fc"] > 0
        colors = np.full(len(df), COLORS['neutral'])
        colors[sig_mask & up_mask] = COLORS['discordant']
        colors[sig_mask & ~up_mask] = COLORS['concordant']

        ax.scatter(df["median_log2fc"], df["neg_log_repro"], c=colors,
                   s=4, alpha=0.6, edgecolors='none', rasterized=True)

        # Label top genes — use offset positioning to avoid overlap
        top_up = df[df["median_log2fc"] > 0].nlargest(5, "neg_log_repro")
        top_down = df[df["median_log2fc"] < 0].nlargest(5, "neg_log_repro")
        labeled = pd.concat([top_up, top_down])

        # Simple vertical staggering: sort by y, assign staggered text positions
        labeled = labeled.sort_values("neg_log_repro", ascending=False)
        used_y = []
        min_gap = y_cap * 0.07

        for _, row in labeled.iterrows():
            gene_x = row["median_log2fc"]
            gene_y = row["neg_log_repro"]
            # Find a text position that doesn't overlap
            text_y = gene_y
            for uy in used_y:
                if abs(text_y - uy) < min_gap:
                    text_y = uy - min_gap
            used_y.append(text_y)

            # Wider offset to prevent labels from clipping axes
            # Extra offset on negative side (0.16) to keep labels like OXTR clear of y-axis
            text_x = gene_x + (0.12 * fc_lim if gene_x > 0 else -0.16 * fc_lim)
            # Clamp text_x to stay inside plot area
            text_x = max(-fc_lim * 0.88, min(fc_lim * 0.92, text_x))
            ha = 'left' if gene_x > 0 else 'right'

            ax.annotate(row["gene"], xy=(gene_x, gene_y),
                        xytext=(text_x, max(0.1, text_y)),
                        fontsize=4.5, ha=ha, va='center',
                        arrowprops=dict(arrowstyle='-', lw=0.3, color='gray',
                                        shrinkA=0, shrinkB=2),
                        clip_on=True)

        ax.axvline(0, color='gray', linewidth=0.3, alpha=0.5)
        ax.set_xlabel(r"Median log$_2$FC (discordant vs concordant)", fontsize=6)
        if idx == 0:
            ax.set_ylabel("-log10(1 - reproducibility)", fontsize=6)
        ax.set_title(COHORT_LABELS[cohort_key], fontsize=7,
                     fontweight='bold', color=cohort_color(cohort_key))

        # Identical axis ranges across both panels
        ax.set_xlim(-fc_lim, fc_lim)
        ax.set_ylim(bottom=-0.1, top=y_cap)

        # Count annotations
        n_up = int((sig_mask & up_mask).sum())
        n_down = int((sig_mask & ~up_mask).sum())
        ax.text(0.98, 0.98, f'{n_up} up', transform=ax.transAxes, fontsize=5,
                ha='right', va='top', color=COLORS['discordant'])
        ax.text(0.02, 0.98, f'{n_down} down', transform=ax.transAxes, fontsize=5,
                ha='left', va='top', color=COLORS['concordant'])

    # Color legend (shared)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['discordant'],
               label='Up in discordant (repro > 50%)', markersize=4, markeredgewidth=0),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['concordant'],
               label='Down in discordant (repro > 50%)', markersize=4, markeredgewidth=0),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['neutral'],
               label='Not reproducible', markersize=4, markeredgewidth=0),
    ]
    axes[1].legend(handles=legend_elements, loc='center right', fontsize=4.5,
                   framealpha=0.9, handletextpad=0.3)

    fig.tight_layout()
    return fig


# ============================================================
# Figure 4: Gene Predictability
# ============================================================

def figure4(config, out_dir, logger):
    """Generate Figure 4: Determinants of gene predictability."""
    logger.info("Figure 4: Gene Predictability")

    fig_a = _fig4a_morans_vs_pearson(config, logger)
    save_figure(fig_a, out_dir / "fig4a_morans_vs_pearson")

    fig_b = _fig4b_coefficient_plot(config, logger)
    save_figure(fig_b, out_dir / "fig4b_ols_coefficients")

    fig_d = _fig4d_heldout_scatter(config, logger)
    save_figure(fig_d, out_dir / "fig4d_heldout_validation")

    logger.info("  Figure 4 complete.")


def _fig4a_morans_vs_pearson(config, logger):
    """Scatter: Moran's I vs Pearson, colored + shaped by localization.

    Shape encoding per localization for colorblind accessibility.
    Annotates both r and R^2. Labels extreme outliers in validation.
    """
    base = Path(config["output_dir"])

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, FULL_WIDTH * 0.38))

    # Shape encoding per localization category
    loc_markers = {
        'Membrane': 'o',
        'Cytoplasm': 's',
        'Nucleus': '^',
        'Secreted/Extracellular': 'D',
        'Unknown': 'v',
    }

    for idx, cohort_key in enumerate(["discovery", "validation"]):
        ax = axes[idx]
        cohort_name = config["cohorts"][cohort_key]["name"]

        gf = pd.read_csv(base / "phase3" / "gene_predictability" / cohort_name / "gene_features.csv")

        # Merge small categories into existing ones for plotting
        if "primary_localization" in gf.columns:
            gf["plot_loc"] = gf["primary_localization"].replace({
                'ER/Golgi': 'Cytoplasm', 'Mitochondria': 'Cytoplasm', 'Other': 'Unknown',
            })
            for loc in ['Membrane', 'Cytoplasm', 'Nucleus', 'Secreted/Extracellular', 'Unknown']:
                color = LOC_COLORS.get(loc, COLORS['neutral'])
                marker = loc_markers.get(loc, 'o')
                mask = gf["plot_loc"] == loc
                if mask.any():
                    ax.scatter(gf.loc[mask, "spatial_autocorrelation"],
                               gf.loc[mask, "mean_pearson"],
                               c=color, s=12, alpha=0.6, label=loc,
                               marker=marker, edgecolors='none')
        else:
            ax.scatter(gf["spatial_autocorrelation"], gf["mean_pearson"],
                       s=8, alpha=0.5, color=cohort_color(cohort_key))

        # OLS regression line
        valid = gf["spatial_autocorrelation"].notna() & gf["mean_pearson"].notna()
        if valid.sum() > 10:
            x = gf.loc[valid, "spatial_autocorrelation"]
            y = gf.loc[valid, "mean_pearson"]
            slope, intercept, r, p, se = stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, slope * x_line + intercept, 'k--', linewidth=0.8, alpha=0.5,
                    label='OLS fit')
            # Annotate r and R^2 — discovery upper-left (legend is lower-right),
            # validation lower-right (no legend)
            r2 = r**2
            if idx == 0:  # discovery — metrics bottom-right
                ax.text(0.95, 0.05, f'r = {r:.3f}\nR\u00b2 = {r2:.3f}',
                        transform=ax.transAxes, fontsize=5, ha='right', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  edgecolor='gray', alpha=0.8))
            else:  # validation
                ax.text(0.95, 0.05, f'r = {r:.3f}\nR\u00b2 = {r2:.3f}',
                        transform=ax.transAxes, fontsize=5, ha='right', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  edgecolor='gray', alpha=0.8))

        ax.set_xlabel("Spatial autocorrelation (Moran's I)", fontsize=6)
        ax.set_ylabel("Mean Pearson r", fontsize=6)
        ax.set_title(COHORT_LABELS[cohort_key], fontsize=7,
                     fontweight='bold', color=cohort_color(cohort_key))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Label outliers in validation panel
        if cohort_key == "validation" and "primary_localization" in gf.columns:
            for _, row in gf.nsmallest(2, "mean_pearson").iterrows():
                ax.annotate(row["gene"],
                            xy=(row["spatial_autocorrelation"], row["mean_pearson"]),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=4, color='gray', ha='left')

    # Shared legend on left panel (upper-left)
    axes[0].legend(fontsize=4, loc='upper left', framealpha=0.8, markerscale=1.2,
                   handletextpad=0.3, columnspacing=0.5)

    fig.tight_layout()
    return fig


def _fig4b_coefficient_plot(config, logger):
    """Forest plot: top OLS coefficients +/- 95% CI.

    Shows only the most interpretable predictors. Full model in Supplementary.
    Reference category stated: Cytoplasm (localization), Enzyme (function).
    Bold outline for significant (p<0.05), thin gray for non-significant.
    """
    base = Path(config["output_dir"])

    fig, ax = plt.subplots(figsize=(HALF_WIDTH * 1.1, HALF_WIDTH * 0.6))

    # Key predictors to display (ordered by interpretability)
    key_features = [
        'spatial_autocorrelation',
        'mean_expression',
        'cv_expression',
        'pathway_count',
        'loc_Secreted/Extracellular',
        'loc_Membrane',
        'loc_Nucleus',
        'loc_Unknown',
    ]

    # Human-readable labels with reference category noted
    feature_labels = {
        'spatial_autocorrelation': "Moran's I",
        'mean_expression': 'Mean expression',
        'cv_expression': 'Expression CV',
        'pathway_count': 'Pathway count',
        'loc_Secreted/Extracellular': 'Secreted/Extracellular',
        'loc_Membrane': 'Membrane',
        'loc_Nucleus': 'Nucleus',
        'loc_Unknown': 'Unknown localization',
    }

    cohort_data = {}
    for cohort_key in ["discovery", "validation"]:
        cohort_name = config["cohorts"][cohort_key]["name"]
        ols = load_json(base / "phase3" / "gene_predictability" / cohort_name / "ols_results.json")
        if ols:
            coefs = {c["feature"]: c for c in ols["coefficients"] if c["feature"] != "intercept"}
            cohort_data[cohort_key] = coefs

    # Filter to key features that exist in at least one cohort
    features = [f for f in key_features
                if any(f in cd for cd in cohort_data.values())]

    offsets = {"discovery": -0.15, "validation": 0.15}

    for cohort_key, coefs in cohort_data.items():
        color = cohort_color(cohort_key)
        offset = offsets[cohort_key]
        for i, feat in enumerate(features):
            if feat in coefs:
                c = coefs[feat]
                ci = 1.96 * c["std_error"]
                is_sig = c["p_value"] < 0.05
                ax.errorbar(c["coefficient"], i + offset, xerr=ci,
                            fmt='o', color=color, markersize=5 if is_sig else 3.5,
                            markeredgecolor='black' if is_sig else 'gray',
                            markeredgewidth=0.8 if is_sig else 0.3,
                            capsize=2, capthick=0.5, elinewidth=0.5,
                            alpha=1.0 if is_sig else 0.4,
                            zorder=5 if is_sig else 3)

    ax.axvline(0, color='gray', linewidth=0.5, linestyle='-', alpha=0.5)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels([feature_labels.get(f, f) for f in features], fontsize=5.5)
    ax.set_xlabel("Standardized \u03b2 \u00b1 95% CI", fontsize=7)
    ax.set_title("Gene predictability: OLS coefficients", fontsize=8, fontweight='bold')

    # Separator line between continuous and categorical predictors
    ax.axhline(3.5, color='lightgray', linewidth=0.5, linestyle='-', alpha=0.6)
    ax.text(0.98, 0.62, 'Localization\n(ref: Cytoplasm)',
            transform=ax.transAxes,
            fontsize=4, ha='right', va='bottom', color='gray', style='italic')

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['discovery'],
               markeredgecolor='black', markeredgewidth=0.8,
               label='Disc. (p < 0.05)', markersize=5),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['discovery'],
               markeredgecolor='gray', markeredgewidth=0.3,
               label='Disc. (n.s.)', markersize=3.5, alpha=0.4),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['validation'],
               markeredgecolor='black', markeredgewidth=0.8,
               label='Val. (p < 0.05)', markersize=5),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['validation'],
               markeredgecolor='gray', markeredgewidth=0.3,
               label='Val. (n.s.)', markersize=3.5, alpha=0.4),
    ]
    ax.legend(handles=legend_elements, fontsize=4.5, loc='lower left', framealpha=0.9)

    fig.tight_layout()
    return fig


def _fig4d_heldout_scatter(config, logger):
    """Scatter: predicted vs observed Pearson (5-fold gene CV within each cohort).

    Single color per cohort (fold distinction adds noise). Standardized 0-1 axes.
    Clarifies that this is within-cohort gene-level cross-validation.
    """
    base = Path(config["output_dir"])

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, FULL_WIDTH * 0.48))

    for idx, cohort_key in enumerate(["discovery", "validation"]):
        ax = axes[idx]
        cohort_name = config["cohorts"][cohort_key]["name"]

        cv_path = base / "phase4" / "heldout_validation" / cohort_name / "cv_predictions.csv"
        if not cv_path.exists():
            continue
        cv = pd.read_csv(cv_path)

        # Single cohort color — fold coloring adds visual noise without information
        color = cohort_color(cohort_key)
        ax.scatter(cv["predicted_pearson"], cv["actual_pearson"],
                   s=8, alpha=0.5, color=color, edgecolors='none', rasterized=True)

        # Identity line (0 to 1)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.5, alpha=0.3)

        r, p = stats.pearsonr(cv["predicted_pearson"], cv["actual_pearson"])
        rho, _ = stats.spearmanr(cv["predicted_pearson"], cv["actual_pearson"])
        # Show Pearson r and Spearman rho
        ax.text(0.05, 0.95, f'r = {r:.3f}\n\u03c1 = {rho:.3f}',
                transform=ax.transAxes, fontsize=5, ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='gray', alpha=0.8))

        ax.set_xlabel("Predicted Pearson r", fontsize=6)
        ax.set_ylabel("Observed Pearson r", fontsize=6)
        ax.set_title(f"{COHORT_LABELS[cohort_key]}: 5-fold gene CV",
                     fontsize=7, fontweight='bold', color=color)

        # Standardized axes
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Annotation: n folds, n genes
        n_folds = cv["fold"].nunique()
        n_genes = len(cv)
        ax.text(0.95, 0.05, f'{n_genes} genes\n{n_folds}-fold CV',
                transform=ax.transAxes, fontsize=4.5, ha='right', va='bottom',
                color='gray')

    fig.subplots_adjust(left=0.08, right=0.98, wspace=0.18)
    return fig


# ============================================================
# Figure 5: Within-Patient & Cross-Cohort
# ============================================================

def figure5(config, out_dir, logger):
    """Generate Figure 5: Within-patient reproducibility and cross-cohort validation.

    3 panels: a (gene scatter), c (pathway correlation), d (bridge gene scatter).
    Combined layout in fig5_combined_acd.
    """
    logger.info("Figure 5: Within-Patient & Cross-Cohort")

    fig_a = _fig5a_gene_scatter(config, logger)
    save_figure(fig_a, out_dir / "fig5a_within_patient_gene_scatter")

    fig_c = _fig5c_pathway_correlation(config, logger)
    save_figure(fig_c, out_dir / "fig5c_pathway_correlation")

    fig_d = _fig5d_bridge_gene_scatter(config, logger)
    save_figure(fig_d, out_dir / "fig5d_bridge_gene_scatter")

    # Combined 2x2: top = 5a (two scatters), bottom = 5c + 5d
    fig_combined = _fig5_combined_acd(config, logger)
    if fig_combined is not None:
        save_figure(fig_combined, out_dir / "fig5_combined_acd")

    logger.info("  Figure 5 complete.")


def _fig5a_gene_scatter(config, logger):
    """Scatter: per-gene mean |residual| section A vs B.

    Shows P02 (best, r=0.999, validation) and P05 (worst, r=0.822, discovery)
    to demonstrate the range of within-patient reproducibility.
    """
    base = Path(config["output_dir"])

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, FULL_WIDTH * 0.42))
    # Best and worst patients for honest representation
    patients = [
        ("P02", "TENX97", "TENX95", "validation"),
        ("P05", "TENX199", "TENX198", "discovery"),
    ]

    wp_path = base / "figure_data" / "within_patient_per_gene.csv"
    if not wp_path.exists():
        for ax in axes:
            ax.text(0.5, 0.5, 'Pre-computation needed', transform=ax.transAxes, ha='center')
        return fig

    wp_df = pd.read_csv(wp_path)

    for idx, (patient_id, sec_a, sec_b, coh_key) in enumerate(patients):
        ax = axes[idx]
        color = cohort_color(coh_key)
        a_data = wp_df[wp_df["sample_id"] == sec_a].set_index("gene")
        b_data = wp_df[wp_df["sample_id"] == sec_b].set_index("gene")

        common = sorted(set(a_data.index) & set(b_data.index))
        x = a_data.loc[common, "mean_abs_residual"].values
        y = b_data.loc[common, "mean_abs_residual"].values
        genes = np.array(common)

        ax.scatter(x, y, s=4, alpha=0.5, color=color, edgecolors='none')

        # Identity line
        lims = [0, max(x.max(), y.max()) * 1.05]
        ax.plot(lims, lims, 'k--', linewidth=0.5, alpha=0.3)

        r, p = stats.pearsonr(x, y)
        # P02 (tight diagonal): upper left. P05 (spread out): upper right
        stat_pos = 'upper left' if patient_id == "P02" else 'upper right'
        annotate_r(ax, r, p, pos=stat_pos, fontsize=6)

        ax.set_xlabel(f'Mean |residual| ({sec_a})', fontsize=6)
        ax.set_ylabel(f'Mean |residual| ({sec_b})', fontsize=6)
        ax.set_title(f'{patient_id} ({COHORT_LABELS[coh_key]})',
                     fontsize=7, color=color)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        # Label outlier genes deviating from diagonal (for P05, the worse patient)
        if patient_id == "P05":
            resid = np.abs(x - y)
            top_idx = np.argsort(resid)[-4:]  # 4 largest deviations
            for ti in top_idx:
                ax.annotate(genes[ti], xy=(x[ti], y[ti]),
                            xytext=(5, -5), textcoords='offset points',
                            fontsize=4, color='gray', ha='left', va='top')

    add_panel_label(axes[0], 'a')
    fig.tight_layout()
    return fig


def _fig5c_pathway_correlation(config, logger):
    """Boxplot: within-patient vs between-patient pathway profile correlation.

    Shows that pathway enrichment profiles (Cohen's d across pathways) are
    more similar within-patient than between-patient, demonstrating
    pathway-level reproducibility of discordance.
    """
    base = Path(config["output_dir"])
    repro_dir = base / "figure_data" / "reproducibility"

    pairwise_path = repro_dir / "pathway_repro_pairwise.csv"
    summary_path = repro_dir / "pathway_repro_summary.json"

    if not pairwise_path.exists():
        fig, ax = plt.subplots(figsize=(HALF_WIDTH, HALF_WIDTH * 0.8))
        ax.text(0.5, 0.5, 'Pre-computation needed',
                transform=ax.transAxes, ha='center', va='center')
        return fig

    pairwise_df = pd.read_csv(pairwise_path)
    pw_summary = load_json(summary_path)

    fig, ax = plt.subplots(figsize=(HALF_WIDTH, HALF_WIDTH * 0.8))

    within = pairwise_df[pairwise_df["is_within_patient"]]["pearson_r"].values
    between = pairwise_df[~pairwise_df["is_within_patient"]]["pearson_r"].values

    bp = ax.boxplot([within, between], positions=[0, 1], widths=0.5,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='black', linewidth=1))
    bp['boxes'][0].set_facecolor(COLORS['concordant'])
    bp['boxes'][0].set_alpha(0.3)
    bp['boxes'][1].set_facecolor(COLORS['neutral'])
    bp['boxes'][1].set_alpha(0.3)

    for pos_idx, (data, color) in enumerate([
        (within, COLORS['concordant']),
        (between, COLORS['neutral']),
    ]):
        jitter = np.random.RandomState(42).uniform(-0.12, 0.12, len(data))
        ax.scatter(np.full_like(data, pos_idx) + jitter, data, s=12,
                   alpha=0.6, color=color, edgecolors='none', zorder=3)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Within-\npatient', 'Between-\npatient'], fontsize=6)
    ax.set_ylabel('Pathway profile Pearson r', fontsize=7)
    ax.set_title('Pathway enrichment reproducibility', fontsize=7, fontweight='bold')
    ax.tick_params(labelsize=5)

    if pw_summary:
        p = pw_summary.get("mannwhitney_p")
        rb = pw_summary.get("rank_biserial")
        if p is not None:
            p_str = f'p = {p:.4f}' if p >= 0.001 else 'p < 0.001'
            rb_str = f'r_b = {rb:.2f}' if rb is not None else ''
            ax.text(0.5, 0.95, f'MW {p_str}\n{rb_str}',
                    transform=ax.transAxes, fontsize=5, ha='center', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='gray', alpha=0.9))

    add_panel_label(ax, 'c')
    fig.tight_layout()
    return fig


def _fig5_combined_acd(config, logger):
    """Combined 2x2 figure: top row = 5a (two gene scatters), bottom row = 5c + 5d."""
    base = Path(config["output_dir"])

    fig = plt.figure(figsize=(FULL_WIDTH, FULL_WIDTH * 0.85))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30)

    # --- Top row: 5a — within-patient gene scatter (P02 and P05) ---
    wp_path = base / "figure_data" / "within_patient_per_gene.csv"
    if not wp_path.exists():
        return None
    wp_df = pd.read_csv(wp_path)

    patients = [
        ("P02", "TENX97", "TENX95", "validation"),
        ("P05", "TENX199", "TENX198", "discovery"),
    ]
    for idx, (patient_id, sec_a, sec_b, coh_key) in enumerate(patients):
        ax = fig.add_subplot(gs[0, idx])
        color = cohort_color(coh_key)
        a_data = wp_df[wp_df["sample_id"] == sec_a].set_index("gene")
        b_data = wp_df[wp_df["sample_id"] == sec_b].set_index("gene")
        common = sorted(set(a_data.index) & set(b_data.index))
        x = a_data.loc[common, "mean_abs_residual"].values
        y = b_data.loc[common, "mean_abs_residual"].values
        genes = np.array(common)

        ax.scatter(x, y, s=4, alpha=0.5, color=color, edgecolors='none')
        lims = [0, max(x.max(), y.max()) * 1.05]
        ax.plot(lims, lims, 'k--', linewidth=0.5, alpha=0.3)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        r, p = stats.pearsonr(x, y)
        stat_pos = 'upper left' if patient_id == "P02" else 'upper right'
        annotate_r(ax, r, p, pos=stat_pos, fontsize=5)

        ax.set_xlabel(f'Mean |residual| ({sec_a})', fontsize=6)
        ax.set_ylabel(f'Mean |residual| ({sec_b})', fontsize=6)
        ax.set_title(f'{patient_id} ({COHORT_LABELS[coh_key]})',
                     fontsize=7, color=color)

        if patient_id == "P05":
            resid = np.abs(x - y)
            top_idx = np.argsort(resid)[-4:]
            for ti in top_idx:
                ax.annotate(genes[ti], xy=(x[ti], y[ti]),
                            xytext=(5, -5), textcoords='offset points',
                            fontsize=4, color='gray', ha='left', va='top')

    # --- Bottom left: 5c — pathway correlation boxplot ---
    repro_dir = base / "figure_data" / "reproducibility"
    pairwise_path = repro_dir / "pathway_repro_pairwise.csv"
    summary_path = repro_dir / "pathway_repro_summary.json"

    ax_c = fig.add_subplot(gs[1, 0])
    if pairwise_path.exists():
        pairwise_df = pd.read_csv(pairwise_path)
        pw_summary = load_json(summary_path)

        within = pairwise_df[pairwise_df["is_within_patient"]]["pearson_r"].values
        between = pairwise_df[~pairwise_df["is_within_patient"]]["pearson_r"].values

        bp = ax_c.boxplot([within, between], positions=[0, 1], widths=0.5,
                          patch_artist=True, showfliers=False,
                          medianprops=dict(color='black', linewidth=1))
        bp['boxes'][0].set_facecolor(COLORS['concordant'])
        bp['boxes'][0].set_alpha(0.3)
        bp['boxes'][1].set_facecolor(COLORS['neutral'])
        bp['boxes'][1].set_alpha(0.3)

        for pos_idx, (data, color) in enumerate([
            (within, COLORS['concordant']),
            (between, COLORS['neutral']),
        ]):
            jitter = np.random.RandomState(42).uniform(-0.12, 0.12, len(data))
            ax_c.scatter(np.full_like(data, pos_idx) + jitter, data, s=10,
                         alpha=0.6, color=color, edgecolors='none', zorder=3)

        ax_c.set_xticks([0, 1])
        ax_c.set_xticklabels(['Within-\npatient', 'Between-\npatient'], fontsize=6)
        ax_c.set_ylabel('Pathway profile Pearson r', fontsize=7)
        ax_c.set_title('Pathway enrichment reproducibility', fontsize=7, fontweight='bold')
        ax_c.tick_params(labelsize=5)

        if pw_summary:
            mw_p = pw_summary.get("mannwhitney_p")
            rb = pw_summary.get("rank_biserial")
            if mw_p is not None:
                p_str = f'p = {mw_p:.4f}' if mw_p >= 0.001 else 'p < 0.001'
                rb_str = f'r_b = {rb:.2f}' if rb is not None else ''
                ax_c.text(0.5, 0.95, f'MW {p_str}\n{rb_str}',
                          transform=ax_c.transAxes, fontsize=5, ha='center', va='top',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                    edgecolor='gray', alpha=0.9))

    # --- Bottom right: 5d — bridge gene scatter ---
    ax_d = fig.add_subplot(gs[1, 1])
    bg_path = base / "phase4" / "bridge_genes" / "bridge_gene_pearson.csv"
    bg_json_path = base / "phase4" / "bridge_genes" / "bridge_gene_correlation.json"

    if bg_path.exists():
        bg_df = pd.read_csv(bg_path)
        bg_json = load_json(bg_json_path)
        both_sig = set(bg_json.get("de_overlap", {}).get("both_sig_genes", []))

        is_de = bg_df["gene"].isin(both_sig)
        ax_d.scatter(bg_df.loc[~is_de, "pearson_discovery"],
                     bg_df.loc[~is_de, "pearson_validation"],
                     s=10, alpha=0.6, color=COLORS['neutral'], label='Not DE in both',
                     edgecolors='none')
        ax_d.scatter(bg_df.loc[is_de, "pearson_discovery"],
                     bg_df.loc[is_de, "pearson_validation"],
                     s=14, alpha=0.7, color=COLORS['discordant'],
                     label=f'DE in both ({is_de.sum()})',
                     edgecolors='white', linewidth=0.3)

        ax_d.plot([0, 1], [0, 1], 'k--', linewidth=0.5, alpha=0.3)

        top3 = bg_df.nlargest(3, "pearson_discovery")
        bottom3 = bg_df.nsmallest(3, "pearson_validation")
        # Per-gene offsets to avoid overlap (fanned out)
        label_offsets = {
            'EPCAM':  (3, 6),
            'TFAP2A': (3, -12),
            'HMGA1':  (3, -26),
            'MUC6':   (6, -10),
            'CPA3':   (-10, 8),
            'S100A8':  (6, 6),
        }
        labeled = set()
        for _, row in pd.concat([top3, bottom3]).iterrows():
            if row["gene"] in labeled:
                continue
            labeled.add(row["gene"])
            is_top = row["pearson_discovery"] > bg_df["pearson_discovery"].median()
            xoff, yoff = label_offsets.get(row["gene"], (-8 if is_top else 6, -6 if is_top else 4))
            ha = 'right' if xoff < 0 else 'left'
            ax_d.annotate(row["gene"],
                          xy=(row["pearson_discovery"], row["pearson_validation"]),
                          xytext=(xoff, yoff), textcoords='offset points',
                          fontsize=4, ha=ha, va='center', color='#333333',
                          arrowprops=dict(arrowstyle='-', lw=0.3, color='gray',
                                          shrinkA=0, shrinkB=2))

        r_val = bg_json["cross_cohort_pearson"]["prediction_quality_r"]
        p_val = bg_json["cross_cohort_pearson"]["prediction_quality_p"]
        annotate_r(ax_d, r_val, p_val, pos='upper left', fontsize=5)

        ax_d.set_xlabel("Pearson r (Discovery)", fontsize=7)
        ax_d.set_ylabel("Pearson r (Validation)", fontsize=7)
        ax_d.set_title("90 bridge genes: cross-cohort", fontsize=7, fontweight='bold')
        ax_d.legend(fontsize=4.5, loc='lower right')

    return fig


def _fig5d_bridge_gene_scatter(config, logger):
    """Scatter: discovery vs validation Pearson for 90 bridge genes.

    Labels top-performing and worst-performing bridge genes.
    Increased opacity for non-DE genes for better visibility.
    """
    base = Path(config["output_dir"])

    fig, ax = plt.subplots(figsize=(HALF_WIDTH, HALF_WIDTH * 0.9))

    bg_df = pd.read_csv(base / "phase4" / "bridge_genes" / "bridge_gene_pearson.csv")
    bg_json = load_json(base / "phase4" / "bridge_genes" / "bridge_gene_correlation.json")
    both_sig = set(bg_json.get("de_overlap", {}).get("both_sig_genes", []))

    # Color by DE status — increased opacity for non-DE
    is_de = bg_df["gene"].isin(both_sig)
    ax.scatter(bg_df.loc[~is_de, "pearson_discovery"],
               bg_df.loc[~is_de, "pearson_validation"],
               s=12, alpha=0.6, color=COLORS['neutral'], label='Not DE in both',
               edgecolors='none')
    ax.scatter(bg_df.loc[is_de, "pearson_discovery"],
               bg_df.loc[is_de, "pearson_validation"],
               s=18, alpha=0.7, color=COLORS['discordant'], label=f'DE in both ({is_de.sum()})',
               edgecolors='white', linewidth=0.3)

    # Identity line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.5, alpha=0.3)

    # Label top 3 (best discovery r) and bottom 3 (worst validation r)
    top3 = bg_df.nlargest(3, "pearson_discovery")
    bottom3 = bg_df.nsmallest(3, "pearson_validation")
    labeled = set()
    for _, row in pd.concat([top3, bottom3]).iterrows():
        if row["gene"] in labeled:
            continue
        labeled.add(row["gene"])
        # Offset direction: top genes go left, bottom genes go right
        is_top = row["pearson_discovery"] > bg_df["pearson_discovery"].median()
        xoff = -8 if is_top else 6
        yoff = -6 if is_top else 4
        ax.annotate(row["gene"],
                    xy=(row["pearson_discovery"], row["pearson_validation"]),
                    xytext=(xoff, yoff), textcoords='offset points',
                    fontsize=4, ha='right' if is_top else 'left',
                    va='top' if is_top else 'bottom', color='#333333',
                    arrowprops=dict(arrowstyle='-', lw=0.3, color='gray',
                                    shrinkA=0, shrinkB=2))

    r = bg_json["cross_cohort_pearson"]["prediction_quality_r"]
    p = bg_json["cross_cohort_pearson"]["prediction_quality_p"]
    annotate_r(ax, r, p, pos='upper left', fontsize=6)

    ax.set_xlabel("Pearson r (Discovery)", fontsize=7)
    ax.set_ylabel("Pearson r (Validation)", fontsize=7)
    ax.set_title("90 bridge genes: cross-cohort replication", fontsize=8, fontweight='bold')
    ax.legend(fontsize=5, loc='lower right')

    add_panel_label(ax, 'd')
    fig.tight_layout()
    return fig


# ============================================================
# Figure 6: Encoder Comparison (Optional)
# ============================================================

COLOR_COAD = "#7B68EE"  # Medium slate blue


def figure6(config, out_dir, logger):
    """Generate Figure 6: Cross-cancer generalization."""
    logger.info("Figure 6: Cross-Cancer Generalization")

    fig_a = _fig6a_coad_morans_scatter(config, logger)
    save_figure(fig_a, out_dir / "fig6a_coad_morans_scatter")

    fig_b = _fig6b_cross_tissue_scatter(config, logger)
    save_figure(fig_b, out_dir / "fig6b_cross_tissue_scatter")

    fig_c = _fig6c_cross_cancer_pathway(config, logger)
    save_figure(fig_c, out_dir / "fig6c_cross_cancer_pathway")

    logger.info("  Figure 6 complete.")


def _fig6a_coad_morans_scatter(config, logger):
    """COAD: Spatial autocorrelation (Moran's I) vs gene predictability."""
    base = Path(config["output_dir"])
    coad_dir = base / "coad"

    gf = pd.read_csv(coad_dir / "gene_features.csv").dropna(
        subset=["morans_i", "mean_pearson"]
    )

    fig, ax = plt.subplots(figsize=(HALF_WIDTH, HALF_WIDTH * 0.85))
    ax.scatter(gf["morans_i"], gf["mean_pearson"], s=6, alpha=0.6,
               color=COLOR_COAD, edgecolors="none", zorder=2)

    # OLS trend line — COAD
    m, b = np.polyfit(gf["morans_i"], gf["mean_pearson"], 1)
    x_range = np.linspace(gf["morans_i"].min(), gf["morans_i"].max(), 100)
    ax.plot(x_range, m * x_range + b, color=COLOR_COAD, linewidth=1,
            alpha=0.8, zorder=1)

    # IDC trend for comparison
    try:
        idc_gf = pd.read_csv(
            base / "phase3" / "gene_predictability" / "biomarkers" / "gene_features.csv"
        )
        if "spatial_autocorrelation" in idc_gf.columns:
            m_idc, b_idc = np.polyfit(
                idc_gf["spatial_autocorrelation"], idc_gf["mean_pearson"], 1
            )
            x_idc = np.linspace(gf["morans_i"].min(), gf["morans_i"].max(), 100)
            ax.plot(x_idc, m_idc * x_idc + b_idc, color=COLORS["discovery"],
                    linewidth=1, linestyle="--", alpha=0.6,
                    label="IDC (discovery)", zorder=1)
            ax.legend(fontsize=6, frameon=False)
    except Exception:
        pass

    rho, p = stats.spearmanr(gf["morans_i"], gf["mean_pearson"])
    if p < 1e-100:
        rho_text = f"Spearman \u03c1 = {rho:.3f}\np < 1e-100"
    else:
        rho_text = f"Spearman \u03c1 = {rho:.3f}\np = {p:.2e}"
    ax.text(0.05, 0.95, rho_text, transform=ax.transAxes, fontsize=6,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.8))

    # Annotate outlier genes (largest residuals from OLS fit)
    gf = gf.copy()
    gf["ols_resid"] = gf["mean_pearson"] - (m * gf["morans_i"] + b)
    n_outliers = 3
    top_pos = gf.nlargest(n_outliers, "ols_resid")
    top_neg = gf.nsmallest(n_outliers, "ols_resid")
    outliers = pd.concat([top_pos, top_neg])
    gene_offsets = {"SLPI": (5, 12)}  # per-gene overrides
    for _, row in outliers.iterrows():
        offset_y = 12 if row["ols_resid"] > 0 else -4
        dx, dy = gene_offsets.get(row["gene"], (-8, offset_y))
        ax.annotate(row["gene"], (row["morans_i"], row["mean_pearson"]),
                    textcoords="offset points", xytext=(dx, dy),
                    fontsize=4.5, fontstyle="italic", alpha=0.9,
                    arrowprops=dict(arrowstyle="-", lw=0.3, alpha=0.5))

    ax.set_xlabel("Mean Moran's I")
    ax.set_ylabel("Mean Pearson r")
    ax.set_title("COAD: Spatial autocorrelation vs predictability", fontsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig


def _fig6b_cross_tissue_scatter(config, logger):
    """Cross-tissue gene predictability: IDC vs COAD for shared genes."""
    base = Path(config["output_dir"])
    coad_dir = base / "coad"

    cross_df = pd.read_csv(coad_dir / "cross_tissue_gene_predictability.csv")

    fig, ax = plt.subplots(figsize=(HALF_WIDTH, HALF_WIDTH * 0.85))

    ax.scatter(cross_df["idc_r"], cross_df["coad_r"], s=12, alpha=0.7,
               color=COLOR_COAD, edgecolors="white", linewidths=0.3, zorder=2)

    # Identity line
    lims = [min(cross_df["idc_r"].min(), cross_df["coad_r"].min()) - 0.05,
            max(cross_df["idc_r"].max(), cross_df["coad_r"].max()) + 0.05]
    ax.plot(lims, lims, color="#999999", linewidth=0.8, linestyle="--",
            zorder=0, label="Identity")

    # OLS regression line
    m_fit, b_fit = np.polyfit(cross_df["idc_r"], cross_df["coad_r"], 1)
    x_fit = np.linspace(lims[0], lims[1], 100)
    ax.plot(x_fit, m_fit * x_fit + b_fit, color=COLOR_COAD, linewidth=1,
            alpha=0.8, zorder=1, label="OLS fit")
    ax.legend(fontsize=5, frameon=False, loc="lower right")

    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Annotate correlations
    r, p_r = stats.pearsonr(cross_df["idc_r"], cross_df["coad_r"])
    rho, p_rho = stats.spearmanr(cross_df["idc_r"], cross_df["coad_r"])
    ax.text(0.05, 0.95, f"Pearson r = {r:.3f}\nSpearman \u03c1 = {rho:.3f}",
            transform=ax.transAxes, fontsize=6, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.8))

    # Label key genes
    label_genes = {
        "CD3E": (8, -5),
        "CD8A": (-12, -6),
        "ACTA2": (8, -10),
        "VWF": (-10, -7),
        "CD24": (-14, -4),
        "CTSG": (8, 6),
        "CEACAM6": (-14, 5),
        "EGFR": (-12, 6),
    }
    for gene, (dx, dy) in label_genes.items():
        row = cross_df[cross_df["gene"] == gene]
        if len(row) == 0:
            continue
        ax.annotate(gene, (row["idc_r"].values[0], row["coad_r"].values[0]),
                    textcoords="offset points", xytext=(dx, dy),
                    fontsize=4.5, fontstyle="italic",
                    arrowprops=dict(arrowstyle="-", lw=0.3, alpha=0.5))

    ax.set_xlabel("IDC mean Pearson r")
    ax.set_ylabel("COAD mean Pearson r")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    n_genes = len(cross_df)
    ax.set_title(f"Cross-tissue gene predictability ({n_genes} shared genes)",
                 fontsize=7)
    fig.tight_layout()
    return fig


def _fig6c_cross_cancer_pathway(config, logger):
    """Paired horizontal bar chart: IDC vs COAD pathway Cohen's d."""
    base = Path(config["output_dir"])

    # Load pathway consistency (mean Cohen's d across samples)
    # Use primary IDC Xenium discovery cohort, not Visium generalization
    idc_df = pd.read_csv(base / "phase3" / "pathways" / "biomarkers" / "pathway_consistency.csv")
    coad_df = pd.read_csv(base / "coad" / "pathway_consistency.csv")

    # Intersect pathways
    shared = set(idc_df["pathway"]) & set(coad_df["pathway"])
    idc_df = idc_df[idc_df["pathway"].isin(shared)].set_index("pathway")
    coad_df = coad_df[coad_df["pathway"].isin(shared)].set_index("pathway")

    # Merge and sort by IDC effect size descending
    merged = pd.DataFrame({
        "IDC": idc_df["mean_cohens_d"],
        "COAD": coad_df["mean_cohens_d"],
    }).sort_values("IDC", ascending=True)  # ascending for barh (bottom = highest)

    n_pathways = len(merged)
    logger.info(f"  6c: {n_pathways} shared pathways between IDC and COAD")

    # Clean pathway names
    labels = [p.replace("HALLMARK_", "").replace("_", " ").title()
              for p in merged.index]

    # Colors: colorblind-friendly pair
    c_idc = '#2171B5'   # strong blue
    c_coad = '#D95F0E'  # strong orange

    bar_height = 0.35
    y = np.arange(n_pathways)

    fig_h = max(3.5, n_pathways * 0.22)
    fig, ax = plt.subplots(figsize=(HALF_WIDTH, fig_h))

    ax.barh(y + bar_height / 2, merged["IDC"], height=bar_height,
            color=c_idc, label="IDC", edgecolor='white', linewidth=0.3)
    ax.barh(y - bar_height / 2, merged["COAD"], height=bar_height,
            color=c_coad, label="COAD", edgecolor='white', linewidth=0.3)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=5.5)
    ax.set_xlabel("Mean Cohen's d", fontsize=7)
    ax.set_title("Pathway enrichment in discordant spots", fontsize=7)

    # Clean axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', length=0)

    ax.legend(fontsize=6, frameon=False, loc='lower right')
    fig.tight_layout()
    return fig


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate main figures")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--figure", type=int, nargs="*", default=None,
                        help="Which figures to generate (1-6). Default: all.")
    args = parser.parse_args()

    config = load_config(args.config)

    log_dir = Path(config["output_dir"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(
        log_level=config.get("log_level", "INFO"),
        log_file=str(log_dir / "18_main_figures.log"),
    )

    out_dir = Path(config["output_dir"]) / "figures" / "main"
    out_dir.mkdir(parents=True, exist_ok=True)

    figures_to_gen = args.figure or [1, 2, 3, 4, 5, 6]
    overall_start = time.time()

    figure_funcs = {
        1: figure1,
        2: figure2,
        3: figure3,
        4: figure4,
        5: figure5,
        6: figure6,
    }

    for fig_num in figures_to_gen:
        if fig_num in figure_funcs:
            t0 = time.time()
            try:
                figure_funcs[fig_num](config, out_dir, logger)
                logger.info(f"  Time: {format_time(time.time() - t0)}")
            except Exception as e:
                logger.error(f"  Figure {fig_num} failed: {e}")
                import traceback
                traceback.print_exc()

    total_time = time.time() - overall_start
    logger.info(f"\nTotal time: {format_time(total_time)}")
    logger.info("Done.")


if __name__ == "__main__":
    main()

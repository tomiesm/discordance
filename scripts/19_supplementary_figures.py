#!/usr/bin/env python3
"""
Script 19: Generate supplementary figures.

Available figures: S1, S3-S14, S17, S18, S22.

Usage:
    python scripts/19_supplementary_figures.py                  # all figures
    python scripts/19_supplementary_figures.py --figure 1       # just S1
    python scripts/19_supplementary_figures.py --figure 1 4 7   # S1, S4, S7

Output:
    outputs/figures/supplementary/figS{N}*.pdf / .png
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.plotting import (
    COLORS, COHORT_LABELS, ENCODER_LABELS, LOC_COLORS, PATIENT_COLORS,
    setup_style, save_figure, load_config, load_json,
    load_discordance_spatial, spatial_scatter, annotate_r, cohort_color,
    FULL_WIDTH, HALF_WIDTH,
)
from src.utils import setup_logging, format_time

setup_style()


# ============================================================
# Helper: load full discordance parquet with all columns
# ============================================================

def load_discordance_full(sample_id, cohort_name, config):
    """Load full discordance parquet for a sample."""
    base = Path(config["output_dir"])
    path = base / "phase2" / "scores" / cohort_name / f"{sample_id}_discordance.parquet"
    return pd.read_parquet(path)


# ============================================================
# S1: Quality Control
# ============================================================

def supp_figure_s1(config, out_dir, logger):
    """S1: Quality control — UMI distribution and spot counts per sample."""
    logger.info("Supp Figure S1: Quality Control")

    all_samples = []
    for cohort_key in ["discovery", "validation"]:
        cc = config["cohorts"][cohort_key]
        for sid in cc["samples"]:
            all_samples.append({
                "sample_id": sid,
                "cohort_key": cohort_key,
                "cohort_name": cc["name"],
            })

    # Collect per-spot stats — vectorized (no iterrows)
    frames = []
    for s in all_samples:
        df = load_discordance_full(s["sample_id"], s["cohort_name"], config)
        frames.append(pd.DataFrame({
            "sample_id": s["sample_id"],
            "cohort": s["cohort_key"],
            "total_expr": df["total_expr"].values,
        }))
    spot_df = pd.concat(frames, ignore_index=True)

    sample_order = [s["sample_id"] for s in all_samples]
    colors = [cohort_color(s["cohort_key"]) for s in all_samples]

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS['discovery'], alpha=0.6, label='Discovery'),
                       Patch(facecolor=COLORS['validation'], alpha=0.6, label='Validation')]

    # --- S1a: UMI distribution per sample ---
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, FULL_WIDTH * 0.35))

    data_by_sample = [spot_df[spot_df["sample_id"] == sid]["total_expr"].values
                      for sid in sample_order]
    bp = ax.boxplot(data_by_sample, labels=sample_order, patch_artist=True,
                    showfliers=False, widths=0.6)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xticklabels(sample_order, rotation=45, ha='right', fontsize=5)
    ax.set_ylabel("Total expression (log1p UMI)", fontsize=6)
    ax.set_title("Per-spot expression distribution by sample", fontsize=8, fontweight='bold')
    ax.legend(handles=legend_elements, fontsize=5, loc='upper right', framealpha=0.9)
    fig.tight_layout()
    save_figure(fig, out_dir / "figS1a_umi_distribution")

    # --- S1b: Spot count per sample ---
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, FULL_WIDTH * 0.3))
    spot_counts = spot_df.groupby("sample_id").size().reindex(sample_order)
    ax.bar(range(len(sample_order)), spot_counts.values, color=colors, alpha=0.7)
    ax.set_xticks(range(len(sample_order)))
    ax.set_xticklabels(sample_order, rotation=45, ha='right', fontsize=5)
    ax.set_ylabel("Number of spots", fontsize=6)
    ax.set_title("Spot count per sample", fontsize=8, fontweight='bold')
    for i, v in enumerate(spot_counts.values):
        ax.text(i, v + 100, str(int(v)), ha='center', fontsize=4)
    ax.legend(handles=legend_elements, fontsize=5, loc='upper right', framealpha=0.9)
    fig.tight_layout()
    save_figure(fig, out_dir / "figS1b_spot_counts")

    logger.info("  S1 complete.")


# ============================================================
# S3: Per-Gene Prediction Performance
# ============================================================

def supp_figure_s3(config, out_dir, logger):
    """S3: Per-gene prediction heatmap and encoder-regressor comparison."""
    logger.info("Supp Figure S3: Per-Gene Prediction Performance")
    base = Path(config["output_dir"])
    encoders = [e["name"] for e in config["encoders"]]
    regressors = [r["name"] for r in config["regressors"]]

    for cohort_key in ["discovery", "validation"]:
        cohort_name = config["cohorts"][cohort_key]["name"]
        n_folds = config["cohorts"][cohort_key]["n_lopo_folds"]

        # Load per-gene Pearson for each config
        gene_list_path = Path(config["v3_data_dir"]) / f"gene_list_{cohort_name}.json"
        with open(gene_list_path) as f:
            gene_names = json.load(f)

        n_genes = len(gene_names)
        configs = []
        pearson_matrix = []

        for enc in encoders:
            for reg in regressors:
                config_label = f"{ENCODER_LABELS[enc]}\n{reg.title()}"
                configs.append(config_label)

                # Aggregate per-gene Pearson across folds
                gene_pearsons = np.zeros(n_genes)
                fold_count = np.zeros(n_genes)

                for fold in range(n_folds):
                    fold_dir = base / "predictions" / cohort_name / enc / reg / f"fold{fold}"
                    metrics_path = fold_dir / "metrics.json"
                    m = load_json(metrics_path)
                    if m:
                        for gi, gene in enumerate(gene_names):
                            if gene in m and isinstance(m[gene], dict) and "pearson" in m[gene]:
                                gene_pearsons[gi] += m[gene]["pearson"]
                                fold_count[gi] += 1
                # Average across folds
                mask = fold_count > 0
                gene_pearsons[mask] /= fold_count[mask]

                pearson_matrix.append(gene_pearsons)

        pearson_matrix = np.array(pearson_matrix).T  # genes × configs

        # Sort by mean Pearson
        mean_per_gene = pearson_matrix.mean(axis=1)
        sort_idx = np.argsort(mean_per_gene)[::-1]
        pearson_sorted = pearson_matrix[sort_idx]

        # Heatmap
        fig, ax = plt.subplots(figsize=(HALF_WIDTH, FULL_WIDTH * 0.9))
        im = ax.imshow(pearson_sorted, aspect='auto', cmap='viridis',
                        vmin=0.0, vmax=1.0, interpolation='nearest')
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, fontsize=4, rotation=45, ha='right')
        ax.set_ylabel(f"Genes (sorted by mean r, n={n_genes})", fontsize=6)
        ax.set_title(f"Per-gene Pearson: {COHORT_LABELS[cohort_key]}", fontsize=7,
                     fontweight='bold', color=cohort_color(cohort_key))

        # Y-axis: show a few gene labels
        n_show = min(20, n_genes)
        ytick_pos = np.linspace(0, len(sort_idx) - 1, n_show, dtype=int)
        ax.set_yticks(ytick_pos)
        ax.set_yticklabels([gene_names[sort_idx[i]] for i in ytick_pos], fontsize=3)

        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label='Pearson r')
        fig.tight_layout()
        suffix = "disc" if cohort_key == "discovery" else "val"
        save_figure(fig, out_dir / f"figS3a_gene_heatmap_{suffix}")

    # --- S3c: Ridge vs MLP per-gene Pearson scatter ---
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH * 0.75, FULL_WIDTH * 0.35))
    for idx, cohort_key in enumerate(["discovery", "validation"]):
        ax = axes[idx]
        cohort_name = config["cohorts"][cohort_key]["name"]
        n_folds = config["cohorts"][cohort_key]["n_lopo_folds"]
        gene_list_path = Path(config["v3_data_dir"]) / f"gene_list_{cohort_name}.json"
        with open(gene_list_path) as f:
            gene_names = json.load(f)

        # Compute per-gene Pearson for ridge and MLP from metrics files
        # Average across all 3 encoders and all folds
        ridge_pearsons = np.zeros(len(gene_names))
        mlp_pearsons = np.zeros(len(gene_names))
        ridge_counts = np.zeros(len(gene_names))
        mlp_counts = np.zeros(len(gene_names))

        for enc in encoders:
            for fold in range(n_folds):
                for reg, arr, cnt in [("ridge", ridge_pearsons, ridge_counts),
                                      ("mlp", mlp_pearsons, mlp_counts)]:
                    m = load_json(base / "predictions" / cohort_name / enc / reg / f"fold{fold}" / "metrics.json")
                    if m:
                        for gi, gene in enumerate(gene_names):
                            if gene in m and isinstance(m[gene], dict) and "pearson" in m[gene]:
                                arr[gi] += m[gene]["pearson"]
                                cnt[gi] += 1

        mask_r = ridge_counts > 0
        ridge_pearsons[mask_r] /= ridge_counts[mask_r]
        mask_m = mlp_counts > 0
        mlp_pearsons[mask_m] /= mlp_counts[mask_m]

        x_vals = ridge_pearsons
        y_vals = mlp_pearsons

        ax.scatter(x_vals, y_vals, s=4, alpha=0.5, color=cohort_color(cohort_key),
                   edgecolors='none', rasterized=True)
        lim = [min(x_vals.min(), y_vals.min()) - 0.05, max(x_vals.max(), y_vals.max()) + 0.05]
        ax.plot(lim, lim, 'k--', linewidth=0.5, alpha=0.3)
        ax.set_xlabel("Ridge Pearson r", fontsize=6)
        ax.set_ylabel("MLP Pearson r", fontsize=6)
        ax.set_title(COHORT_LABELS[cohort_key], fontsize=7, fontweight='bold',
                     color=cohort_color(cohort_key))
        r, p = stats.pearsonr(x_vals, y_vals)
        annotate_r(ax, r, p, fontsize=6)

    fig.suptitle("Ridge vs MLP per-gene Pearson", fontsize=8, fontweight='bold', y=0.98)
    fig.subplots_adjust(wspace=0.25, left=0.10, right=0.97, bottom=0.15, top=0.88)
    save_figure(fig, out_dir / "figS3c_ridge_vs_mlp")

    logger.info("  S3 complete.")


# ============================================================
# S4: Conditional Discordance
# ============================================================

def supp_figure_s4(config, out_dir, logger):
    """S4: Raw vs conditional discordance — expression confound removal."""
    logger.info("Supp Figure S4: Conditional Discordance")

    sample_id = "TENX193"
    cohort_name = config["cohorts"]["discovery"]["name"]
    df = load_discordance_full(sample_id, cohort_name, config)

    # Average across ridge encoders
    raw_cols = [c for c in df.columns if c.startswith("D_raw_") and c.endswith("_ridge")]
    cond_cols = [c for c in df.columns if c.startswith("D_cond_") and c.endswith("_ridge")]
    df["D_raw"] = df[raw_cols].mean(axis=1)
    df["D_cond"] = df[cond_cols].mean(axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH * 0.65, FULL_WIDTH * 0.55))

    # S4a: D_raw vs total expression
    ax = axes[0, 0]
    ax.scatter(df["total_expr"], df["D_raw"], s=0.3, alpha=0.2, color=COLORS["discovery"],
               rasterized=True)
    rho, p = stats.spearmanr(df["total_expr"], df["D_raw"])
    annotate_r(ax, rho, p, pos='upper right')
    ax.set_xlabel("Total expression", fontsize=6)
    ax.set_ylabel("D (raw)", fontsize=6)
    ax.set_title("Raw discordance", fontsize=7, fontweight='bold')

    # S4b: D_cond vs total expression
    ax = axes[0, 1]
    ax.scatter(df["total_expr"], df["D_cond"], s=0.3, alpha=0.2, color=COLORS["discovery"],
               rasterized=True)
    rho, p = stats.spearmanr(df["total_expr"], df["D_cond"])
    annotate_r(ax, rho, p, pos='upper right')
    ax.set_xlabel("Total expression", fontsize=6)
    ax.set_ylabel("D (conditional)", fontsize=6)
    ax.set_title("After conditioning", fontsize=7, fontweight='bold')

    # S4c: D_raw spatial map
    x, y = df["x"].values, df["y"].values
    spatial_scatter(axes[1, 0], x, y, df["D_raw"].values, cmap='magma', s=0.3,
                    title="D (raw) spatial")

    # S4d: D_cond spatial map
    spatial_scatter(axes[1, 1], x, y, df["D_cond"].values, cmap='magma', s=0.3,
                    title="D (cond) spatial")

    fig.suptitle(f"Conditional discordance: {sample_id}", fontsize=8,
                 fontweight='bold', y=0.98)
    fig.subplots_adjust(hspace=0.35, wspace=0.25, left=0.08, right=0.95, bottom=0.06, top=0.90)
    save_figure(fig, out_dir / "figS4_conditional_discordance")

    logger.info("  S4 complete.")


# ============================================================
# S5: Gate 2.1 Detail
# ============================================================

def supp_figure_s5(config, out_dir, logger):
    """S5: Multi-model agreement detail — pairwise correlation matrices."""
    logger.info("Supp Figure S5: Gate 2.1 Detail")
    base = Path(config["output_dir"])

    gate_data = load_json(base / "phase2" / "gate2_1_agreement.json")

    # Representative samples
    representatives = {
        "High agreement": ("NCBI785", "validation"),
        "Borderline": ("TENX198", "discovery"),
        "Low agreement": ("TENX197", "discovery"),
    }

    # S5a: Pairwise correlation matrices
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, FULL_WIDTH * 0.3),
                             gridspec_kw={"wspace": 0.15})

    for ax_idx, (label, (sid, ckey)) in enumerate(representatives.items()):
        ax = axes[ax_idx]
        cohort_name = config["cohorts"][ckey]["name"]

        # Load discordance data and compute pairwise correlations
        df = load_discordance_full(sid, cohort_name, config)
        d_cond_cols = [c for c in df.columns if c.startswith("D_cond_")]

        corr_matrix = df[d_cond_cols].corr(method='spearman')

        im = ax.imshow(corr_matrix.values, cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_xticks(range(len(d_cond_cols)))
        ax.set_xticklabels([c.replace("D_cond_", "") for c in d_cond_cols],
                           fontsize=3, rotation=90)
        ax.set_yticks(range(len(d_cond_cols)))
        ax.set_yticklabels([c.replace("D_cond_", "") for c in d_cond_cols], fontsize=3)

        sample_data = gate_data["cohorts"][ckey]["samples"][sid]
        med_rho = sample_data["median_rho"]
        status = "PASS" if sample_data["pass"] else "FAIL"
        ax.set_title(f"{label}\n{sid} (ρ={med_rho:.2f}, {status})", fontsize=6)

    fig.subplots_adjust(wspace=0.15, left=0.06, right=0.88, bottom=0.18, top=0.85)
    cbar_ax = fig.add_axes([0.90, 0.22, 0.015, 0.58])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label("Spearman ρ", fontsize=6)
    save_figure(fig, out_dir / "figS5a_pairwise_correlation_matrices")

    # S5b: Spot count vs median ρ
    fig, ax = plt.subplots(figsize=(HALF_WIDTH, HALF_WIDTH * 0.8))

    from matplotlib.lines import Line2D
    # Collect points for label adjustment
    label_points = []
    for cohort_key in ["discovery", "validation"]:
        samples = gate_data["cohorts"][cohort_key]["samples"]
        for sid, sdata in samples.items():
            color = cohort_color(cohort_key)
            marker = 'o' if sdata["pass"] else 'x'
            ax.scatter(sdata["n_spots"], sdata["median_rho"],
                       c=color, marker=marker, s=25, alpha=0.8)
            label_points.append((sdata["n_spots"], sdata["median_rho"], sid))

    # Simple greedy label placement to avoid overlaps
    all_x = [p[0] for p in label_points]
    x_max = max(all_x) if all_x else 1
    label_points.sort(key=lambda p: (-p[1], p[0]))  # top to bottom
    placed = []
    for xp, yp, sid in label_points:
        # Place label to the left for rightmost points
        if xp > x_max * 0.8:
            ox, oy = -3, 3
            ha = 'right'
        else:
            ox, oy = 3, 3
            ha = 'left'
        # Check against already placed labels and nudge if too close
        for px, py in placed:
            if abs(xp + ox - px) < 1500 and abs(yp + oy * 0.003 - py) < 0.025:
                oy += 8
        placed.append((xp + ox, yp + oy * 0.003))
        ax.annotate(sid, (xp, yp), fontsize=3.5, alpha=0.7,
                    xytext=(ox, oy), textcoords='offset points', ha=ha)

    ax.axhline(gate_data["threshold"], color='gray', linestyle='--', linewidth=0.5,
               label=f'Threshold (ρ = {gate_data["threshold"]})')
    ax.set_xlabel("Spot count", fontsize=6)
    ax.set_ylabel("Median pairwise ρ", fontsize=6)
    ax.set_title("Spot count vs agreement", fontsize=8, fontweight='bold')
    r_spots = [gate_data["cohorts"][ck]["samples"][s]["n_spots"]
               for ck in ["discovery", "validation"]
               for s in gate_data["cohorts"][ck]["samples"]]
    r_rhos = [gate_data["cohorts"][ck]["samples"][s]["median_rho"]
              for ck in ["discovery", "validation"]
              for s in gate_data["cohorts"][ck]["samples"]]
    r, p = stats.spearmanr(r_spots, r_rhos)
    annotate_r(ax, r, p, pos='lower right', fontsize=6)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['discovery'],
               label='Discovery (pass)', markersize=5, markeredgewidth=0),
        Line2D([0], [0], marker='x', color=COLORS['discovery'],
               label='Discovery (fail)', markersize=5, markeredgewidth=1.5),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['validation'],
               label='Validation (pass)', markersize=5, markeredgewidth=0),
        Line2D([0], [0], marker='x', color=COLORS['validation'],
               label='Validation (fail)', markersize=5, markeredgewidth=1.5),
    ]
    ax.legend(handles=legend_elements, fontsize=4.5, loc='upper right', framealpha=0.9)
    fig.tight_layout()
    save_figure(fig, out_dir / "figS5b_spotcount_vs_agreement")

    logger.info("  S5 complete.")


# ============================================================
# S6: Gate 2.2 Spatial Detail
# ============================================================

def supp_figure_s6(config, out_dir, logger):
    """S6: Spatial autocorrelation detail — all sample D_cond maps + Moran's I."""
    logger.info("Supp Figure S6: Gate 2.2 Spatial Detail")
    base = Path(config["output_dir"])
    gate_spatial = load_json(base / "phase2" / "gate2_2_spatial.json")

    # S6a: Grid of spatial maps for all 18 samples
    all_samples = []
    for cohort_key in ["discovery", "validation"]:
        cc = config["cohorts"][cohort_key]
        for sid in cc["samples"]:
            all_samples.append({"sample_id": sid, "cohort_key": cohort_key,
                                "cohort_name": cc["name"]})

    n = len(all_samples)
    ncols = 6
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(FULL_WIDTH, FULL_WIDTH * nrows / ncols * 0.95))
    axes_flat = axes.flatten()

    for i, s in enumerate(all_samples):
        ax = axes_flat[i]
        disc_df = load_discordance_spatial(s["sample_id"], s["cohort_name"], config)
        spatial_scatter(ax, disc_df["x"].values, disc_df["y"].values,
                        disc_df["D_cond"].values, cmap='magma', s=0.3,
                        colorbar=False)
        morans = gate_spatial["cohorts"][s["cohort_key"]]["samples"][s["sample_id"]]["morans_i"]
        ax.set_title(f"{s['sample_id']}\nI={morans:.3f}", fontsize=5,
                     color=cohort_color(s["cohort_key"]))

    # Hide unused axes
    for i in range(n, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle("D_cond spatial maps — all samples", fontsize=8,
                 fontweight='bold', y=1.01)
    fig.tight_layout()
    save_figure(fig, out_dir / "figS6a_spatial_maps_all")

    # S6b: Moran's I vs sample size
    fig, ax = plt.subplots(figsize=(HALF_WIDTH, HALF_WIDTH * 0.8))
    label_points = []
    for cohort_key in ["discovery", "validation"]:
        samples = gate_spatial["cohorts"][cohort_key]["samples"]
        for sid, sdata in samples.items():
            color = cohort_color(cohort_key)
            ax.scatter(sdata["n_spots"], sdata["morans_i"],
                       c=color, s=25, alpha=0.8)
            label_points.append((sdata["n_spots"], sdata["morans_i"], sid))

    # Place labels with overlap avoidance
    all_x = [p[0] for p in label_points]
    x_max = max(all_x) if all_x else 1
    label_points.sort(key=lambda p: (-p[1], p[0]))  # top to bottom
    placed = []
    for xp, yp, sid in label_points:
        if xp > x_max * 0.8:
            ox, oy = -3, 3
            ha = 'right'
        else:
            ox, oy = 3, 3
            ha = 'left'
        for px, py in placed:
            if abs(xp + ox - px) < 1500 and abs(yp + oy * 0.003 - py) < 0.02:
                oy += 8
        placed.append((xp + ox, yp + oy * 0.003))
        ax.annotate(sid, (xp, yp), fontsize=3.5, alpha=0.7,
                    xytext=(ox, oy), textcoords='offset points', ha=ha)

    ax.set_xlabel("Number of spots", fontsize=6)
    ax.set_ylabel("Moran's I", fontsize=6)
    ax.set_title("Spatial autocorrelation vs sample size", fontsize=8, fontweight='bold')

    all_spots = [gate_spatial["cohorts"][ck]["samples"][s]["n_spots"]
                 for ck in ["discovery", "validation"]
                 for s in gate_spatial["cohorts"][ck]["samples"]]
    all_morans = [gate_spatial["cohorts"][ck]["samples"][s]["morans_i"]
                  for ck in ["discovery", "validation"]
                  for s in gate_spatial["cohorts"][ck]["samples"]]
    r, p = stats.spearmanr(all_spots, all_morans)
    annotate_r(ax, r, p, pos='upper right', fontsize=6)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS['discovery'], alpha=0.8, label='Discovery'),
                       Patch(facecolor=COLORS['validation'], alpha=0.8, label='Validation')]
    ax.legend(handles=legend_elements, fontsize=5, loc='center right', framealpha=0.9)
    fig.tight_layout()
    save_figure(fig, out_dir / "figS6b_morans_vs_samplesize")

    logger.info("  S6 complete.")


# ============================================================
# S7: Dual-Track Detail
# ============================================================

def supp_figure_s7(config, out_dir, logger):
    """S7: Dual-track scatter plots for all samples."""
    logger.info("Supp Figure S7: Dual-Track Detail")
    base = Path(config["output_dir"])

    # Compute dual-track per spot for all samples using averaged residuals
    all_samples = []
    for cohort_key in ["discovery", "validation"]:
        cc = config["cohorts"][cohort_key]
        for sid in cc["samples"]:
            all_samples.append({"sample_id": sid, "cohort_key": cohort_key,
                                "cohort_name": cc["name"]})

    # Load residuals and compute track A/B for all samples
    pred_dir = base / "predictions"
    encoder_names = [e["name"] for e in config["encoders"]]
    seed = config["seed"]

    sample_dt_data = {}
    for cohort_key in ["discovery", "validation"]:
        cc = config["cohorts"][cohort_key]
        cohort_name = cc["name"]
        n_folds = cc["n_lopo_folds"]

        # Load and average residuals across encoders
        all_enc_residuals = []
        spot_ids = None
        for enc in encoder_names:
            fold_residuals = []
            fold_spots = []
            for fold in range(n_folds):
                fold_dir = pred_dir / cohort_name / enc / "ridge" / f"fold{fold}"
                residuals = np.load(fold_dir / "test_residuals.npy")
                with open(fold_dir / "test_spot_ids.json") as f:
                    spots = json.load(f)
                fold_residuals.append(residuals)
                fold_spots.extend(spots)
            all_enc_residuals.append(np.vstack(fold_residuals))
            if spot_ids is None:
                spot_ids = fold_spots

        residuals = np.mean(all_enc_residuals, axis=0)
        n_genes = residuals.shape[1]

        # Split genes
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n_genes)
        mid = n_genes // 2
        genes_A, genes_B = indices[:mid], indices[mid:]

        from src.discordance import compute_mean_absolute_discordance
        D_A = compute_mean_absolute_discordance(residuals[:, genes_A])
        D_B = compute_mean_absolute_discordance(residuals[:, genes_B])

        for sid in cc["samples"]:
            prefix = f"{sid}_"
            mask = np.array([s.startswith(prefix) for s in spot_ids])
            if mask.sum() > 0:
                sample_dt_data[sid] = {
                    "D_A": D_A[mask],
                    "D_B": D_B[mask],
                    "cohort_key": cohort_key,
                }

    # S7a: Grid of scatter plots
    n = len(all_samples)
    ncols = 6
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(FULL_WIDTH, FULL_WIDTH * nrows / ncols))
    axes_flat = axes.flatten()

    for i, s in enumerate(all_samples):
        ax = axes_flat[i]
        sid = s["sample_id"]
        if sid in sample_dt_data:
            dt = sample_dt_data[sid]
            ax.scatter(dt["D_A"], dt["D_B"], s=0.3, alpha=0.2,
                       color=cohort_color(s["cohort_key"]), rasterized=True)
            rho, _ = stats.spearmanr(dt["D_A"], dt["D_B"])
            ax.set_title(f"{sid}\nρ={rho:.3f}", fontsize=5,
                         color=cohort_color(s["cohort_key"]))
            # Add identity line
            lim = [0, max(dt["D_A"].max(), dt["D_B"].max())]
            ax.plot(lim, lim, 'k--', linewidth=0.3, alpha=0.3)
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(n, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle("Dual-track scatter: Track A vs Track B discordance", fontsize=8,
                 fontweight='bold', y=1.01)
    fig.tight_layout()
    save_figure(fig, out_dir / "figS7a_dual_track_all_samples")

    # S7b: Subsampling stability curve
    csv_path = base / "figure_data" / "subsampling_curve.csv"
    fig, ax = plt.subplots(figsize=(HALF_WIDTH, HALF_WIDTH * 0.85))

    if not csv_path.exists():
        ax.text(0.5, 0.5, 'Pre-computation needed\n(run script 17)',
                transform=ax.transAxes, ha='center', fontsize=8)
    else:
        df_sub = pd.read_csv(csv_path)
        gene_counts = sorted(df_sub["n_genes_per_track"].unique())

        for cohort_key in ["discovery", "validation"]:
            color = cohort_color(cohort_key)
            cdf = df_sub[df_sub["cohort"] == cohort_key]
            summary = cdf.groupby("n_genes_per_track")["spearman_rho"].agg(
                median='median',
                q25=lambda x: x.quantile(0.25),
                q75=lambda x: x.quantile(0.75),
            ).reset_index()

            ax.plot(summary["n_genes_per_track"], summary["median"],
                    color=color, linewidth=1.5, marker='o', markersize=3,
                    label=COHORT_LABELS[cohort_key], zorder=3)
            ax.fill_between(summary["n_genes_per_track"], summary["q25"], summary["q75"],
                            color=color, alpha=0.15, zorder=2)

        v3_med = df_sub[df_sub["n_genes_per_track"] == 140]["spearman_rho"].median()
        ax.scatter([140], [v3_med], marker='D', s=40, color='black', zorder=5,
                   edgecolors='white', linewidths=0.5)
        ax.annotate(f'140 genes,\nρ = {v3_med:.2f}', xy=(140, v3_med),
                    xytext=(100, v3_med - 0.12), fontsize=5,
                    arrowprops=dict(arrowstyle='->', color='black', lw=0.5))

        ax.set_xticks(gene_counts)
        ax.set_xticklabels([str(g) for g in gene_counts], fontsize=5)
        ax.set_xlim(0, max(gene_counts) + 10)
        ax.set_ylim(-0.05, 1.05)

    ax.set_xlabel("Genes per track", fontsize=7)
    ax.set_ylabel("Dual-track Spearman ρ", fontsize=7)
    ax.set_title("Discordance stability vs gene count", fontsize=8)
    ax.legend(fontsize=5, loc='lower right')
    fig.tight_layout()
    save_figure(fig, out_dir / "figS7b_subsampling_curve")

    logger.info("  S7 complete.")


# ============================================================
# S8: Differential Expression Detail
# ============================================================

def supp_figure_s8(config, out_dir, logger):
    """S8: Per-sample volcano plots and DE reproducibility."""
    logger.info("Supp Figure S8: Differential Expression Detail")
    base = Path(config["output_dir"])

    all_samples = []
    for cohort_key in ["discovery", "validation"]:
        cc = config["cohorts"][cohort_key]
        for sid in cc["samples"]:
            all_samples.append({"sample_id": sid, "cohort_key": cohort_key,
                                "cohort_name": cc["name"]})

    # S8a: Per-sample volcano plots
    n = len(all_samples)
    ncols = 6
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(FULL_WIDTH, FULL_WIDTH * nrows / ncols))
    axes_flat = axes.flatten()

    for i, s in enumerate(all_samples):
        ax = axes_flat[i]
        de_path = base / "phase3" / "de" / s["cohort_name"] / "per_sample" / f"{s['sample_id']}_unmatched_de.csv"
        if de_path.exists():
            de = pd.read_csv(de_path)
            neg_log10_fdr = -np.log10(de["fdr"].clip(lower=1e-300))
            sig = de["fdr"] < 0.05

            ax.scatter(de.loc[~sig, "log2fc"], neg_log10_fdr[~sig],
                       s=1, alpha=0.3, color=COLORS["neutral"], rasterized=True)
            ax.scatter(de.loc[sig, "log2fc"], neg_log10_fdr[sig],
                       s=1, alpha=0.5, color=COLORS["discordant"], rasterized=True)
            ax.axhline(-np.log10(0.05), color='gray', linestyle='--', linewidth=0.3)
            n_sig = sig.sum()
            ax.set_title(f"{s['sample_id']}\n{n_sig} sig", fontsize=5,
                         color=cohort_color(s["cohort_key"]))
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(n, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle("Per-sample volcano plots (unmatched DE)", fontsize=8,
                 fontweight='bold', y=1.01)
    fig.tight_layout()
    save_figure(fig, out_dir / "figS8a_per_sample_volcanos")

    # S8b: Matching quality — unmatched fraction per sample
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, FULL_WIDTH * 0.3))
    sample_order = [s["sample_id"] for s in all_samples]
    unmatched_fracs = []
    bar_colors = []

    for s in all_samples:
        mq_path = (base / "phase3" / "de" / s["cohort_name"] / "per_sample"
                   / f"{s['sample_id']}_matching_quality.json")
        mq = load_json(mq_path)
        if mq:
            unmatched_fracs.append(mq["frac_unmatched"])
        else:
            unmatched_fracs.append(0)
        bar_colors.append(cohort_color(s["cohort_key"]))

    ax.bar(range(len(sample_order)), unmatched_fracs, color=bar_colors, alpha=0.7)
    ax.set_xticks(range(len(sample_order)))
    ax.set_xticklabels(sample_order, rotation=45, ha='right', fontsize=5)
    ax.set_ylabel("Fraction unmatched", fontsize=6)
    ax.set_title("Matching quality: unmatched fraction per sample", fontsize=8,
                 fontweight='bold')
    ax.axhline(0.5, color='red', linestyle='--', linewidth=0.5, alpha=0.5,
               label='Quality threshold')
    for i, v in enumerate(unmatched_fracs):
        if v > 0.5:
            ax.annotate(f'{v:.2f}', (i, v + 0.02), fontsize=4, ha='center', color='red')
    fig.tight_layout()
    save_figure(fig, out_dir / "figS8b_matching_quality")

    logger.info("  S8 complete.")


# ============================================================
# S9: Pathway Detail
# ============================================================

def supp_figure_s9(config, out_dir, logger):
    """S9: Full pathway heatmap and pathway overlap with gene panel."""
    logger.info("Supp Figure S9: Pathway Detail")
    base = Path(config["output_dir"])

    # Clean pathway names (same dict as main Fig 3c)
    _pw_names = {
        'ADIPOGENESIS': 'Adipogenesis', 'ALLOGRAFT_REJECTION': 'Allograft rejection',
        'APOPTOSIS': 'Apoptosis', 'CHOLESTEROL_HOMEOSTASIS': 'Cholesterol homeostasis',
        'COAGULATION': 'Coagulation', 'COMPLEMENT': 'Complement',
        'DNA_REPAIR': 'DNA repair', 'E2F_TARGETS': 'E2F targets',
        'EPITHELIAL_MESENCHYMAL_TRANSITION': 'EMT',
        'ESTROGEN_RESPONSE_EARLY': 'Estrogen response (early)',
        'ESTROGEN_RESPONSE_LATE': 'Estrogen response (late)',
        'FATTY_ACID_METABOLISM': 'Fatty acid metabolism',
        'G2M_CHECKPOINT': 'G2M checkpoint', 'GLYCOLYSIS': 'Glycolysis',
        'HEME_METABOLISM': 'Heme metabolism', 'HYPOXIA': 'Hypoxia',
        'IL2_STAT5_SIGNALING': 'IL-2/STAT5 signaling',
        'IL6_JAK_STAT3_SIGNALING': 'IL-6/JAK/STAT3 signaling',
        'INFLAMMATORY_RESPONSE': 'Inflammatory response',
        'INTERFERON_ALPHA_RESPONSE': 'IFN-\u03b1 response',
        'INTERFERON_GAMMA_RESPONSE': 'IFN-\u03b3 response',
        'KRAS_SIGNALING_DN': 'KRAS signaling (down)',
        'KRAS_SIGNALING_UP': 'KRAS signaling (up)',
        'MITOTIC_SPINDLE': 'Mitotic spindle', 'MTORC1_SIGNALING': 'mTORC1 signaling',
        'MYC_TARGETS_V1': 'MYC targets (v1)', 'MYC_TARGETS_V2': 'MYC targets (v2)',
        'NOTCH_SIGNALING': 'Notch signaling',
        'OXIDATIVE_PHOSPHORYLATION': 'Oxidative phosphorylation',
        'P53_PATHWAY': 'p53 pathway', 'PI3K_AKT_MTOR_SIGNALING': 'PI3K/AKT/mTOR signaling',
        'PROTEIN_SECRETION': 'Protein secretion',
        'REACTIVE_OXYGEN_SPECIES_PATHWAY': 'ROS pathway',
        'TGF_BETA_SIGNALING': 'TGF-\u03b2 signaling',
        'TNFA_SIGNALING_VIA_NFKB': 'TNF-\u03b1/NF-\u03baB signaling',
        'UNFOLDED_PROTEIN_RESPONSE': 'Unfolded protein response',
        'WNT_BETA_CATENIN_SIGNALING': 'Wnt/\u03b2-catenin signaling',
        'XENOBIOTIC_METABOLISM': 'Xenobiotic metabolism',
        'APICAL_JUNCTION': 'Apical junction', 'APICAL_SURFACE': 'Apical surface',
        'ANGIOGENESIS': 'Angiogenesis', 'ANDROGEN_RESPONSE': 'Androgen response',
        'BILE_ACID_METABOLISM': 'Bile acid metabolism',
        'HEDGEHOG_SIGNALING': 'Hedgehog signaling',
        'PANCREAS_BETA_CELLS': 'Pancreas beta cells', 'PEROXISOME': 'Peroxisome',
        'SPERMATOGENESIS': 'Spermatogenesis',
        'UV_RESPONSE_DN': 'UV response (down)', 'UV_RESPONSE_UP': 'UV response (up)',
    }

    def clean_pathway(name):
        key = name.replace('HALLMARK_', '')
        return _pw_names.get(key, key.replace('_', ' ').title())

    # S9a: Combined pathway bar chart
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, FULL_WIDTH * 0.6),
                             gridspec_kw={'width_ratios': [1, 1]})

    for idx, cohort_key in enumerate(["discovery", "validation"]):
        ax = axes[idx]
        cohort_name = config["cohorts"][cohort_key]["name"]
        pc = pd.read_csv(base / "phase3" / "pathways" / cohort_name / "pathway_consistency.csv")

        # Sort by mean Cohen's d
        pc = pc.sort_values("mean_cohens_d", ascending=True)

        bar_colors = []
        for _, row in pc.iterrows():
            if row["reproducibility"] >= 0.5:
                bar_colors.append(COLORS["discordant"] if row["mean_cohens_d"] > 0
                                  else COLORS["concordant"])
            else:
                bar_colors.append(COLORS["neutral"])

        ax.barh(range(len(pc)), pc["mean_cohens_d"].values, color=bar_colors, alpha=0.7)
        ax.set_yticks(range(len(pc)))
        ax.set_yticklabels([clean_pathway(p) for p in pc["pathway"].values], fontsize=4)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_xlabel("Mean Cohen's d", fontsize=6)
        ax.set_title(COHORT_LABELS[cohort_key], fontsize=7, fontweight='bold',
                     color=cohort_color(cohort_key))

        # Direction labels
        ax.text(0.98, 0.02, 'Enriched in\ndiscordant \u2192', transform=ax.transAxes,
                fontsize=3.5, ha='right', va='bottom', color='gray', fontstyle='italic')
        ax.text(0.02, 0.02, '\u2190 Depleted in\ndiscordant', transform=ax.transAxes,
                fontsize=3.5, ha='left', va='bottom', color='gray', fontstyle='italic')

        # Reproducibility annotation (% of samples significant)
        for i, (_, row) in enumerate(pc.iterrows()):
            repro = row["reproducibility"]
            star = '**' if repro > 0.9 else '*' if repro > 0.7 else ''
            if star:
                ax.text(row["mean_cohens_d"], i, f' {star}', fontsize=5, va='center',
                        ha='left' if row["mean_cohens_d"] > 0 else 'right',
                        fontweight='bold', color='dimgray')

    fig.suptitle("All pathway enrichments in discordant vs concordant spots", fontsize=8,
                 fontweight='bold', y=1.01)
    fig.tight_layout()
    save_figure(fig, out_dir / "figS9a_full_pathway_heatmap")

    # S9b: Pathway overlap with gene panel
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, FULL_WIDTH * 0.5))
    cohort_name = config["cohorts"]["discovery"]["name"]
    po = pd.read_csv(base / "phase3" / "pathways" / cohort_name / "pathway_overlap.csv")
    po = po.sort_values("n_overlap", ascending=True)

    bar_colors = [COLORS["discovery"] if row["passes_threshold"] else COLORS["neutral"]
                  for _, row in po.iterrows()]
    ax.barh(range(len(po)), po["n_overlap"].values, color=bar_colors, alpha=0.7)
    ax.set_yticks(range(len(po)))
    ax.set_yticklabels([clean_pathway(p) for p in po["pathway"].values], fontsize=4)
    ax.set_xlabel("Number of panel genes in pathway", fontsize=6)
    ax.set_title("Pathway overlap with 280-gene discovery panel", fontsize=8,
                 fontweight='bold')
    ax.axvline(5, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.text(5.5, len(po) * 0.35, 'min. threshold\n(n = 5)', fontsize=4, color='gray',
            va='top')
    fig.tight_layout()
    save_figure(fig, out_dir / "figS9b_pathway_overlap")

    logger.info("  S9 complete.")


# ============================================================
# S10: Deconvolution Detail
# ============================================================

def supp_figure_s10(config, out_dir, logger):
    """S10: Per-cohort cell type effect sizes with dumbbell plot.

    Shows mean Cohen's d per cell type per cohort, with discovery/validation
    comparison. Same ordering as main Fig 3b for consistency.
    """
    logger.info("Supp Figure S10: Deconvolution Detail")
    base = Path(config["output_dir"])

    # Load celltype summary from both cohorts
    disc_path = (base / "phase3" / "deconvolution"
                 / config["cohorts"]["discovery"]["name"] / "celltype_summary.csv")
    val_path = (base / "phase3" / "deconvolution"
                / config["cohorts"]["validation"]["name"] / "celltype_summary.csv")

    if not disc_path.exists() and not val_path.exists():
        logger.info("  No celltype summary data found. Skipping S10.")
        return

    fig, ax = plt.subplots(figsize=(HALF_WIDTH * 1.2, HALF_WIDTH * 0.85))

    disc_ct = pd.read_csv(disc_path).set_index("cell_type") if disc_path.exists() else pd.DataFrame()
    val_ct = pd.read_csv(val_path).set_index("cell_type") if val_path.exists() else pd.DataFrame()

    # Union of all cell types, ordered by max absolute d (largest at top)
    all_types = sorted(set(disc_ct.index) | set(val_ct.index))
    type_max_abs_d = {}
    d_col = "mean_cohens_d" if "mean_cohens_d" in disc_ct.columns else "cohens_d"
    for ct in all_types:
        d1 = abs(disc_ct.loc[ct, d_col]) if ct in disc_ct.index else 0
        d2 = abs(val_ct.loc[ct, d_col]) if ct in val_ct.index else 0
        type_max_abs_d[ct] = max(d1, d2)
    all_types = sorted(all_types, key=lambda ct: type_max_abs_d[ct])

    for i, ct in enumerate(all_types):
        has_disc = ct in disc_ct.index
        has_val = ct in val_ct.index
        d_disc = disc_ct.loc[ct, d_col] if has_disc else None
        d_val = val_ct.loc[ct, d_col] if has_val else None

        # Connecting line
        if d_disc is not None and d_val is not None:
            ax.plot([d_disc, d_val], [i, i], color='gray', linewidth=0.8, zorder=1)

        if d_disc is not None:
            ax.scatter(d_disc, i, color=COLORS['discovery'], s=35, zorder=3,
                       edgecolors=COLORS['discovery'], linewidth=1.0)
        if d_val is not None:
            ax.scatter(d_val, i, marker='D', color=COLORS['validation'], s=30,
                       zorder=3, edgecolors=COLORS['validation'], linewidth=1.0)

    ax.axvline(0, color='gray', linewidth=0.5, linestyle='-', alpha=0.5)
    ax.set_yticks(range(len(all_types)))
    ax.set_yticklabels([ct.replace('_', ' ').title() for ct in all_types], fontsize=6)
    ax.set_xlabel("Cohen's d (discordant vs concordant)", fontsize=7)
    ax.set_title("Cell type enrichment summary", fontsize=8, fontweight='bold')

    # Direction labels
    ax.text(0.98, 0.02, 'Enriched in\ndiscordant \u2192', transform=ax.transAxes,
            fontsize=4.5, ha='right', va='bottom', color='gray', fontstyle='italic')
    ax.text(0.02, 0.02, '\u2190 Depleted in\ndiscordant', transform=ax.transAxes,
            fontsize=4.5, ha='left', va='bottom', color='gray', fontstyle='italic')

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['discovery'],
               markeredgecolor=COLORS['discovery'], label='Discovery', markersize=5),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=COLORS['validation'],
               markeredgecolor=COLORS['validation'], label='Validation', markersize=5),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=5, framealpha=0.9)

    fig.tight_layout()
    save_figure(fig, out_dir / "figS10_deconvolution_detail")

    logger.info("  S10 complete.")


# ============================================================
# S11: Gene Predictability Detail
# ============================================================

def supp_figure_s11(config, out_dir, logger):
    """S11: Predictability by protein localization (boxplots with merged categories)."""
    logger.info("Supp Figure S11: Gene Predictability Detail")
    base = Path(config["output_dir"])

    merge_map = {'ER/Golgi': 'Cytoplasm', 'Mitochondria': 'Cytoplasm', 'Other': 'Unknown'}

    # S11b: Predictability by localization (boxplots with merged categories)
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, FULL_WIDTH * 0.35))

    cat_order = ['Membrane', 'Cytoplasm', 'Nucleus', 'Secreted/\nExtracellular', 'Unknown']
    cat_raw = ['Membrane', 'Cytoplasm', 'Nucleus', 'Secreted/Extracellular', 'Unknown']

    # First pass for shared y-axis
    all_pearson = []
    for cohort_key in ["discovery", "validation"]:
        cohort_name = config["cohorts"][cohort_key]["name"]
        gf = pd.read_csv(base / "phase3" / "gene_predictability" / cohort_name / "gene_features.csv")
        if "mean_pearson" in gf.columns:
            all_pearson.extend(gf["mean_pearson"].dropna().values)
    y_min = max(0, np.percentile(all_pearson, 0.5) - 0.05) if all_pearson else 0
    y_max = min(1.0, np.percentile(all_pearson, 99.5) + 0.05) if all_pearson else 1.0

    for idx, cohort_key in enumerate(["discovery", "validation"]):
        ax = axes[idx]
        cohort_name = config["cohorts"][cohort_key]["name"]
        gf = pd.read_csv(base / "phase3" / "gene_predictability" / cohort_name / "gene_features.csv")

        if "primary_localization" not in gf.columns:
            continue
        gf["plot_loc"] = gf["primary_localization"].replace(merge_map)

        rng = np.random.RandomState(42)
        for j, (cat_label, cat) in enumerate(zip(cat_order, cat_raw)):
            vals = gf[gf["plot_loc"] == cat]["mean_pearson"].values
            color = LOC_COLORS.get(cat, COLORS['neutral'])
            n = len(vals)

            if n >= 5:
                bp = ax.boxplot(vals, positions=[j], widths=0.5, patch_artist=True,
                                showfliers=False)
                bp['boxes'][0].set_facecolor(color)
                bp['boxes'][0].set_alpha(0.4)
                bp['medians'][0].set_color('black')
                bp['medians'][0].set_linewidth(1.0)

            jitter_x = rng.normal(j, 0.08, n)
            ax.scatter(jitter_x, vals, s=3, alpha=0.5, color=color, zorder=3,
                       edgecolors='none')
            # n annotation
            ax.text(j, y_min - 0.01, f'n={n}', ha='center', va='top', fontsize=3.5,
                    color='gray')

        ax.set_xticks(range(len(cat_order)))
        ax.set_xticklabels(cat_order, fontsize=4.5)
        ax.set_ylabel("Mean Pearson r", fontsize=6)
        ax.set_title(COHORT_LABELS[cohort_key], fontsize=7, fontweight='bold',
                     color=cohort_color(cohort_key))
        ax.set_ylim(y_min - 0.04, y_max + 0.04)

        # Mann-Whitney: Secreted vs all others
        secreted = gf[gf["plot_loc"] == "Secreted/Extracellular"]["mean_pearson"].values
        others = gf[gf["plot_loc"] != "Secreted/Extracellular"]["mean_pearson"].values
        if len(secreted) > 2 and len(others) > 2:
            _, mw_p = stats.mannwhitneyu(secreted, others, alternative='less')
            star = '***' if mw_p < 0.001 else '**' if mw_p < 0.01 else '*' if mw_p < 0.05 else 'n.s.'
            sec_idx = cat_raw.index('Secreted/Extracellular')
            ax.text(sec_idx, y_max - 0.02, f'{star}', ha='center', va='bottom',
                    fontsize=5, fontweight='bold')

    fig.suptitle("Prediction quality by protein localization (all categories)",
                 fontsize=8, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_figure(fig, out_dir / "figS11b_localization_predictability")

    logger.info("  S11 complete.")


# ============================================================
# S12: Within-Patient Detail
# ============================================================

def supp_figure_s12(config, out_dir, logger):
    """S12: Within-patient gene-level scatter plots and spatial binning."""
    logger.info("Supp Figure S12: Within-Patient Detail")
    base = Path(config["output_dir"])

    # S12a: Gene-level scatter for all multi-section patients
    wp_gene = pd.read_csv(base / "figure_data" / "within_patient_per_gene.csv")

    # Get all patients with multiple sections
    patients = wp_gene["patient_id"].unique()
    n_patients = len(patients)

    fig, axes = plt.subplots(2, 4, figsize=(FULL_WIDTH, FULL_WIDTH * 0.5))
    axes_flat = axes.flatten()

    for p_idx, patient_id in enumerate(patients):
        if p_idx >= len(axes_flat):
            break
        ax = axes_flat[p_idx]
        p_data = wp_gene[wp_gene["patient_id"] == patient_id]
        samples = sorted(p_data["sample_id"].unique())

        if len(samples) >= 2:
            # Compare first two sections
            s1_data = p_data[p_data["sample_id"] == samples[0]].set_index("gene")
            s2_data = p_data[p_data["sample_id"] == samples[1]].set_index("gene")
            common_genes = s1_data.index.intersection(s2_data.index)

            x = s1_data.loc[common_genes, "mean_abs_residual"].values
            y = s2_data.loc[common_genes, "mean_abs_residual"].values

            cohort = p_data["cohort"].iloc[0]
            color = COLORS["discovery"] if cohort == "biomarkers" else COLORS["validation"]

            ax.scatter(x, y, s=3, alpha=0.4, color=color, edgecolors='none',
                       rasterized=True)
            r, p_val = stats.pearsonr(x, y)
            annotate_r(ax, r, p_val, fontsize=5)

            lim = [0, max(x.max(), y.max()) * 1.1]
            ax.plot(lim, lim, 'k--', linewidth=0.3, alpha=0.3)
            ax.set_xlabel(samples[0], fontsize=5)
            ax.set_ylabel(samples[1], fontsize=5)
            ax.set_title(f"{patient_id} (r={r:.3f})", fontsize=6,
                         fontweight='bold', color=color)

    for i in range(n_patients, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle("Within-patient gene-level residual correlation",
                 fontsize=8, fontweight='bold', y=0.98)
    fig.subplots_adjust(wspace=0.25, hspace=0.35, left=0.06, right=0.97, bottom=0.06, top=0.90)
    save_figure(fig, out_dir / "figS12a_within_patient_gene_scatter")

    logger.info("  S12 complete.")


# ============================================================
# S13: Matched vs Unmatched DE (control figure, moved from main Fig 3e)
# ============================================================

def supp_figure_s13(config, out_dir, logger):
    """S13: Scatter of unmatched vs matched log2FC (confound control).

    Shows that DE signal persists when controlling for morphological similarity.
    Moved from main Figure 3 panel e to supplementary.
    """
    logger.info("Supp Figure S13: Matched vs Unmatched DE")
    base = Path(config["output_dir"])

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH * 0.75, FULL_WIDTH * 0.35))

    for idx, cohort_key in enumerate(["discovery", "validation"]):
        ax = axes[idx]
        cohort_name = config["cohorts"][cohort_key]["name"]

        unmatched_path = base / "phase3" / "de" / cohort_name / "meta_de_unmatched.csv"
        matched_path = base / "phase3" / "de" / cohort_name / "meta_de_matched.csv"
        if not unmatched_path.exists() or not matched_path.exists():
            ax.text(0.5, 0.5, 'Data N/A', transform=ax.transAxes, ha='center')
            continue

        unmatched = pd.read_csv(unmatched_path)
        matched = pd.read_csv(matched_path)

        merged = unmatched[["gene", "median_log2fc"]].rename(
            columns={"median_log2fc": "unmatched_fc"}
        ).merge(
            matched[["gene", "median_log2fc"]].rename(
                columns={"median_log2fc": "matched_fc"}
            ),
            on="gene", how="inner"
        )

        ax.scatter(merged["unmatched_fc"], merged["matched_fc"],
                   s=4, alpha=0.5, color=cohort_color(cohort_key), rasterized=True)

        # Identity line
        all_vals = np.concatenate([merged["unmatched_fc"].values,
                                    merged["matched_fc"].values])
        lim = [np.min(all_vals) * 1.1, np.max(all_vals) * 1.1]
        ax.plot(lim, lim, 'k--', linewidth=0.5, alpha=0.3, label='y = x')
        ax.set_xlim(lim); ax.set_ylim(lim)

        r, p = stats.pearsonr(merged["unmatched_fc"], merged["matched_fc"])
        annotate_r(ax, r, p, pos='upper left', fontsize=6)

        ax.set_xlabel("Unmatched log2FC", fontsize=6)
        ax.set_ylabel("Morphology-matched log2FC", fontsize=6)
        ax.set_title(COHORT_LABELS[cohort_key], fontsize=7, fontweight='bold',
                     color=cohort_color(cohort_key))

    fig.suptitle("DE control: unmatched vs morphology-matched log2FC",
                 fontsize=8, fontweight='bold', y=0.98)
    fig.subplots_adjust(wspace=0.25, left=0.10, right=0.97, bottom=0.15, top=0.88)
    save_figure(fig, out_dir / "figS13_matched_vs_unmatched")

    logger.info("  S13 complete.")


# ============================================================
# S14: Per-sample cell type Cohen's d (macrophage outlier figure)
# ============================================================

def supp_figure_s14(config, out_dir, logger):
    """S14: Per-sample Cohen's d for all cell types, highlighting NCBI783 macrophage outlier.

    Panel a: Forest plot of per-sample macrophage d across both cohorts.
    Panel b: Forest plot for all cell types (mean ± individual samples).
    """
    logger.info("Supp Figure S14: Per-sample cell type Cohen's d")
    base = Path(config["output_dir"])

    # --- Compute per-sample Cohen's d for each cell type ---
    records = []
    for cohort_key in ["discovery", "validation"]:
        cohort_name = config["cohorts"][cohort_key]["name"]
        score_dir = base / "phase2" / "scores" / cohort_name
        ct_dir = base / "phase3" / "deconvolution" / cohort_name / "per_sample"

        # Get patient mapping for labels
        patient_map = {}
        for pat, samples in config["cohorts"][cohort_key]["patient_mapping"].items():
            for s in samples:
                patient_map[s] = pat

        for ct_file in sorted(ct_dir.glob("*_celltype_scores.csv")):
            sample = ct_file.stem.replace("_celltype_scores", "")
            scores = pd.read_csv(ct_file)
            scores["barcode"] = scores["barcode"].astype(str)

            disc = pd.read_parquet(score_dir / f"{sample}_discordance.parquet")
            disc["barcode"] = disc["barcode"].astype(str)
            ridge_cols = [c for c in disc.columns
                          if c.startswith("D_cond_") and c.endswith("_ridge")]
            disc["D_cond_avg"] = disc[ridge_cols].mean(axis=1)

            merged = disc[["barcode", "D_cond_avg"]].merge(scores, on="barcode", how="inner")
            q25, q75 = merged["D_cond_avg"].quantile([0.25, 0.75])
            conc_mask = merged["D_cond_avg"] <= q25
            disc_mask = merged["D_cond_avg"] >= q75

            # All cell type columns (exclude non-celltype columns)
            skip_cols = {"barcode", "D_cond_avg", "spot_id", "sample_id", "x", "y",
                         "total_expr", "D_cond"}
            ct_cols = [c for c in merged.columns if c not in skip_cols
                       and not c.startswith("D_")]
            for ct in ct_cols:
                conc_vals = merged.loc[conc_mask, ct].values
                disc_vals = merged.loc[disc_mask, ct].values
                pooled_std = np.sqrt((conc_vals.var() + disc_vals.var()) / 2)
                d = (disc_vals.mean() - conc_vals.mean()) / pooled_std if pooled_std > 0 else 0.0
                records.append({
                    "sample": sample,
                    "patient": patient_map.get(sample, "?"),
                    "cohort": cohort_key,
                    "cell_type": ct,
                    "cohens_d": d,
                })

    df = pd.DataFrame(records)

    # ---- Panel a: Macrophage per-sample strip/forest plot ----
    mac = df[df["cell_type"] == "macrophage"].copy()
    mac["label"] = mac["sample"] + " (" + mac["patient"] + ")"
    mac = mac.sort_values("cohens_d").reset_index(drop=True)

    fig_a, ax = plt.subplots(figsize=(HALF_WIDTH * 1.1, HALF_WIDTH * 0.9))

    for i, (_, row) in enumerate(mac.iterrows()):
        color = cohort_color(row["cohort"])
        marker = "D" if row["sample"] == "NCBI783" else "o"
        ms = 6 if row["sample"] == "NCBI783" else 4
        zorder = 10 if row["sample"] == "NCBI783" else 5
        edgecolor = "black" if row["sample"] == "NCBI783" else "none"
        edgewidth = 0.8 if row["sample"] == "NCBI783" else 0
        ax.scatter(row["cohens_d"], i, marker=marker, color=color,
                   s=ms**2, zorder=zorder, edgecolors=edgecolor, linewidths=edgewidth)

    ax.set_yticks(range(len(mac)))
    ax.set_yticklabels(mac["label"].values, fontsize=5)
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)

    # Cohort means
    for cohort_key in ["discovery", "validation"]:
        sub = mac[mac["cohort"] == cohort_key]
        mean_d = sub["cohens_d"].mean()
        median_d = sub["cohens_d"].median()
        ax.axvline(mean_d, color=cohort_color(cohort_key), linewidth=1.0,
                   linestyle="-", alpha=0.6, label=f"{COHORT_LABELS[cohort_key]} mean={mean_d:.2f}")
        ax.axvline(median_d, color=cohort_color(cohort_key), linewidth=1.0,
                   linestyle=":", alpha=0.6, label=f"  median={median_d:.2f}")

    ax.set_xlabel("Cohen's d (macrophage score: discordant vs concordant)", fontsize=6)
    ax.set_title("Per-sample macrophage enrichment in discordant spots", fontsize=8, fontweight="bold")
    ax.legend(fontsize=5, loc="upper left", framealpha=0.9)

    # Annotate NCBI783 outlier (uses actual patient label from data)
    ncbi783_row = mac[mac["sample"] == "NCBI783"]
    if not ncbi783_row.empty:
        ncbi783_idx = ncbi783_row.index[0]
        ncbi783_d = mac.loc[ncbi783_idx, "cohens_d"]
        ncbi783_pat = mac.loc[ncbi783_idx, "patient"]
        ax.annotate(f"NCBI783 ({ncbi783_pat})", xy=(ncbi783_d, ncbi783_idx),
                    xytext=(ncbi783_d + 0.3, ncbi783_idx + 1.5),
                    fontsize=5, ha="left", va="center",
                    arrowprops=dict(arrowstyle="->", lw=0.5, color="gray"))

    fig_a.tight_layout()
    save_figure(fig_a, out_dir / "figS14a_macrophage_per_sample")

    # ---- Panel b: All cell types summary ----
    # Mean d per cell type per cohort, with individual sample dots + jitter
    ct_order = (df.groupby("cell_type")["cohens_d"]
                .apply(lambda x: abs(x.mean())).sort_values(ascending=True).index.tolist())

    fig_b, ax2 = plt.subplots(figsize=(HALF_WIDTH * 1.1, FULL_WIDTH * 0.45))

    y_positions = {}
    for i, ct in enumerate(ct_order):
        y_positions[ct] = i

    rng = np.random.RandomState(42)
    for cohort_key, offset in [("discovery", -0.15), ("validation", 0.15)]:
        sub = df[df["cohort"] == cohort_key]
        color = cohort_color(cohort_key)
        for ct in ct_order:
            ct_data = sub[sub["cell_type"] == ct]
            if ct_data.empty:
                continue
            y_base = y_positions[ct] + offset

            # Vertical jitter to separate overlapping dots
            jitter = rng.uniform(-0.08, 0.08, len(ct_data))
            y_jittered = [y_base + j for j in jitter]

            # Individual samples as small dots
            ax2.scatter(ct_data["cohens_d"].values, y_jittered,
                        s=10, alpha=0.5, color=color, edgecolors="none", zorder=3)

            # Mean as larger diamond marker (more distinct from circles)
            mean_d = ct_data["cohens_d"].mean()
            ax2.scatter(mean_d, y_base, marker="D", color=color, s=40,
                        edgecolors="white", linewidths=0.7, zorder=5)

    ax2.axvline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax2.set_yticks(range(len(ct_order)))
    ax2.set_yticklabels([ct.replace("_", " ").title() for ct in ct_order], fontsize=5)
    ax2.set_xlabel("Cohen's d (discordant vs concordant)", fontsize=6)
    ax2.set_title("Cell type enrichment in discordant spots (per sample)", fontsize=8, fontweight="bold")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor=cohort_color("discovery"),
               label="Discovery mean", markersize=5, markeredgecolor="white", markeredgewidth=0.5),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=cohort_color("discovery"),
               label="Discovery samples", markersize=3, markeredgewidth=0, alpha=0.5),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=cohort_color("validation"),
               label="Validation mean", markersize=5, markeredgecolor="white", markeredgewidth=0.5),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=cohort_color("validation"),
               label="Validation samples", markersize=3, markeredgewidth=0, alpha=0.5),
    ]
    ax2.legend(handles=legend_elements, fontsize=4.5, loc="lower right", framealpha=0.9)

    # Reference note to panel a (below x-axis label, in figure coords)
    fig_b.text(0.12, 0.01, "Individual sample identifiers shown in panel a for macrophage.",
               fontsize=4, color="gray", ha="left", va="bottom")

    fig_b.tight_layout()
    save_figure(fig_b, out_dir / "figS14b_celltype_per_sample")

    logger.info("  S14 complete.")


# ============================================================
# S17: Pathway-Level Reproducibility (Analysis 3)
# ============================================================

def supp_figure_s17(config, out_dir, logger):
    """S17: Pathway-level reproducibility — enrichment heatmap and pairwise comparison."""
    logger.info("S17: Pathway-Level Reproducibility")
    base = Path(config["output_dir"])
    repro_dir = base / "figure_data" / "reproducibility"

    # --- S17a: Pathway enrichment heatmap ---
    profiles_df = pd.read_csv(repro_dir / "pathway_repro_profiles.csv")

    # Build patient mapping for row ordering
    sample_to_patient = {}
    for cohort_key in ["discovery", "validation"]:
        mapping = config["cohorts"][cohort_key].get("patient_mapping", {})
        for pid, sids in mapping.items():
            for sid in sids:
                sample_to_patient[sid] = pid

    for cohort_key in ["discovery", "validation"]:
        cohort_name = config["cohorts"][cohort_key]["name"]
        coh_df = profiles_df[profiles_df["cohort"] == cohort_key]
        if coh_df.empty:
            continue

        pivot = coh_df.pivot(index="section_id", columns="pathway", values="cohens_d")

        # Order sections by patient, then by sample name
        sections = sorted(pivot.index, key=lambda s: (sample_to_patient.get(s, 'Z'), s))
        pivot = pivot.loc[sections]

        # Clean pathway names
        pathways = [p.replace('HALLMARK_', '').replace('_', ' ').title()
                    for p in pivot.columns]

        fig, ax = plt.subplots(figsize=(FULL_WIDTH, FULL_WIDTH * 0.35))
        vmax = np.percentile(np.abs(pivot.values), 95)
        im = ax.imshow(pivot.values, aspect='auto', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, interpolation='nearest')

        ax.set_yticks(range(len(sections)))
        ylabels = [f'{s} ({sample_to_patient.get(s, "?")})' for s in sections]
        ax.set_yticklabels(ylabels, fontsize=4)

        # Color y-tick labels by patient
        for idx, tick_label in enumerate(ax.get_yticklabels()):
            pid = sample_to_patient.get(sections[idx], 'unknown')
            tick_label.set_color(PATIENT_COLORS.get(pid, '#333333'))

        ax.set_xticks(range(len(pathways)))
        ax.set_xticklabels(pathways, fontsize=3.5, rotation=90, ha='right')

        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Cohen's d (disc. vs conc.)", fontsize=5)
        cbar.ax.tick_params(labelsize=4)

        cohort_label = COHORT_LABELS.get(cohort_key, cohort_key)
        ax.set_title(f'Pathway enrichment: {cohort_label}', fontsize=7, fontweight='bold')

        fig.tight_layout()
        save_figure(fig, out_dir / f"figS17{'a' if cohort_key == 'discovery' else 'b'}_pathway_heatmap_{cohort_key}")

    logger.info("  S17 complete.")


# ============================================================
# S18: COAD Generalization Overview (3 panels)
# ============================================================

COLOR_COAD = "#7B68EE"  # Medium slate blue

def supp_figure_s18(config, out_dir, logger):
    """S18: COAD generalization — Moran's I scatter, pathway heatmap, predictability comparison."""
    logger.info("S18: COAD Generalization Overview")
    base = Path(config["output_dir"])
    coad_dir = base / "coad"

    if not coad_dir.exists():
        logger.warning("  COAD output directory not found — run script 20 first. Skipping.")
        return

    # --- S18a: Moran's I vs Predictability scatter ---
    gf_path = coad_dir / "gene_features.csv"
    if not gf_path.exists():
        logger.warning("  gene_features.csv not found. Skipping S18.")
        return

    gf = pd.read_csv(gf_path).dropna(subset=["morans_i", "mean_pearson"])

    fig, ax = plt.subplots(figsize=(HALF_WIDTH, HALF_WIDTH * 0.85))
    ax.scatter(gf["morans_i"], gf["mean_pearson"], s=6, alpha=0.6,
               color=COLOR_COAD, edgecolors="none", zorder=2)

    # OLS trend line
    m, b = np.polyfit(gf["morans_i"], gf["mean_pearson"], 1)
    x_range = np.linspace(gf["morans_i"].min(), gf["morans_i"].max(), 100)
    ax.plot(x_range, m * x_range + b, color=COLOR_COAD, linewidth=1, alpha=0.8, zorder=1)

    # IDC trend for comparison
    try:
        idc_bio_gf = pd.read_csv(base / "phase3" / "gene_predictability" / "biomarkers" / "gene_features.csv")
        if "spatial_autocorrelation" in idc_bio_gf.columns:
            m_idc, b_idc = np.polyfit(idc_bio_gf["spatial_autocorrelation"],
                                       idc_bio_gf["mean_pearson"], 1)
            x_idc = np.linspace(gf["morans_i"].min(), gf["morans_i"].max(), 100)
            ax.plot(x_idc, m_idc * x_idc + b_idc, color=COLORS["discovery"],
                    linewidth=1, linestyle="--", alpha=0.6, label="IDC (discovery)", zorder=1)
            ax.legend(fontsize=6, frameon=False)
    except Exception:
        pass

    rho, p = stats.spearmanr(gf["morans_i"], gf["mean_pearson"])
    # Use "Spearman ρ" label instead of generic "r"
    if p < 1e-100:
        rho_text = f"Spearman ρ = {rho:.3f}\np < 1e-100"
    else:
        rho_text = f"Spearman ρ = {rho:.3f}\np = {p:.2e}"
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
    for _, row in outliers.iterrows():
        offset_y = 8 if row["ols_resid"] > 0 else -8
        ax.annotate(row["gene"], (row["morans_i"], row["mean_pearson"]),
                    textcoords="offset points", xytext=(5, offset_y),
                    fontsize=4.5, fontstyle="italic", alpha=0.9,
                    arrowprops=dict(arrowstyle="-", lw=0.3, alpha=0.5))

    ax.set_xlabel("Mean Moran's I")
    ax.set_ylabel("Mean Pearson r")
    ax.set_title("COAD: Spatial autocorrelation vs predictability")
    save_figure(fig, out_dir / "figS18a_coad_morans_scatter")
    plt.close(fig)

    # --- S18b: Pathway Enrichment Heatmap ---
    pw_path = coad_dir / "pathway_de.csv"
    if pw_path.exists():
        pw_de = pd.read_csv(pw_path)
        pivot = pw_de.pivot(index="sample_id", columns="pathway", values="cohens_d")
        fdr_pivot = pw_de.pivot(index="sample_id", columns="pathway", values="fdr")

        # Select pathways: significant in ≥1 sample, or top 15 by |mean d|
        mean_d = pivot.abs().mean().sort_values(ascending=False)
        sig_pathways = fdr_pivot.columns[fdr_pivot.min() < 0.05].tolist()
        top_pathways = sig_pathways if len(sig_pathways) >= 5 else mean_d.head(15).index.tolist()

        pivot = pivot[top_pathways]
        fdr_pivot = fdr_pivot[top_pathways]
        short_names = [p.replace("HALLMARK_", "").replace("_", " ").title() for p in top_pathways]

        n_rows = pivot.shape[0]
        fig_h = max(2.5, 0.45 * n_rows + 0.8)
        fig, ax = plt.subplots(figsize=(FULL_WIDTH, fig_h))
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                if fdr_pivot.values[i, j] < 0.05:
                    ax.text(j, i, "*", ha="center", va="center", fontsize=8, fontweight="bold")

        ax.set_xticks(range(len(short_names)))
        ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=5)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=6)
        ax.set_title("COAD: Pathway enrichment in discordant spots (Cohen's d)")

        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Cohen's d", fontsize=6)
        cbar.ax.tick_params(labelsize=5)

        fig.tight_layout()
        save_figure(fig, out_dir / "figS18b_coad_pathway_heatmap")
        plt.close(fig)
    else:
        logger.warning("  pathway_de.csv not found. Skipping S18b.")

    # --- S18c: Predictability Distribution Comparison ---
    try:
        idc_bio = pd.read_csv(base / "phase3" / "gene_predictability" / "biomarkers" / "gene_features.csv")
        idc_jan = pd.read_csv(base / "phase3" / "gene_predictability" / "10x_janesick" / "gene_features.csv")
    except FileNotFoundError:
        logger.warning("  IDC gene_features not found. Skipping S18c.")
        return

    fig, ax = plt.subplots(figsize=(HALF_WIDTH, HALF_WIDTH * 0.85))

    data = [idc_bio["mean_pearson"].values, idc_jan["mean_pearson"].values,
            gf["mean_pearson"].values]
    labels = ["IDC Discovery\n(280 genes)", "IDC Validation\n(280 genes)", "COAD\n(351 genes)"]
    colors_list = [COLORS["discovery"], COLORS["validation"], COLOR_COAD]

    parts = ax.violinplot(data, positions=[0, 1, 2], showextrema=False, showmedians=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors_list[i])
        pc.set_alpha(0.4)

    bp = ax.boxplot(data, positions=[0, 1, 2], widths=0.3, patch_artist=False,
                    showfliers=False, zorder=3)
    for element in ["boxes", "whiskers", "caps"]:
        for line in bp[element]:
            line.set_color("#333333")
            line.set_linewidth(0.8)
    for line in bp["medians"]:
        line.set_color("#333333")
        line.set_linewidth(1.2)

    for i, d in enumerate(data):
        med = np.median(d)
        ax.text(i, med + 0.02, f"{med:.3f}", ha="center", va="bottom", fontsize=5)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("Per-gene mean Pearson r")
    ax.set_title("Gene predictability: IDC vs COAD")
    ax.axhline(y=0, color="#999999", linewidth=0.5, linestyle="--", zorder=0)

    save_figure(fig, out_dir / "figS18c_coad_predictability_comparison")
    plt.close(fig)

    logger.info("  S18 complete.")


# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate supplementary figures")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--figure", type=int, nargs="*", default=None,
                        help="Which supplementary figures to generate. "
                             "Available: 1,3-14,17,18,22. Default: all.")
    args = parser.parse_args()

    config = load_config(args.config)

    log_dir = Path(config["output_dir"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(
        log_level=config.get("log_level", "INFO"),
        log_file=str(log_dir / "19_supplementary_figures.log"),
    )

    out_dir = Path(config["output_dir"]) / "figures" / "supplementary"
    out_dir.mkdir(parents=True, exist_ok=True)

    figures_to_gen = args.figure or [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18, 22]
    overall_start = time.time()

    figure_funcs = {
        1: supp_figure_s1,
        3: supp_figure_s3,
        4: supp_figure_s4,
        5: supp_figure_s5,
        6: supp_figure_s6,
        7: supp_figure_s7,
        8: supp_figure_s8,
        9: supp_figure_s9,
        10: supp_figure_s10,
        11: supp_figure_s11,
        12: supp_figure_s12,
        13: supp_figure_s13,
        14: supp_figure_s14,
        17: supp_figure_s17,
        18: supp_figure_s18,
    }

    for fig_num in figures_to_gen:
        if fig_num in figure_funcs:
            t0 = time.time()
            try:
                figure_funcs[fig_num](config, out_dir, logger)
                logger.info(f"  Time: {format_time(time.time() - t0)}")
            except Exception as e:
                logger.error(f"  S{fig_num} failed: {e}")
                import traceback
                traceback.print_exc()

    total_time = time.time() - overall_start
    logger.info(f"\nTotal time: {format_time(total_time)}")
    logger.info("Done.")


if __name__ == "__main__":
    main()

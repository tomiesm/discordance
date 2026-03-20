#!/usr/bin/env python3
"""
Figure 1: Study overview schematic.

Programmatic generation using matplotlib drawing primitives + embedded
real data thumbnails. Flat, clean, colorblind-safe design.

Panels:
  a) Prediction pipeline (H&E → encoders → ridge → D_cond)
  b) Spatial evidence (5 maps: H&E, observed, predicted, residual, quartiles)
  c) Biology of discordant spots (volcano, pathway, cell type thumbnails)
  d) Cross-cohort replication diagram
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.plotting import (
    setup_style, save_figure, FULL_WIDTH, COLORS,
    load_config, load_discordance_spatial, load_expression_for_gene,
    load_he_thumbnail,
)

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('discordance')

# ============================================================
# Color scheme (colorblind-safe, no red-green)
# ============================================================

C_PIPE_FILL = '#E8EEF5'     # steel blue tint (methods)
C_PIPE_EDGE = '#4A7FB5'     # steel blue border
C_DISC_FILL = '#FDEDED'     # vermilion tint (discordance - novel)
C_DISC_EDGE = '#D94A4A'     # vermilion border
C_BIO_FILL  = '#E5F3F3'     # teal tint (biology)
C_BIO_EDGE  = '#2CA6A4'     # teal border
C_DATA_FILL = '#F5F5F5'     # gray tint (data inputs)
C_DATA_EDGE = '#9E9E9E'     # gray border
C_ARROW     = '#424242'     # dark gray arrows
C_TEXT      = '#212121'     # near-black text
C_ANNOT     = '#616161'     # annotation text
C_DISC_COL  = COLORS['discovery']   # teal
C_VAL_COL   = COLORS['validation']  # coral
C_COAD_COL  = '#7B68AE'    # purple for COAD

# Quartile colors (blue→red, colorblind-safe)
Q_COLORS = ['#4A90D9', '#A8C4E0', '#E8A0A0', '#D94A4A']

# ============================================================
# Drawing helpers
# ============================================================

def draw_box(ax, x, y, w, h, fill, edge, linewidth=0.8, radius=0.015, zorder=2):
    """Draw a rounded rectangle in axes fraction coordinates."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fill, edgecolor=edge, linewidth=linewidth,
        transform=ax.transAxes, zorder=zorder, clip_on=False,
    )
    ax.add_patch(box)
    return box


def draw_arrow_between(ax, x0, y0, x1, y1, color=C_ARROW, lw=1.0,
                       style='->', head_w=4, head_l=4, zorder=3):
    """Draw an arrow in axes fraction coordinates."""
    arrow = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=f'{style},head_width={head_w},head_length={head_l}',
        color=color, linewidth=lw,
        transform=ax.transAxes, zorder=zorder, clip_on=False,
        mutation_scale=1,
    )
    ax.add_patch(arrow)
    return arrow


def text_in_box(ax, x, y, lines, fontsize=6, weight='normal', color=C_TEXT,
                ha='center', va='center', linespacing=1.3):
    """Place multi-line text at axes fraction position."""
    txt = '\n'.join(lines) if isinstance(lines, (list, tuple)) else lines
    ax.text(x, y, txt, fontsize=fontsize, fontweight=weight, color=color,
            ha=ha, va=va, transform=ax.transAxes, zorder=5,
            linespacing=linespacing, clip_on=False)


def load_and_crop_image(path, padding=10):
    """Load PNG, auto-crop whitespace, return numpy array."""
    img = Image.open(path).convert('RGB')
    arr = np.array(img)
    # Find non-white bounding box
    mask = arr.min(axis=2) < 250
    if mask.any():
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        rmin = max(0, rmin - padding)
        rmax = min(arr.shape[0], rmax + padding)
        cmin = max(0, cmin - padding)
        cmax = min(arr.shape[1], cmax + padding)
        arr = arr[rmin:rmax, cmin:cmax]
    return arr


# ============================================================
# Panel a: Prediction pipeline
# ============================================================

def panel_a_pipeline(fig, ax):
    """Draw the compressed prediction pipeline."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # --- H&E patch image ---
    he_path = Path('outputs/figures/schematic_crops/crop_he_patch.png')
    if he_path.exists():
        he_img = load_and_crop_image(he_path, padding=2)
        ax_he = fig.add_axes([0.02, 0.77, 0.055, 0.18])
        ax_he.imshow(he_img)
        ax_he.axis('off')
        # Border
        for sp in ax_he.spines.values():
            sp.set_visible(True)
            sp.set_color(C_DATA_EDGE)
            sp.set_linewidth(0.8)

    # Row positions (fraction of panel axes)
    top_y = 0.62   # top row center
    bot_y = 0.22   # bottom row center
    box_h = 0.30
    box_h2 = 0.24

    # --- H&E label ---
    text_in_box(ax, 0.06, top_y - 0.22, ['H&E patch', '224×224 px'],
                fontsize=5.5, color=C_ANNOT)

    # Arrow: H&E → Encoders
    draw_arrow_between(ax, 0.115, top_y, 0.155, top_y, lw=0.8)

    # --- Foundation Model Encoders box ---
    enc_x, enc_w = 0.16, 0.22
    draw_box(ax, enc_x, top_y - box_h/2, enc_w, box_h, C_PIPE_FILL, C_PIPE_EDGE)
    text_in_box(ax, enc_x + enc_w/2, top_y + 0.06,
                ['Foundation Model Encoders'], fontsize=6, weight='bold', color=C_PIPE_EDGE)
    text_in_box(ax, enc_x + enc_w/2, top_y - 0.04,
                ['UNI2-h · Virchow2 · H-Optimus-0'], fontsize=5, color=C_TEXT)
    text_in_box(ax, enc_x + enc_w/2, top_y - 0.12,
                ['1,536 / 2,560 / 1,536-d'], fontsize=4.5, color=C_ANNOT)

    # Arrow: Encoders → Ridge
    draw_arrow_between(ax, enc_x + enc_w + 0.01, top_y,
                       enc_x + enc_w + 0.04, top_y, lw=0.8)

    # --- Ridge Regression box ---
    ridge_x = enc_x + enc_w + 0.05
    ridge_w = 0.13
    draw_box(ax, ridge_x, top_y - box_h2/2, ridge_w, box_h2, C_PIPE_FILL, C_PIPE_EDGE)
    text_in_box(ax, ridge_x + ridge_w/2, top_y + 0.04,
                ['Ridge Regression'], fontsize=6, weight='bold', color=C_PIPE_EDGE)
    text_in_box(ax, ridge_x + ridge_w/2, top_y - 0.06,
                ['PCA(256), LOPO-CV'], fontsize=5, color=C_ANNOT)

    # Arrow: Ridge → Predicted
    pred_x = ridge_x + ridge_w + 0.01
    draw_arrow_between(ax, pred_x, top_y, pred_x + 0.03, top_y, lw=0.8)

    # --- Predicted ŷ label ---
    text_in_box(ax, pred_x + 0.06, top_y,
                ['Predicted ŷ'], fontsize=6, weight='bold', color=C_PIPE_EDGE)
    text_in_box(ax, pred_x + 0.06, top_y - 0.09,
                ['per gene, per spot'], fontsize=4.5, color=C_ANNOT)

    # Arrow: Predicted ŷ down to comparison
    compare_x = 0.72
    compare_y = 0.42
    draw_arrow_between(ax, pred_x + 0.06, top_y - 0.16,
                       compare_x, compare_y + 0.06,
                       color=C_PIPE_EDGE, lw=0.8)

    # --- Xenium input box (bottom-left) ---
    xen_x, xen_w, xen_h = 0.42, 0.15, 0.22
    draw_box(ax, xen_x, bot_y - xen_h/2, xen_w, xen_h, C_DATA_FILL, C_DATA_EDGE)
    text_in_box(ax, xen_x + xen_w/2, bot_y + 0.03,
                ['Xenium In Situ'], fontsize=6, weight='bold', color=C_TEXT)
    text_in_box(ax, xen_x + xen_w/2, bot_y - 0.06,
                ['280 genes / spot'], fontsize=5, color=C_ANNOT)

    # Arrow: Xenium → Observed y
    draw_arrow_between(ax, xen_x + xen_w + 0.01, bot_y,
                       xen_x + xen_w + 0.04, bot_y, lw=0.8, color=C_DATA_EDGE)

    # --- Observed y label ---
    text_in_box(ax, 0.645, bot_y,
                ['Observed y'], fontsize=6, weight='bold', color=C_DATA_EDGE)

    # Arrow: Observed y → comparison
    draw_arrow_between(ax, 0.685, bot_y + 0.05,
                       compare_x, compare_y - 0.06,
                       color=C_DATA_EDGE, lw=0.8)

    # --- Comparison circle (ŷ − y) ---
    circle = plt.Circle((compare_x, compare_y), 0.035,
                         facecolor='white', edgecolor=C_DISC_EDGE,
                         linewidth=1.5, transform=ax.transAxes,
                         zorder=4, clip_on=False)
    ax.add_patch(circle)
    ax.text(compare_x, compare_y, 'ŷ−y', fontsize=5.5, fontweight='bold',
            color=C_DISC_EDGE, ha='center', va='center',
            transform=ax.transAxes, zorder=5)

    # Arrow: comparison → D_cond
    draw_arrow_between(ax, compare_x + 0.045, compare_y,
                       compare_x + 0.08, compare_y,
                       color=C_DISC_EDGE, lw=1.2)

    # --- D_cond box (PROMINENT) ---
    dc_x = compare_x + 0.09
    dc_w, dc_h = 0.185, 0.42
    dc_cy = compare_y
    draw_box(ax, dc_x, dc_cy - dc_h/2, dc_w, dc_h,
             C_DISC_FILL, C_DISC_EDGE, linewidth=1.5, radius=0.02)
    text_in_box(ax, dc_x + dc_w/2, dc_cy + 0.10,
                ['Discordance'], fontsize=7, weight='bold', color=C_DISC_EDGE)
    text_in_box(ax, dc_x + dc_w/2, dc_cy + 0.02,
                ['Score'], fontsize=7, weight='bold', color=C_DISC_EDGE)
    text_in_box(ax, dc_x + dc_w/2, dc_cy - 0.07,
                ['D_cond'], fontsize=7, weight='bold', color=C_DISC_EDGE)
    text_in_box(ax, dc_x + dc_w/2, dc_cy - 0.16,
                ['depth-corrected', 'mean |ŷ − y|'], fontsize=5, color=C_ANNOT)

    logger.info("  Panel a: pipeline complete")


# ============================================================
# Panel b: Spatial evidence
# ============================================================

def _setup_spatial_ax(ax, x_coords, y_coords, bg_color='#F0F0F0'):
    """Configure axes for spatial scatter plots with consistent limits."""
    pad_x = (x_coords.max() - x_coords.min()) * 0.03
    pad_y = (y_coords.max() - y_coords.min()) * 0.03
    ax.set_xlim(x_coords.min() - pad_x, x_coords.max() + pad_x)
    ax.set_ylim(y_coords.max() + pad_y, y_coords.min() - pad_y)  # inverted
    ax.set_aspect('equal')
    ax.set_facecolor(bg_color)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color('#D0D0D0')
        sp.set_linewidth(0.4)


def panel_b_spatial(fig, config):
    """Render 4 spatial maps from TENX193 raw data.

    Maps: H&E tissue | Gene expression (GATA3) | D_cond (continuous) | D_cond quartiles
    """
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm
    from matplotlib.lines import Line2D

    sample_id = "TENX193"
    cohort_name = "biomarkers"

    # Load discordance data
    disc_df = load_discordance_spatial(sample_id, cohort_name, config)
    x_coords = disc_df['x'].values
    y_coords = disc_df['y'].values
    d_cond = disc_df['D_cond'].values

    # Load GATA3 expression
    barcodes_expr, gata3_vals = load_expression_for_gene(sample_id, 'GATA3', config)
    bc_to_expr = dict(zip(barcodes_expr, gata3_vals))
    gata3_aligned = np.array([bc_to_expr.get(bc, np.nan) for bc in disc_df['barcode']])

    # Compute quartiles for D_cond
    quartiles = np.digitize(d_cond, np.percentile(d_cond, [25, 50, 75])) + 1  # 1-4

    # Load H&E
    he_img = load_he_thumbnail(sample_id, config)

    # --- Layout: 4 maps with colorbars ---
    map_y = 0.40
    map_h = 0.22
    map_w = 0.195
    gap = 0.02
    start_x = 0.04

    positions = []
    for i in range(4):
        px = start_x + i * (map_w + gap)
        positions.append([px, map_y, map_w, map_h])

    # Spot size: median NN distance = 365 px in data space.
    # With ~28k data range mapped to ~1.4 inches, each spot should be ~1.8 pts diameter.
    spot_size = 5.5

    # --- Map 1: H&E tissue ---
    ax1 = fig.add_axes(positions[0])
    if he_img is not None:
        # Map H&E to spatial coordinate space using scalefactors
        from src.plotting import get_scalefactors
        sf = get_scalefactors(sample_id, config)
        tissue_sf = sf.get('tissue_downscaled_fullres_scalef',
                          sf.get('tissue_hires_scalef', 1.0))
        img_h, img_w = he_img.shape[:2]
        # Image pixel i corresponds to spatial coord i / tissue_sf
        he_extent = [0, img_w / tissue_sf, img_h / tissue_sf, 0]
        ax1.imshow(he_img, extent=he_extent, aspect='equal', interpolation='bilinear')
    _setup_spatial_ax(ax1, x_coords, y_coords, bg_color='white')
    ax1.set_facecolor('white')
    ax1.set_title('H&E tissue', fontsize=5.5, fontweight='bold', color=C_TEXT, pad=2)

    # --- Map 2: Observed GATA3 expression ---
    valid = ~np.isnan(gata3_aligned)
    vmin_expr = np.nanpercentile(gata3_aligned[valid], 2)
    vmax_expr = np.nanpercentile(gata3_aligned[valid], 98)

    ax2 = fig.add_axes(positions[1])
    sc2 = ax2.scatter(x_coords[valid], y_coords[valid], c=gata3_aligned[valid],
                      cmap='viridis', s=spot_size, alpha=0.9,
                      vmin=vmin_expr, vmax=vmax_expr, edgecolors='none', rasterized=True)
    _setup_spatial_ax(ax2, x_coords, y_coords)
    ax2.set_title('GATA3 expression', fontsize=5.5, fontweight='bold', color=C_TEXT, pad=2)

    # --- Map 3: D_cond (continuous) ---
    vmin_d = np.percentile(d_cond, 2)
    vmax_d = np.percentile(d_cond, 98)

    ax3 = fig.add_axes(positions[2])
    sc3 = ax3.scatter(x_coords, y_coords, c=d_cond,
                      cmap='magma', s=spot_size, alpha=0.9,
                      vmin=vmin_d, vmax=vmax_d, edgecolors='none', rasterized=True)
    _setup_spatial_ax(ax3, x_coords, y_coords)
    ax3.set_title('Discordance (D_cond)', fontsize=5.5, fontweight='bold', color=C_TEXT, pad=2)

    # --- Map 4: D_cond quartiles ---
    ax4 = fig.add_axes(positions[3])
    for q in range(1, 5):
        mask = quartiles == q
        ax4.scatter(x_coords[mask], y_coords[mask], c=Q_COLORS[q-1],
                    s=spot_size, alpha=0.9, edgecolors='none', rasterized=True,
                    label=f'Q{q}')
    _setup_spatial_ax(ax4, x_coords, y_coords)
    ax4.set_title('D_cond quartiles', fontsize=5.5, fontweight='bold', color=C_TEXT, pad=2)

    # Quartile legend below map 4
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=Q_COLORS[i],
                              markersize=3.5, label=f'Q{i+1}') for i in range(4)]
    ax4.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.13),
               ncol=4, fontsize=4.5, frameon=False, handletextpad=0.2, columnspacing=0.6)

    # --- Shared colorbars below maps 2 and 3 ---
    cbar_h = 0.010
    cbar_y = map_y - 0.030

    # GATA3 colorbar
    cax2 = fig.add_axes([positions[1][0] + 0.01, cbar_y,
                         positions[1][2] - 0.02, cbar_h])
    cb2 = plt.colorbar(sc2, cax=cax2, orientation='horizontal')
    cb2.ax.tick_params(labelsize=4)
    cb2.set_label('Expression', fontsize=4.5, labelpad=1)

    # D_cond colorbar
    cax3 = fig.add_axes([positions[2][0] + 0.01, cbar_y,
                         positions[2][2] - 0.02, cbar_h])
    cb3 = plt.colorbar(sc3, cax=cax3, orientation='horizontal')
    cb3.ax.tick_params(labelsize=4)
    cb3.set_label('D_cond', fontsize=4.5, labelpad=1)

    # --- Connecting arrows between maps ---
    for i in range(3):
        x_start = positions[i][0] + positions[i][2] + 0.002
        x_end = positions[i+1][0] - 0.002
        y_mid = map_y + map_h / 2
        fig.patches.append(FancyArrowPatch(
            (x_start, y_mid), (x_end, y_mid),
            arrowstyle='->,head_width=3,head_length=3',
            color=C_ANNOT, linewidth=0.6,
            transform=fig.transFigure, clip_on=False,
            mutation_scale=1,
        ))

    # Annotation below maps
    fig.text(0.50, cbar_y - 0.022,
             'Spatially coherent regions  \u00b7  Moran\'s I = 0.47\u20130.70, p < 0.001 in all 18 samples',
             fontsize=5, color=C_ANNOT, ha='center', va='top', style='italic')

    logger.info("  Panel b: spatial maps complete")


# ============================================================
# Panel c: Biology of discordant spots
# ============================================================

def panel_c_biology(fig):
    """Embed cropped thumbnails from existing figure panels."""
    panels = [
        ('outputs/figures/main/fig3d_volcano.png', 'Differential expression'),
        ('outputs/figures/main/fig3c_pathway_heatmap.png', 'Pathway enrichment'),
        ('outputs/figures/main/fig3b_deconvolution_dumbbell.png', 'Cell type composition'),
    ]

    thumb_y = 0.04
    thumb_h = 0.24
    gap = 0.008
    start_x = 0.02
    # Variable widths: volcano is wide (2.8:1), heatmap medium (1.4:1), dumbbell narrow (1.2:1)
    thumb_widths = [0.20, 0.175, 0.155]

    px = start_x
    for i, (path, title) in enumerate(panels):
        tw = thumb_widths[i]
        ax = fig.add_axes([px, thumb_y, tw, thumb_h])

        p = Path(path)
        if p.exists():
            img = load_and_crop_image(p, padding=5)
            ax.imshow(img)
        ax.axis('off')
        ax.set_title(title, fontsize=5, fontweight='bold', color=C_BIO_EDGE, pad=2)

        # Light border
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(C_BIO_EDGE)
            sp.set_linewidth(0.5)
        px += tw + gap

    # Finding callout below thumbnails
    total_w = sum(thumb_widths) + 2 * gap
    callout_x = start_x + total_w / 2
    fig.text(callout_x, thumb_y - 0.01,
             'EMT + immune activation  \u00b7  epithelial depletion  \u00b7  macrophage enrichment',
             fontsize=4.5, color=C_BIO_EDGE, ha='center', va='top',
             fontweight='bold', style='italic')

    logger.info("  Panel c: biology thumbnails complete")


# ============================================================
# Panel d: Cross-cohort replication
# ============================================================

def panel_d_replication(fig):
    """Draw the replication diagram with stacked cohort boxes."""
    ax = fig.add_axes([0.58, 0.02, 0.40, 0.28])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Three stacked boxes
    cohorts = [
        ('Discovery', 'IDC, n = 11, 4 patients', C_DISC_COL, 0.78),
        ('Validation', 'IDC, n = 7, 4 patients', C_VAL_COL, 0.48),
        ('COAD', 'CRC, n = 4, 4 patients', C_COAD_COL, 0.18),
    ]

    box_w = 0.75
    box_h = 0.18
    box_x = 0.12

    for name, detail, color, cy in cohorts:
        # Light fill version of the color
        rgb = mcolors.to_rgb(color)
        fill = tuple(c * 0.15 + 0.85 for c in rgb)

        draw_box(ax, box_x, cy - box_h/2, box_w, box_h, fill, color, linewidth=1.2)
        text_in_box(ax, box_x + box_w/2, cy + 0.02,
                    [name], fontsize=6.5, weight='bold', color=color)
        text_in_box(ax, box_x + box_w/2, cy - 0.05,
                    [detail], fontsize=4.5, color=C_ANNOT)

    # Arrows between boxes
    ax.annotate('', xy=(box_x + box_w/2, 0.69),
                xytext=(box_x + box_w/2, 0.78 - box_h/2 - 0.01),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=0.8),
                transform=ax.transAxes, annotation_clip=False)
    ax.text(box_x + box_w + 0.02, 0.73, 'replicates', fontsize=4.5,
            color=C_ANNOT, style='italic', rotation=0, va='center',
            transform=ax.transAxes)

    ax.annotate('', xy=(box_x + box_w/2, 0.39),
                xytext=(box_x + box_w/2, 0.48 - box_h/2 - 0.01),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=0.8),
                transform=ax.transAxes, annotation_clip=False)
    ax.text(box_x + box_w + 0.02, 0.43, 'generalizes', fontsize=4.5,
            color=C_ANNOT, style='italic', rotation=0, va='center',
            transform=ax.transAxes)

    # Badge at bottom
    ax.text(box_x + box_w/2, 0.03,
            '25/25 shared pathways replicate',
            fontsize=5.5, fontweight='bold', color=C_TEXT,
            ha='center', va='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0F0F0',
                      edgecolor=C_DATA_EDGE, linewidth=0.5))

    logger.info("  Panel d: replication diagram complete")


# ============================================================
# Main figure assembly
# ============================================================

def main():
    logger.info("Figure 1: Study Overview Schematic")
    setup_style()
    config = load_config()

    # Figure dimensions: 180mm wide × ~120mm tall
    fig_h = FULL_WIDTH * 0.67  # aspect ratio ~1.5:1
    fig = plt.figure(figsize=(FULL_WIDTH, fig_h), facecolor='white')

    # Layout: 3 rows
    #   Row 1 (pipeline):  y = 0.70 to 1.0  (30%)
    #   Row 2 (spatial):   y = 0.32 to 0.70 (38%)
    #   Row 3 (bio+repl):  y = 0.0  to 0.32 (32%)
    row1_bot = 0.70
    row2_bot = 0.32

    # Subtle separator lines
    fig.patches.append(mpatches.Rectangle(
        (0.02, row1_bot - 0.002), 0.96, 0.001,
        facecolor='#E0E0E0', transform=fig.transFigure, zorder=1))
    fig.patches.append(mpatches.Rectangle(
        (0.02, row2_bot - 0.002), 0.96, 0.001,
        facecolor='#E0E0E0', transform=fig.transFigure, zorder=1))

    # --- Panel a: Pipeline (top row) ---
    ax_a = fig.add_axes([0.0, row1_bot, 1.0, 1.0 - row1_bot])
    ax_a.axis('off')
    panel_a_pipeline(fig, ax_a)

    # --- Panel b: Spatial evidence (middle row) ---
    panel_b_spatial(fig, config)

    # --- Panel c: Biology (bottom-left) ---
    panel_c_biology(fig)

    # --- Panel d: Replication (bottom-right) ---
    panel_d_replication(fig)

    # --- Panel labels (8pt bold lowercase) ---
    label_props = dict(fontsize=9, fontweight='bold', color=C_TEXT,
                       ha='left', va='top')
    fig.text(0.01, 0.99, 'a', **label_props)
    fig.text(0.01, row1_bot - 0.005, 'b', **label_props)
    fig.text(0.01, row2_bot - 0.005, 'c', **label_props)
    fig.text(0.57, row2_bot - 0.005, 'd', **label_props)

    # --- Row titles ---
    fig.text(0.04, 0.98, 'Morpho-transcriptomic prediction pipeline',
             fontsize=6.5, fontweight='bold', color=C_PIPE_EDGE, va='top')
    fig.text(0.04, row1_bot - 0.015, 'Discordance reveals spatially coherent tissue regions',
             fontsize=6.5, fontweight='bold', color=C_DISC_EDGE, va='top')
    fig.text(0.04, row2_bot - 0.015, 'Biology of discordant spots',
             fontsize=6.5, fontweight='bold', color=C_BIO_EDGE, va='top')
    fig.text(0.60, row2_bot - 0.015, 'Cross-cohort replication',
             fontsize=6.5, fontweight='bold', color=C_TEXT, va='top')

    # --- Save ---
    out_path = 'outputs/figures/main/fig1_schematic'
    save_figure(fig, out_path, formats=('pdf', 'png', 'svg'), dpi=300)
    logger.info(f"  Saved: {out_path}.pdf / .png")
    logger.info("Done.")


if __name__ == '__main__':
    main()

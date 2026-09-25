"""
Shared plotting utilities for v3 figure generation.

Provides consistent color palette, journal-quality style setup,
and reusable helper functions for spatial maps, forest plots,
heatmaps, and other common figure elements.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import scanpy as sc

# ============================================================
# Color palette (from v3_figure_table_plan.md)
# ============================================================

COLORS = {
    # Cohorts
    'discovery': '#2CA6A4',
    'validation': '#E8785E',
    # Discordance status
    'concordant': '#4A90D9',
    'discordant': '#D94A4A',
    'neutral': '#B0B0B0',
    # Encoders
    'uni': '#1B2A4A',
    'virchow2': '#2D6A2E',
    'hoptimus0': '#D4710A',
}

COHORT_LABELS = {
    'discovery': 'Discovery (Biomarkers)',
    'validation': 'Validation (10x+Janesick)',
}

ENCODER_LABELS = {
    'uni': 'UNI2-h',
    'virchow2': 'Virchow2',
    'hoptimus0': 'H-Optimus-0',
}

# Localization category colors (ColorBrewer Set2, colorblind-safe)
LOC_COLORS = {
    'Membrane': '#66C2A5',
    'Cytoplasm': '#FC8D62',
    'Nucleus': '#8DA0CB',
    'Secreted/Extracellular': '#E78AC3',
    'Unknown': '#A6D854',
}

# Patient identity colors (Set1-derived, colorblind-safe)
PATIENT_COLORS = {
    'P01': '#E41A1C',
    'P02': '#377EB8',
    'P03': '#4DAF4A',
    'P04': '#984EA3',
    'P05': '#FF7F00',
    'P06': '#A65628',
    'P07': '#F781BF',
    'P08': '#999999',
}


# ============================================================
# Style setup
# ============================================================

def setup_style():
    """Configure matplotlib rcParams for journal-quality figures."""
    plt.rcParams.update({
        # Font
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
        'mathtext.fontset': 'dejavusans',
        'font.size': 7,
        'axes.titlesize': 8,
        'axes.labelsize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6,
        # Lines
        'axes.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size': 2,
        'ytick.major.size': 2,
        'lines.linewidth': 1.0,
        'lines.markersize': 3,
        # Layout
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        # Colormap
        'image.cmap': 'viridis',
    })


# ============================================================
# Figure dimensions (mm → inches)
# ============================================================

MM_PER_INCH = 25.4
FULL_WIDTH = 180 / MM_PER_INCH    # ~7.09 inches
HALF_WIDTH = 88 / MM_PER_INCH     # ~3.46 inches
COLUMN_1_5 = 140 / MM_PER_INCH    # ~5.51 inches


# ============================================================
# Save helper
# ============================================================

def save_figure(fig, path, formats=('pdf', 'png'), dpi=300):
    """Save figure in multiple formats.

    Args:
        fig: matplotlib Figure.
        path: Base path without extension (e.g., 'outputs/figures/main/fig1').
        formats: Tuple of format strings.
        dpi: DPI for raster formats.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(str(path) + f'.{fmt}', dpi=dpi, bbox_inches='tight',
                    pad_inches=0.05, facecolor='white')
    plt.close(fig)


# ============================================================
# Data loading helpers
# ============================================================

def load_config(config_path="config.yaml"):
    """Load v3 config."""
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_json(path):
    """Load JSON file, return None if missing."""
    path = Path(path)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_discordance_spatial(sample_id, cohort_name, config):
    """Load discordance parquet and compute average D_cond across 3 ridge encoders.

    Returns:
        DataFrame with columns: spot_id, barcode, x, y, D_cond, total_expr
    """
    base = Path(config["output_dir"])
    pq_path = base / "phase2" / "scores" / cohort_name / f"{sample_id}_discordance.parquet"
    df = pd.read_parquet(pq_path)

    # Average D_cond across 3 ridge encoders
    ridge_cols = [c for c in df.columns if c.startswith('D_cond_') and c.endswith('_ridge')]
    df['D_cond'] = df[ridge_cols].mean(axis=1)

    return df[['spot_id', 'barcode', 'x', 'y', 'D_cond', 'total_expr']].copy()


def load_he_thumbnail(sample_id, config):
    """Load H&E thumbnail image from h5ad file.

    Returns:
        (H, W, 3) uint8 numpy array.
    """
    hest_dir = Path(config["hest_dir"])
    h5ad_path = hest_dir / "st" / f"{sample_id}.h5ad"
    adata = sc.read_h5ad(str(h5ad_path), backed='r')
    # Navigate spatial storage
    spatial = adata.uns.get('spatial', {})
    for lib_key in spatial:
        images = spatial[lib_key].get('images', {})
        if 'downscaled_fullres' in images:
            img = np.array(images['downscaled_fullres'])
            adata.file.close()
            return img
    adata.file.close()
    return None


def get_scalefactors(sample_id, config):
    """Get spatial scale factors from h5ad for coordinate mapping."""
    hest_dir = Path(config["hest_dir"])
    h5ad_path = hest_dir / "st" / f"{sample_id}.h5ad"
    adata = sc.read_h5ad(str(h5ad_path), backed='r')
    spatial = adata.uns.get('spatial', {})
    for lib_key in spatial:
        sf = spatial[lib_key].get('scalefactors', {})
        if sf:
            adata.file.close()
            return sf
    adata.file.close()
    return {}


def load_expression_for_gene(sample_id, gene_name, config):
    """Load expression values for a single gene from h5ad.

    Returns:
        (barcodes, values) tuple.
    """
    hest_dir = Path(config["hest_dir"])
    h5ad_path = hest_dir / "st" / f"{sample_id}.h5ad"
    adata = sc.read_h5ad(str(h5ad_path))
    if gene_name in adata.var_names:
        x_sub = adata[:, gene_name].X
        if hasattr(x_sub, 'todense'):
            vals = np.array(x_sub.todense()).ravel()
        elif hasattr(x_sub, 'toarray'):
            vals = np.array(x_sub.toarray()).ravel()
        else:
            vals = np.asarray(x_sub).ravel()
    else:
        vals = np.zeros(adata.n_obs)
    barcodes = list(adata.obs_names)
    return barcodes, vals


# ============================================================
# Plot helpers
# ============================================================

def spatial_scatter(ax, x, y, values, cmap='viridis', vmin=None, vmax=None,
                    s=1, alpha=0.8, title=None, colorbar=True, cbar_label=None):
    """Standard spatial scatter plot with consistent styling.

    Args:
        ax: matplotlib Axes.
        x, y: coordinate arrays.
        values: color values (or single color string).
        cmap: colormap name.
        vmin, vmax: color limits.
        s: point size.
        alpha: transparency.
        title: panel title.
        colorbar: whether to add colorbar.
        cbar_label: colorbar label.
    """
    sc_plot = ax.scatter(x, y, c=values, cmap=cmap, s=s, alpha=alpha,
                         vmin=vmin, vmax=vmax, edgecolors='none', rasterized=True)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=8, fontweight='bold')
    if colorbar and not isinstance(values, str):
        cbar = plt.colorbar(sc_plot, ax=ax, fraction=0.046, pad=0.04)
        if cbar_label:
            cbar.set_label(cbar_label, fontsize=6)
        cbar.ax.tick_params(labelsize=5)
    return sc_plot


def annotate_r(ax, r, p=None, pos='upper left', fontsize=6):
    """Add correlation annotation to axes.

    Args:
        ax: matplotlib Axes.
        r: correlation coefficient.
        p: p-value (optional).
        pos: 'upper left', 'upper right', 'lower left', 'lower right'.
        fontsize: annotation font size.
    """
    if p is not None and p < 1e-100:
        text = f'r = {r:.3f}\np < 1e-100'
    elif p is not None:
        text = f'r = {r:.3f}\np = {p:.2e}'
    else:
        text = f'r = {r:.3f}'

    pos_map = {
        'upper left': (0.05, 0.95),
        'upper right': (0.95, 0.95),
        'lower left': (0.05, 0.05),
        'lower right': (0.95, 0.05),
    }
    ha_map = {
        'upper left': 'left', 'upper right': 'right',
        'lower left': 'left', 'lower right': 'right',
    }
    va_map = {
        'upper left': 'top', 'upper right': 'top',
        'lower left': 'bottom', 'lower right': 'bottom',
    }
    xy = pos_map.get(pos, (0.05, 0.95))
    ax.text(xy[0], xy[1], text, transform=ax.transAxes,
            fontsize=fontsize, ha=ha_map.get(pos, 'left'),
            va=va_map.get(pos, 'top'),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='gray', alpha=0.8))


def add_panel_label(ax, label, x=-0.15, y=1.05, fontsize=10):
    """Add panel label (a, b, c, ...) to axes."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold', va='bottom', ha='right')


def cohort_color(key):
    """Get color for a cohort key."""
    return COLORS.get(key, COLORS['neutral'])

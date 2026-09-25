"""
Cell type deconvolution for Phase 3.

Strategy 1: Signature-based scoring (mean expression of marker genes per cell type).
Strategy 2: CellViT nuclear morphometry (per-spot nuclear features from segmentation).
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.spatial import cKDTree


# Literature-derived cell type signatures
CELL_TYPE_SIGNATURES = {
    'epithelial': ['EPCAM', 'KRT18', 'KRT19', 'KRT8', 'CDH1', 'MUC1'],
    'fibroblast': ['COL1A1', 'COL3A1', 'DCN', 'FAP', 'ACTA2', 'VIM', 'LUM'],
    'T_cell': ['CD3D', 'CD3E', 'CD8A', 'CD4', 'GZMB', 'PRF1'],
    'macrophage': ['CD68', 'CD163', 'CSF1R', 'CD14', 'ITGAM'],
    'B_cell': ['CD79A', 'CD79B', 'MS4A1', 'CD19'],
    'endothelial': ['PECAM1', 'VWF', 'CDH5', 'ERG'],
    'NK_cell': ['NCAM1', 'NKG7', 'GNLY', 'KLRD1'],
    'mast_cell': ['KIT', 'TPSAB1', 'TPSB2', 'CPA3'],
}


def get_available_signatures(gene_names: List[str],
                             signatures: Optional[Dict[str, List[str]]] = None,
                             min_genes: int = 2) -> Dict[str, List[str]]:
    """Filter signatures to those with sufficient gene overlap.

    Args:
        gene_names: Available gene names in the expression data
        signatures: Cell type signatures dict. Defaults to CELL_TYPE_SIGNATURES.
        min_genes: Minimum number of genes required per signature

    Returns:
        Dict mapping cell type to list of available genes
    """
    if signatures is None:
        signatures = CELL_TYPE_SIGNATURES

    gene_set = set(gene_names)
    available = {}

    for cell_type, genes in signatures.items():
        overlap = [g for g in genes if g in gene_set]
        if len(overlap) >= min_genes:
            available[cell_type] = overlap

    return available


def compute_celltype_scores(expression: np.ndarray,
                            gene_names: List[str],
                            signatures: Optional[Dict[str, List[str]]] = None,
                            min_genes: int = 2) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Compute per-spot cell type scores as mean expression of signature genes.

    Args:
        expression: Dense expression matrix (N_spots, N_genes), log1p-normalized
        gene_names: Gene names matching columns
        signatures: Cell type signatures. Defaults to CELL_TYPE_SIGNATURES.
        min_genes: Minimum genes per signature

    Returns:
        scores_df: DataFrame (N_spots, N_cell_types) with cell type scores
        used_genes: Dict mapping cell type to list of genes actually used
    """
    available = get_available_signatures(gene_names, signatures, min_genes)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    scores = {}
    for cell_type, genes in available.items():
        idx = [gene_to_idx[g] for g in genes]
        scores[cell_type] = np.mean(expression[:, idx], axis=1)

    scores_df = pd.DataFrame(scores)
    return scores_df, available


def gene_availability_report(gene_names: List[str],
                             signatures: Optional[Dict[str, List[str]]] = None
                             ) -> pd.DataFrame:
    """Report which signature genes are available in the expression data.

    Args:
        gene_names: Available gene names
        signatures: Cell type signatures. Defaults to CELL_TYPE_SIGNATURES.

    Returns:
        DataFrame with cell_type, gene, available columns
    """
    if signatures is None:
        signatures = CELL_TYPE_SIGNATURES

    gene_set = set(gene_names)
    records = []

    for cell_type, genes in signatures.items():
        for gene in genes:
            records.append({
                'cell_type': cell_type,
                'gene': gene,
                'available': gene in gene_set,
            })

    return pd.DataFrame(records)


def find_cellvit_file(sample_id: str, cellvit_dir: str) -> Optional[Path]:
    """Find the CellViT segmentation file for a sample.

    Args:
        sample_id: Sample identifier (e.g., 'TENX96')
        cellvit_dir: Directory containing CellViT files

    Returns:
        Path to the JSON/GeoJSON file, or None if not found
    """
    cellvit_path = Path(cellvit_dir)
    if not cellvit_path.exists():
        return None

    # Search for matching files (JSON or GeoJSON)
    for pattern in [f'*{sample_id}*.geojson', f'*{sample_id}*.json']:
        matches = list(cellvit_path.glob(pattern))
        if matches:
            return matches[0]

    # Check in subdirectories
    for pattern in [f'**/*{sample_id}*.geojson', f'**/*{sample_id}*.json']:
        matches = list(cellvit_path.glob(pattern))
        if matches:
            return matches[0]

    return None


def load_cellvit_nuclei(cellvit_path: Path) -> List[dict]:
    """Load nuclei from a CellViT JSON/GeoJSON file.

    Args:
        cellvit_path: Path to the segmentation file

    Returns:
        List of nucleus dicts with keys: centroid, class_label, contour, area, etc.
    """
    with open(cellvit_path, 'r') as f:
        data = json.load(f)

    nuclei = []

    if 'features' in data:
        # GeoJSON format
        for feat in data['features']:
            props = feat.get('properties', {})
            geom = feat.get('geometry', {})

            # Extract centroid
            if 'centroid' in props:
                centroid = props['centroid']
            elif geom.get('type') == 'Point':
                centroid = geom['coordinates']
            elif geom.get('type') == 'Polygon':
                coords = np.array(geom['coordinates'][0])
                centroid = coords.mean(axis=0).tolist()
            else:
                continue

            # Extract class label
            class_label = props.get('type_tissue',
                         props.get('class', props.get('classification', 'unknown')))

            # Compute area from polygon if available
            area = props.get('area', None)
            if area is None and geom.get('type') == 'Polygon':
                coords = np.array(geom['coordinates'][0])
                # Shoelace formula
                x, y = coords[:, 0], coords[:, 1]
                area = 0.5 * abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))

            nuclei.append({
                'centroid': centroid,
                'class_label': str(class_label),
                'area': area,
            })

    elif 'nuc' in data:
        # HEST-specific dict format
        for nuc_id, nuc_data in data['nuc'].items():
            centroid = nuc_data.get('centroid', None)
            if centroid is None:
                continue

            class_label = nuc_data.get('type', 'unknown')
            # Map numeric labels
            label_map = {
                0: 'background', 1: 'neoplastic', 2: 'inflammatory',
                3: 'connective', 4: 'necrotic', 5: 'non_neoplastic_epithelial'
            }
            if isinstance(class_label, (int, float)):
                class_label = label_map.get(int(class_label), str(class_label))

            contour = nuc_data.get('contour', None)
            area = None
            if contour is not None:
                coords = np.array(contour)
                if len(coords) >= 3:
                    x, y = coords[:, 0], coords[:, 1]
                    area = 0.5 * abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))

            nuclei.append({
                'centroid': centroid,
                'class_label': class_label,
                'area': area,
            })

    return nuclei


def extract_nuclear_morphometry(cellvit_path: Path,
                                 spot_coords: np.ndarray,
                                 spot_radius_px: float = 56.0
                                 ) -> pd.DataFrame:
    """Extract per-spot nuclear features from CellViT segmentation.

    For each spot, find all nuclei within the spot radius and compute
    aggregate nuclear morphometry features.

    Args:
        cellvit_path: Path to CellViT JSON/GeoJSON file
        spot_coords: (N_spots, 2) array of spot centroid coordinates in pixels
        spot_radius_px: Spot radius in pixels (Visium ~56px at 20x, Xenium varies)

    Returns:
        DataFrame with columns: nuclear_count, mean_nuclear_area,
        var_nuclear_area, frac_neoplastic, frac_inflammatory, frac_stromal,
        nuclear_density
    """
    nuclei = load_cellvit_nuclei(cellvit_path)
    if not nuclei:
        return pd.DataFrame()

    # Build KD-tree of nucleus centroids
    nuc_centroids = np.array([n['centroid'] for n in nuclei])
    nuc_tree = cKDTree(nuc_centroids)

    # Pre-compute nucleus properties
    nuc_areas = np.array([n.get('area') or 0 for n in nuclei], dtype=float)
    nuc_classes = [n['class_label'] for n in nuclei]

    records = []
    for i in range(len(spot_coords)):
        # Find nuclei within spot radius
        idx_in_spot = nuc_tree.query_ball_point(spot_coords[i], r=spot_radius_px)

        n_nuclei = len(idx_in_spot)
        if n_nuclei == 0:
            records.append({
                'nuclear_count': 0,
                'mean_nuclear_area': np.nan,
                'var_nuclear_area': np.nan,
                'frac_neoplastic': np.nan,
                'frac_inflammatory': np.nan,
                'frac_stromal': np.nan,
                'nuclear_density': 0,
            })
            continue

        areas = nuc_areas[idx_in_spot]
        classes = [nuc_classes[j] for j in idx_in_spot]

        spot_area = np.pi * spot_radius_px ** 2
        class_counts = pd.Series(classes).value_counts()

        records.append({
            'nuclear_count': n_nuclei,
            'mean_nuclear_area': float(np.mean(areas)) if np.any(areas > 0) else np.nan,
            'var_nuclear_area': float(np.var(areas)) if np.any(areas > 0) else np.nan,
            'frac_neoplastic': class_counts.get('neoplastic', 0) / n_nuclei,
            'frac_inflammatory': class_counts.get('inflammatory', 0) / n_nuclei,
            'frac_stromal': class_counts.get('connective', 0) / n_nuclei,
            'nuclear_density': n_nuclei / spot_area,
        })

    return pd.DataFrame(records)

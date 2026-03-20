#!/usr/bin/env python3
"""
Script 21: IDC Visium Cross-Technology Generalization

Single orchestration script for IDC Visium analysis. Demonstrates that the
discordance framework generalizes across spatial transcriptomics technologies
(targeted Xenium → whole-transcriptome Visium) within the same tissue type.

Pipeline:
  1. QC & gene panel computation
  2. Embedding extraction (3 encoders, GPU)
  3. LOSO cross-validation with ridge regression
  4. Discordance score computation
  5. Moran's I per gene (spatial autocorrelation)
  6. Gene features aggregation
  7. Pathway enrichment
  8. Analysis results (A-D, cross-technology)
  9. Figure generation

Output:
    outputs/idc_visium/
    outputs/figures/idc_visium/

Usage:
    python scripts/21_idc_visium_generalization.py
    python scripts/21_idc_visium_generalization.py --skip-embeddings
    python scripts/21_idc_visium_generalization.py --skip-figures
    python scripts/21_idc_visium_generalization.py --gpu 0
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, seed_everything, format_time, checkpoint_exists
from src.embeddings import get_encoder
from src.regressors import get_regressor
from src.discordance import (
    compute_mean_absolute_discordance,
    compute_conditional_discordance,
)
from src.spatial import build_spatial_weights, morans_i
from src.pathways import (
    load_gene_sets,
    compute_studentized_gene_residuals,
    compute_studentized_pathway_scores,
)
from src.de_analysis import partition_spots_by_discordance
from src.plotting import (
    setup_style, save_figure, annotate_r,
    FULL_WIDTH, HALF_WIDTH, COLORS,
)

# ============================================================
# Constants
# ============================================================

VISIUM_SAMPLES = [
    "TENX13", "TENX14", "TENX39", "TENX53", "TENX68",
    "NCBI776", "NCBI681", "NCBI682", "NCBI683", "NCBI684",
]
HEST_DIR = "./data/hest"
OUTPUT_DIR = "./outputs/idc_visium"
IDC_OUTPUT_DIR = "./outputs"
FIGURE_DIR = "./outputs/figures/idc_visium"
HALLMARK_GMT = "data/gene_sets/h.all.v2024.1.Hs.symbols.gmt"
SEED = 42

CONTROL_PREFIXES = [
    "UnassignedCodeword", "BLANK", "DeprecatedCodeword",
    "NegControlCodeword", "NegControlProbe", "antisense_",
]

ENCODER_CONFIGS = [
    {"name": "uni", "model_id": "MahmoodLab/UNI2-h", "embed_dim": 1536, "batch_size_per_gpu": 64},
    {"name": "virchow2", "model_id": "paige-ai/Virchow2", "embed_dim": 2560, "batch_size_per_gpu": 64},
    {"name": "hoptimus0", "model_id": "bioptimus/H-optimus-0", "embed_dim": 1536, "batch_size_per_gpu": 96},
]

REGRESSOR_CONFIG = {"name": "ridge", "type": "ridge_fixed", "pca_components": 256}

COLOR_VISIUM = "#2E8B57"  # Sea green

N_BINS = 10
MIN_BIN_SIZE = 30
PATHWAY_MIN_OVERLAP = 5
SPATIAL_N_NEIGHBORS = 6


# ============================================================
# Utility: Patch Dataset (from scripts/02_extract_embeddings.py)
# ============================================================

class SamplePatchDataset(Dataset):
    """Load patches for a single sample from HEST HDF5 file."""

    def __init__(self, sample_id, hest_dir, transform=None):
        self.sample_id = sample_id
        self.patches_path = Path(hest_dir) / "patches" / f"{sample_id}.h5"
        self.transform = transform

        if not self.patches_path.exists():
            raise FileNotFoundError(f"Patches not found: {self.patches_path}")

        with h5py.File(self.patches_path, "r") as f:
            if "barcode" in f:
                raw = f["barcode"][:]
                if raw.dtype == object or isinstance(raw[0][0], bytes):
                    self.spot_ids = [
                        s[0].decode() if isinstance(s[0], bytes) else str(s[0])
                        for s in raw
                    ]
                else:
                    self.spot_ids = [str(s[0]) for s in raw]
            else:
                n = len(f["img"])
                self.spot_ids = [f"{sample_id}_{i}" for i in range(n)]

            self.n_spots = len(self.spot_ids)

    def __len__(self):
        return self.n_spots

    def __getitem__(self, idx):
        with h5py.File(self.patches_path, "r") as f:
            patch = f["img"][idx]

        patch = patch.astype(np.float32) / 255.0
        image = torch.from_numpy(patch).permute(2, 0, 1)

        if self.transform is not None:
            image = self.transform(image)

        return self.spot_ids[idx], image


# ============================================================
# Utility: Expression/Embedding Alignment (from scripts/03)
# ============================================================

def align_expression_embeddings(expr_df, spot_to_sample, embed_data, sample_ids):
    """Align expression and embedding data by matching spot IDs."""
    aligned_embeds = []
    aligned_exprs = []
    aligned_spots = []

    for sid in sample_ids:
        if sid not in embed_data:
            continue

        embeddings, embed_spot_ids = embed_data[sid]
        prefix = f"{sid}_"

        sample_expr_spots = {
            s: s[len(prefix):] for s in expr_df.index if s.startswith(prefix)
        }
        barcode_to_prefixed = {v: k for k, v in sample_expr_spots.items()}
        embed_idx = {b: i for i, b in enumerate(embed_spot_ids)}

        for barcode in embed_spot_ids:
            if barcode in barcode_to_prefixed:
                emb = embeddings[embed_idx[barcode]]
                if np.any(np.isnan(emb)):
                    continue
                prefixed = barcode_to_prefixed[barcode]
                aligned_embeds.append(emb)
                aligned_exprs.append(expr_df.loc[prefixed].values)
                aligned_spots.append(prefixed)

    X = np.array(aligned_embeds, dtype=np.float32)
    Y = np.array(aligned_exprs, dtype=np.float32)

    nan_mask = np.isnan(Y).any(axis=1) | np.isnan(X).any(axis=1)
    if nan_mask.any():
        X = X[~nan_mask]
        Y = Y[~nan_mask]
        aligned_spots = [s for s, m in zip(aligned_spots, nan_mask) if not m]

    return X, Y, aligned_spots


def compute_per_gene_pearson(Y_true, Y_pred, gene_names):
    """Compute Pearson correlation per gene."""
    results = {}
    pearsons = []

    for i, gene in enumerate(gene_names):
        y_t = Y_true[:, i]
        y_p = Y_pred[:, i]

        if np.std(y_t) < 1e-10 or np.std(y_p) < 1e-10:
            r, p = 0.0, 1.0
        else:
            r, p = pearsonr(y_t, y_p)
            if np.isnan(r):
                r = 0.0

        results[gene] = {"pearson": float(r), "pvalue": float(p)}
        pearsons.append(r)

    pearsons = np.array(pearsons)
    results["__mean_pearson__"] = float(np.mean(pearsons))
    results["__median_pearson__"] = float(np.median(pearsons))
    results["__n_positive__"] = int(np.sum(pearsons > 0))
    results["__n_genes__"] = len(gene_names)
    results["__n_spots__"] = int(Y_true.shape[0])

    return results


# ============================================================
# Step 1: QC & Gene Panel
# ============================================================

def run_qc_and_gene_panel(logger):
    """Load Visium IDC samples, compute gene panel, and QC stats."""
    logger.info("\n=== Step 1: QC & Gene Panel ===")
    hest_dir = Path(HEST_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    gene_sets = {}
    qc_records = []

    for sid in VISIUM_SAMPLES:
        h5ad_path = hest_dir / "st" / f"{sid}.h5ad"
        adata = sc.read_h5ad(h5ad_path)

        all_genes = list(adata.var_names)
        real_genes = [g for g in all_genes if not any(g.startswith(p) for p in CONTROL_PREFIXES)]
        gene_sets[sid] = set(real_genes)

        total_counts = np.array(adata.X.sum(axis=1)).flatten()
        genes_per_spot = np.array((adata.X > 0).sum(axis=1)).flatten()

        qc_records.append({
            "sample_id": sid,
            "n_spots": adata.n_obs,
            "n_total_genes": len(all_genes),
            "n_real_genes": len(real_genes),
            "n_control_probes": len(all_genes) - len(real_genes),
            "median_total_counts": float(np.median(total_counts)),
            "mean_total_counts": float(np.mean(total_counts)),
            "median_genes_per_spot": float(np.median(genes_per_spot)),
        })
        logger.info(f"  {sid}: {adata.n_obs} spots, {len(real_genes)} real genes, "
                     f"median counts={np.median(total_counts):.0f}")

    # Gene panel intersection
    panel = sorted(set.intersection(*gene_sets.values()))
    logger.info(f"  Visium panel (intersection): {len(panel)} genes")

    # Cross-technology overlaps with IDC Xenium panels
    idc_panels = {}
    for name in ["biomarkers", "10x_janesick", "bridge"]:
        panel_path = f"data/v3/gene_list_{name}.json"
        if Path(panel_path).exists():
            with open(panel_path) as f:
                idc_panels[name] = set(json.load(f))

    idc_union = idc_panels.get("biomarkers", set()) | idc_panels.get("10x_janesick", set())
    panel_set = set(panel)

    overlap_union = sorted(panel_set & idc_union)
    overlap_bridge = sorted(panel_set & idc_panels.get("bridge", set()))
    overlap_bio = sorted(panel_set & idc_panels.get("biomarkers", set()))
    overlap_jan = sorted(panel_set & idc_panels.get("10x_janesick", set()))

    logger.info(f"  Overlap with IDC Xenium union ({len(idc_union)}): {len(overlap_union)}")
    logger.info(f"  Overlap with IDC Xenium bridge (90): {len(overlap_bridge)}")
    logger.info(f"  Overlap with biomarkers (280): {len(overlap_bio)}")
    logger.info(f"  Overlap with 10x_janesick (280): {len(overlap_jan)}")

    # Save
    with open(out_dir / "gene_panel.json", "w") as f:
        json.dump(panel, f)

    cross_tech = {
        "xenium_union_overlap": overlap_union,
        "bridge_overlap": overlap_bridge,
        "biomarkers_overlap": overlap_bio,
        "janesick_overlap": overlap_jan,
        "visium_panel_size": len(panel),
        "xenium_union_size": len(idc_union),
    }
    with open(out_dir / "cross_technology_genes.json", "w") as f:
        json.dump(cross_tech, f, indent=2)

    qc_summary = {
        "samples": qc_records,
        "n_samples": len(VISIUM_SAMPLES),
        "gene_panel_size": len(panel),
        "cross_technology_overlap": len(overlap_union),
    }
    with open(out_dir / "qc_summary.json", "w") as f:
        json.dump(qc_summary, f, indent=2)

    logger.info(f"  Saved gene_panel.json, cross_technology_genes.json, qc_summary.json")
    return panel


# ============================================================
# Step 2: Embedding Extraction
# ============================================================

def run_embedding_extraction(gpu_id, logger):
    """Extract embeddings for Visium IDC samples using 3 encoders."""
    logger.info("\n=== Step 2: Embedding Extraction ===")
    hest_dir = Path(HEST_DIR).resolve()
    out_dir = Path(OUTPUT_DIR)

    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    logger.info(f"  Device: {device}")

    for enc_config in ENCODER_CONFIGS:
        enc_name = enc_config["name"]
        logger.info(f"\n  Encoder: {enc_name}")

        need_extraction = []
        for sid in VISIUM_SAMPLES:
            emb_path = out_dir / "embeddings" / sid / f"{enc_name}_embeddings.h5"
            if checkpoint_exists(str(emb_path)):
                logger.info(f"    {sid}/{enc_name}: already exists, skipping")
            else:
                need_extraction.append(sid)

        if not need_extraction:
            logger.info(f"    All samples done for {enc_name}")
            continue

        encoder = get_encoder(enc_name, enc_config["model_id"], device=device,
                              use_mixed_precision=False)
        batch_size = enc_config.get("batch_size_per_gpu", 64)

        for sid in need_extraction:
            logger.info(f"    Extracting {sid}/{enc_name}...")
            t0 = time.time()

            dataset = SamplePatchDataset(sid, hest_dir)
            logger.info(f"      Spots: {dataset.n_spots}")

            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False,
                num_workers=4, pin_memory=True, drop_last=False,
            )

            all_embeddings = []
            all_spot_ids = []

            for spot_ids_batch, images_batch in tqdm(
                dataloader, desc=f"      {sid}/{enc_name}", leave=False
            ):
                embeddings_batch = encoder.encode(images_batch)
                all_embeddings.append(embeddings_batch)
                all_spot_ids.extend(spot_ids_batch)

            embeddings_matrix = np.vstack(all_embeddings)

            n_nan = np.isnan(embeddings_matrix).sum()
            if n_nan > 0:
                logger.error(f"      NaN detected: {n_nan} values")
                continue

            emb_dir = out_dir / "embeddings" / sid
            emb_dir.mkdir(parents=True, exist_ok=True)
            emb_path = emb_dir / f"{enc_name}_embeddings.h5"

            with h5py.File(emb_path, "w") as f:
                f.create_dataset("embeddings", data=embeddings_matrix,
                                 dtype="float32", compression="gzip", compression_opts=4)
                f.create_dataset("spot_ids", data=np.array(all_spot_ids, dtype="S"),
                                 dtype=h5py.string_dtype())
                f.attrs["encoder_name"] = enc_name
                f.attrs["model_id"] = enc_config["model_id"]
                f.attrs["embed_dim"] = enc_config["embed_dim"]
                f.attrs["sample_id"] = sid
                f.attrs["n_spots"] = len(all_spot_ids)
                f.attrs["extraction_date"] = datetime.now().isoformat()

            logger.info(f"      Done: {embeddings_matrix.shape} in {format_time(time.time() - t0)}")

        del encoder
        torch.cuda.empty_cache()

    logger.info("  Embedding extraction complete.")


# ============================================================
# Step 3: LOSO Ridge Prediction
# ============================================================

def build_loso_folds(sample_ids):
    """Generate leave-one-sample-out folds."""
    folds = []
    for i, test_sample in enumerate(sample_ids):
        train_samples = [s for s in sample_ids if s != test_sample]
        folds.append({
            "fold": i,
            "test_samples": [test_sample],
            "train_samples": train_samples,
        })
    return folds


def load_embeddings_for_samples(sample_ids, encoder_name, embed_dir):
    """Load embeddings for multiple samples."""
    result = {}
    for sid in sample_ids:
        path = embed_dir / sid / f"{encoder_name}_embeddings.h5"
        with h5py.File(path, "r") as f:
            embeddings = f["embeddings"][:]
            spot_ids = [
                s.decode() if isinstance(s, bytes) else s for s in f["spot_ids"][:]
            ]
        result[sid] = (embeddings, spot_ids)
    return result


def run_loso_prediction(gene_panel, logger):
    """LOSO cross-validation with ridge regression."""
    logger.info("\n=== Step 3: LOSO Ridge Prediction ===")
    from src.data import load_v3_task

    out_dir = Path(OUTPUT_DIR)
    embed_dir = out_dir / "embeddings"
    pred_dir = out_dir / "predictions"
    gene_list_path = out_dir / "gene_panel.json"

    folds = build_loso_folds(VISIUM_SAMPLES)
    logger.info(f"  {len(folds)} LOSO folds, {len(ENCODER_CONFIGS)} encoders, ridge only")

    # Load expression data once
    logger.info("  Loading expression data...")
    expr_df, gene_names, spot_to_sample = load_v3_task(
        sample_ids=VISIUM_SAMPLES,
        hest_dir=HEST_DIR,
        gene_list_path=str(gene_list_path),
        normalize=True,
    )
    logger.info(f"  Expression: {expr_df.shape[0]} spots × {expr_df.shape[1]} genes")

    seed_everything(SEED)

    for enc_config in ENCODER_CONFIGS:
        enc_name = enc_config["name"]
        logger.info(f"\n  Encoder: {enc_name}")

        embed_data = load_embeddings_for_samples(VISIUM_SAMPLES, enc_name, embed_dir)
        embed_dim = next(iter(embed_data.values()))[0].shape[1]
        logger.info(f"  Embeddings loaded: {embed_dim} dims")

        for fold in folds:
            fold_idx = fold["fold"]
            fold_out = pred_dir / enc_name / f"fold{fold_idx}"

            if (fold_out / "metrics.json").exists():
                logger.info(f"    Fold {fold_idx}: already exists, skipping")
                continue

            train_samples = fold["train_samples"]
            test_samples = fold["test_samples"]
            logger.info(f"    Fold {fold_idx}: train={train_samples}, test={test_samples}")

            X_train, Y_train, train_spots = align_expression_embeddings(
                expr_df, spot_to_sample, embed_data, train_samples)
            X_test, Y_test, test_spots = align_expression_embeddings(
                expr_df, spot_to_sample, embed_data, test_samples)

            logger.info(f"      Train: {X_train.shape[0]} spots, Test: {X_test.shape[0]} spots")

            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc = scaler.transform(X_test)

            reg = get_regressor(REGRESSOR_CONFIG)
            t0 = time.time()
            reg.fit(X_train_sc, Y_train)
            Y_pred = reg.predict(X_test_sc)
            logger.info(f"      Train+Predict: {format_time(time.time() - t0)}")

            metrics = compute_per_gene_pearson(Y_test, Y_pred, gene_names)
            logger.info(f"      Mean Pearson: {metrics['__mean_pearson__']:.4f} "
                        f"(median: {metrics['__median_pearson__']:.4f}, "
                        f"{metrics['__n_positive__']}/{metrics['__n_genes__']} positive)")

            residuals = Y_test - Y_pred
            fold_out.mkdir(parents=True, exist_ok=True)
            np.save(fold_out / "test_predictions.npy", Y_pred.astype(np.float32))
            np.save(fold_out / "test_targets.npy", Y_test.astype(np.float32))
            np.save(fold_out / "test_residuals.npy", residuals.astype(np.float32))
            with open(fold_out / "test_spot_ids.json", "w") as f:
                json.dump(test_spots, f)
            with open(fold_out / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

    logger.info("  LOSO prediction complete.")


# ============================================================
# Step 4: Discordance Scores
# ============================================================

def aggregate_folds(pred_dir, encoder_name, n_folds):
    """Aggregate residuals/targets across folds."""
    all_residuals, all_targets, all_spot_ids = [], [], []
    for fold_idx in range(n_folds):
        fold_dir = pred_dir / encoder_name / f"fold{fold_idx}"
        residuals = np.load(fold_dir / "test_residuals.npy")
        targets = np.load(fold_dir / "test_targets.npy")
        with open(fold_dir / "test_spot_ids.json") as f:
            spots = json.load(f)
        all_residuals.append(residuals)
        all_targets.append(targets)
        all_spot_ids.extend(spots)
    return np.vstack(all_residuals), np.vstack(all_targets), all_spot_ids


def load_spatial_coords(sample_id, hest_dir):
    """Load spatial coordinates from h5ad."""
    import anndata as ad
    h5ad_path = Path(hest_dir) / "st" / f"{sample_id}.h5ad"
    adata = ad.read_h5ad(h5ad_path)
    coords = adata.obsm["spatial"]
    barcodes = list(adata.obs.index)
    return {b: coords[i] for i, b in enumerate(barcodes)}


def run_discordance_scores(logger):
    """Compute discordance scores for all Visium IDC samples."""
    logger.info("\n=== Step 4: Discordance Scores ===")
    out_dir = Path(OUTPUT_DIR)
    pred_dir = out_dir / "predictions"
    scores_dir = out_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    n_folds = len(VISIUM_SAMPLES)
    encoder_names = [e["name"] for e in ENCODER_CONFIGS]

    all_coords = {}
    for sid in VISIUM_SAMPLES:
        all_coords[sid] = load_spatial_coords(sid, HEST_DIR)

    config_names = []
    config_discordance = {}

    for enc_name in encoder_names:
        config_name = f"{enc_name}_ridge"
        config_names.append(config_name)
        logger.info(f"  Config: {config_name}")

        residuals, targets, spot_ids = aggregate_folds(pred_dir, enc_name, n_folds)
        logger.info(f"    Aggregated: {residuals.shape[0]} spots × {residuals.shape[1]} genes")

        D_raw = compute_mean_absolute_discordance(residuals)
        total_expr = targets.sum(axis=1)
        D_cond = compute_conditional_discordance(D_raw, total_expr,
                                                  n_bins=N_BINS, min_bin_size=MIN_BIN_SIZE)

        logger.info(f"    D_raw: mean={D_raw.mean():.4f}, D_cond: mean={D_cond.mean():.4f}")

        for i, sid_spot in enumerate(spot_ids):
            if sid_spot not in config_discordance:
                config_discordance[sid_spot] = {}
            config_discordance[sid_spot][config_name] = (
                D_raw[i], D_cond[i], total_expr[i],
            )

    for sid in VISIUM_SAMPLES:
        prefix = f"{sid}_"
        sample_spots = [s for s in config_discordance if s.startswith(prefix)]

        rows = []
        for spot_id in sample_spots:
            barcode = spot_id[len(prefix):]
            coord = all_coords[sid].get(barcode, np.array([np.nan, np.nan]))

            row = {
                "spot_id": spot_id, "sample_id": sid, "barcode": barcode,
                "x": coord[0], "y": coord[1],
                "total_expr": config_discordance[spot_id][config_names[0]][2],
            }
            for cn in config_names:
                d_raw, d_cond, _ = config_discordance[spot_id][cn]
                row[f"D_raw_{cn}"] = d_raw
                row[f"D_cond_{cn}"] = d_cond
            rows.append(row)

        df = pd.DataFrame(rows)
        out_path = scores_dir / f"{sid}_discordance.parquet"
        df.to_parquet(out_path, index=False)
        logger.info(f"  Saved {sid}: {len(df)} spots")

    logger.info("  Discordance scores complete.")


# ============================================================
# Step 5: Moran's I Per Gene
# ============================================================

def compute_morans_i_per_gene(gene_panel, logger):
    """Compute spatial autocorrelation per gene per sample, average across samples."""
    logger.info("\n=== Step 5: Moran's I Per Gene ===")
    hest_dir = Path(HEST_DIR)

    gene_morans = {g: [] for g in gene_panel}

    for sid in VISIUM_SAMPLES:
        t0 = time.time()
        adata = sc.read_h5ad(hest_dir / "st" / f"{sid}.h5ad")
        available = [g for g in gene_panel if g in adata.var_names]
        adata = adata[:, available]

        X = adata.X
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X = np.log1p(X)

        coords = adata.obsm["spatial"]
        W = build_spatial_weights(coords, n_neighbors=SPATIAL_N_NEIGHBORS)

        for i, gene in enumerate(available):
            I = morans_i(X[:, i], W)
            gene_morans[gene].append(I)

        elapsed = time.time() - t0
        logger.info(f"  {sid}: {len(available)} genes computed in {format_time(elapsed)}")

    # Average across samples
    result = {}
    for gene in gene_panel:
        vals = gene_morans[gene]
        result[gene] = float(np.mean(vals)) if vals else np.nan

    logger.info(f"  Mean Moran's I: {np.nanmean(list(result.values())):.4f}")
    return result


# ============================================================
# Step 6: Gene Features
# ============================================================

def compute_gene_features(gene_panel, morans_dict, logger):
    """Compute per-gene prediction quality averaged across encoders."""
    logger.info("\n=== Step 6: Gene Features ===")
    out_dir = Path(OUTPUT_DIR)
    pred_dir = out_dir / "predictions"
    n_folds = len(VISIUM_SAMPLES)
    encoder_names = [e["name"] for e in ENCODER_CONFIGS]

    gene_pearsons = {g: [] for g in gene_panel}

    for enc_name in encoder_names:
        residuals, targets, spot_ids = aggregate_folds(pred_dir, enc_name, n_folds)
        predictions = targets - residuals

        metrics_path = pred_dir / enc_name / "fold0" / "metrics.json"
        with open(metrics_path) as f:
            metrics = json.load(f)
        gene_names = [k for k in metrics.keys() if not k.startswith("__")]

        for i, gene in enumerate(gene_names):
            y_t = targets[:, i]
            y_p = predictions[:, i]
            if np.std(y_t) < 1e-10 or np.std(y_p) < 1e-10:
                r = 0.0
            else:
                r, _ = pearsonr(y_t, y_p)
                if np.isnan(r):
                    r = 0.0
            gene_pearsons[gene].append(r)

    records = []
    for gene in gene_panel:
        pearsons = gene_pearsons.get(gene, [])
        mean_r = float(np.mean(pearsons)) if pearsons else 0.0
        records.append({
            "gene": gene,
            "mean_pearson": mean_r,
            "morans_i": morans_dict.get(gene, np.nan),
        })

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "gene_features.csv", index=False)
    logger.info(f"  Gene features: mean Pearson r = {df['mean_pearson'].mean():.4f} "
                f"(median = {df['mean_pearson'].median():.4f})")
    logger.info(f"  Saved gene_features.csv")
    return df


# ============================================================
# Step 7: Pathway Enrichment
# ============================================================

def aggregate_residuals_for_sample(sample_id, pred_dir, n_folds, encoder_names):
    """Load and aggregate residuals across folds and encoders for a sample."""
    sample_prefix = f"{sample_id}_"
    all_residuals = {}

    for enc_name in encoder_names:
        enc_residuals = []
        enc_targets = []
        enc_spots = []

        for fold_idx in range(n_folds):
            fold_dir = pred_dir / enc_name / f"fold{fold_idx}"
            residuals = np.load(fold_dir / "test_residuals.npy")
            targets = np.load(fold_dir / "test_targets.npy")
            with open(fold_dir / "test_spot_ids.json") as f:
                spots = json.load(f)

            for i, sid in enumerate(spots):
                if sid.startswith(sample_prefix):
                    enc_residuals.append(residuals[i])
                    enc_targets.append(targets[i])
                    enc_spots.append(sid)

        for i, sid in enumerate(enc_spots):
            if sid not in all_residuals:
                all_residuals[sid] = {"residuals": [], "target": enc_targets[i]}
            all_residuals[sid]["residuals"].append(enc_residuals[i])

    spot_ids = sorted(all_residuals.keys())
    residuals = np.array([np.mean(all_residuals[s]["residuals"], axis=0) for s in spot_ids])
    targets = np.array([all_residuals[s]["target"] for s in spot_ids])

    metrics_path = pred_dir / encoder_names[0] / "fold0" / "metrics.json"
    with open(metrics_path) as f:
        metrics = json.load(f)
    gene_names = [k for k in metrics.keys() if not k.startswith("__")]

    return residuals, targets, gene_names, spot_ids


def pathway_de_test(pathway_scores, disc_mask, conc_mask, pathway_name):
    """Wilcoxon rank-sum test on pathway scores."""
    disc_vals = pathway_scores[disc_mask]
    conc_vals = pathway_scores[conc_mask]

    mean_disc = np.mean(disc_vals)
    mean_conc = np.mean(conc_vals)

    try:
        stat, pval = stats.mannwhitneyu(disc_vals, conc_vals, alternative="two-sided")
    except ValueError:
        pval = 1.0

    # Pooled within-group SD (consistent with wilcoxon_de)
    n_d, n_c = len(disc_vals), len(conc_vals)
    var_d = np.var(disc_vals, ddof=1) if n_d > 1 else 0
    var_c = np.var(conc_vals, ddof=1) if n_c > 1 else 0
    pooled_var = ((var_d * (n_d - 1) + var_c * (n_c - 1)) / (n_d + n_c - 2)) if (n_d + n_c) > 2 else 0
    pooled_sd = np.sqrt(pooled_var) if pooled_var > 0 else 1e-6
    cohens_d = (mean_disc - mean_conc) / pooled_sd

    return {
        "pathway": pathway_name,
        "mean_disc": float(mean_disc),
        "mean_conc": float(mean_conc),
        "delta": float(mean_disc - mean_conc),
        "cohens_d": float(cohens_d),
        "pval": float(pval),
    }


def run_pathway_enrichment(logger):
    """Compute pathway enrichment for Visium IDC samples."""
    logger.info("\n=== Step 7: Pathway Enrichment ===")
    out_dir = Path(OUTPUT_DIR)
    pred_dir = out_dir / "predictions"
    scores_dir = out_dir / "scores"
    n_folds = len(VISIUM_SAMPLES)
    encoder_names = [e["name"] for e in ENCODER_CONFIGS]

    gene_sets = load_gene_sets(HALLMARK_GMT)
    logger.info(f"  Loaded {len(gene_sets)} Hallmark gene sets")

    all_records = []

    for sid in VISIUM_SAMPLES:
        logger.info(f"  Sample: {sid}")

        residuals, targets, gene_names, spot_ids = aggregate_residuals_for_sample(
            sid, pred_dir, n_folds, encoder_names)
        logger.info(f"    {len(spot_ids)} spots × {len(gene_names)} genes")

        disc_parquet = scores_dir / f"{sid}_discordance.parquet"
        disc_df = pd.read_parquet(disc_parquet)
        d_cond_cols = [c for c in disc_df.columns if c.startswith("D_cond_")]
        disc_df["D_cond_mean"] = disc_df[d_cond_cols].mean(axis=1)

        disc_df = disc_df.set_index("spot_id")
        aligned_dcond = disc_df.loc[spot_ids, "D_cond_mean"].values

        disc_mask, conc_mask = partition_spots_by_discordance(aligned_dcond)
        logger.info(f"    Discordant: {disc_mask.sum()}, Concordant: {conc_mask.sum()}")

        stud_resid = compute_studentized_gene_residuals(residuals, targets, n_bins=20)
        pw_scores = compute_studentized_pathway_scores(
            stud_resid, gene_names, gene_sets, min_overlap=PATHWAY_MIN_OVERLAP)

        logger.info(f"    Testable pathways: {len(pw_scores)}")

        sample_records = []
        for pw_name, scores in pw_scores.items():
            result = pathway_de_test(scores, disc_mask, conc_mask, pw_name)
            result["sample_id"] = sid
            sample_records.append(result)

        if sample_records:
            pvals = [r["pval"] for r in sample_records]
            _, fdr_vals, _, _ = multipletests(pvals, method="fdr_bh")
            for r, fdr in zip(sample_records, fdr_vals):
                r["fdr"] = float(fdr)
            n_sig = sum(1 for r in sample_records if r["fdr"] < 0.05)
            logger.info(f"    Significant at FDR<0.05: {n_sig}/{len(sample_records)}")

        all_records.extend(sample_records)

    pw_de_df = pd.DataFrame(all_records)
    pw_de_df.to_csv(out_dir / "pathway_de.csv", index=False)
    logger.info(f"  Saved pathway_de.csv ({len(pw_de_df)} records)")


# ============================================================
# Step 8: Analysis Results
# ============================================================

def compile_analysis_results(gene_features_df, logger):
    """Compute all 4 cross-technology analyses and save to analysis_results.json."""
    logger.info("\n=== Step 8: Analysis Results ===")
    out_dir = Path(OUTPUT_DIR)
    idc_dir = Path(IDC_OUTPUT_DIR)
    results = {}

    # --- Analysis A: Moran's I vs Predictability ---
    gf = gene_features_df.dropna(subset=["morans_i", "mean_pearson"])
    rho, p_rho = spearmanr(gf["morans_i"], gf["mean_pearson"])
    r_pearson, p_pearson = pearsonr(gf["morans_i"], gf["mean_pearson"])

    results["analysis_a_morans_predictability"] = {
        "n_genes": len(gf),
        "spearman_rho": float(rho),
        "spearman_p": float(p_rho),
        "pearson_r": float(r_pearson),
        "pearson_p": float(p_pearson),
    }
    logger.info(f"  A: Moran's I vs Predictability: ρ={rho:.3f} (p={p_rho:.2e})")

    # --- Analysis B: Pathway Enrichment Summary ---
    pw_de = pd.read_csv(out_dir / "pathway_de.csv")
    n_pathways = pw_de["pathway"].nunique()
    n_sig_any = pw_de[pw_de["fdr"] < 0.05]["pathway"].nunique()

    consistency_records = []
    for pw, grp in pw_de.groupby("pathway"):
        signs = np.sign(grp["cohens_d"].values)
        n_pos = (signs > 0).sum()
        n_neg = (signs < 0).sum()
        direction = "up" if n_pos > n_neg else "down"
        consistency = max(n_pos, n_neg) / len(signs)
        consistency_records.append({
            "pathway": pw,
            "direction": direction,
            "consistency": float(consistency),
            "n_sig": int((grp["fdr"] < 0.05).sum()),
            "mean_cohens_d": float(grp["cohens_d"].mean()),
        })

    consistency_df = pd.DataFrame(consistency_records)
    consistency_df.to_csv(out_dir / "pathway_consistency.csv", index=False)

    results["analysis_b_pathway_enrichment"] = {
        "n_pathways_tested": int(n_pathways),
        "n_pathways_sig_any_sample": int(n_sig_any),
        "n_samples": len(VISIUM_SAMPLES),
        "mean_direction_consistency": float(consistency_df["consistency"].mean()),
        "top_pathways": consistency_df.sort_values("n_sig", ascending=False).head(10)[
            ["pathway", "mean_cohens_d", "n_sig", "consistency"]
        ].to_dict("records"),
    }
    logger.info(f"  B: {n_pathways} pathways tested, {n_sig_any} sig in ≥1 sample")

    # --- Analysis C: Predictability Distribution Comparison ---
    # Load cross-technology gene list to compare on shared genes only
    with open(out_dir / "cross_technology_genes.json") as f:
        ct = json.load(f)
    shared_genes = set(ct["xenium_union_overlap"])

    # Visium gene features filtered to shared genes
    visium_shared = gene_features_df[gene_features_df["gene"].isin(shared_genes)]
    visium_r = visium_shared["mean_pearson"].values

    # IDC Xenium gene features filtered to shared genes
    idc_bio = pd.read_csv(idc_dir / "phase3" / "gene_predictability" / "biomarkers" / "gene_features.csv")
    idc_jan = pd.read_csv(idc_dir / "phase3" / "gene_predictability" / "10x_janesick" / "gene_features.csv")
    idc_bio_shared = idc_bio[idc_bio["gene"].isin(shared_genes)]
    idc_jan_shared = idc_jan[idc_jan["gene"].isin(shared_genes)]

    results["analysis_c_predictability_distribution"] = {
        "visium": {
            "n_genes": len(visium_r),
            "mean": float(np.mean(visium_r)),
            "median": float(np.median(visium_r)),
            "q25": float(np.percentile(visium_r, 25)),
            "q75": float(np.percentile(visium_r, 75)),
        },
        "idc_discovery": {
            "n_genes": len(idc_bio_shared),
            "mean": float(idc_bio_shared["mean_pearson"].mean()),
            "median": float(idc_bio_shared["mean_pearson"].median()),
            "q25": float(idc_bio_shared["mean_pearson"].quantile(0.25)),
            "q75": float(idc_bio_shared["mean_pearson"].quantile(0.75)),
        },
        "idc_validation": {
            "n_genes": len(idc_jan_shared),
            "mean": float(idc_jan_shared["mean_pearson"].mean()),
            "median": float(idc_jan_shared["mean_pearson"].median()),
            "q25": float(idc_jan_shared["mean_pearson"].quantile(0.25)),
            "q75": float(idc_jan_shared["mean_pearson"].quantile(0.75)),
        },
        "visium_full_panel": {
            "n_genes": len(gene_features_df),
            "mean": float(gene_features_df["mean_pearson"].mean()),
            "median": float(gene_features_df["mean_pearson"].median()),
        },
    }
    logger.info(f"  C: Visium shared={np.mean(visium_r):.3f}, "
                f"Xenium disc={idc_bio_shared['mean_pearson'].mean():.3f}, "
                f"Xenium val={idc_jan_shared['mean_pearson'].mean():.3f}")

    # --- Analysis D: Cross-Technology Gene Predictability ---
    visium_gf = gene_features_df.set_index("gene")
    idc_bio_gf = idc_bio.set_index("gene")
    idc_jan_gf = idc_jan.set_index("gene")

    cross_records = []
    for gene in shared_genes:
        visium_val = visium_gf.loc[gene, "mean_pearson"] if gene in visium_gf.index else np.nan

        idc_vals = []
        if gene in idc_bio_gf.index:
            idc_vals.append(idc_bio_gf.loc[gene, "mean_pearson"])
        if gene in idc_jan_gf.index:
            idc_vals.append(idc_jan_gf.loc[gene, "mean_pearson"])

        idc_val = float(np.mean(idc_vals)) if idc_vals else np.nan

        if not np.isnan(visium_val) and not np.isnan(idc_val):
            cross_records.append({"gene": gene, "visium_r": visium_val, "xenium_r": idc_val})

    cross_df = pd.DataFrame(cross_records)

    if len(cross_df) >= 5:
        r_cross, p_cross = pearsonr(cross_df["visium_r"], cross_df["xenium_r"])
        rho_cross, p_rho_cross = spearmanr(cross_df["visium_r"], cross_df["xenium_r"])
    else:
        r_cross = rho_cross = p_cross = p_rho_cross = np.nan

    results["analysis_d_cross_technology"] = {
        "n_shared_genes": len(cross_df),
        "pearson_r": float(r_cross) if not np.isnan(r_cross) else None,
        "pearson_p": float(p_cross) if not np.isnan(p_cross) else None,
        "spearman_rho": float(rho_cross) if not np.isnan(rho_cross) else None,
        "spearman_p": float(p_rho_cross) if not np.isnan(p_rho_cross) else None,
    }
    cross_df.to_csv(out_dir / "cross_technology_gene_predictability.csv", index=False)

    logger.info(f"  D: {len(cross_df)} shared genes, Pearson r={r_cross:.3f} (p={p_cross:.2e}), "
                f"Spearman ρ={rho_cross:.3f}")

    with open(out_dir / "analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("  Saved analysis_results.json")
    return results


# ============================================================
# Step 9: Figure Generation
# ============================================================

def generate_figures(gene_features_df, logger):
    """Generate cross-technology analysis figures."""
    import matplotlib.pyplot as plt

    logger.info("\n=== Step 9: Figure Generation ===")
    setup_style()

    out_dir = Path(OUTPUT_DIR)
    fig_dir = Path(FIGURE_DIR)
    fig_dir.mkdir(parents=True, exist_ok=True)

    idc_dir = Path(IDC_OUTPUT_DIR)

    # --- Moran's I vs Predictability Scatter ---
    logger.info("  Fig 1: Moran's I vs Predictability scatter")
    gf = gene_features_df.dropna(subset=["morans_i", "mean_pearson"])

    fig, ax = plt.subplots(figsize=(HALF_WIDTH, HALF_WIDTH * 0.85))
    ax.scatter(gf["morans_i"], gf["mean_pearson"], s=3, alpha=0.3,
               color=COLOR_VISIUM, edgecolors="none", zorder=2)

    # OLS trend line
    m, b = np.polyfit(gf["morans_i"], gf["mean_pearson"], 1)
    x_range = np.linspace(gf["morans_i"].min(), gf["morans_i"].max(), 100)
    ax.plot(x_range, m * x_range + b, color=COLOR_VISIUM, linewidth=1, alpha=0.8, zorder=1)

    # IDC Xenium trend for comparison
    try:
        idc_bio_gf = pd.read_csv(idc_dir / "phase3" / "gene_predictability" / "biomarkers" / "gene_features.csv")
        if "spatial_autocorrelation" in idc_bio_gf.columns:
            m_idc, b_idc = np.polyfit(idc_bio_gf["spatial_autocorrelation"],
                                       idc_bio_gf["mean_pearson"], 1)
            x_idc = np.linspace(gf["morans_i"].min(), gf["morans_i"].max(), 100)
            ax.plot(x_idc, m_idc * x_idc + b_idc, color=COLORS["discovery"],
                    linewidth=1, linestyle="--", alpha=0.6, label="IDC Xenium (discovery)", zorder=1)
            ax.legend(fontsize=6, frameon=False)
    except Exception:
        pass

    rho, p = spearmanr(gf["morans_i"], gf["mean_pearson"])
    if p < 1e-100:
        rho_text = f"Spearman ρ = {rho:.3f}\np < 1e-100"
    else:
        rho_text = f"Spearman ρ = {rho:.3f}\np = {p:.2e}"
    ax.text(0.02, 0.99, rho_text, transform=ax.transAxes, fontsize=6,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.8))

    # Annotate outlier genes
    gf = gf.copy()
    gf["ols_resid"] = gf["mean_pearson"] - (m * gf["morans_i"] + b)
    n_outliers = 3
    top_pos = gf.nlargest(n_outliers, "ols_resid")
    top_neg = gf.nsmallest(n_outliers, "ols_resid")
    outliers = pd.concat([top_pos, top_neg])
    gene_offsets = {}
    for _, row in outliers.iterrows():
        gene = row["gene"]
        if gene.startswith("TUBA") or gene.startswith("LSM"):
            gene_offsets[gene] = (5, -10)
    for _, row in outliers.iterrows():
        gene = row["gene"]
        default_y = 8 if row["ols_resid"] > 0 else -8
        dx, dy = gene_offsets.get(gene, (5, default_y))
        ax.annotate(gene, (row["morans_i"], row["mean_pearson"]),
                    textcoords="offset points", xytext=(dx, dy),
                    fontsize=4.5, fontstyle="italic", alpha=0.9,
                    arrowprops=dict(arrowstyle="-", lw=0.3, alpha=0.5))

    ax.set_xlabel("Mean Moran's I")
    ax.set_ylabel("Mean Pearson r")
    ax.set_title("IDC Visium: Spatial autocorrelation vs predictability")
    save_figure(fig, fig_dir / "fig_visium_morans_scatter")
    plt.close(fig)

    # --- Pathway Enrichment Heatmap ---
    logger.info("  Fig 2: Pathway enrichment heatmap")
    pw_de = pd.read_csv(out_dir / "pathway_de.csv")

    pivot = pw_de.pivot(index="sample_id", columns="pathway", values="cohens_d")
    fdr_pivot = pw_de.pivot(index="sample_id", columns="pathway", values="fdr")

    # Filter to pathways significant in at least 1 sample, or top 20 by |mean d|
    mean_d = pivot.abs().mean().sort_values(ascending=False)
    sig_pathways = fdr_pivot.columns[fdr_pivot.min() < 0.05].tolist()
    if len(sig_pathways) < 5:
        top_pathways = mean_d.head(20).index.tolist()
    else:
        top_pathways = sig_pathways
    # Cap at 25 pathways for readability
    if len(top_pathways) > 25:
        top_pathways = mean_d.loc[top_pathways].sort_values(ascending=False).head(25).index.tolist()

    pivot = pivot[top_pathways]
    fdr_pivot = fdr_pivot[top_pathways]

    short_names = [p.replace("HALLMARK_", "").replace("_", " ").title() for p in top_pathways]

    # Determine colormap: sequential if all positive, diverging otherwise
    all_vals = pivot.values.flatten()
    all_vals = all_vals[~np.isnan(all_vals)]
    if np.all(all_vals >= 0):
        cmap = "Reds"
        vmin, vmax = 0, np.ceil(np.nanmax(all_vals) * 10) / 10
    else:
        cmap = "RdBu_r"
        abs_max = np.ceil(np.nanmax(np.abs(all_vals)) * 10) / 10
        vmin, vmax = -abs_max, abs_max

    fig_height = max(2.5, FULL_WIDTH * 0.35)
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, fig_height))
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(short_names)))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=5)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=6)
    ax.set_title("IDC Visium: Pathway enrichment in discordant spots (Cohen's d)")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Cohen's d", fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    fig.tight_layout()
    save_figure(fig, fig_dir / "fig_visium_pathway_heatmap")
    plt.close(fig)

    # --- Predictability Distribution Comparison ---
    logger.info("  Fig 3: Predictability distribution comparison")

    # Load shared gene lists
    with open(out_dir / "cross_technology_genes.json") as f:
        ct = json.load(f)
    shared_genes = set(ct["xenium_union_overlap"])

    idc_bio = pd.read_csv(idc_dir / "phase3" / "gene_predictability" / "biomarkers" / "gene_features.csv")
    idc_jan = pd.read_csv(idc_dir / "phase3" / "gene_predictability" / "10x_janesick" / "gene_features.csv")

    # Filter to shared genes
    visium_shared = gene_features_df[gene_features_df["gene"].isin(shared_genes)]
    idc_bio_shared = idc_bio[idc_bio["gene"].isin(shared_genes)]
    idc_jan_shared = idc_jan[idc_jan["gene"].isin(shared_genes)]

    fig, ax = plt.subplots(figsize=(HALF_WIDTH, HALF_WIDTH * 0.85))

    data = [idc_bio_shared["mean_pearson"].values,
            idc_jan_shared["mean_pearson"].values,
            visium_shared["mean_pearson"].values]
    labels = [f"IDC Xenium\nDiscovery\n({len(idc_bio_shared)} genes)",
              f"IDC Xenium\nValidation\n({len(idc_jan_shared)} genes)",
              f"IDC Visium\n({len(visium_shared)} genes)"]
    colors = [COLORS["discovery"], COLORS["validation"], COLOR_VISIUM]

    parts = ax.violinplot(data, positions=[0, 1, 2], showextrema=False, showmedians=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
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
    ax.set_xticklabels(labels, fontsize=5)
    ax.set_ylabel("Per-gene mean Pearson r")
    ax.set_title("Gene predictability: Xenium vs Visium (shared genes)")
    ax.axhline(y=0, color="#999999", linewidth=0.5, linestyle="--", zorder=0)

    save_figure(fig, fig_dir / "fig_visium_predictability_comparison")
    plt.close(fig)

    # --- Cross-Technology Gene Predictability Scatter ---
    logger.info("  Fig 4: Cross-technology gene predictability scatter")
    cross_path = out_dir / "cross_technology_gene_predictability.csv"
    if cross_path.exists():
        cross_df = pd.read_csv(cross_path)

        fig, ax = plt.subplots(figsize=(HALF_WIDTH, HALF_WIDTH))

        ax.scatter(cross_df["xenium_r"], cross_df["visium_r"], s=12, alpha=0.7,
                   color=COLOR_VISIUM, edgecolors="white", linewidths=0.3, zorder=2)

        # Identity line
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, color="#999999", linewidth=0.8, linestyle="--",
                zorder=0, label="Identity")

        # OLS regression line
        m_fit, b_fit = np.polyfit(cross_df["xenium_r"], cross_df["visium_r"], 1)
        x_fit = np.linspace(lims[0], lims[1], 100)
        ax.plot(x_fit, m_fit * x_fit + b_fit, color=COLOR_VISIUM, linewidth=1,
                alpha=0.8, zorder=1, label="OLS fit")
        ax.legend(fontsize=5, frameon=False, loc="lower right")

        ax.set_xlim(lims)
        ax.set_ylim(lims)

        # Annotate correlations
        r, p_r = pearsonr(cross_df["xenium_r"], cross_df["visium_r"])
        rho, p_rho = spearmanr(cross_df["xenium_r"], cross_df["visium_r"])
        ax.text(0.05, 0.95, f"Pearson r = {r:.3f}\nSpearman ρ = {rho:.3f}",
                transform=ax.transAxes, fontsize=6, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="gray", alpha=0.8))

        # Label key IDC genes
        label_genes = {
            "ERBB2": (8, -5),
            "ESR1": (-12, 8),
            "CD68": (8, 5),
            "EPCAM": (-14, -4),
            "KRT7": (8, 6),
            "MKI67": (-12, 6),
            "ACTA2": (8, -8),
            "VIM": (-10, -7),
        }
        for gene, (dx, dy) in label_genes.items():
            row = cross_df[cross_df["gene"] == gene]
            if len(row) == 0:
                continue
            ax.annotate(gene, (row["xenium_r"].values[0], row["visium_r"].values[0]),
                        textcoords="offset points", xytext=(dx, dy),
                        fontsize=5, fontweight="bold", fontstyle="italic",
                        arrowprops=dict(arrowstyle="-", lw=0.4, alpha=0.6))

        ax.set_xlabel("IDC Xenium mean Pearson r")
        ax.set_ylabel("IDC Visium mean Pearson r")
        ax.set_title(f"Cross-technology gene predictability ({len(cross_df)} shared genes)")
        ax.set_aspect("equal")

        save_figure(fig, fig_dir / "fig_visium_cross_technology_scatter")
        plt.close(fig)
    else:
        logger.warning("  cross_technology_gene_predictability.csv not found, skipping")

    logger.info("  Figure generation complete.")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="IDC Visium cross-technology generalization")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip embedding extraction (use existing)")
    parser.add_argument("--skip-predictions", action="store_true",
                        help="Skip training/prediction (use existing)")
    parser.add_argument("--skip-figures", action="store_true",
                        help="Skip figure generation")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    args = parser.parse_args()

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(
        log_level="INFO",
        log_file=str(log_dir / "21_idc_visium_generalization.log"),
    )

    logger.info("=" * 60)
    logger.info("IDC Visium Cross-Technology Generalization")
    logger.info("=" * 60)
    logger.info(f"Samples: {VISIUM_SAMPLES}")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info(f"Figures: {FIGURE_DIR}")
    t0 = time.time()

    seed_everything(SEED)

    # Step 1: QC & Gene Panel
    gene_panel = run_qc_and_gene_panel(logger)

    # Step 2: Embedding Extraction
    if not args.skip_embeddings:
        run_embedding_extraction(args.gpu, logger)
    else:
        logger.info("\n=== Step 2: Embedding Extraction (SKIPPED) ===")

    # Step 3: LOSO Prediction
    if not args.skip_predictions:
        run_loso_prediction(gene_panel, logger)
    else:
        logger.info("\n=== Step 3: LOSO Prediction (SKIPPED) ===")

    # Check if predictions exist (needed for steps 4-9)
    pred_check = Path(OUTPUT_DIR) / "predictions" / "uni" / "fold0" / "metrics.json"
    if not pred_check.exists():
        logger.info("\nPredictions not found — skipping steps 4-9.")
        logger.info("Run without --skip-embeddings and --skip-predictions first.")
        logger.info(f"\nTotal time: {format_time(time.time() - t0)}")
        return

    # Step 4: Discordance Scores
    run_discordance_scores(logger)

    # Step 5: Moran's I
    morans_dict = compute_morans_i_per_gene(gene_panel, logger)

    # Step 6: Gene Features
    gene_features_df = compute_gene_features(gene_panel, morans_dict, logger)

    # Step 7: Pathway Enrichment
    run_pathway_enrichment(logger)

    # Step 8: Analysis Results
    results = compile_analysis_results(gene_features_df, logger)

    # Step 9: Figures
    if not args.skip_figures:
        generate_figures(gene_features_df, logger)
    else:
        logger.info("\n=== Step 9: Figure Generation (SKIPPED) ===")

    logger.info(f"\nTotal time: {format_time(time.time() - t0)}")
    logger.info("Done.")


if __name__ == "__main__":
    main()

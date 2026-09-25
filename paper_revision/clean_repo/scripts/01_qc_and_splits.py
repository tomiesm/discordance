#!/usr/bin/env python3
"""
Script 01: QC, Gene Validation, and LOPO Splits — Two-Panel Design

Phase 0 data preparation for discovery/validation cohorts:
  - Filters control probes, validates real gene panels per cohort
  - Saves per-cohort gene lists + 90-gene bridge list
  - Saves per-cohort patient mappings and LOPO splits (4 folds each)
  - QC metrics per sample
  - Batch effect assessment (PCA pseudo-bulk)
  - Pathway coverage per panel
  - Bridge gene analysis

Usage:
    python scripts/01_qc_and_splits.py
    python scripts/01_qc_and_splits.py --config config.yaml
"""

import argparse
import json
import sys
import time
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.utils import setup_logging, format_time


def load_v3_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def is_control_probe(gene_name, control_prefixes):
    """Check if a gene name is a control/blank probe."""
    return any(gene_name.startswith(p) for p in control_prefixes)


# ── Step 1: Load all samples and validate genes ─────────────────────────

def validate_genes(config, logger):
    """Load all h5ad files, filter controls, compute per-cohort gene panels."""
    hest_dir = Path(config["hest_dir"]).resolve()
    control_prefixes = config["control_prefixes"]

    logger.info("=" * 60)
    logger.info("Step 1: Gene validation (two-panel)")
    logger.info("=" * 60)

    sample_info = {}
    sample_real_genes = {}  # sid -> set of real gene names

    # Load all 18 samples
    all_samples = config["all_samples"]
    for sid in all_samples:
        adata_path = hest_dir / "st" / f"{sid}.h5ad"
        if not adata_path.exists():
            logger.error(f"  MISSING: {adata_path}")
            continue

        adata = sc.read_h5ad(adata_path)
        all_genes = list(adata.var_names)
        real_genes = [g for g in all_genes if not is_control_probe(g, control_prefixes)]
        control_genes = [g for g in all_genes if is_control_probe(g, control_prefixes)]

        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        total_counts = X.sum(axis=1)
        genes_detected = (X > 0).sum(axis=1)

        sample_real_genes[sid] = set(real_genes)
        sample_info[sid] = {
            "n_spots": adata.n_obs,
            "n_total_probes": len(all_genes),
            "n_real_genes": len(real_genes),
            "n_control_probes": len(control_genes),
            "median_counts": float(np.median(total_counts)),
            "mean_counts": float(np.mean(total_counts)),
            "median_genes_detected": float(np.median(genes_detected)),
        }

        logger.info(f"  {sid}: {adata.n_obs:,} spots, "
                     f"{len(real_genes)} real genes + {len(control_genes)} controls, "
                     f"median counts={np.median(total_counts):.0f}")

    # Per-cohort gene panels
    cohort_genes = {}
    for cohort_key, cohort_cfg in config["cohorts"].items():
        cohort_name = cohort_cfg["name"]
        sids = cohort_cfg["samples"]
        gene_sets = [sample_real_genes[sid] for sid in sids if sid in sample_real_genes]

        intersection = set.intersection(*gene_sets)
        union = set.union(*gene_sets)
        cohort_genes[cohort_key] = sorted(intersection)

        logger.info(f"\n  {cohort_name} ({len(sids)} samples):")
        logger.info(f"    Intersection: {len(intersection)} real genes")
        logger.info(f"    Union: {len(union)} real genes")

        # Check for per-sample deviations
        for sid in sids:
            if sid not in sample_real_genes:
                continue
            extra = sample_real_genes[sid] - intersection
            if extra:
                logger.info(f"    {sid}: +{len(extra)} extra genes beyond intersection")

    # Bridge genes
    disc_set = set(cohort_genes["discovery"])
    val_set = set(cohort_genes["validation"])
    bridge = sorted(disc_set & val_set)
    only_disc = sorted(disc_set - val_set)
    only_val = sorted(val_set - disc_set)

    logger.info(f"\n  Cross-panel summary:")
    logger.info(f"    Discovery panel: {len(cohort_genes['discovery'])} genes")
    logger.info(f"    Validation panel: {len(cohort_genes['validation'])} genes")
    logger.info(f"    Bridge (shared): {len(bridge)} genes")
    logger.info(f"    Discovery-only: {len(only_disc)} genes")
    logger.info(f"    Validation-only: {len(only_val)} genes")

    return cohort_genes, bridge, sample_info


# ── Step 2: Save gene lists, patient mappings, and splits ────────────────

def save_outputs(config, cohort_genes, bridge, logger):
    """Save gene lists, patient mappings, and LOPO splits for both cohorts."""
    v3_dir = Path(config["v3_data_dir"])
    v3_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'=' * 60}")
    logger.info("Step 2: Save gene lists, mappings, and LOPO splits")
    logger.info(f"{'=' * 60}")

    # Gene lists
    for cohort_key, genes in cohort_genes.items():
        cohort_name = config["cohorts"][cohort_key]["name"]
        gene_path = v3_dir / f"gene_list_{cohort_name}.json"
        with open(gene_path, "w") as f:
            json.dump(genes, f, indent=2)
        logger.info(f"  Saved {len(genes)} genes -> {gene_path}")

    bridge_path = v3_dir / "gene_list_bridge.json"
    with open(bridge_path, "w") as f:
        json.dump(bridge, f, indent=2)
    logger.info(f"  Saved {len(bridge)} bridge genes -> {bridge_path}")

    # Patient mappings and LOPO splits per cohort
    for cohort_key, cohort_cfg in config["cohorts"].items():
        cohort_name = cohort_cfg["name"]
        mapping = cohort_cfg["patient_mapping"]

        # Save patient mapping
        map_path = v3_dir / f"patient_mapping_{cohort_name}.json"
        with open(map_path, "w") as f:
            json.dump(mapping, f, indent=2)
        logger.info(f"  Saved patient mapping ({len(mapping)} patients) -> {map_path}")

        # Verify all samples accounted for
        mapped = set()
        for samples in mapping.values():
            mapped.update(samples)
        config_samples = set(cohort_cfg["samples"])
        assert mapped == config_samples, \
            f"Mismatch in {cohort_name}: {mapped.symmetric_difference(config_samples)}"

        # Generate LOPO splits
        splits_dir = v3_dir / f"lopo_splits_{cohort_name}"
        splits_dir.mkdir(parents=True, exist_ok=True)

        patients = sorted(mapping.keys())
        assert len(patients) == cohort_cfg["n_lopo_folds"], \
            f"Expected {cohort_cfg['n_lopo_folds']} patients in {cohort_name}, got {len(patients)}"

        for fold_idx, test_patient in enumerate(patients):
            test_samples = mapping[test_patient]
            train_samples = []
            train_patients = []

            for patient, samples in mapping.items():
                if patient != test_patient:
                    train_samples.extend(samples)
                    train_patients.append(patient)

            split = {
                "fold": fold_idx,
                "cohort": cohort_name,
                "test_patient": test_patient,
                "test_samples": test_samples,
                "train_patients": train_patients,
                "train_samples": train_samples,
                "n_test_samples": len(test_samples),
                "n_train_samples": len(train_samples),
            }

            split_path = splits_dir / f"fold_{fold_idx}.json"
            with open(split_path, "w") as f:
                json.dump(split, f, indent=2)

            logger.info(f"  [{cohort_name}] Fold {fold_idx}: test={test_patient} "
                         f"({len(test_samples)} samples), "
                         f"train={len(train_samples)} samples")


# ── Step 3: Batch effect assessment ─────────────────────────────────────

def assess_batch_effects(config, cohort_genes, logger):
    """PCA pseudo-bulk colored by cohort and patient."""
    hest_dir = Path(config["hest_dir"]).resolve()
    output_dir = Path(config["output_dir"]) / "phase0"
    output_dir.mkdir(parents=True, exist_ok=True)
    control_prefixes = config["control_prefixes"]

    logger.info(f"\n{'=' * 60}")
    logger.info("Step 3: Batch effect assessment")
    logger.info(f"{'=' * 60}")

    # Build reverse mappings
    sample_to_cohort = {}
    sample_to_patient = {}
    for cohort_key, cohort_cfg in config["cohorts"].items():
        for sid in cohort_cfg["samples"]:
            sample_to_cohort[sid] = cohort_cfg["name"]
        for patient, sids in cohort_cfg["patient_mapping"].items():
            for sid in sids:
                sample_to_patient[sid] = patient

    # Use bridge genes for cross-cohort PCA (fair comparison)
    bridge_genes = sorted(set(cohort_genes["discovery"]) & set(cohort_genes["validation"]))
    logger.info(f"  Using {len(bridge_genes)} bridge genes for cross-cohort PCA")

    pseudobulk = []
    sample_labels = []
    cohort_labels = []
    patient_labels = []

    for sid in config["all_samples"]:
        adata_path = hest_dir / "st" / f"{sid}.h5ad"
        if not adata_path.exists():
            continue

        adata = sc.read_h5ad(adata_path)
        available = [g for g in bridge_genes if g in adata.var_names]
        if len(available) < 10:
            logger.warning(f"  {sid}: only {len(available)} bridge genes available, skipping PCA")
            continue

        subset = adata[:, available]
        X = subset.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.log1p(X)

        pseudobulk.append(X.mean(axis=0))
        sample_labels.append(sid)
        cohort_labels.append(sample_to_cohort.get(sid, "unknown"))
        patient_labels.append(sample_to_patient.get(sid, "unknown"))

    pseudobulk = np.array(pseudobulk)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(5, len(pseudobulk)))
    pc_coords = pca.fit_transform(pseudobulk)

    logger.info(f"  PCA explained variance (bridge genes): {pca.explained_variance_ratio_[:3].round(3)}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    cohort_colors = {"biomarkers": "#ff7f0e", "10x_janesick": "#1f77b4"}

    ax = axes[0]
    for cohort in sorted(set(cohort_labels)):
        mask = [c == cohort for c in cohort_labels]
        ax.scatter(pc_coords[mask, 0], pc_coords[mask, 1],
                   label=cohort, c=cohort_colors.get(cohort, "gray"), s=80, alpha=0.8)
        for i, m in enumerate(mask):
            if m:
                ax.annotate(sample_labels[i], (pc_coords[i, 0], pc_coords[i, 1]),
                            fontsize=6, alpha=0.7)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title(f"Pseudo-bulk PCA by Cohort ({len(bridge_genes)} bridge genes)")
    ax.legend()

    ax = axes[1]
    unique_patients = sorted(set(patient_labels))
    cmap = plt.colormaps.get_cmap("tab10")
    for idx, patient in enumerate(unique_patients):
        mask = [p == patient for p in patient_labels]
        ax.scatter(pc_coords[mask, 0], pc_coords[mask, 1],
                   label=patient, c=[cmap(idx)], s=80, alpha=0.8)
        for i, m in enumerate(mask):
            if m:
                ax.annotate(sample_labels[i], (pc_coords[i, 0], pc_coords[i, 1]),
                            fontsize=6, alpha=0.7)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("Pseudo-bulk PCA by Patient")
    ax.legend(fontsize=7)

    plt.tight_layout()
    fig_path = output_dir / "batch_effect_pca.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {fig_path}")

    # Silhouette
    if len(set(cohort_labels)) > 1 and len(pseudobulk) > 3:
        from sklearn.metrics import silhouette_score
        sil = silhouette_score(pc_coords[:, :min(3, pc_coords.shape[1])], cohort_labels)
        logger.info(f"  Silhouette score (cohort): {sil:.3f} "
                     f"({'Strong batch' if sil > 0.5 else 'Moderate' if sil > 0.25 else 'Weak/none'})")


# ── Step 4: Pathway coverage per panel ──────────────────────────────────

def assess_pathway_coverage(config, cohort_genes, logger):
    """Check Hallmark pathway coverage for each panel."""
    output_dir = Path(config["output_dir"]) / "phase0"
    output_dir.mkdir(parents=True, exist_ok=True)
    v3_dir = Path(config["v3_data_dir"])

    logger.info(f"\n{'=' * 60}")
    logger.info("Step 4: Pathway coverage per panel")
    logger.info(f"{'=' * 60}")

    gmt_path = Path(config["phase3"]["hallmark_gmt"])
    if not gmt_path.exists():
        logger.warning(f"  GMT file not found: {gmt_path}")
        return {}

    # Parse GMT
    pathways = OrderedDict()
    with open(gmt_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            pathways[parts[0]] = parts[2:]

    results = {}
    for cohort_key in ["discovery", "validation"]:
        cohort_name = config["cohorts"][cohort_key]["name"]
        gene_set = set(cohort_genes[cohort_key])

        logger.info(f"\n  [{cohort_name}] {len(gene_set)} genes")

        rows = []
        usable = 0
        for pw_name, pw_genes in pathways.items():
            overlap = [g for g in pw_genes if g in gene_set]
            ok = len(overlap) >= 5
            if ok:
                usable += 1
            rows.append({
                "pathway": pw_name,
                "total_genes": len(pw_genes),
                "panel_genes": len(overlap),
                "fraction": round(len(overlap) / len(pw_genes), 3) if pw_genes else 0,
                "usable": ok,
                "overlap_genes": ";".join(overlap),
            })

        df = pd.DataFrame(rows).sort_values("panel_genes", ascending=False)
        csv_path = v3_dir / f"pathway_coverage_{cohort_name}.csv"
        df.to_csv(csv_path, index=False)

        logger.info(f"    Usable pathways (>=5 genes): {usable}/{len(pathways)}")
        # Show top 5
        for _, row in df.head(5).iterrows():
            logger.info(f"    {row['pathway']}: {row['panel_genes']}/{row['total_genes']}")

        results[cohort_key] = df

    # Bridge gene pathway coverage
    bridge_set = set(cohort_genes["discovery"]) & set(cohort_genes["validation"])
    logger.info(f"\n  [bridge] {len(bridge_set)} shared genes")
    bridge_usable = 0
    for pw_name, pw_genes in pathways.items():
        overlap = [g for g in pw_genes if g in bridge_set]
        if len(overlap) >= 5:
            bridge_usable += 1
    logger.info(f"    Bridge usable pathways: {bridge_usable}/{len(pathways)}")

    return results


# ── Step 5: QC report ───────────────────────────────────────────────────

def write_qc_report(config, cohort_genes, bridge, sample_info, pathway_results, logger):
    """Write comprehensive QC report."""
    output_dir = Path(config["output_dir"]) / "phase0"
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Phase 0 QC Report: v3 IDC Xenium — Two-Panel Design\n")

    # Overview
    disc_cfg = config["cohorts"]["discovery"]
    val_cfg = config["cohorts"]["validation"]
    lines.append("## Design Overview\n")
    lines.append("| | Discovery | Validation | Bridge |")
    lines.append("|---|-----------|------------|--------|")
    lines.append(f"| Cohort | {disc_cfg['name']} | {val_cfg['name']} | — |")
    lines.append(f"| Samples | {len(disc_cfg['samples'])} | {len(val_cfg['samples'])} | — |")
    lines.append(f"| Patients | {len(disc_cfg['patient_mapping'])} | {len(val_cfg['patient_mapping'])} | — |")
    lines.append(f"| Real genes | {len(cohort_genes['discovery'])} | {len(cohort_genes['validation'])} | {len(bridge)} |")

    total_disc_spots = sum(sample_info.get(s, {}).get("n_spots", 0) for s in disc_cfg["samples"])
    total_val_spots = sum(sample_info.get(s, {}).get("n_spots", 0) for s in val_cfg["samples"])
    lines.append(f"| Total spots | {total_disc_spots:,} | {total_val_spots:,} | — |")
    lines.append("")

    # Per-cohort sample tables
    for cohort_key in ["discovery", "validation"]:
        cohort_cfg = config["cohorts"][cohort_key]
        cohort_name = cohort_cfg["name"]

        # Reverse patient mapping
        s2p = {}
        for p, sids in cohort_cfg["patient_mapping"].items():
            for sid in sids:
                s2p[sid] = p

        lines.append(f"\n## {cohort_name.title()} Cohort ({len(cohort_genes[cohort_key])} genes)\n")
        lines.append("| Sample | Spots | Real Genes | Controls | Median Counts | Median Genes | Patient |")
        lines.append("|--------|------:|----------:|---------:|--------------:|-------------:|---------|")

        for sid in cohort_cfg["samples"]:
            info = sample_info.get(sid, {})
            lines.append(
                f"| {sid} | {info.get('n_spots', 0):,} | {info.get('n_real_genes', 0)} | "
                f"{info.get('n_control_probes', 0)} | "
                f"{info.get('median_counts', 0):.0f} | "
                f"{info.get('median_genes_detected', 0):.0f} | {s2p.get(sid, '?')} |"
            )

    # LOPO splits
    lines.append("\n## LOPO Splits\n")
    for cohort_key in ["discovery", "validation"]:
        cohort_cfg = config["cohorts"][cohort_key]
        cohort_name = cohort_cfg["name"]
        patients = sorted(cohort_cfg["patient_mapping"].keys())

        lines.append(f"\n### {cohort_name.title()} ({len(patients)}-fold)\n")
        lines.append("| Fold | Test Patient | Test Samples | Train Samples |")
        lines.append("|-----:|:-------------|-------------:|--------------:|")
        for fold, patient in enumerate(patients):
            n_test = len(cohort_cfg["patient_mapping"][patient])
            n_train = len(cohort_cfg["samples"]) - n_test
            lines.append(f"| {fold} | {patient} | {n_test} | {n_train} |")

    # Pathway coverage
    lines.append("\n## Pathway Coverage\n")
    for cohort_key in ["discovery", "validation"]:
        cohort_name = config["cohorts"][cohort_key]["name"]
        if cohort_key in pathway_results:
            df = pathway_results[cohort_key]
            usable = df[df["usable"]].shape[0]
            lines.append(f"**{cohort_name.title()}:** {usable}/{len(df)} usable Hallmark pathways (>=5 genes)\n")

            lines.append("| Pathway | Panel/Total | Fraction |")
            lines.append("|---------|:------------|:---------|")
            for _, row in df.head(10).iterrows():
                lines.append(f"| {row['pathway']} | {row['panel_genes']}/{row['total_genes']} | {row['fraction']:.0%} |")
            lines.append("")

    # Bridge genes
    lines.append(f"\n## Bridge Genes ({len(bridge)})\n")
    lines.append("These 90 genes are shared between panels and enable cross-cohort comparison.\n")
    # List in columns
    for i in range(0, len(bridge), 6):
        chunk = bridge[i:i+6]
        lines.append("  ".join(f"`{g}`" for g in chunk))

    report_text = "\n".join(lines)
    report_path = output_dir / "qc_report.md"
    with open(report_path, "w") as f:
        f.write(report_text)

    logger.info(f"\nQC report saved: {report_path}")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="v3 QC — two-panel design")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_v3_config(args.config)

    log_dir = Path(config["output_dir"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(
        log_level=config.get("log_level", "INFO"),
        log_file=str(log_dir / "01_qc_and_splits.log"),
    )

    logger.info("=" * 60)
    logger.info("v3 Phase 0: QC — Two-Panel Discovery/Validation Design")
    logger.info("=" * 60)

    start = time.time()

    # Step 1: Validate genes
    cohort_genes, bridge, sample_info = validate_genes(config, logger)

    # Step 2: Save gene lists, patient mappings, LOPO splits
    save_outputs(config, cohort_genes, bridge, logger)

    # Step 3: Batch effect assessment
    assess_batch_effects(config, cohort_genes, logger)

    # Step 4: Pathway coverage
    pathway_results = assess_pathway_coverage(config, cohort_genes, logger)

    # Step 5: QC report
    write_qc_report(config, cohort_genes, bridge, sample_info, pathway_results, logger)

    elapsed = time.time() - start
    logger.info(f"\nPhase 0 complete in {format_time(elapsed)}")


if __name__ == "__main__":
    main()

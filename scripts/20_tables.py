#!/usr/bin/env python3
"""
Script 20: Generate tables (main + supplementary).

Output:
    outputs/figures/tables/
        table1_dataset_characteristics.csv
        table2_gate_results.csv
        table3_pathway_replication.csv
        table4_ols_coefficients.csv
        supp_table_s1_per_gene_performance.csv
        supp_table_s2_bridge_genes.csv
        supp_table_s3_full_de_results.csv
        supp_table_s4_within_patient.csv
        supp_table_s5_pathway_overlap.csv
        supp_table_s6_encoder_specific.csv
        supp_table_s8_subsampling.csv

Usage:
    python scripts/20_tables.py
    python scripts/20_tables.py --table 1 2    # specific tables
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.utils import setup_logging, format_time
from src.plotting import load_config, load_json


def main():
    parser = argparse.ArgumentParser(description="Generate tables")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--table", type=str, nargs="*", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    base = Path(config["output_dir"])
    out_dir = base / "figures" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_dir = base / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(
        log_level=config.get("log_level", "INFO"),
        log_file=str(log_dir / "20_tables.log"),
    )

    logger.info("Generating tables...")
    overall_start = time.time()

    all_tables = args.table or ["1", "2", "3", "4", "S1", "S2", "S3", "S4", "S5", "S6", "S8"]

    # ===== Table 1: Dataset Characteristics =====
    if "1" in all_tables:
        logger.info("Table 1: Dataset Characteristics")
        records = []
        for cohort_key in ["discovery", "validation"]:
            cohort_config = config["cohorts"][cohort_key]
            cohort_name = cohort_config["name"]
            for patient_id, patient_samples in cohort_config["patient_mapping"].items():
                for sid in patient_samples:
                    h5ad_path = Path(config["hest_dir"]) / "st" / f"{sid}.h5ad"
                    n_spots = 0
                    median_umi = 0
                    n_genes_detected = 0
                    if h5ad_path.exists():
                        adata = sc.read_h5ad(str(h5ad_path))
                        tissue_mask = adata.obs.get("in_tissue", pd.Series([True] * adata.n_obs))
                        n_spots = int(tissue_mask.sum()) if tissue_mask.dtype == bool else adata.n_obs
                        if hasattr(adata.X, 'toarray'):
                            X = adata.X.toarray()
                        else:
                            X = np.array(adata.X)
                        median_umi = float(np.median(X.sum(axis=1)))
                        n_genes_detected = int((X > 0).any(axis=0).sum())

                    source = "Janesick" if sid.startswith("NCBI") else ("10x Public" if sid.startswith("TENX9") else "Biomarkers")
                    has_serial = len(patient_samples) > 1
                    records.append({
                        "Sample ID": sid,
                        "Cohort": cohort_key.title(),
                        "Patient": patient_id,
                        "Source": source,
                        "Spots": n_spots,
                        "Genes detected": n_genes_detected,
                        "Median UMI": f"{median_umi:.0f}",
                        "Serial sections": "Yes" if has_serial else "No",
                    })

        table1 = pd.DataFrame(records)
        table1.to_csv(out_dir / "table1_dataset_characteristics.csv", index=False)
        logger.info(f"  Saved table 1 ({len(table1)} rows)")

    # ===== Table 2: Gate Results =====
    if "2" in all_tables:
        logger.info("Table 2: Gate Results")
        gate1 = load_json(base / "phase2" / "gate2_1_agreement.json")
        gate2 = load_json(base / "phase2" / "gate2_2_spatial.json")
        gate3 = load_json(base / "phase2" / "gate2_3_dual_track.json")

        records = []
        for cohort_key in ["discovery", "validation"]:
            if gate1:
                g1 = gate1["cohorts"][cohort_key]
                rhos = [s["median_rho"] for s in g1["samples"].values()]
                records.append({
                    "Gate": "2.1 Multi-model agreement",
                    "Cohort": cohort_key.title(),
                    "Criterion": "Median ρ > 0.4 in ≥50% samples",
                    "Pass/Total": f"{g1['n_pass']}/{g1['n_total']}",
                    "Fraction": f"{g1['fraction_pass']:.1%}",
                    "Median (range)": f"{np.median(rhos):.3f} ({min(rhos):.3f}-{max(rhos):.3f})",
                    "Result": "PASS" if g1["gate_pass"] else "FAIL",
                })

            if gate2:
                g2 = gate2["cohorts"][cohort_key]
                morans = [s["morans_i"] for s in g2["samples"].values()]
                records.append({
                    "Gate": "2.2 Spatial structure",
                    "Cohort": cohort_key.title(),
                    "Criterion": "Moran's I p<0.01 in ≥50%",
                    "Pass/Total": f"{g2['n_pass']}/{g2['n_total']}",
                    "Fraction": f"{g2['fraction_pass']:.1%}",
                    "Median (range)": f"{np.median(morans):.3f} ({min(morans):.3f}-{max(morans):.3f})",
                    "Result": "PASS" if g2["gate_pass"] else "FAIL",
                })

            if gate3:
                g3 = gate3["cohorts"][cohort_key]
                records.append({
                    "Gate": "2.3 Dual-track concordance",
                    "Cohort": cohort_key.title(),
                    "Criterion": "Median ρ > 0.15",
                    "Pass/Total": "-",
                    "Fraction": "-",
                    "Median (range)": f"{g3['overall_median_rho']:.3f}",
                    "Result": "PASS" if g3["gate_pass"] else "FAIL",
                })

        table2 = pd.DataFrame(records)
        table2.to_csv(out_dir / "table2_gate_results.csv", index=False)
        logger.info(f"  Saved table 2 ({len(table2)} rows)")

    # ===== Table 3: Pathway Replication =====
    if "3" in all_tables:
        logger.info("Table 3: Pathway Replication")
        disc_pw = pd.read_csv(base / "phase3" / "pathways" / config["cohorts"]["discovery"]["name"] / "pathway_consistency.csv")
        val_pw = pd.read_csv(base / "phase3" / "pathways" / config["cohorts"]["validation"]["name"] / "pathway_consistency.csv")

        # Also load overlap info
        disc_overlap = pd.read_csv(base / "phase3" / "pathways" / config["cohorts"]["discovery"]["name"] / "pathway_overlap.csv")
        val_overlap = pd.read_csv(base / "phase3" / "pathways" / config["cohorts"]["validation"]["name"] / "pathway_overlap.csv")

        merged = disc_pw[["pathway", "mean_cohens_d", "direction_consistency", "reproducibility"]].rename(
            columns={"mean_cohens_d": "d_disc", "direction_consistency": "dc_disc", "reproducibility": "repro_disc"}
        ).merge(
            val_pw[["pathway", "mean_cohens_d", "direction_consistency", "reproducibility"]].rename(
                columns={"mean_cohens_d": "d_val", "direction_consistency": "dc_val", "reproducibility": "repro_val"}
            ),
            on="pathway", how="outer"
        ).fillna(0)

        # Add overlap counts
        if "pathway" in disc_overlap.columns and "n_overlap" in disc_overlap.columns:
            merged = merged.merge(
                disc_overlap[["pathway", "n_overlap"]].rename(columns={"n_overlap": "n_genes_disc"}),
                on="pathway", how="left"
            )
        if "pathway" in val_overlap.columns and "n_overlap" in val_overlap.columns:
            merged = merged.merge(
                val_overlap[["pathway", "n_overlap"]].rename(columns={"n_overlap": "n_genes_val"}),
                on="pathway", how="left"
            )

        # Determine replication
        merged["replicates"] = (
            (merged["repro_disc"] > 0.5) &
            (merged["repro_val"] > 0.5) &
            (np.sign(merged["d_disc"]) == np.sign(merged["d_val"]))
        )

        merged = merged.sort_values("d_disc", ascending=False)
        merged["pathway_clean"] = merged["pathway"].str.replace("HALLMARK_", "").str.replace("_", " ").str.title()

        table3 = merged[["pathway_clean", "d_disc", "d_val", "dc_disc", "dc_val", "replicates"]].copy()
        table3.columns = ["Pathway", "Cohen's d (disc)", "Cohen's d (val)",
                          "Direction consistency (disc)", "Direction consistency (val)", "Replicates"]
        table3.to_csv(out_dir / "table3_pathway_replication.csv", index=False)
        logger.info(f"  Saved table 3 ({len(table3)} rows)")

    # ===== Table 4: OLS Coefficients =====
    if "4" in all_tables:
        logger.info("Table 4: OLS Coefficients")
        records = []
        for cohort_key in ["discovery", "validation"]:
            cohort_name = config["cohorts"][cohort_key]["name"]
            ols = load_json(base / "phase3" / "gene_predictability" / cohort_name / "ols_results.json")
            if ols:
                for c in ols["coefficients"]:
                    records.append({
                        "Cohort": cohort_key.title(),
                        "Feature": c["feature"],
                        "β": f"{c['coefficient']:.4f}",
                        "SE": f"{c['std_error']:.4f}",
                        "t": f"{c['t_statistic']:.2f}",
                        "p-value": f"{c['p_value']:.2e}" if c["p_value"] is not None else "N/A",
                        "Significant": "***" if c["p_value"] is not None and c["p_value"] < 0.001
                                       else "**" if c["p_value"] is not None and c["p_value"] < 0.01
                                       else "*" if c["p_value"] is not None and c["p_value"] < 0.05
                                       else "",
                    })
                # Add model summary row
                records.append({
                    "Cohort": cohort_key.title(),
                    "Feature": "MODEL SUMMARY",
                    "β": f"R²={ols['r_squared']:.3f}",
                    "SE": f"Adj R²={ols['adj_r_squared']:.3f}",
                    "t": f"N={ols['n_genes']}",
                    "p-value": f"Features={ols['n_features']}",
                    "Significant": "",
                })

        table4 = pd.DataFrame(records)
        table4.to_csv(out_dir / "table4_ols_coefficients.csv", index=False)
        logger.info(f"  Saved table 4 ({len(table4)} rows)")

    # ===== Supp Table S1: Per-Gene Performance =====
    if "S1" in all_tables:
        logger.info("Supp Table S1: Per-Gene Performance")
        all_rows = []
        for cohort_key in ["discovery", "validation"]:
            cohort_name = config["cohorts"][cohort_key]["name"]

            enc_df = pd.read_csv(base / "phase4" / "encoder_consistency" / cohort_name / "per_gene_pearson_by_encoder.csv")
            gf_path = base / "phase3" / "gene_predictability" / cohort_name / "gene_features.csv"
            gf = pd.read_csv(gf_path) if Path(gf_path).exists() else pd.DataFrame()

            merged = enc_df.copy()
            if not gf.empty:
                merge_cols = [c for c in gf.columns if c != "mean_pearson"]
                merged = merged.merge(gf[merge_cols], on="gene", how="left")

            merged.insert(0, "cohort", cohort_key.title())
            all_rows.append(merged)

        s1 = pd.concat(all_rows, ignore_index=True)
        s1.to_csv(out_dir / "supp_table_s1_per_gene_performance.csv", index=False)
        logger.info(f"  Saved S1 ({len(s1)} rows)")

    # ===== Supp Table S2: Bridge Gene Detail =====
    if "S2" in all_tables:
        logger.info("Supp Table S2: Bridge Gene Detail")
        bg = pd.read_csv(base / "phase4" / "bridge_genes" / "bridge_gene_pearson.csv")
        bg_json = load_json(base / "phase4" / "bridge_genes" / "bridge_gene_correlation.json")
        both_sig = set(bg_json.get("de_overlap", {}).get("both_sig_genes", []))

        # Add DE info
        disc_meta = pd.read_csv(base / "phase3" / "de" / config["cohorts"]["discovery"]["name"] / "meta_de_unmatched.csv")
        val_meta = pd.read_csv(base / "phase3" / "de" / config["cohorts"]["validation"]["name"] / "meta_de_unmatched.csv")

        bg = bg.merge(
            disc_meta[["gene", "median_log2fc", "reproducibility"]].rename(
                columns={"median_log2fc": "log2fc_disc", "reproducibility": "repro_disc"}
            ), on="gene", how="left"
        ).merge(
            val_meta[["gene", "median_log2fc", "reproducibility"]].rename(
                columns={"median_log2fc": "log2fc_val", "reproducibility": "repro_val"}
            ), on="gene", how="left"
        )

        bg["de_both"] = bg["gene"].isin(both_sig)
        bg["direction_consistent"] = np.sign(bg["log2fc_disc"].fillna(0)) == np.sign(bg["log2fc_val"].fillna(0))

        bg.to_csv(out_dir / "supp_table_s2_bridge_genes.csv", index=False)
        logger.info(f"  Saved S2 ({len(bg)} rows)")

    # ===== Supp Table S3: Full DE Results =====
    if "S3" in all_tables:
        logger.info("Supp Table S3: Full DE Results")
        all_de = []
        for cohort_key in ["discovery", "validation"]:
            cohort_name = config["cohorts"][cohort_key]["name"]
            de_dir = base / "phase3" / "de" / cohort_name / "per_sample"
            if de_dir.exists():
                for f in sorted(de_dir.glob("*_unmatched_de.csv")):
                    sample_id = f.stem.replace("_unmatched_de", "")
                    df = pd.read_csv(f)
                    df.insert(0, "sample_id", sample_id)
                    df.insert(0, "cohort", cohort_key.title())
                    df.insert(len(df.columns), "de_type", "unmatched")
                    all_de.append(df)
                for f in sorted(de_dir.glob("*_matched_de.csv")):
                    sample_id = f.stem.replace("_matched_de", "")
                    df = pd.read_csv(f)
                    df.insert(0, "sample_id", sample_id)
                    df.insert(0, "cohort", cohort_key.title())
                    df.insert(len(df.columns), "de_type", "matched")
                    all_de.append(df)

        if all_de:
            s3 = pd.concat(all_de, ignore_index=True)
            s3.to_csv(out_dir / "supp_table_s3_full_de_results.csv", index=False)
            logger.info(f"  Saved S3 ({len(s3)} rows)")

    # ===== Supp Table S4: Within-Patient Detail =====
    if "S4" in all_tables:
        logger.info("Supp Table S4: Within-Patient Detail")
        gene_corr = pd.read_csv(base / "phase3" / "within_patient" / "gene_residual_correlation.csv")
        spatial_corr = pd.read_csv(base / "phase3" / "within_patient" / "spatial_binned_correlation.csv")
        dice = pd.read_csv(base / "phase3" / "within_patient" / "dice_top_quartile.csv")

        merged = gene_corr.merge(
            spatial_corr[["patient_id", "section_a", "section_b", "spearman_r", "n_common_bins"]],
            on=["patient_id", "section_a", "section_b"], how="left"
        ).merge(
            dice[["patient_id", "section_a", "section_b", "dice_coeff"]],
            on=["patient_id", "section_a", "section_b"], how="left"
        )

        merged.columns = ["Patient", "Cohort", "Section A", "Section B",
                          "Gene r", "Gene p-value", "N genes",
                          "Spatial ρ", "N common bins", "Dice"]
        merged.to_csv(out_dir / "supp_table_s4_within_patient.csv", index=False)
        logger.info(f"  Saved S4 ({len(merged)} rows)")

    # ===== Supp Table S5: Pathway-Gene Overlap =====
    if "S5" in all_tables:
        logger.info("Supp Table S5: Pathway-Gene Overlap")
        all_overlap = []
        for cohort_key in ["discovery", "validation"]:
            cohort_name = config["cohorts"][cohort_key]["name"]
            overlap_path = base / "phase3" / "pathways" / cohort_name / "pathway_overlap.csv"
            if overlap_path.exists():
                df = pd.read_csv(overlap_path)
                df.insert(0, "cohort", cohort_key.title())
                all_overlap.append(df)

        if all_overlap:
            s5 = pd.concat(all_overlap, ignore_index=True)
            s5.to_csv(out_dir / "supp_table_s5_pathway_overlap.csv", index=False)
            logger.info(f"  Saved S5 ({len(s5)} rows)")

    # ===== Supp Table S6: Encoder-Specific Genes =====
    if "S6" in all_tables:
        logger.info("Supp Table S6: Encoder-Specific Genes")
        all_spec = []
        for cohort_key in ["discovery", "validation"]:
            cohort_name = config["cohorts"][cohort_key]["name"]
            spec_path = base / "phase4" / "encoder_consistency" / cohort_name / "encoder_specific_genes.csv"
            if spec_path.exists():
                df = pd.read_csv(spec_path)
                df.insert(0, "cohort", cohort_key.title())
                all_spec.append(df)

        if all_spec:
            s6 = pd.concat(all_spec, ignore_index=True)
            s6.to_csv(out_dir / "supp_table_s6_encoder_specific.csv", index=False)
            logger.info(f"  Saved S6 ({len(s6)} rows)")

    # ===== Supp Table S8: Subsampling Results =====
    if "S8" in all_tables:
        logger.info("Supp Table S8: Subsampling Results")
        sub_path = base / "figure_data" / "subsampling_curve.csv"
        if sub_path.exists():
            df = pd.read_csv(sub_path)
            summary = df.groupby(["cohort", "n_genes_per_track"])["spearman_rho"].agg(
                n_splits='count',
                median_rho='median',
                q25=lambda x: x.quantile(0.25),
                q75=lambda x: x.quantile(0.75),
                mean_rho='mean',
                std_rho='std',
            ).reset_index()
            # Average across samples for cleaner table
            clean = summary.groupby(["cohort", "n_genes_per_track"]).agg({
                "median_rho": "median",
                "q25": "median",
                "q75": "median",
                "mean_rho": "mean",
            }).reset_index()
            clean.columns = ["Cohort", "Genes per track", "Median ρ", "Q25", "Q75", "Mean ρ"]
            clean.to_csv(out_dir / "supp_table_s8_subsampling.csv", index=False)
            logger.info(f"  Saved S8 ({len(clean)} rows)")

    total_time = time.time() - overall_start
    logger.info(f"\nTotal time: {format_time(total_time)}")
    logger.info("Done.")


if __name__ == "__main__":
    main()

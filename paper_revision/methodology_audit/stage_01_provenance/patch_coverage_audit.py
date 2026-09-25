"""Audit supplied patch membership and coordinates; no prediction/biology changes."""
from datetime import datetime, timezone
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import shapely
import yaml

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
H = ROOT / "data/hest"


def strings(x):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in np.asarray(x).ravel()]


def index(g):
    return strings(g[g.attrs["_index"]][:])


def main():
    cfg = yaml.safe_load((ROOT / "paper_revision/clean_repo/config.yaml").read_text())
    source = json.loads((OUT / "sources/hest_source_manifest.json").read_text())
    sources = {r["relative_path"]: r for r in source["files"] if r["status"] == 200}
    samples = cfg["all_samples"] + ["TENX111", "TENX147", "TENX148", "TENX149",
                "TENX13", "TENX14", "TENX39", "TENX53", "TENX68", "NCBI776",
                "NCBI681", "NCBI682", "NCBI683", "NCBI684"]
    cohorts = {s: c["name"] for c in cfg["cohorts"].values() for s in c["samples"]}
    summary, distributions, rows, issues = [], [], [], []
    (OUT / "figures").mkdir(exist_ok=True)
    with PdfPages(OUT / "figures/patch_coverage_all_sections.pdf") as pdf:
        for sid in samples:
            with h5py.File(H / "st" / f"{sid}.h5ad", "r") as f:
                ids, genes = index(f["obs"]), index(f["var"])
                xy = f["obsm/spatial"][:]
                st = f["uns/spatial"]
                sg = st[list(st.keys())[0]]
                thumb = sg["images/downscaled_fullres"][:]
                thumb_scale = float(sg["scalefactors/tissue_downscaled_fullres_scalef"][()])
                diam = float(sg["scalefactors/spot_diameter_fullres"][()])
                counts = f["X"][:] if sid in cohorts else None
            lookup = {v: i for i, v in enumerate(ids)}
            with h5py.File(H / "patches" / f"{sid}.h5", "r") as f:
                pids = strings(f["barcode"][:])
                pxy = f["coords"][:]
                at = dict(f["img"].attrs)
                src = int(at["patch_size_src"])
                pixel = float(at["pixel_size"])
                size_float = int(at["patch_size_target"]) * float(at["downsample"])
            ix = np.array([lookup[s] for s in pids])
            retained = np.zeros(len(ids), dtype=bool); retained[ix] = True
            top_float = xy - size_float // 2
            top = top_float.astype(int)
            coord_difference = np.abs(top[ix] - pxy)
            max_delta = float(coord_difference.max())
            mdsrc = sources.get(f"metadata/{sid}.json")
            masksrc = sources.get(f"tissue_seg/{sid}_contours.geojson")
            md = json.loads(Path(mdsrc["path"]).read_text()) if mdsrc else {}
            width = md.get("fullres_px_width", md.get("fullres_width", thumb.shape[1] / thumb_scale))
            height = md.get("fullres_px_height", md.get("fullres_height", thumb.shape[0] / thumb_scale))
            on_slide = ((top_float[:, 0] + size_float >= 0) & (top_float[:, 0] < width)
                        & (top_float[:, 1] + size_float >= 0) & (top_float[:, 1] < height))
            wholly_on_slide = ((top[:, 0] >= 0) & (top[:, 1] >= 0)
                               & (top[:, 0] + src <= width) & (top[:, 1] + src <= height))
            area_frac = np.full(len(ids), np.nan)
            invalid = 0
            repair_area_delta = 0.0
            repair_union_difference = 0.0
            if masksrc:
                gj = json.loads(Path(masksrc["path"]).read_text())
                geoms = [shapely.geometry.shape(f["geometry"]) for f in gj["features"]]
                invalid = sum(not g.is_valid for g in geoms)
                valid_geoms = [shapely.make_valid(g) if not g.is_valid else g for g in geoms]
                repair_area_delta = sum(abs(a.area - b.area) for a, b in zip(geoms, valid_geoms))
                union = shapely.union_all(valid_geoms)
                if invalid:
                    buffer_union = shapely.union_all([g.buffer(0) for g in geoms])
                    repair_union_difference = union.symmetric_difference(buffer_union).area
                    issues.append({"sample": sid, "issue": "invalid_tissue_geometry_repaired_in_memory",
                                   "count": invalid, "area_change_px2": repair_area_delta,
                                   "make_valid_vs_buffer_union_difference_px2": repair_union_difference})
                shapely.prepare(union)
                # Chunking limits transient geometry memory.
                for start in range(0, len(ids), 1024):
                    t = top[start:start + 1024]
                    boxes = shapely.box(t[:, 0], t[:, 1], t[:, 0] + src, t[:, 1] + src)
                    interior = shapely.covers(union, boxes)
                    intersects = shapely.intersects(union, boxes)
                    boundary = intersects & ~interior
                    fractions = interior.astype(float)
                    fractions[boundary] = shapely.area(shapely.intersection(boxes[boundary], union)) / src ** 2
                    area_frac[start:start + len(t)] = fractions
            expected = on_slide & (area_frac >= .15)
            mask_available = np.isfinite(area_frac).all()
            false_included = retained & ~expected if mask_available else np.zeros(len(ids), bool)
            false_excluded = ~retained & expected if mask_available else np.zeros(len(ids), bool)
            if max_delta:
                issues.append({"sample": sid, "issue": "coordinate_reconstruction_difference", "max_abs_px": max_delta})
            if mask_available and (false_included.any() or false_excluded.any()):
                issues.append({"sample": sid, "issue": "mask_rule_membership_difference",
                               "saved_but_rule_excluded": int(false_included.sum()),
                               "rule_included_but_no_patch": int(false_excluded.sum()),
                               "mask_origin": masksrc["origin"]})
            frame = pd.DataFrame({"sample": sid, "barcode": ids, "x": xy[:, 0], "y": xy[:, 1],
                                  "has_patch": retained, "tissue_fraction": area_frac,
                                  "overlaps_slide": on_slide, "fully_on_slide": wholly_on_slide,
                                  "expected_patch_default_rule": expected,
                                  "default_rule_available": mask_available})
            if counts is not None:
                panel = json.loads((ROOT / "paper_revision/clean_repo/data/v3" / f"gene_list_{cohorts[sid]}.json").read_text())
                gi = {g: i for i, g in enumerate(genes)}
                pidx = [gi[g] for g in panel]
                real = [i for i, g in enumerate(genes) if not any(g.startswith(p) for p in cfg["control_prefixes"])]
                assert np.isfinite(counts).all() and (counts >= 0).all()
                frame["noncontrol_counts"] = counts[:, real].sum(axis=1, dtype=np.float64)
                frame["panel_counts"] = counts[:, pidx].sum(axis=1, dtype=np.float64)
                frame["panel_genes_detected"] = (counts[:, pidx] > 0).sum(axis=1)
                frame["panel_sum_log1p"] = np.log1p(counts[:, pidx].astype(np.float64)).sum(axis=1)
                frame["noncontrol_genes_detected"] = (counts[:, real] > 0).sum(axis=1)
                for group, mask in [("retained", retained), ("excluded", ~retained)]:
                    rec = {"sample": sid, "cohort": cohorts[sid], "group": group, "n": int(mask.sum())}
                    for col in ["noncontrol_counts", "panel_counts", "panel_genes_detected", "panel_sum_log1p", "noncontrol_genes_detected"]:
                        v = frame.loc[mask, col].to_numpy()
                        for label, val in [("mean", np.mean(v)), ("median", np.median(v)),
                                           ("q25", np.quantile(v, .25)), ("q75", np.quantile(v, .75))]:
                            rec[col + "_" + label] = float(val)
                    distributions.append(rec)
                fractional = int(np.count_nonzero(counts != np.floor(counts)))
            else:
                fractional = None
            rec = {"sample": sid, "cohort": cohorts.get(sid, "coad" if sid in samples[18:22] else "idc_visium"),
                   "raw_spots": len(ids), "patches": len(pids), "no_patch": int((~retained).sum()),
                   "max_patch_coordinate_difference_px": max_delta,
                   "patch_size_src": src, "pixel_size_um": pixel, "patch_footprint_um": src * pixel,
                   "nominal_output_pixel_um": pixel * float(at["downsample"]),
                   "h5ad_display_spot_diameter_um": diam * pixel,
                   "mask_origin": masksrc["origin"] if masksrc else "unavailable",
                   "metadata_origin": mdsrc["origin"] if mdsrc else "unavailable",
                   "invalid_contour_count": invalid,
                   "geometry_repair_area_change_px2": repair_area_delta,
                   "make_valid_vs_buffer_union_difference_px2": repair_union_difference,
                   "mask_rule_available": bool(mask_available),
                   "mask_rule_exact_match": bool(mask_available and np.array_equal(expected, retained)),
                   "excluded_no_slide_overlap": int((~retained & ~on_slide).sum()),
                   "excluded_below_mask_threshold": int((~retained & on_slide & (area_frac < .15)).sum()),
                   "retained_below_mask_threshold": int((retained & (area_frac < .15)).sum()),
                   "retained_partially_outside_slide": int((retained & ~wholly_on_slide).sum()),
                   "saved_but_rule_excluded": int(false_included.sum()),
                   "rule_included_but_no_patch": int(false_excluded.sum()),
                   "fractional_count_entries": fractional}
            summary.append(rec); rows.append(frame)
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            for ax in axes:
                ax.imshow(thumb); ax.set_axis_off()
            axes[0].scatter(xy[retained, 0] * thumb_scale, xy[retained, 1] * thumb_scale,
                            c="#008080", s=1, alpha=.3, label="Supplied patch")
            axes[0].scatter(xy[~retained, 0] * thumb_scale, xy[~retained, 1] * thumb_scale,
                            c="#e66101", s=3, alpha=.75, label="No patch")
            axes[0].legend(loc="upper right", fontsize=7)
            axes[0].set_title(f"{sid}: {len(pids):,}/{len(ids):,} locations retained")
            sc = axes[1].scatter(xy[:, 0] * thumb_scale, xy[:, 1] * thumb_scale,
                                 c=area_frac, cmap="viridis", vmin=0, vmax=1, s=2, alpha=.6)
            fig.colorbar(sc, ax=axes[1], shrink=.65, label="Patch fraction intersecting H&E mask")
            axes[1].set_title(f"Default mask-rule disagreements: {int(false_included.sum()+false_excluded.sum())}")
            fig.tight_layout(); pdf.savefig(fig)
            if sid in ["TENX193", "TENX95", "NCBI784", "NCBI785", "NCBI783", "NCBI776"]:
                fig.savefig(OUT / "figures" / f"{sid}_coverage.png", dpi=160)
            plt.close(fig)
            print(f"{sid}: coordinate delta={max_delta:g}; mask match={rec['mask_rule_exact_match']}; "
                  f"outside={rec['excluded_no_slide_overlap']}; low-mask={rec['excluded_below_mask_threshold']}; "
                  f"unexplained={int(false_excluded.sum())}", flush=True)
    pd.DataFrame(summary).to_csv(OUT / "coverage_summary.csv", index=False)
    pd.DataFrame(distributions).to_csv(OUT / "idc_coverage_distributions.csv", index=False)
    pd.concat(rows, ignore_index=True).to_parquet(OUT / "spot_coverage.parquet", index=False)
    (OUT / "coverage_checks.json").write_text(json.dumps({"completed_utc": datetime.now(timezone.utc).isoformat(),
              "n_sections": len(summary), "coordinate_exact_sections": sum(r["max_patch_coordinate_difference_px"] == 0 for r in summary),
              "mask_exact_sections": sum(r["mask_rule_exact_match"] for r in summary), "issues": issues}, indent=2) + "\n")


if __name__ == "__main__":
    main()

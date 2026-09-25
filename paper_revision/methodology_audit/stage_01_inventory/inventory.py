"""Read-only inventory of existing IDC inputs and corrected output membership."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT / "paper_revision/clean_repo"
OUT = Path(__file__).resolve().parent
inputs = {}
checks = []


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def record(path, checksum=False):
    path = Path(path)
    st = path.stat()
    item = {"bytes": st.st_size, "mtime_ns": st.st_mtime_ns}
    if checksum:
        item["sha256"] = sha(path)
    inputs[str(path)] = item
    return path


def read_json(path):
    return json.loads(record(path, True).read_text())


def check(name, passed, detail=None):
    checks.append({"check": name, "passed": bool(passed), "detail": detail})


def strings(values):
    return [v.decode() if isinstance(v, bytes) else str(v)
            for v in np.asarray(values).reshape(-1)]


def frame_index(group):
    key = group.attrs["_index"]
    if isinstance(key, bytes):
        key = key.decode()
    return strings(group[key][:])


def dump(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def main():
    config = yaml.safe_load(record(REPO / "config.yaml", True).read_text())
    hest = Path(config["hest_dir"])
    manifest = read_json(ROOT / "paper_revision/original_snapshot_manifest.json")
    preservation = []
    for folder, entries in manifest.items():
        for rel, expected in entries.items():
            path = ROOT / folder / rel
            match = path.is_file() and sha(path) == expected["sha256"]
            preservation.append({"path": str(path), "unchanged": match})
            check(f"original:{folder}/{rel}", match)
            if folder == "CIBM_submission":
                copy = ROOT / "paper_revision" / folder / rel
                match = copy.is_file() and sha(copy) == expected["sha256"]
                preservation.append({"path": str(copy), "unchanged": match})
                check(f"copied_submission:{rel}", match)
    dump("preservation.json", preservation)

    metadata = pd.read_csv(record(hest / config["hest_metadata"], True))
    fold_rows, section_rows, encoder_rows, gene_rows = [], [], [], []
    for cohort_key, cc in config["cohorts"].items():
        name = cc["name"]
        genes = read_json(REPO / "data/v3" / f"gene_list_{name}.json")
        check(f"{name}:unique_genes", len(genes) == len(set(genes)))
        sample_patient = {s: p for p, ss in cc["patient_mapping"].items() for s in ss}
        mapping = read_json(REPO / "data/v3" / f"patient_mapping_{name}.json")
        check(f"{name}:saved_mapping", mapping == cc["patient_mapping"])
        check(f"{name}:samples_in_mapping", set(sample_patient) == set(cc["samples"]))
        test_membership = {}
        test_visits = Counter()
        for f in range(cc["n_lopo_folds"]):
            split = read_json(REPO / "data/v3" / f"lopo_splits_{name}" / f"fold_{f}.json")
            test_visits.update(split["test_samples"])
            tr, te = set(split["train_samples"]), set(split["test_samples"])
            trp = {sample_patient[s] for s in tr}
            tep = {sample_patient[s] for s in te}
            okay = (not tr.intersection(te) and not trp.intersection(tep)
                    and tr | te == set(cc["samples"])
                    and tep == {split["test_patient"]}
                    and trp == set(split["train_patients"]))
            check(f"{name}:fold{f}:patient_partition", okay)
            fold_rows.append({"cohort": name, "fold": f,
                              "test_patient": split["test_patient"],
                              "n_train_sections": len(tr), "n_test_sections": len(te),
                              "patient_partition_pass": bool(okay)})
            encoder_ids = []
            for enc in config["encoders"]:
                encname = enc["name"]
                fp = REPO / "outputs/predictions" / name / encname / "ridge" / f"fold{f}"
                ids = read_json(fp / "test_spot_ids.json")
                check(f"{name}:fold{f}:{encname}:unique_ids", len(ids) == len(set(ids)))
                check(f"{name}:fold{f}:{encname}:only_test_sections",
                      all(any(s.startswith(sid + "_") for sid in te) for s in ids))
                for file in ["test_targets.npy", "test_predictions.npy", "test_residuals.npy"]:
                    arr = np.load(record(fp / file), mmap_mode="r")
                    check(f"{name}:fold{f}:{encname}:{file}:shape",
                          arr.shape == (len(ids), len(genes)), list(arr.shape))
                    del arr
                encoder_ids.append(ids)
                for sid in te:
                    test_membership[(sid, encname)] = [s[len(sid) + 1:]
                                                      for s in ids if s.startswith(sid + "_")]
            check(f"{name}:fold{f}:encoder_id_order", all(x == encoder_ids[0] for x in encoder_ids))
        check(f"{name}:each_section_tested_once",
              set(test_visits) == set(cc["samples"]) and all(v == 1 for v in test_visits.values()))

        real_panels = []
        for sid in cc["samples"]:
            with h5py.File(record(hest / "st" / f"{sid}.h5ad"), "r") as f:
                obs = frame_index(f["obs"])
                var = frame_index(f["var"])
                raw_set = set(obs)
                check(f"{sid}:unique_raw_ids", len(obs) == len(raw_set))
                check(f"{sid}:unique_raw_genes", len(var) == len(set(var)))
                available = set(var)
                real = {g for g in var if not any(g.startswith(p) for p in config["control_prefixes"])}
                real_panels.append(real)
                check(f"{sid}:all_modeled_genes_available", set(genes) <= available)
                in_tissue = f["obs/in_tissue"][:] if "in_tissue" in f["obs"] else None
                tissue_set = {b for b, t in zip(obs, in_tissue) if t == 1} if in_tissue is not None else None

            patch_path = hest / "patches" / f"{sid}.h5"
            with h5py.File(record(patch_path), "r") as f:
                barcode_key = "barcode" if "barcode" in f else "barcodes" if "barcodes" in f else None
                patch_ids = strings(f[barcode_key][:]) if barcode_key else []
                patch_shape = list(f["img"].shape)
                check(f"{sid}:patch_barcodes_available", barcode_key is not None)
                check(f"{sid}:patch_id_count", len(patch_ids) == patch_shape[0])
                check(f"{sid}:unique_patch_ids", len(patch_ids) == len(set(patch_ids)))
                check(f"{sid}:patches_subset_raw", set(patch_ids) <= raw_set)

            score_file = REPO / "outputs/phase2/scores" / name / f"{sid}_discordance.parquet"
            scores = pd.read_parquet(record(score_file))
            score_ids = strings(scores["barcode"])
            check(f"{sid}:unique_score_ids", len(score_ids) == len(set(score_ids)))
            check(f"{sid}:score_coords_finite", np.isfinite(scores[["x", "y"]].to_numpy()).all())
            check(f"{sid}:score_prefixed_ids", strings(scores["spot_id"]) == [sid + "_" + x for x in score_ids])
            for enc in config["encoders"]:
                encname = enc["name"]
                ep = REPO / "outputs/embeddings" / sid / f"{encname}_embeddings.h5"
                with h5py.File(record(ep), "r") as f:
                    ids = strings(f["spot_ids"][:])
                    arr = f["embeddings"]
                    check(f"{sid}:{encname}:embedding_shape",
                          arr.shape == (len(ids), enc["embed_dim"]), list(arr.shape))
                    finite = np.zeros(len(ids), dtype=bool)
                    nan_rows = 0
                    for start in range(0, len(ids), 1024):
                        batch = arr[start:start + 1024]
                        finite[start:start + len(batch)] = np.isfinite(batch).all(axis=1)
                        nan_rows += int(np.isnan(batch).any(axis=1).sum())
                check(f"{sid}:{encname}:unique_embedding_ids", len(ids) == len(set(ids)))
                check(f"{sid}:{encname}:patch_embedding_id_order", ids == patch_ids)
                test_ids = test_membership[(sid, encname)]
                expected = [s for s, fin in zip(ids, finite) if fin and s in raw_set]
                check(f"{sid}:{encname}:finite_alignment_reproduces_test_ids", expected == test_ids)
                check(f"{sid}:{encname}:score_test_set", set(score_ids) == set(test_ids))
                encoder_rows.append({"cohort": name, "sample": sid, "encoder": encname,
                                     "embedding_rows": len(ids), "nonfinite_embedding_rows": int((~finite).sum()),
                                     "nan_embedding_rows": nan_rows,
                                     "embedding_ids_absent_raw": len(set(ids) - raw_set),
                                     "test_spots": len(test_ids), "expected_test_spots": len(expected),
                                     "test_id_order_matches_alignment": expected == test_ids})

            m = metadata.loc[metadata["id"] == sid]
            check(f"{sid}:metadata_unique", len(m) == 1)
            md = m.iloc[0] if len(m) == 1 else {}
            section_rows.append({"cohort": name, "sample": sid, "patient": sample_patient[sid],
                                 "source_patient_label": md.get("patient", ""),
                                 "source_dataset": md.get("dataset_title", ""),
                                 "preservation": md.get("preservation_method", ""),
                                 "raw_spots": len(obs),
                                 "raw_in_tissue": len(tissue_set) if tissue_set is not None else None,
                                 "raw_features": len(var), "noncontrol_features": len(real),
                                 "modeled_genes": len(genes), "supplied_patches": len(patch_ids),
                                 "patch_shape": "x".join(map(str, patch_shape[1:])),
                                 "scored_spots": len(score_ids),
                                 "raw_spots_without_patch": len(raw_set - set(patch_ids)),
                                 "patch_spots_not_scored": len(set(patch_ids) - set(score_ids)),
                                 "in_tissue_without_patch": len(tissue_set - set(patch_ids)) if tissue_set is not None else None,
                                 "retained_fraction": len(score_ids) / len(obs)})
            print(f"{sid}: raw={len(obs)} patches={len(patch_ids)} scored={len(score_ids)}", flush=True)
        common = set.intersection(*real_panels)
        check(f"{name}:modeled_panel_equals_noncontrol_intersection", common == set(genes),
              {"extra_raw_genes": sorted(common - set(genes)), "extra_modeled_genes": sorted(set(genes) - common)})
        gene_rows.append({"cohort": name, "modeled_genes": len(genes),
                          "raw_noncontrol_intersection": len(common),
                          "raw_intersection_not_modeled": sorted(common - set(genes)),
                          "modeled_not_in_intersection": sorted(set(genes) - common)})

    for file, rows in [("section_inventory.csv", section_rows), ("encoder_inventory.csv", encoder_rows),
                       ("fold_inventory.csv", fold_rows)]:
        pd.DataFrame(rows).to_csv(OUT / file, index=False)
    dump("gene_panel_inventory.json", gene_rows)
    changed = [path for path, st in inputs.items()
               if Path(path).stat().st_size != st["bytes"] or Path(path).stat().st_mtime_ns != st["mtime_ns"]]
    check("inspected_inputs_size_mtime_unchanged_during_run", not changed, changed)
    dump("inputs.json", inputs)
    dump("checks.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
                         "status": "pass" if all(c["passed"] for c in checks) else "issues_found",
                         "n_checks": len(checks), "failed": [c for c in checks if not c["passed"]],
                         "checks": checks})
    frame = pd.DataFrame(section_rows)
    summary = frame.groupby("cohort")[["raw_spots", "supplied_patches", "scored_spots", "raw_spots_without_patch", "patch_spots_not_scored"]].sum()
    dump("cohort_totals.json", json.loads(summary.to_json(orient="index")))
    print(summary.to_string(), flush=True)
    print(json.dumps({"n_checks": len(checks), "failed": [c for c in checks if not c["passed"]]}, indent=2), flush=True)


if __name__ == "__main__":
    main()

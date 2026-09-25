"""Small offline reproduction of archived embeddings; never replaces them."""
import contextlib
import gc
import hashlib
import io
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT / "paper_revision/clean_repo"
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from src.embeddings import get_encoder


def main():
    cfg = yaml.safe_load((REPO / "config.yaml").read_text())
    images, locations = [], []
    for sid in cfg["all_samples"]:
        with h5py.File(ROOT / "data/hest/patches" / f"{sid}.h5", "r") as f:
            for i in [0, len(f["img"]) // 2, len(f["img"]) - 1]:
                images.append(f["img"][i])
                b = f["barcode"][i,0]
                locations.append((sid, i, b.decode() if isinstance(b,bytes) else str(b)))
    tensor = torch.from_numpy(np.stack(images).astype(np.float32) / 255).permute(0,3,1,2)
    rows, loading = [], {}
    for ec in cfg["encoders"]:
        name = ec["name"]
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            encoder = get_encoder(name, model_id=ec["model_id"], device="cuda:0", use_mixed_precision=False)
        log = capture.getvalue()
        loading[name] = log
        if "random initialization" in log.lower() or "could not load" in log.lower():
            (OUT / "encoder_loading.json").write_text(json.dumps(loading,indent=2)+"\n")
            raise RuntimeError(f"{name}: pretrained loading failed; no embedding validation certified")
        result = []
        for start in range(0,len(tensor),8):
            result.append(encoder.encode(tensor[start:start+8]))
        result = np.concatenate(result)
        for (sid, idx, barcode), new in zip(locations, result):
            with h5py.File(REPO / "outputs/embeddings" / sid / f"{name}_embeddings.h5", "r") as f:
                old = f["embeddings"][idx].astype(np.float64)
                b = f["spot_ids"][idx]
                assert (b.decode() if isinstance(b,bytes) else str(b)) == barcode
            new = new.astype(np.float64)
            relative = float(np.linalg.norm(new-old)/np.linalg.norm(old))
            rows.append({"sample":sid,"patch_index":idx,"barcode":barcode,"encoder":name,
                         "relative_l2_error":relative,"max_absolute_error":float(np.max(np.abs(new-old))),
                         "cosine_similarity":float(np.dot(old,new)/(np.linalg.norm(old)*np.linalg.norm(new))),
                         "within_tolerance":relative <= 1e-3})
        print(name, "max relative L2", max(r["relative_l2_error"] for r in rows if r["encoder"]==name), flush=True)
        del encoder,result
        gc.collect();torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(OUT / "embedding_reproduction.csv",index=False)
    (OUT / "encoder_loading.json").write_text(json.dumps(loading,indent=2)+"\n")
    (OUT / "embedding_verification.json").write_text(json.dumps({"n_vectors":len(rows),
          "status":"pass" if all(r["within_tolerance"] for r in rows) else "differences_require_review",
          "relative_l2_tolerance":1e-3,"max_relative_l2_error":max(r["relative_l2_error"] for r in rows),
          "torch_version":torch.__version__,"cuda_device":torch.cuda.get_device_name(0),
          "wrapper_sha256":hashlib.sha256((REPO/'src/embeddings.py').read_bytes()).hexdigest(),
          "scope":"54 fixed patches per encoder across all 18 IDC sections; existing cached pretrained models, no replacements"},indent=2)+"\n")


if __name__ == "__main__":
    main()

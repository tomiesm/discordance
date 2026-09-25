"""Download only missing public metadata/contours into the isolated audit."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import requests
import yaml
from huggingface_hub import get_token

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCE = OUT / "sources"


def fetch(task):
    rel, sha = task
    original = ROOT / "data/hest" / rel
    url = f"https://huggingface.co/datasets/MahmoodLab/hest/resolve/{sha}/{rel}"
    if original.exists():
        path = original
        origin = "existing_local"
    else:
        path = SOURCE / "hest" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        token = get_token()
        headers = {"Authorization": "Bearer " + token} if token else {}
        response = requests.get(url, headers=headers, timeout=60)
        if response.status_code != 200:
            return {"relative_path": rel, "status": response.status_code, "url": url}
        path.write_bytes(response.content)
        origin = "downloaded_pinned_current_revision"
    return {"relative_path": rel, "path": str(path), "origin": origin,
            "url": url if origin != "existing_local" else None,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "status": 200}


if __name__ == "__main__":
    cfg = yaml.safe_load((ROOT / "paper_revision/clean_repo/config.yaml").read_text())
    samples = cfg["all_samples"] + ["TENX111", "TENX147", "TENX148", "TENX149",
                "TENX13", "TENX14", "TENX39", "TENX53", "TENX68", "NCBI776",
                "NCBI681", "NCBI682", "NCBI683", "NCBI684"]
    sha = json.loads((SOURCE / "hest_dataset_info.json").read_text())["sha"]
    tasks = [(f"{folder}/{sid}{suffix}", sha) for sid in samples
             for folder, suffix in [("metadata", ".json"), ("tissue_seg", "_contours.geojson")]]
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch, tasks))
    manifest = {"retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "remote_revision": sha, "files": records}
    (SOURCE / "hest_source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"files": len(records), "downloaded": sum(r.get("origin", "").startswith("downloaded") for r in records),
                      "failed": [r for r in records if r["status"] != 200]}, indent=2))

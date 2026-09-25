#!/usr/bin/env python3
"""
Script 00: Download missing HEST v1.3.0 samples for v3

Downloads 11 Biomarkers IDC Xenium samples not present in v1.2.1.
Uses per-sample allow_patterns to target specific files.

Usage:
    python scripts/00_download.py
    python scripts/00_download.py --verify_only
    python scripts/00_download.py --cellvit   # Also download CellViT segmentation
"""

import os
import argparse
import sys
import time
from pathlib import Path

# CRITICAL: Disable xet download to avoid CAS IO errors
# Must be set BEFORE any huggingface_hub imports
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.utils import setup_logging, format_bytes, format_time


def load_v3_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def check_sample_status(hest_dir, sample_ids, logger):
    """Check which samples have st and patches files."""
    hest_dir = Path(hest_dir)
    status = {}

    for sid in sample_ids:
        st_exists = (hest_dir / "st" / f"{sid}.h5ad").exists()
        patches_exists = (hest_dir / "patches" / f"{sid}.h5").exists()
        status[sid] = {
            "st": st_exists,
            "patches": patches_exists,
            "complete": st_exists and patches_exists,
        }

    n_complete = sum(1 for s in status.values() if s["complete"])
    n_st_only = sum(1 for s in status.values() if s["st"] and not s["patches"])
    n_missing = sum(1 for s in status.values() if not s["st"] and not s["patches"])

    logger.info(f"Sample status: {n_complete}/{len(sample_ids)} complete, "
                f"{n_st_only} st-only, {n_missing} fully missing")

    return status


def download_missing_samples(config, logger, include_cellvit=False):
    """Download missing samples from HEST v1.3.0."""
    from huggingface_hub import snapshot_download

    hest_dir = Path(config["hest_dir"]).resolve()
    repo_id = config["hest_repo_id"]
    sample_ids = config["all_samples"]

    # Check current status
    status = check_sample_status(hest_dir, sample_ids, logger)
    missing = [sid for sid, s in status.items() if not s["complete"]]

    if not missing:
        logger.info("All samples already downloaded.")
        return True

    logger.info(f"\nNeed to download {len(missing)} samples: {missing}")

    # Build allow_patterns for missing samples
    allow_patterns = []
    for sid in missing:
        if not status[sid]["st"]:
            allow_patterns.append(f"st/{sid}.h5ad")
        if not status[sid]["patches"]:
            allow_patterns.append(f"patches/{sid}.h5")

    if include_cellvit:
        for sid in missing:
            allow_patterns.append(f"cellvit_seg/{sid}*")

    # Also ensure metadata is present
    allow_patterns.append("HEST_v1_3_0.csv")
    allow_patterns.append("*.csv")

    logger.info(f"\nDownloading {len(allow_patterns)} file patterns from {repo_id}")
    logger.info(f"Target directory: {hest_dir}")
    logger.info(f"HF_HUB_ENABLE_HF_TRANSFER = {os.environ.get('HF_HUB_ENABLE_HF_TRANSFER', 'not set')}")

    start_time = time.time()

    try:
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=allow_patterns,
            repo_type="dataset",
            local_dir=str(hest_dir),
            resume_download=True,
            max_workers=4,
        )
        elapsed = time.time() - start_time
        logger.info(f"\nDownload complete in {format_time(elapsed)}")
        return True

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"\nDownload failed after {format_time(elapsed)}: {e}")

        if "xet" in str(e).lower() or "cas" in str(e).lower():
            logger.error("CAS/xet error detected. Verify HF_HUB_ENABLE_HF_TRANSFER=0")
            logger.error("Re-run the script to resume from where it stopped.")

        raise


def verify_all_samples(config, logger):
    """Verify all 18 samples are present with correct file sizes."""
    hest_dir = Path(config["hest_dir"]).resolve()
    sample_ids = config["all_samples"]

    logger.info("\n" + "=" * 60)
    logger.info("Verification")
    logger.info("=" * 60)

    all_ok = True
    total_size = 0

    for sid in sample_ids:
        st_path = hest_dir / "st" / f"{sid}.h5ad"
        patches_path = hest_dir / "patches" / f"{sid}.h5"

        st_ok = st_path.exists() and st_path.stat().st_size > 0
        patches_ok = patches_path.exists() and patches_path.stat().st_size > 0

        if st_ok and patches_ok:
            st_size = st_path.stat().st_size
            patches_size = patches_path.stat().st_size
            total_size += st_size + patches_size
            logger.info(f"  {sid}: st={format_bytes(st_size)}, "
                        f"patches={format_bytes(patches_size)}")
        else:
            missing = []
            if not st_ok:
                missing.append("st")
            if not patches_ok:
                missing.append("patches")
            logger.error(f"  {sid}: MISSING {', '.join(missing)}")
            all_ok = False

    logger.info(f"\nTotal data size: {format_bytes(total_size)}")
    logger.info(f"Result: {'ALL OK' if all_ok else 'INCOMPLETE'} "
                f"({sum(1 for sid in sample_ids if (hest_dir / 'st' / f'{sid}.h5ad').exists())}"
                f"/{len(sample_ids)} samples)")

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Download missing HEST v1.3.0 samples for v3"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--verify_only", action="store_true")
    parser.add_argument("--cellvit", action="store_true",
                        help="Also download CellViT segmentation data")
    args = parser.parse_args()

    config = load_v3_config(args.config)

    # Setup logging
    log_dir = Path(config["output_dir"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(
        log_level=config.get("log_level", "INFO"),
        log_file=str(log_dir / "00_download.log"),
    )

    logger.info("=" * 60)
    logger.info("v3 Download: IDC Xenium samples from HEST v1.3.0")
    logger.info("=" * 60)
    logger.info(f"Samples: {len(config['all_samples'])}")
    n_patients = sum(len(c["patient_mapping"]) for c in config["cohorts"].values())
    logger.info(f"Patients: {n_patients}")

    if args.verify_only:
        verify_all_samples(config, logger)
    else:
        download_missing_samples(config, logger, include_cellvit=args.cellvit)
        verify_all_samples(config, logger)

    logger.info("\nDone.")


if __name__ == "__main__":
    main()

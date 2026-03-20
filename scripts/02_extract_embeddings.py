#!/usr/bin/env python3
"""
Script 02: Extract Patch Embeddings for v3

Extracts embeddings from histology patches for each sample using 3 encoders.
Distributes full jobs (sample × encoder) across GPUs — no DataParallel.

Output structure:
    outputs/embeddings/{sample_id}/{encoder}_embeddings.h5

Usage:
    python scripts/02_extract_embeddings.py
    python scripts/02_extract_embeddings.py --sample TENX99 --encoder uni
    python scripts/02_extract_embeddings.py --gpu 0 2 3
"""

import argparse
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import h5py
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.utils import (setup_logging, seed_everything, get_available_gpus,
                       format_time, checkpoint_exists)
from src.embeddings import get_encoder


def load_v3_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


class SamplePatchDataset(Dataset):
    """Load patches for a single sample from HEST HDF5 file."""

    def __init__(self, sample_id, hest_dir, transform=None):
        self.sample_id = sample_id
        self.patches_path = Path(hest_dir) / "patches" / f"{sample_id}.h5"
        self.transform = transform

        if not self.patches_path.exists():
            raise FileNotFoundError(f"Patches not found: {self.patches_path}")

        # Read spot IDs (keep file closed for multiprocessing compatibility)
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


def extract_for_sample(sample_id, encoder, encoder_config, config, gpu_id, logger):
    """Extract embeddings for a single sample using a pre-loaded encoder."""
    encoder_name = encoder_config["name"]
    hest_dir = Path(config["hest_dir"]).resolve()

    output_dir = Path(config["output_dir"]) / "embeddings" / sample_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{encoder_name}_embeddings.h5"

    # Skip if already done
    if checkpoint_exists(str(output_path)):
        logger.info(f"  [GPU {gpu_id}] [{sample_id}/{encoder_name}] Already exists, skipping.")
        return True

    logger.info(f"  [GPU {gpu_id}] [{sample_id}/{encoder_name}] Starting extraction...")
    start = time.time()

    try:
        # Create dataset
        dataset = SamplePatchDataset(sample_id, hest_dir)
        logger.info(f"    Spots: {dataset.n_spots}")

        # Single-GPU batch size
        batch_size = encoder_config.get("batch_size_per_gpu", 64)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=min(config.get("n_workers", 8) // max(config.get("n_gpus", 1), 1), 4),
            pin_memory=config.get("pin_memory", True),
            drop_last=False,
        )

        # Extract
        all_embeddings = []
        all_spot_ids = []

        for spot_ids_batch, images_batch in tqdm(
            dataloader, desc=f"    {sample_id}/{encoder_name}", leave=False
        ):
            embeddings_batch = encoder.encode(images_batch)
            all_embeddings.append(embeddings_batch)
            all_spot_ids.extend(spot_ids_batch)

        embeddings_matrix = np.vstack(all_embeddings)
        logger.info(f"    Shape: {embeddings_matrix.shape}")

        # Verify: no NaN
        n_nan = np.isnan(embeddings_matrix).sum()
        if n_nan > 0:
            logger.error(f"    NaN detected: {n_nan} values in {sample_id}/{encoder_name}")
            return False

        # Verify: spot count
        assert len(all_spot_ids) == dataset.n_spots, \
            f"Spot count mismatch: {len(all_spot_ids)} vs {dataset.n_spots}"

        # Save
        with h5py.File(output_path, "w") as f:
            f.create_dataset(
                "embeddings",
                data=embeddings_matrix,
                dtype="float32",
                compression="gzip",
                compression_opts=4,
            )
            f.create_dataset(
                "spot_ids",
                data=np.array(all_spot_ids, dtype="S"),
                dtype=h5py.string_dtype(),
            )
            f.attrs["encoder_name"] = encoder_name
            f.attrs["model_id"] = encoder_config.get("model_id", "unknown")
            f.attrs["embed_dim"] = encoder_config["embed_dim"]
            f.attrs["sample_id"] = sample_id
            f.attrs["n_spots"] = len(all_spot_ids)
            f.attrs["extraction_date"] = datetime.now().isoformat()

        elapsed = time.time() - start
        logger.info(f"    Done in {format_time(elapsed)} -> {output_path}")

        return True

    except Exception as e:
        logger.error(f"    FAILED [{sample_id}/{encoder_name}]: {e}", exc_info=True)
        return False


def gpu_worker(gpu_id, sample_ids, encoder_configs, config_path):
    """Worker process: process assigned samples on a single GPU.

    Each worker loads the encoder on its own GPU, processes all assigned
    samples for that encoder, then moves to the next encoder.
    """
    config = load_v3_config(config_path)

    # Per-GPU logger
    log_dir = Path(config["output_dir"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(
        log_level=config.get("log_level", "INFO"),
        log_file=str(log_dir / f"02_extract_embeddings_gpu{gpu_id}.log"),
    )

    seed_everything(config["seed"])

    device = f"cuda:{gpu_id}"
    logger.info(f"Worker started on GPU {gpu_id}, {len(sample_ids)} samples, "
                f"{len(encoder_configs)} encoders")

    completed = 0
    failed = 0

    for encoder_config in encoder_configs:
        encoder_name = encoder_config["name"]
        logger.info(f"\n  Loading encoder {encoder_name} on GPU {gpu_id}...")

        encoder = get_encoder(
            encoder_name,
            model_id=encoder_config.get("model_id"),
            device=device,
            use_mixed_precision=config.get("mixed_precision", False),
        )

        for sample_id in sample_ids:
            success = extract_for_sample(
                sample_id, encoder, encoder_config, config, gpu_id, logger
            )
            if success:
                completed += 1
            else:
                failed += 1

        # Free encoder before loading next one
        del encoder
        torch.cuda.empty_cache()

    logger.info(f"\nGPU {gpu_id} done: {completed} completed, {failed} failed")
    return failed


def main():
    parser = argparse.ArgumentParser(
        description="Extract patch embeddings for v3 samples"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--sample", type=str, default=None,
                        help="Process a specific sample only")
    parser.add_argument("--encoder", type=str, default=None,
                        help="Use a specific encoder only")
    parser.add_argument("--gpu", type=int, nargs="+", default=None,
                        help="GPU IDs to use")
    args = parser.parse_args()

    config = load_v3_config(args.config)

    log_dir = Path(config["output_dir"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(
        log_level=config.get("log_level", "INFO"),
        log_file=str(log_dir / "02_extract_embeddings.log"),
    )

    seed_everything(config["seed"])

    # GPU setup
    if args.gpu is not None:
        gpu_ids = args.gpu
    else:
        gpu_ids = get_available_gpus()

    if not gpu_ids:
        logger.warning("No GPUs available, using CPU.")
        gpu_ids = []
    else:
        logger.info(f"Using GPUs: {gpu_ids}")

    # Determine samples and encoders
    samples = [args.sample] if args.sample else config["all_samples"]
    encoders = (
        [e for e in config["encoders"] if e["name"] == args.encoder]
        if args.encoder
        else config["encoders"]
    )

    total = len(samples) * len(encoders)
    logger.info("=" * 60)
    logger.info("v3 Embedding Extraction (per-GPU job distribution)")
    logger.info("=" * 60)
    logger.info(f"Samples: {len(samples)}")
    logger.info(f"Encoders: {[e['name'] for e in encoders]}")
    logger.info(f"GPUs: {gpu_ids}")
    logger.info(f"Total jobs: {total}")

    overall_start = time.time()

    if len(gpu_ids) <= 1:
        # Single GPU or CPU: run sequentially
        gpu_id = gpu_ids[0] if gpu_ids else None
        device = f"cuda:{gpu_id}" if gpu_id is not None else "cpu"
        completed = 0
        failed = 0

        for encoder_config in encoders:
            encoder_name = encoder_config["name"]
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Encoder: {encoder_name}")
            logger.info(f"{'=' * 60}")

            encoder = get_encoder(
                encoder_name,
                model_id=encoder_config.get("model_id"),
                device=device,
                use_mixed_precision=config.get("mixed_precision", False),
            )

            for sample_id in samples:
                success = extract_for_sample(
                    sample_id, encoder, encoder_config, config,
                    gpu_id if gpu_id is not None else "cpu", logger
                )
                if success:
                    completed += 1
                else:
                    failed += 1
                logger.info(f"  Progress: {completed + failed}/{total}")

            del encoder
            torch.cuda.empty_cache()

    else:
        # Multi-GPU: distribute samples across GPUs
        # Assign samples round-robin
        gpu_sample_map = {gpu: [] for gpu in gpu_ids}
        for i, sample_id in enumerate(samples):
            gpu = gpu_ids[i % len(gpu_ids)]
            gpu_sample_map[gpu].append(sample_id)

        for gpu, assigned in gpu_sample_map.items():
            logger.info(f"  GPU {gpu}: {len(assigned)} samples: {assigned}")

        # Spawn worker processes
        mp.set_start_method("spawn", force=True)
        processes = []
        for gpu_id, assigned_samples in gpu_sample_map.items():
            if not assigned_samples:
                continue
            p = mp.Process(
                target=gpu_worker,
                args=(gpu_id, assigned_samples, encoders, args.config),
            )
            p.start()
            processes.append((gpu_id, p))

        # Wait for all workers
        any_failed = False
        for gpu_id, p in processes:
            p.join()
            if p.exitcode != 0:
                logger.error(f"GPU {gpu_id} worker exited with code {p.exitcode}")
                any_failed = True

        # Count results from checkpoint files
        completed = sum(
            1 for s in samples for e in encoders
            if (Path(config["output_dir"]) / "embeddings" / s / f"{e['name']}_embeddings.h5").exists()
        )
        failed = total - completed

    total_time = time.time() - overall_start

    logger.info(f"\n{'=' * 60}")
    logger.info("Summary")
    logger.info(f"{'=' * 60}")
    logger.info(f"Completed: {completed}/{total}")
    logger.info(f"Failed: {failed}/{total}")
    logger.info(f"Total time: {format_time(total_time)}")

    if failed > 0:
        logger.error(f"{failed} jobs failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()

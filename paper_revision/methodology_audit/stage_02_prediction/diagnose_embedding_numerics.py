"""Preserved, bounded follow-up to the failed numerical tolerance check."""
import contextlib
import gc
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
REPO = ROOT / 'paper_revision/clean_repo'
OUT = Path(__file__).resolve().parent / 'embedding_numerics'
sys.path.insert(0, str(REPO))
from src.embeddings import get_encoder
from src.utils import seed_everything


def main():
    OUT.mkdir(exist_ok=True)
    cfg = yaml.safe_load((REPO / 'config.yaml').read_text())
    seed_everything(cfg['seed'])
    images, locations = [], []
    for sid in cfg['all_samples']:
        with h5py.File(ROOT / 'data/hest/patches' / f'{sid}.h5', 'r') as f:
            for i in [0, len(f['img']) // 2, len(f['img']) - 1]:
                images.append(f['img'][i])
                b = f['barcode'][i, 0]
                locations.append({'sample':sid, 'patch_index':i, 'barcode':b.decode() if isinstance(b, bytes) else str(b)})
    pd.DataFrame(locations).to_csv(OUT / 'locations.csv', index=False)
    x = torch.from_numpy(np.stack(images).astype(np.float32) / 255.0).permute(0,3,1,2)
    rows, logs = [], {}
    for ec in cfg['encoders']:
        name = ec['name']
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            encoder = get_encoder(name, model_id=ec['model_id'], device='cuda:2', use_mixed_precision=False)
        logs[name] = capture.getvalue()
        assert 'random initialization' not in logs[name].lower() and 'could not load' not in logs[name].lower()
        new = np.concatenate([encoder.encode(x[start:start+8]) for start in range(0,len(x),8)])
        old = []
        for loc in locations:
            with h5py.File(REPO / 'outputs/embeddings' / loc['sample'] / f'{name}_embeddings.h5', 'r') as f:
                old.append(f['embeddings'][loc['patch_index']])
        old = np.stack(old)
        np.savez_compressed(OUT / f'{name}_vectors.npz', archived=old, deterministic_batch8=new)
        full_indices, full_values = [], []
        for k, loc in enumerate(locations):
            a, b = old[k].astype(float), new[k].astype(float)
            rows.append({**loc, 'encoder':name, 'mode':'deterministic_batch8',
                         'relative_l2_error':np.linalg.norm(a-b)/np.linalg.norm(a),
                         'max_absolute_error':np.max(np.abs(a-b)),
                         'cosine_similarity':np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))})
            if loc['sample'] not in ['NCBI784', 'TENX193']:
                continue
            batch = ec['batch_size_per_gpu']
            start = loc['patch_index'] // batch * batch
            with h5py.File(ROOT / 'data/hest/patches' / f"{loc['sample']}.h5", 'r') as f:
                im = f['img'][start:min(start+batch, len(f['img']))]
            tx = torch.from_numpy(im.astype(np.float32)/255.0).permute(0,3,1,2)
            b = encoder.encode(tx)[loc['patch_index']-start]
            full_indices.append(k); full_values.append(b)
            b = b.astype(float)
            rows.append({**loc, 'encoder':name, 'mode':'original_contiguous_batch',
                         'relative_l2_error':np.linalg.norm(a-b)/np.linalg.norm(a),
                         'max_absolute_error':np.max(np.abs(a-b)),
                         'cosine_similarity':np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))})
        np.savez_compressed(OUT / f'{name}_original_batch_vectors.npz', indices=full_indices, recomputed=np.stack(full_values))
        pd.DataFrame(rows).to_csv(OUT / 'comparisons.csv', index=False)
        print(name, pd.DataFrame(rows).query('encoder == @name').groupby('mode').relative_l2_error.max().to_dict(), flush=True)
        del encoder, new, old, tx
        gc.collect(); torch.cuda.empty_cache()
    (OUT / 'settings.json').write_text(json.dumps({'torch_version':torch.__version__,
        'device':torch.cuda.get_device_name(2),'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
        'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark,
        'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
        'loading_logs':logs},indent=2)+'\n')


if __name__ == '__main__':
    main()

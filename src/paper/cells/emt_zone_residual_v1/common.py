from src.paper.paths import PROJECT_ROOT, STAGE_SOURCE, CELL_SOURCE, ANALYSIS_ROOT, CELL_ROOT, stage_dir, cell_dir
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '4'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from scipy import sparse
HERE = cell_dir('emt_zone_residual_v1')
PROJECT = PROJECT_ROOT
CELLS = HERE.parent / 'emt_cells_v1/results'
SPATIAL = HERE.parent / 'emt_spatial_enrichment_v1/results'
OUT = HERE / 'results'
OUT.mkdir(exist_ok=True)
PROTOCOL = json.loads((CELL_SOURCE / 'emt_zone_residual_v1' / 'protocol.json').read_text())
OLD_PROTOCOL = json.loads((CELL_SOURCE / 'emt_cells_v1/protocol.json').read_text())
TF = ['SNAI1', 'ZEB1', 'ZEB2']
EPI = OLD_PROTOCOL['epithelial_genes']
for line in (PROJECT_ROOT / 'data/gene_sets/h.all.v2024.1.Hs.symbols.gmt').read_text().splitlines():
    fields = line.split('\t')
    if fields[0] == 'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION':
        HALLMARK = fields[2:]

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def grid(a):
    xy = np.asarray(a.obsm['spatial'], float)
    step = float(np.median(np.diff(np.unique(xy[:, 0]))))
    origin = xy.min(axis=0) - step / 2
    ij = np.rint((xy - origin) / step - 0.5).astype(int)
    shape = ij.max(axis=0) + 1
    lookup = np.full(tuple(shape), -1, int)
    lookup[ij[:, 0], ij[:, 1]] = np.arange(len(xy))
    assert (lookup >= 0).all()
    np.testing.assert_allclose(origin + (ij + 0.5) * step, xy, atol=1e-07)
    return (origin, step, lookup)

def assign_grid(xy, origin, step, lookup):
    xy = np.asarray(xy)
    finite = np.isfinite(xy).all(axis=1)
    scaled = np.zeros_like(xy)
    scaled[finite] = (xy[finite] - origin) / step
    scaled[(scaled < 0) & (scaled > -1e-10)] = 0
    ij = np.floor(scaled).astype(int)
    valid = finite & (ij >= 0).all(axis=1) & (ij < np.array(lookup.shape)).all(axis=1)
    out = np.full(len(ij), -1, int)
    out[valid] = lookup[ij[valid, 0], ij[valid, 1]]
    return out

def biological(genes):
    return np.array([not any((g.startswith(p) for p in OLD_PROTOCOL['control_prefixes'])) for g in genes])

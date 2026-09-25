from src.paper.paths import PROJECT_ROOT, STAGE_SOURCE, CELL_SOURCE, ANALYSIS_ROOT, CELL_ROOT, stage_dir, cell_dir
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
OUT = stage_dir('contrasts')
AUDIT = ANALYSIS_ROOT
ROOT = PROJECT_ROOT
REPO = PROJECT_ROOT
FAMILIES = ['biomarkers', '10x_janesick', 'coad', 'idc_visium']
IDC = FAMILIES[:2]
EMT = 'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION'
OUTCOMES = ['observed', 'signed', 'absolute']
SOURCES = {}
CHECKS = []

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def track(path):
    path = Path(path).resolve()
    if str(path) not in SOURCES:
        SOURCES[str(path)] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    return path

def csv(path):
    return pd.read_csv(track(path))

def parquet(path):
    return pd.read_parquet(track(path))

def check(name, error=0.0, tolerance=1e-09):
    passed = bool(np.isfinite(error) and error <= tolerance)
    CHECKS.append(dict(check=name, max_abs=float(error), tolerance=tolerance, passed=passed))
    assert passed, (name, error, tolerance)

def finish(name):
    (OUT / f'{name}_checks.json').write_text(json.dumps(dict(status='pass', checks=CHECKS), indent=2) + '\n')
    (OUT / f'{name}_sources.json').write_text(json.dumps(SOURCES, indent=2) + '\n')

def tails(score):
    a, b = np.quantile(score, [0.25, 0.75])
    assert a < b
    return (score <= a, score >= b)

def strata(count, det):
    q = np.linspace(0, 1, 6)[1:-1]
    return 5 * np.searchsorted(np.quantile(count, q), count, side='right') + np.searchsorted(np.quantile(det, q), det, side='right')

def weights(s, lo, hi):
    a = np.bincount(s[lo], minlength=25)
    b = np.bincount(s[hi], minlength=25)
    ok = (a >= 10) & (b >= 10)
    h = np.divide(2 * a * b, a + b, out=np.zeros(25), where=a + b > 0) * ok
    w1 = np.divide(h, a, out=np.zeros(25), where=a > 0)[s] * lo
    w4 = np.divide(h, b, out=np.zeros(25), where=b > 0)[s] * hi
    return (w1, w4, ok)

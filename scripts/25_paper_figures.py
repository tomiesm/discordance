#!/usr/bin/env python3
"""Draw current result panels from the generated paper tables."""
from pathlib import Path
import os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
for module in ['main_figures','supplement_figures','presentation']:
    subprocess.run([sys.executable,'-B','-m','src.paper.figures.'+module],cwd=ROOT,env={**os.environ,'MPLBACKEND':'Agg'},check=True)

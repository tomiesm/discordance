"""Locations of source, downloaded inputs and generated analysis outputs."""
from pathlib import Path
import os

PROJECT_ROOT = Path(os.environ.get("DISCORDANCE_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
STAGE_SOURCE = Path(__file__).resolve().parent
CELL_SOURCE = STAGE_SOURCE / "cells"
ANALYSIS_ROOT = Path(os.environ.get("DISCORDANCE_ANALYSIS_ROOT", PROJECT_ROOT / "outputs/paper"))
CELL_ROOT = Path(os.environ.get("DISCORDANCE_CELL_ROOT", PROJECT_ROOT / "outputs/cells"))

def stage_dir(name):
    path = ANALYSIS_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path

def cell_dir(name):
    path = CELL_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "results").mkdir(exist_ok=True)
    return path

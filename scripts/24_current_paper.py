#!/usr/bin/env python3
"""Run the computations retained in the current paper, in dependency order."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
def module(name,*args,model=False):return (model,['-m','src.paper.'+name,*args])
def script(name):return (True,['scripts/'+name])
STEPS={
 'download':[script('00_download.py'),module('download_inputs')],
 'models':[script(s) for s in ['01_qc_and_splits.py','02_extract_embeddings.py','03_train_predict.py','04_discordance_scores.py','05_multimodel_agreement.py','06_spatial_structure.py','07_dual_track.py','10_deconvolution.py','11_gene_predictability.py','20_coad_generalization.py','21_idc_visium_generalization.py']],
 'analysis':[
  module('coverage.fetch_sources'),module('coverage.patch_coverage_audit'),module('inventory.inventory'),
  module('prediction.audit_existing_predictions'),module('scores.audit_score'),module('reliability.audit_reliability'),
  module('programs.build_atlas'),module('matching.audit_matching'),module('matching.overlap_bootstrap'),module('genes.gene_audit'),
  module('external.refit_visium',model=True),module('external.analyze_external','--family','coad'),module('external.analyze_external','--family','idc_visium'),
  module('external.idc_comparators'),module('external.compare_replication'),
  *[module('cutoffs.tails','--family',f) for f in ['biomarkers','10x_janesick','coad','idc_visium']],
  module('features.audit'),module('boundary.analyze'),module('boundary.rings')],
 'cells':[module('cells.emt_cells_v1.fetch_vendor'),module('cells.emt_cells_v1.prepare_cells'),module('cells.emt_cells_v1.analyze_cells'),module('cells.emt_zone_residual_v1.prepare'),module('cells.emt_spatial_enrichment_v1.spatial_test')],
 'final':[module('composition.composition'),module('contrasts.boundary_biology'),module('contrasts.patient_profiles'),module('contrasts.tumor_rich_emt'),module('donors.correct'),module('export_tables')]
}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['all',*STEPS],default='all')
    parser.add_argument('--model-python',default=sys.executable,help='Python from the model environment (scikit-learn 1.4.0).')
    parser.add_argument('--list',action='store_true',help='Print commands without downloading, training or calculating.')
    args=parser.parse_args()
    steps=[step for phase,items in STEPS.items() if args.phase in ['all',phase] for step in items]
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','MPLBACKEND':'Agg','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
    for uses_model,arguments in steps:
        command=[args.model_python if uses_model else sys.executable,'-B',*arguments]
        print(shlex.join(command),flush=True)
        if not args.list:subprocess.run(command,cwd=ROOT,env=env,check=True)

if __name__=='__main__':main()

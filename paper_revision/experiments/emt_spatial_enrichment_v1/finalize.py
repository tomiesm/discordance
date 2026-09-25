"""Check deliverables and archive final source/output hashes."""
from spatial_test import *
from importlib.metadata import version
import platform
import re
import fitz


def main():
    for name in ['ANALYSIS_COMPLETE.json','VERIFICATION.json','EXTENT_COMPLETE.json','EXTENT_VERIFICATION.json']:
        result=json.loads((OUT/name).read_text())
        assert result['status'] in ['complete','pass']
    assert (HERE/'logs/unit_tests.log').read_text().strip().endswith('OK')
    for name,expected in [('spatial_enrichment_overview.pdf',1),('scan_adjusted_neighborhoods.pdf',3),('component_extent_maps.pdf',3),('scale_and_nuclear_sensitivity.pdf',1)]:
        with fitz.open(HERE/'figures'/name) as doc:assert len(doc)==expected
    for filename in ['RESULTS.md','README.md']:
        for target in re.findall(r'\]\(([^)]+)\)',(HERE/filename).read_text()):
            if not target.startswith(('http:','https:','#')):assert (HERE/target).exists(),target
    dump(HERE/'environment.json',{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
        'packages':{n:version(n) for n in ['numpy','scipy','pandas','anndata','h5py','pyarrow','statsmodels','matplotlib','PyMuPDF']}})
    final=HERE/'final_source';final.mkdir(exist_ok=True)
    for pattern in ['*.py','*.md','*.json']:
        for file in HERE.glob(pattern):
            if file.name not in ['EXPERIMENT_COMPLETE.json','output_manifest.json']:
                shutil.copy2(file,final/file.name)
    shutil.copytree(HERE/'tests',final/'tests',dirs_exist_ok=True)
    dump(final/'manifest.json',{str(p.relative_to(final)):sha(p) for p in sorted(final.rglob('*')) if p.is_file() and p.name!='manifest.json'})
    paths=[p for directory in [OUT,HERE/'figures'] for p in directory.rglob('*') if p.is_file()]
    dump(HERE/'output_manifest.json',{str(p.relative_to(HERE)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(paths)})
    verification=json.loads((OUT/'VERIFICATION.json').read_text())
    dump(HERE/'EXPERIMENT_COMPLETE.json',{'status':'complete','utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Fixed-phenotype spatial random-labeling comparison with detection/identity/size strata and composition/nuclear sensitivities; exploratory region-extent addendum',
        'n_sections':3,'n_patients':2,'initial_configurations':30,'extent_configurations_reusing_same_draws':9,
        'permutations_per_configuration':1999,'report':'RESULTS.md',
        'primary_spatial_clustering_positive_sections':['NCBI785','NCBI784','NCBI783'],
        'primary_holm_p_each':.0015,
        'exploratory_technical_extent_supported_components':{'NCBI784':[1,2]},
        'interpretation':'Association with measured EMT-related coexpression; unchanged negative aggregate-discordance result',
        'original_files_unchanged':verification['original_files_unchanged'],
        'prior_cell_experiment_outputs_unchanged':verification['previous_experiment_outputs_unchanged'],
        'source_manifest_sha256':sha(final/'manifest.json'),'output_manifest_sha256':sha(HERE/'output_manifest.json')})
    print('EXPERIMENT COMPLETE',flush=True)


if __name__=='__main__':main()

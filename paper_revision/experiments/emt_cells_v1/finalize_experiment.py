"""Record the completed bounded experiment, source and output provenance."""
from common import *
from datetime import datetime, timezone
from importlib.metadata import version
import platform
import re
import shutil
import fitz


def main():
    check = json.loads((RESULTS/'VERIFICATION.json').read_text())
    assert check['status'] == 'pass' and len(check['sections']) == 3
    assert json.loads((RESULTS/'VISIUM_COMPATIBILITY.json').read_text())['exact_source_count_match']
    assert 'Ran 6 tests' in (HERE/'logs/unit_tests.log').read_text()
    assert (HERE/'logs/unit_tests.log').read_text().strip().endswith('OK')
    pdf_counts = {}
    for name, count in [('all_section_evidence.pdf', 9), ('all_candidate_transcript_evidence.pdf', 50), ('overview.pdf', 1)]:
        with fitz.open(HERE/'figures'/name) as document:
            assert len(document) == count
            pdf_counts[name] = len(document)
    for filename in ['RESULTS.md', 'README.md']:
        for target in re.findall(r'\]\(([^)]+)\)', (HERE/filename).read_text()):
            if not target.startswith(('http:', 'https:', '#')):
                assert (HERE/target).exists(), (filename, target)
    dump(HERE/'run_environment.json', {'python': sys.version, 'python_executable': sys.executable,
        'platform': platform.platform(), 'packages': {name: version(name) for name in
        ['numpy', 'scipy', 'pandas', 'scikit-learn', 'anndata', 'h5py', 'pyarrow', 'matplotlib', 'joblib', 'Pillow', 'PyMuPDF']},
        'threads': 4, 'gpu_training': False})
    # Additional read-only inputs not covered by the original raw-input manifest.
    inputs = [PROJECT/'clean_repo/src/discordance.py',
              PROJECT/'clean_repo/data/gene_sets/h.all.v2024.1.Hs.symbols.gmt',
              PROJECT/'data/v3/gene_list_10x_janesick.json', PROJECT/'data/hest/st/NCBI776.h5ad']
    for sample in PROTOCOL['samples']:
        inputs.extend([PROJECT/f'data/hest/metadata/{sample}.json',
                       PROJECT/f'data/hest/thumbnails/{sample}_downscaled_fullres.jpeg',
                       PROJECT/f'paper_revision/clean_repo/outputs/phase2/scores/10x_janesick/{sample}_discordance.parquet'])
    for encoder in ['uni', 'virchow2', 'hoptimus0']:
        for fold in range(4):
            base = PROJECT/f'paper_revision/clean_repo/outputs/predictions/10x_janesick/{encoder}/ridge/fold{fold}'
            inputs.extend(base/name for name in ['test_targets.npy', 'test_predictions.npy', 'test_spot_ids.json'])
    dump(HERE/'additional_input_manifest.json', {'recorded_at_completion': True,
        'note': 'Complements input_manifest.json and source download manifests; this is final provenance, not a pre-run immutability check for these additional inputs.',
        'files': {str(p): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in inputs}})
    final = HERE/'final_source'
    final.mkdir(exist_ok=True)
    for pattern in ['*.py', '*.md', '*.json']:
        for path in HERE.glob(pattern):
            if path.name not in ['EXPERIMENT_COMPLETE.json', 'output_manifest.json']:
                shutil.copy2(path, final/path.name)
    shutil.copytree(HERE/'tests', final/'tests', dirs_exist_ok=True)
    source_manifest = {str(p.relative_to(final)): sha(p) for p in sorted(final.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
    dump(final/'manifest.json', source_manifest)
    paths = [p for directory in [RESULTS, HERE/'figures'] for p in directory.rglob('*') if p.is_file()]
    dump(HERE/'output_manifest.json', {str(p.relative_to(HERE)): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(paths)})
    dump(HERE/'EXPERIMENT_COMPLETE.json', {'status': 'complete', 'utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Three-section/two-patient cell-resolved feasibility, measured candidate neighborhoods, external reference observability, secondary corrected residual associations, broader Visium source compatibility',
        'report': 'RESULTS.md', 'verification': 'results/VERIFICATION.json', 'figure_pages': pdf_counts,
        'original_files_unchanged': check['unchanged_original_files'],
        'validated_emt_zones': False, 'candidate_cells': 2753, 'descriptive_100um_components': 50,
        'broader_visium_tumor_state_modeling_completed': False,
        'paper_edited': False, 'original_code_edited': False,
        'source_snapshot_manifest_sha256': sha(final/'manifest.json'), 'output_manifest_sha256': sha(HERE/'output_manifest.json')})
    print('EXPERIMENT COMPLETE; outputs and source hashed', flush=True)


if __name__ == '__main__':
    main()

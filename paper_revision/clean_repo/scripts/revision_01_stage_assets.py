#!/usr/bin/env python3
"""Stage corrected figures/tables under original names, away from manuscript."""

import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO.parent / 'CIBM_submission'
OUT = REPO / 'outputs'
DEST = OUT / 'revision_01/submission_assets'


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    mappings = {}
    for tex in PAPER.glob('*.tex'):
        for name in re.findall(r'\{([^{}\n]+\.pdf)\}', tex.read_text()):
            if not (PAPER / name).exists():
                continue
            filename = 'fig1_schematic.pdf' if name == 'figures/main_figure.pdf' else Path(name).name
            sources = list((OUT / 'figures').rglob(filename))
            assert len(sources) == 1, (name, sources)
            mappings[name] = sources[0]
    tables = OUT / 'figures/tables'
    renamed = {'table2_ols_coefficients.csv': 'table4_ols_coefficients.csv',
               'supp_table_s7_subsampling.csv': 'supp_table_s8_subsampling.csv',
               'supp_table_s8_gate_results.csv': 'table2_gate_results.csv',
               'supp_table_s9_pathway_replication.csv': 'table3_pathway_replication.csv'}
    coad = {'supp_table_s10_coad_gene_features.csv': 'gene_features.csv',
            'supp_table_s11_coad_pathway_de.csv': 'pathway_de.csv',
            'supp_table_s12_coad_pathway_consistency.csv': 'pathway_consistency.csv',
            'supp_table_s13_coad_cross_tissue_predictability.csv': 'cross_tissue_gene_predictability.csv'}
    for old in (PAPER / 'Tables').glob('*.csv'):
        source = OUT / 'coad' / coad[old.name] if old.name in coad else tables / renamed.get(old.name, old.name)
        assert source.exists(), source
        mappings[f'Tables/{old.name}'] = source
    manifest = []
    for name, source in sorted(mappings.items()):
        destination = DEST / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        manifest.append({'submission_relative_path': name, 'corrected_source': str(source.relative_to(REPO)),
                         'sha256': hashlib.sha256(destination.read_bytes()).hexdigest()})
    with zipfile.ZipFile(DEST / 'Tables/tables.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((DEST / 'Tables').glob('*.csv')):
            archive.write(path, path.name)
    (DEST / 'asset_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (DEST / 'README.md').write_text(
        '# Corrected computation assets — staging only\n\n'
        'These files use the submitted figure/table paths for later integration. '
        'The manuscript and supplementary TeX have not been edited. Existing captions '
        'and biological interpretation must be revised before these assets are used '
        'in a submission. Other review points remain deferred.\n')
    print(f'Staged {len(manifest)} corrected files in {DEST}')


if __name__ == '__main__':
    main()

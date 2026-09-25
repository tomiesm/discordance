"""Download the additional public HEST inputs used in the paper."""
from .paths import PROJECT_ROOT
from huggingface_hub import snapshot_download
import yaml

SAMPLES = ['TENX111','TENX147','TENX148','TENX149','TENX13','TENX14','TENX39','TENX53','TENX68','NCBI776','NCBI681','NCBI682','NCBI683','NCBI684']

def main():
    config=yaml.safe_load((PROJECT_ROOT/'config.yaml').read_text())
    patterns=[f'{folder}/{sample}{suffix}' for sample in SAMPLES for folder,suffix in [('st','.h5ad'),('patches','.h5')]]
    patterns += [f'transcripts/{sample}_transcripts.parquet' for sample in ['NCBI785','NCBI784','NCBI783']]
    snapshot_download(repo_id=config['hest_repo_id'],repo_type='dataset',local_dir=str(PROJECT_ROOT/'data/hest'),revision='7e8d5a0b0aace41d8c8ec0f6ecea80e4ad2a61ec',allow_patterns=patterns)

if __name__ == '__main__':
    main()

"""Retrieve specific public GEO ZIP members using standard HTTP range reads."""
import io
import json
import hashlib
from pathlib import Path
import requests
import zipfile

HERE = Path(__file__).resolve().parent
ARCHIVES = {
    'NCBI785': ('GSM7780153', 'GSM7780153_Xenium_FFPE_Human_Breast_Cancer_Rep1_outs.zip'),
    'NCBI784': ('GSM7780154', 'GSM7780154_Xenium_FFPE_Human_Breast_Cancer_Rep2_outs.zip'),
    'NCBI783': ('GSM7780155', 'GSM7780155_Xenium_V1_FFPE_Preview_Human_Breast_Cancer_Sample_2_outs.zip')}


class RangeReader(io.RawIOBase):
    def __init__(self, url):
        self.url, self.position = url, 0
        self.session = requests.Session()
        r = self.session.get(url, headers={'Range': 'bytes=0-0'}, timeout=(30, 60))
        assert r.status_code == 206
        self.size = int(r.headers['Content-Range'].split('/')[-1])
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position
    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        return self.position
    def read(self, n=-1):
        n = self.size-self.position if n < 0 else min(n, self.size-self.position)
        if n <= 0: return b''
        start, end = self.position, self.position+n-1
        r = self.session.get(self.url, headers={'Range': f'bytes={start}-{end}'}, timeout=(30, 120))
        r.raise_for_status()
        assert r.status_code == 206 and len(r.content) == n, (r.status_code, len(r.content), n)
        self.position += n
        return r.content


def main():
    for sample, (gsm, filename) in ARCHIVES.items():
        out = HERE / 'sources/vendor' / sample
        out.mkdir(parents=True, exist_ok=True)
        url = f'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM7780nnn/{gsm}/suppl/{filename}'
        manifest = []
        with zipfile.ZipFile(RangeReader(url)) as archive:
            (out / 'archive_members.json').write_text(json.dumps([dict(name=f.filename, bytes=f.file_size, compressed=f.compress_size) for f in archive.infolist()], indent=2)+'\n')
            for member in archive.infolist():
                basename = Path(member.filename).name
                selected = basename in ['cell_feature_matrix.h5', 'cells.parquet', 'cells.csv.gz',
                                        'cell_boundaries.parquet', 'nucleus_boundaries.parquet',
                                        'analysis_summary.html', 'experiment.xenium']
                if not selected: continue
                path = out / basename
                print('FETCH', sample, member.filename, member.file_size, flush=True)
                if not path.exists():
                    data = archive.read(member)  # ZIP CRC checked by zipfile.
                    path.with_suffix(path.suffix+'.partial').write_bytes(data)
                    path.with_suffix(path.suffix+'.partial').rename(path)
                assert path.stat().st_size == member.file_size
                manifest.append(dict(url=url, member=member.filename, bytes=member.file_size,
                                     crc32=member.CRC, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                print('SAVED', sample, basename, flush=True)
        (out / 'download_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__': main()

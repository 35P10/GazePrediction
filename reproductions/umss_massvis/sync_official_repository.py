"""Download or verify the complete pinned UMSS DaRUS V2 deposit."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import time
from urllib.request import urlopen

EXPERIMENT = Path(__file__).resolve().parent
MANIFEST = EXPERIMENT / 'official_repository_manifest.json'
DESTINATION = EXPERIMENT / 'official_repository'
URL = 'https://darus.uni-stuttgart.de/api/access/datafile/{}'


def matches(path, entry):
    if not path.is_file() or path.stat().st_size != entry['bytes']:
        return False
    digest = hashlib.md5()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest() == entry['md5']


def sync(*, verify_only=False, workers=4):
    manifest = json.loads(MANIFEST.read_text())
    entries = manifest['files']

    def process(entry):
        relative = Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe repository path: ' + str(relative))
        path = DESTINATION / relative
        if matches(path, entry):
            return False
        if verify_only:
            raise ValueError('Missing or modified official file: ' + str(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.part')
        for attempt in range(3):
            try:
                # DaRUS MD5 checksums describe the uploaded original, whose
                # byte size can differ from its generated .tab representation.
                url = URL.format(entry['id'])
                if entry['download_format'] == 'original':
                    url += '?format=original'
                with urlopen(url, timeout=120) as response, temporary.open('wb') as stream:
                    shutil.copyfileobj(response, stream, 1024 * 1024)
                if not matches(temporary, entry):
                    raise ValueError('Official checksum/size mismatch: ' + entry['path'])
                temporary.replace(path)
                print('Downloaded and verified: ' + entry['path'], flush=True)
                return True
            except Exception:
                if temporary.exists():
                    temporary.unlink()
                if attempt == 2:
                    raise
                time.sleep(2)

    downloaded = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process, entry) for entry in entries]
        for future in as_completed(futures):
            downloaded += int(future.result())
    print(f"Verified {len(entries)} files from DaRUS V{manifest['version']} "
          f"({downloaded} downloaded): {DESTINATION}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    sync(verify_only=args.verify_only, workers=args.workers)

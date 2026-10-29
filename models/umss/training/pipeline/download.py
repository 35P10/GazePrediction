"""Download everything UMSS stage-1 training needs, into <work>/downloads and <work>/upstream.

Files already present are skipped on re-runs. Checksums are checked by the separate `verify` stage.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
MANIFEST = HERE.parent / 'training_sources.json'
sys.path.insert(0, str(HERE.parent.parent))  # models/umss: reference/ source setup
from reference.setup_umss import setup as setup_official_code  # noqa: E402


def fetch(url, dest):
    dest = Path(dest)
    if dest.is_file():
        print('ok      ', dest.name)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print('download', url, flush=True)
    # Download to a temporary file so an interrupted download is not mistaken for a finished one.
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp, urlopen(url, timeout=120) as response:
        shutil.copyfileobj(response, tmp, 1 << 20)
    os.replace(tmp.name, dest)
    return dest


def extract(archive, target, marker):
    if marker.exists():
        return
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        # Skip macOS metadata.
        z.extractall(target, [member for member in z.namelist() if not member.startswith('__MACOSX/')
                              and not os.path.basename(member).startswith('._')])
    marker.touch()


def main(work):
    work = Path(work)
    downloads = work / 'downloads'
    manifest = json.loads(MANIFEST.read_text())
    setup_official_code(work / 'upstream')
    for item in manifest['archives']:
        archive = fetch(item['url'], downloads / 'archives' / (item['name'] + '.zip'))
        extract(archive, downloads / item['extract_to'], downloads / ('.extracted_' + item['name']))
    for item in manifest['files']:
        fetch(item['url'], downloads / item['dest'])
    print('All downloads in', downloads)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', default='/work')
    main(parser.parse_args().work)

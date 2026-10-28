"""Download and verify everything UMSS stage-1 training needs, into <work>/downloads and <work>/upstream.

Every file is checked against the SHA-256 pinned in training_sources.json (the official code
against the DaRUS MD5s in umss_sources.json). Finished items are skipped on re-runs.
"""
import argparse
import hashlib
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


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url, dest, expected):
    dest = Path(dest)
    if dest.is_file() and sha256(dest) == expected:
        print('ok      ', dest.name)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print('download', url, flush=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp, urlopen(url, timeout=120) as response:
        shutil.copyfileobj(response, tmp, 1 << 20)
    if sha256(tmp.name) != expected:
        os.unlink(tmp.name)
        raise ValueError('Checksum mismatch for %s; the source changed or the download is corrupt' % url)
    os.replace(tmp.name, dest)
    return dest


def extract(archive, target, marker):
    if marker.exists():
        return
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.namelist():
            # Skip macOS metadata; refuse paths escaping the target.
            if member.startswith('__MACOSX/') or os.path.basename(member).startswith('._'):
                continue
            if os.path.isabs(member) or '..' in Path(member).parts:
                raise ValueError('Unsafe path in %s: %s' % (archive, member))
            z.extract(member, target)
    marker.write_text(sha256(archive) + '\n')


def main(work):
    work = Path(work)
    downloads = work / 'downloads'
    manifest = json.loads(MANIFEST.read_text())
    setup_official_code(work / 'upstream')
    for item in manifest['archives']:
        archive = fetch(item['url'], downloads / 'archives' / (item['name'] + '.zip'), item['sha256'])
        extract(archive, downloads / item['extract_to'], downloads / ('.extracted_' + item['name']))
    for item in manifest['files']:
        fetch(item['url'], downloads / item['dest'], item['sha256'])
    print('All downloads verified in', downloads)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', default='/work')
    main(parser.parse_args().work)

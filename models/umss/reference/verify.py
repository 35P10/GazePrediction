"""Integrity checks for UMSS, kept out of the main flow: run them on their own when needed.

    python models/umss/reference/verify.py                  # bundled official sources
    python models/umss/reference/verify.py --work /work     # a training run: upstream/ and downloads/

Official files are compared with the DaRUS MD5s in ``umss_sources.json``; training downloads with the
SHA-256s in ``training/training_sources.json``.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

if __package__:
    from .setup_umss import DEFAULT_DESTINATION, MANIFEST, sha256
else:  # Run as a script, also inside the Docker image.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from reference.setup_umss import DEFAULT_DESTINATION, MANIFEST, sha256

TRAINING_MANIFEST = Path(__file__).resolve().parents[1] / 'training' / 'training_sources.json'


def verify_sources(destination=DEFAULT_DESTINATION):
    """Every official file present with its pinned MD5, and the recorded manifest equal to the pinned one."""
    destination = Path(destination)
    manifest = json.loads(MANIFEST.read_text())
    if json.loads((destination / 'sources.json').read_text()) != manifest:
        raise ValueError('UMSS source manifest differs from the pinned DaRUS V2 files')
    for entry in manifest['files']:
        path = destination / entry['path']
        if not path.is_file() or hashlib.md5(path.read_bytes()).hexdigest() != entry['md5']:
            raise ValueError('UMSS source checksum mismatch: ' + str(path))


def verify_downloads(work):
    """Training archives and files in ``<work>/downloads`` with their pinned SHA-256s."""
    downloads = Path(work) / 'downloads'
    manifest = json.loads(TRAINING_MANIFEST.read_text())
    expected = [(downloads / 'archives' / (item['name'] + '.zip'), item['sha256']) for item in manifest['archives']]
    expected += [(downloads / item['dest'], item['sha256']) for item in manifest['files']]
    for path, digest in expected:
        if not path.is_file() or sha256(path) != digest:
            raise ValueError('Download checksum mismatch: ' + str(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--work', default=None, help='training run folder (verifies upstream/ and downloads/)')
    args = parser.parse_args()
    if args.work is None:
        verify_sources()
    else:
        verify_sources(Path(args.work) / 'upstream')
        verify_downloads(args.work)
    print('UMSS integrity verified')

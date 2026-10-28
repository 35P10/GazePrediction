"""Download the small official UMSS source files, verifying the pinned DaRUS V2 checksums.

Original Code/scripts paths are preserved. No datasets, pretrained weights,
dependencies or inference are downloaded/run.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen

MANIFEST = Path(__file__).with_name('umss_sources.json')
DEFAULT_DESTINATION = Path(__file__).resolve().parents[1] / 'training' / 'official_reference'


def verify_source_file(path, relative_path):
    """Verify one original file without requiring the full neural source tree."""
    entries = json.loads(MANIFEST.read_text())['files']
    entry = next(item for item in entries if item['path'] == relative_path)
    path = Path(path)
    if not path.is_file() or hashlib.md5(path.read_bytes()).hexdigest() != entry['md5']:
        raise ValueError('UMSS source checksum mismatch: ' + str(path))


def verify_sources(destination, manifest):
    expected = json.loads(MANIFEST.read_text())
    if manifest != expected:
        raise ValueError('UMSS source manifest differs from the pinned DaRUS V2 files')
    for entry in manifest['files']:
        path = Path(destination) / entry['path']
        if not path.is_file() or hashlib.md5(path.read_bytes()).hexdigest() != entry['md5']:
            raise ValueError('UMSS source checksum mismatch: ' + str(path))


def load_definitions(repository, relative, names, namespace):
    """Verify and load named definitions without executing source driver code."""
    path = Path(repository) / relative
    verify_source_file(path, relative)
    if path.suffix == '.ipynb':
        notebook = json.loads(path.read_text())
        sources = [''.join(cell['source']) for cell in notebook['cells']
                   if cell['cell_type'] == 'code']
    else:
        sources = [path.read_text()]
    nodes = [node for source in sources for node in ast.parse(source).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if len(nodes) != len(names) or {node.name for node in nodes} != set(names):
        raise ValueError('Missing or duplicate official definitions: ' + str(path))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def _populate(destination, manifest):
    for entry in manifest['files']:
        path = destination / entry['path']
        if path.is_file():
            verify_source_file(path, entry['path'])
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        cached = DEFAULT_DESTINATION / entry['path']
        if cached.is_file():
            verify_source_file(cached, entry['path'])
            shutil.copyfile(cached, path)
        else:
            with urlopen('https://darus.uni-stuttgart.de/api/access/datafile/' + str(entry['id']), timeout=60) as response:
                content = response.read()
            if hashlib.md5(content).hexdigest() != entry['md5']:
                raise ValueError('UMSS download checksum mismatch: ' + entry['path'])
            path.write_bytes(content)
    verify_sources(destination, manifest)
    temporary = destination / 'sources.json.part'
    temporary.write_text(json.dumps(manifest, indent=2) + '\n')
    temporary.replace(destination / 'sources.json')


def setup(destination):
    destination = Path(destination).resolve()
    manifest = json.loads(MANIFEST.read_text())
    if destination.exists():
        previous = json.loads((destination / 'sources.json').read_text())
        # Existing runs may have the earlier, smaller pinned source subset.
        # Extend only a recognized manifest, and verify every existing file.
        if (previous.get('dataset') != manifest['dataset']
                or previous.get('version') != manifest['version']
                or any(entry not in manifest['files'] for entry in previous['files'])):
            raise ValueError('UMSS source manifest differs from the pinned DaRUS V2 files')
        _populate(destination, manifest)
        print('Verified UMSS sources:', destination)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        staging = Path(temporary) / 'umss'
        staging.mkdir()
        _populate(staging, manifest)
        staging.rename(destination)
    print('Official sources ready; supply a trained MD-EAM checkpoint separately:', destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', default=str(DEFAULT_DESTINATION))
    setup(parser.parse_args().destination)

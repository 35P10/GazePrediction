"""Access to the official UMSS sources (DaRUS V2): download and loading.

Original Code/scripts paths are preserved and the authors' files are never edited; this module only
downloads them and loads definitions from them. Checksums are checked separately by ``verify.py``.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys
from urllib.request import urlopen

MANIFEST = Path(__file__).with_name('umss_sources.json')
DEFAULT_DESTINATION = Path(__file__).resolve().parents[1] / 'training' / 'official_reference'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_definitions(repository, relative, names, namespace, inside=None):
    """Load named official definitions without executing their driver code.

    ``inside`` names an official function whose nested definitions are loaded instead of top-level ones.
    """
    path = Path(repository) / relative
    if path.suffix == '.ipynb':
        sources = [''.join(cell['source']) for cell in json.loads(path.read_text())['cells']
                   if cell['cell_type'] == 'code']
    else:
        sources = [path.read_text()]
    bodies = [ast.parse(source).body for source in sources]
    if inside is not None:
        bodies = [node.body for body in bodies for node in body
                  if isinstance(node, ast.FunctionDef) and node.name == inside]
    nodes = [node for body in bodies for node in body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def load_md_eam(repository, checkpoint):
    """Official MD-EAM network (``md_sem``) with the official pre/post-processing functions.

    Needs Keras 2.3.1 / TensorFlow 1.x. Returns ``(model, preprocess_images, postprocess_predictions)``.
    """
    scripts = Path(repository).resolve() / 'Code/scripts'
    # The official modules use bare imports (``from multiduration_models import ...``).
    for folder in (scripts / 'saliency', scripts / 'src'):
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
    from multiduration_models import md_sem  # official
    from sal_imp_utilities import preprocess_images, postprocess_predictions  # official
    # All four training outputs share the same decoder; one is enough for inference.
    model = md_sem(input_shape=(240, 320, 3), nb_timestep=3, n_outs=1, verbose=False, print_shapes=False)
    model.load_weights(str(checkpoint))
    return model, preprocess_images, postprocess_predictions


def setup(destination):
    """Copy (from the bundled reference) or download the missing official files into ``destination``."""
    destination = Path(destination).resolve()
    manifest = json.loads(MANIFEST.read_text())
    for entry in manifest['files']:
        path = destination / entry['path']
        if path.is_file():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        bundled = DEFAULT_DESTINATION / entry['path']
        if bundled.is_file():
            shutil.copyfile(bundled, path)
        else:
            with urlopen('https://darus.uni-stuttgart.de/api/access/datafile/' + str(entry['id']), timeout=60) as response:
                path.write_bytes(response.read())
    (destination / 'sources.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Official sources ready; supply a trained MD-EAM checkpoint separately:', destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', default=str(DEFAULT_DESTINATION))
    setup(parser.parse_args().destination)

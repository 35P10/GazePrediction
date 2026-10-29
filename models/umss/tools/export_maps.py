"""Host-side staging and Docker export of UMSS saliency maps."""
import json
import os
from pathlib import Path

from ..model import PACKAGE_ROOT
from ..reference.setup_umss import sha256


def work_dir():
    """``$UMSS_WORK`` as ``training/run.sh`` resolves it (default ``artifacts/runs/umss/massvis_md_eam_validation``)."""
    default = PACKAGE_ROOT.parents[1] / 'artifacts' / 'runs' / 'umss' / 'massvis_md_eam_validation'
    return Path(os.environ.get('UMSS_WORK', default)).resolve()


def _cached(out, images, checkpoint):
    """An earlier export of these same images with this same checkpoint."""
    try:
        provenance = json.loads((out / 'export_metadata.json').read_text())
        return (provenance['checkpoint_sha256'] == sha256(checkpoint)
                and provenance['image_sha256'] == {p.stem: sha256(p) for p in images})
    except (FileNotFoundError, KeyError):
        return False


def export_maps(images, name, *, work=None):
    """MD-EAM maps of ``images`` (paths) from the trained checkpoint in ``<work>/ckpt``, as the ``.npz`` files
    ``UMSS(maps_dir=...)`` reads. Needs the GPU machine: copies the images under ``<work>/export/<name>`` (the only
    folder the container sees) and runs ``training/run.sh export``. Returns ``<work>/predictions/<name>``;
    reuses an earlier export of the same images with the same checkpoint."""
    import shutil
    import subprocess

    work = Path(work).resolve() if work is not None else work_dir()
    images = [Path(p) for p in images]
    stems = [p.stem for p in images]
    if len(set(stems)) != len(stems):
        raise ValueError('Image names must be unique: maps are looked up by file stem')
    out = work / 'predictions' / name
    checkpoint = work / 'ckpt' / 'md_eam.hdf5'
    if _cached(out, images, checkpoint):
        return out
    charts = work / 'export' / name
    charts.mkdir(parents=True, exist_ok=True)
    for image in images:
        shutil.copyfile(image, charts / image.name)
    command = ['bash', str(PACKAGE_ROOT / 'training' / 'run.sh'), 'export', '--images', f'/work/export/{name}',
               '--out', f'/work/predictions/{name}']
    # Stream the Docker output (into a notebook cell rather than the Jupyter server log).
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          env={**os.environ, 'UMSS_WORK': str(work)}) as process:
        for line in process.stdout:
            print(line, end='')
    if process.returncode:
        raise RuntimeError(f'run.sh export failed (exit {process.returncode})')
    return out

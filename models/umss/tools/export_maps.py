"""Host-side staging and Docker export of UMSS saliency maps."""
import json
from pathlib import Path

from ..sampling import PACKAGE_ROOT
from .files import sha256


def work_dir():
    """``$UMSS_WORK`` as ``training/run.sh`` resolves it (default ``artifacts/runs/umss/massvis_md_eam_validation``)."""
    import os
    return Path(os.environ.get('UMSS_WORK', PACKAGE_ROOT.parents[1] / 'artifacts' / 'runs' / 'umss' / 'massvis_md_eam_validation')).resolve()


def export_maps(images, name, *, work=None):
    """MD-EAM maps of ``images`` (paths) from the trained checkpoint in ``<work>/ckpt``, as the ``.npz`` files
    ``UMSS(maps_dir=...)`` reads. Needs the GPU machine: copies the images under ``<work>/export/<name>`` (the only
    folder the container sees) and runs ``training/run.sh export``. Returns ``<work>/predictions/<name>``;
    reuses a complete export only when its recorded checkpoint and image hashes match."""
    import os
    import shutil
    import subprocess

    work = Path(work).resolve() if work is not None else work_dir()
    images = [Path(p) for p in images]
    stems = [p.stem for p in images]
    if len(set(stems)) != len(stems):
        raise ValueError('Image names must be unique: maps are looked up by file stem')
    out = work / 'predictions' / name
    metadata = out / 'export_metadata.json'
    checkpoint = work / 'ckpt/md_eam.hdf5'
    if metadata.is_file() and checkpoint.is_file() and all((out / f'{stem}.npz').is_file() for stem in stems):
        provenance = json.loads(metadata.read_text())
        if (provenance['checkpoint_sha256'] == sha256(checkpoint)
                and all(provenance['image_sha256'].get(p.stem) == sha256(p) for p in images)):
            return out
    if not (work / 'ckpt' / 'md_eam.hdf5').is_file():
        raise FileNotFoundError(f'No trained checkpoint in {work / "ckpt"}: export runs on the GPU machine after '
                                'training/run.sh; elsewhere copy the .npz files instead')
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
    missing = [stem for stem in stems if not (out / f'{stem}.npz').is_file()]
    if missing:
        raise RuntimeError(f'run.sh export did not write {len(missing)} files, e.g. {missing[0]}.npz')
    return out

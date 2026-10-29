"""UMSS on images: MD-EAM temporal saliency maps (stage 1) and sampled scanpaths (stage 2).

Maps come either from exported ``.npz`` files (``maps_dir``, no TensorFlow needed) or from the trained
checkpoint run through the official network (legacy Keras 2.3.1 / TF 1.x, imported only when used).
"""
from pathlib import Path
import sys
from typing import Iterable, TypedDict, Union

import numpy as np

from .reference.setup_umss import DEFAULT_DESTINATION as OFFICIAL_MODEL, sha256
from .sampling import GRID, BOUNDARIES, DURATION_MS, validate_volume, sample_scanpath

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PACKAGE_ROOT.parents[1] / 'artifacts' / 'checkpoints' / 'umss' / 'md_eam.hdf5'
DATASET = 'doi:10.18419/DARUS-3361'
VERSION = '2.0'
SCANPATH_COLUMNS = ['image_id', 'run', 'seed', 'fix_idx', 'x', 'y', 'grid_x', 'grid_y', 'duration', 'slice']


class ScanpathInput(TypedDict):
    """One image and its output identifier; no questions or AOIs."""
    image_id: str
    image_path: Union[str, Path]


class UMSS:
    def __init__(self, *, repository=None, checkpoint=None, maps_dir=None, sigma=200.):
        if checkpoint is not None and maps_dir is not None:
            raise ValueError('Choose checkpoint or maps_dir')
        self.repository = Path(repository or OFFICIAL_MODEL).resolve()
        self.checkpoint = Path(checkpoint or DEFAULT_CHECKPOINT).resolve()
        self.maps_dir = Path(maps_dir).resolve() if maps_dir is not None else None
        self.sigma = sigma
        self._network = None

    @property
    def settings(self):
        """Run settings for provenance records."""
        import scipy
        if self.maps_dir is not None:
            source = dict(maps_dir=str(self.maps_dir))
        else:
            source = dict(checkpoint=str(self.checkpoint),
                          checkpoint_sha256=sha256(self.checkpoint) if self.checkpoint.is_file() else None)
        return dict(dataset=DATASET, version=VERSION, grid=list(GRID), sigma=self.sigma,
                    temporal_boundaries_seconds=list(BOUNDARIES), duration_ms=list(DURATION_MS),
                    protocol='fixed-length-exgaussian-v1', python=sys.version.split()[0],
                    numpy=np.__version__, scipy=scipy.__version__, **source)

    # ---- stage 1: temporal saliency maps

    def predict_maps(self, image_path):
        """The three maps on the 480x640 grid, with the image width and height."""
        if self.maps_dir is not None:
            return self._read_maps(image_path)
        return self._infer_maps(image_path)

    def _read_maps(self, image_path):
        """Maps exported for this image: ``<maps_dir>/<image stem>.npz``."""
        with np.load(self.maps_dir / (Path(image_path).stem + '.npz'), allow_pickle=False) as saved:
            return dict(maps=validate_volume(saved['maps']), width=int(saved['width']), height=int(saved['height']))

    def _infer_maps(self, image_path):
        """Official preprocessing, network and postprocessing, then a resize to the sampling grid."""
        import cv2
        from PIL import Image
        with Image.open(image_path) as image:
            width, height = image.size
        network, preprocess, postprocess = self._load_network()
        predictions = network.predict(preprocess([str(image_path)], 240, 320))
        if isinstance(predictions, list):
            predictions = predictions[0]
        # Official postprocessing removes the letterbox at the image size; the resize keeps floats.
        maps = [cv2.resize(postprocess(predictions[0, t, :, :, 0], height, width), GRID[::-1]) for t in range(3)]
        return dict(maps=validate_volume(np.stack(maps)), width=width, height=height)

    def _load_network(self):
        if self._network is None:
            from .reference.setup_umss import load_md_eam
            self._network = load_md_eam(self.repository, self.checkpoint)
        return self._network

    # ---- stage 2: scanpaths

    def predict(self, image_path, *, n_fix=8, seed=0):
        """One scanpath for one image."""
        return self._scanpath(self.predict_maps(image_path), n_fix=n_fix, seed=seed)

    def _scanpath(self, prediction, *, n_fix, seed):
        """Sample on the grid and add ``points`` in image pixels."""
        result = sample_scanpath(prediction['maps'], n_fix=n_fix, seed=seed, sigma=self.sigma)
        width, height = prediction['width'], prediction['height']
        points = [(x / GRID[1] * width, y / GRID[0] * height) for x, y in result['grid_points']]
        result.update(width=width, height=height, seed=seed, points=points)
        return result

    def run(self, inputs: Iterable[ScanpathInput], output_dir: Union[str, Path, None] = None,
            *, n_runs=100, n_fix=10, seed=17300) -> Union[Path, dict]:
        """
        - ``n_runs`` scanpaths per image; run r uses seed + r for every image.

        - Without ``output_dir`` returns ``{'maps': {image_id: maps}, 'scanpaths': DataFrame}``. 
        - With it, writes ``maps/<image_id>.npy`` and ``scanpaths.csv`` there and returns the folder.
        
        - ``n_fix=None`` selects the official UMSS-S2 variable-length sampler.
        """
        import pandas as pd
        if output_dir is not None:
            output_dir = Path(output_dir)
            (output_dir / 'maps').mkdir(parents=True, exist_ok=True)
        maps, rows = {}, []
        for item in inputs:
            image_id = item['image_id']
            prediction = self.predict_maps(item['image_path'])
            if output_dir is None:
                maps[image_id] = prediction['maps']
            else:
                np.save(output_dir / 'maps' / (image_id + '.npy'), prediction['maps'], allow_pickle=False)
            for run in range(n_runs):
                scanpath = self._scanpath(prediction, n_fix=n_fix, seed=seed + run)
                rows += _scanpath_rows(image_id, run, scanpath)
        scanpaths = pd.DataFrame(rows, columns=SCANPATH_COLUMNS)
        if output_dir is None:
            return dict(maps=maps, scanpaths=scanpaths)
        scanpaths.to_csv(output_dir / 'scanpaths.csv', index=False)
        return output_dir


def _scanpath_rows(image_id, run, scanpath):
    """One table row per fixation."""
    fixations = zip(scanpath['points'], scanpath['grid_points'], scanpath['durations'], scanpath['slices'])
    return [dict(image_id=image_id, run=run, seed=scanpath['seed'], fix_idx=index, x=x, y=y,
                 grid_x=grid_x, grid_y=grid_y, duration=duration, slice=slice_index)
            for index, ((x, y), (grid_x, grid_y), duration, slice_index) in enumerate(fixations)]


def create_predictor(**settings):
    return UMSS(**settings)

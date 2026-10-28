"""Image-only MD-EAM inference and UMSS scanpath generation; neural imports are lazy."""
from pathlib import Path
import json
import sys
from typing import Iterable, TypedDict, Union

from .sampling import (PACKAGE_ROOT, OFFICIAL_MODEL, DATASET, VERSION, GRID, BOUNDARIES, DURATION_MS,
                    validate_volume, sample_scanpath)
from .tools.files import sha256


class ScanpathInput(TypedDict):
    """One image and its output identifier; no questions or AOIs."""
    image_id: str
    image_path: Union[str, Path]


class UMSS:
    def __init__(self, *, repository=None, checkpoint=None, maps_dir=None, sigma=200.):
        import math
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError('Positive finite sigma required')
        if checkpoint is not None and maps_dir is not None:
            raise ValueError('Choose checkpoint or maps_dir')
        self.repository = Path(repository or OFFICIAL_MODEL).resolve()
        self.checkpoint = Path(checkpoint or PACKAGE_ROOT.parents[1] / 'artifacts' / 'checkpoints' / 'umss' / 'md_eam.hdf5').resolve()
        self.maps_dir = Path(maps_dir).resolve() if maps_dir is not None else None
        self.sigma = sigma
        self._model = None
        self._preprocess = None
        self._postprocess = None

    @property
    def settings(self):
        import numpy as np
        import scipy
        provenance = (dict(maps_dir=str(self.maps_dir)) if self.maps_dir is not None else
                      dict(checkpoint=str(self.checkpoint), checkpoint_sha256=sha256(self.checkpoint)
                           if self.checkpoint.is_file() else None))
        return dict(dataset=DATASET, version=VERSION, grid=list(GRID), sigma=self.sigma,
                    temporal_boundaries_seconds=list(BOUNDARIES), duration_ms=list(DURATION_MS),
                    protocol='fixed-length-exgaussian-v1', python=sys.version.split()[0],
                    numpy=np.__version__, scipy=scipy.__version__, **provenance)

    def input_fingerprint(self, image_path):
        """Bind resumed runs to their actual map files or checkpoint and source files."""
        if self.maps_dir is not None:
            path = self.maps_dir / (Path(image_path).stem + '.npz')
            return dict(path=str(path), sha256=sha256(path))
        self._check_checkpoint()
        manifest = json.loads((self.repository / 'sources.json').read_text())
        from .reference.setup_umss import verify_sources
        verify_sources(self.repository, manifest)
        return dict(checkpoint_sha256=sha256(self.checkpoint), sources_sha256=sha256(self.repository / 'sources.json'))

    def _check_checkpoint(self):
        if not self.checkpoint.is_file():
            raise FileNotFoundError('UMSS MD-EAM checkpoint missing. The official deposit has no trained checkpoint; '
                                    'supply checkpoint=... or maps_dir=...; see models/umss/README.md')

    def _load(self):
        if self._model is not None:
            return
        self._check_checkpoint()
        source = self.repository / 'Code/scripts/src'
        preprocessing = self.repository / 'Code/scripts/saliency'
        if not (source / 'multiduration_models.py').is_file():
            raise FileNotFoundError('Run python models/umss/reference/setup_umss.py first; see models/umss/README.md')
        from .reference.setup_umss import verify_sources
        verify_sources(self.repository, json.loads((self.repository / 'sources.json').read_text()))
        try:
            import keras
            import tensorflow as tf
        except ImportError as error:
            raise RuntimeError('UMSS checkpoint inference needs the separate legacy environment; see models/umss/README.md') from error
        if not keras.__version__.startswith('2.3.') or not tf.__version__.startswith('1.'):
            raise RuntimeError('Official UMSS requires Keras 2.3.1 / TensorFlow 1.x; see models/umss/README.md')
        # Upstream uses bare imports. Refuse collisions with another model checkout.
        for name in ['multiduration_models', 'sal_imp_utilities', 'xception_custom',
                     'attentive_convlstm_new', 'dcn_resnet_new', 'gaussian_prior_new']:
            module = sys.modules.get(name)
            expected = preprocessing if name == 'sal_imp_utilities' else source
            if module is not None and Path(module.__file__).resolve().parent != expected:
                raise RuntimeError('UMSS upstream module collision; restart the dedicated kernel')
        paths = [str(source), str(preprocessing)]
        sys.path[:0] = paths
        try:
            from multiduration_models import md_sem
            from sal_imp_utilities import preprocess_images, postprocess_predictions
            # All four training outputs share the same decoder; one is enough for inference.
            model = md_sem(input_shape=(240, 320, 3), nb_timestep=3, n_outs=1,
                           verbose=False, print_shapes=False)
            model.load_weights(str(self.checkpoint))
            self._model = model
            self._preprocess, self._postprocess = preprocess_images, postprocess_predictions
        finally:
            for path in paths:
                sys.path.remove(path)

    def predict_maps(self, image_path):
        import numpy as np
        from PIL import Image
        with Image.open(image_path) as image:
            width, height = image.size
        if self.maps_dir is not None:
            with np.load(self.maps_dir / (Path(image_path).stem + '.npz'), allow_pickle=False) as saved:
                if (int(saved['width']), int(saved['height'])) != (width, height):
                    raise ValueError('UMSS map image dimensions differ from the chart')
                if str(saved['image_sha256'].item()) != sha256(image_path):
                    raise ValueError('UMSS maps were generated for a different image')
                volume = saved['maps'].copy()
        else:
            self._load()
            predictions = self._model.predict(self._preprocess([str(image_path)], 240, 320))
            if isinstance(predictions, list):
                predictions = predictions[0]
            # Remove letterboxing before converting to the sampling grid; preserve floats.
            import cv2
            volume = np.stack([cv2.resize(self._postprocess(predictions[0, t, :, :, 0], height, width),
                                          (GRID[1], GRID[0])) for t in range(3)])
        return dict(maps=validate_volume(volume), width=width, height=height)

    def predict(self, image_path, *, n_fix=8, seed=0):
        return self._scanpath(self.predict_maps(image_path), n_fix=n_fix, seed=seed)

    def _scanpath(self, prediction, *, n_fix, seed):
        result = sample_scanpath(prediction['maps'], n_fix=n_fix, seed=seed, sigma=self.sigma)
        result.update(width=prediction['width'], height=prediction['height'], seed=seed)
        result['points'] = [(x / GRID[1] * result['width'], y / GRID[0] * result['height'])
                            for x, y in result['grid_points']]
        return result

    def run(self, inputs: Iterable[ScanpathInput], output_dir: Union[str, Path, None] = None,
            *, n_runs=100, n_fix=10, seed=17300) -> Union[Path, dict]:
        """Return maps and scanpaths in memory, or save them when output_dir is supplied.

        Run r uses seed + r for every image. No center fixation is inserted.
        n_fix=None selects the released UMSS-S2 variable-length sampler.
        """
        import numpy as np
        import pandas as pd
        if type(n_runs) is not int or n_runs < 1 or type(seed) is not int or seed < 0:
            raise ValueError('Positive integer n_runs and nonnegative integer seed required')
        if n_fix is not None and (type(n_fix) is not int or n_fix < 1):
            raise ValueError('Positive integer n_fix (or None) required')
        maps = {}
        if output_dir is not None:
            output_dir = Path(output_dir)
            (output_dir / 'maps').mkdir(parents=True, exist_ok=True)
        rows = []
        for item in inputs:
            prediction = self.predict_maps(item['image_path'])
            if output_dir is None:
                maps[item['image_id']] = prediction['maps']
            else:
                np.save(output_dir / 'maps' / (item['image_id'] + '.npy'), prediction['maps'], allow_pickle=False)
            for run in range(n_runs):
                result = self._scanpath(prediction, n_fix=n_fix, seed=seed + run)
                rows.extend(dict(image_id=item['image_id'], run=run, seed=seed+run, fix_idx=i,
                                 x=x, y=y, grid_x=gx, grid_y=gy, duration=duration, slice=slice_index)
                            for i, ((x, y), (gx, gy), duration, slice_index) in enumerate(zip(
                                result['points'], result['grid_points'], result['durations'], result['slices'])))
        scanpaths = pd.DataFrame(rows, columns=['image_id','run','seed','fix_idx','x','y','grid_x','grid_y','duration','slice'])
        if output_dir is None:
            return dict(maps=maps, scanpaths=scanpaths)
        scanpaths.to_csv(output_dir / 'scanpaths.csv', index=False)
        return output_dir


def create_predictor(**settings):
    return UMSS(**settings)

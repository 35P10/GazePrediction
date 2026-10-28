"""Shared saliency metrics and human references for training evaluation and reproduction.

Original function definitions are verified and loaded without running their
standalone drivers or importing their unused Keras dependencies.
"""
from functools import lru_cache

import numpy as np
from PIL import Image

if __package__ != 'evaluation':
    from ..reference.setup_umss import DEFAULT_DESTINATION, load_definitions
else:  # Standalone training Docker image.
    from reference.setup_umss import DEFAULT_DESTINATION, load_definitions

SIGMA = 19


@lru_cache(maxsize=None)
def saliency_functions(repository):
    return load_definitions(repository, 'Code/scripts/saliency/eval.py',
                        ('cc_npy', 'nss_npy', 'kl_npy', 'sim_npy'), {'np': np})


def score_saliency(repository, prediction, heatmap, fixations):
    """Table 2 normalization and the original four NumPy metric functions."""
    functions = saliency_functions(str(repository))
    prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min())
    if heatmap.max() > 1:
        heatmap = heatmap.astype(np.float32) / 255.
    return {
        'NSS': float(functions['nss_npy'](fixations, prediction)),
        'CC': float(functions['cc_npy'](heatmap, prediction)),
        'KL': float(functions['kl_npy'](heatmap, prediction)),
        'SIM': float(functions['sim_npy'](heatmap, prediction)),
    }


@lru_cache(maxsize=None)
def _fixmap_function(repository):
    from scipy import ndimage
    namespace = load_definitions(repository, 'Code/scripts/metrics/saliency_saltinet_csv.py',
                             ('make_fixmap_and_heatmap',),
                             {'np': np, 'Image': Image, 'ndimage': ndimage})
    return namespace['make_fixmap_and_heatmap']


def make_fixmap_and_heatmap(repository, width, height, coords, sigma=SIGMA):
    """Run the original heatmap function; empty buckets produce zero-valued images."""
    with np.errstate(invalid='ignore', divide='ignore'):
        return _fixmap_function(str(repository))(width, height, coords, sigma=sigma)


def accumulated_coords(files, limit_ms):
    """Human coordinates ending strictly before the duration limit used by UMSS."""
    import csv
    coords = []
    for path in files:
        elapsed = 0
        with open(path) as stream:
            for row in csv.reader(stream):
                elapsed += int(float(row[3]))
                if elapsed < limit_ms:
                    coords.append([int(float(row[1])), int(float(row[2]))])
    return coords



"""Shared saliency metrics and human references for training evaluation and reproduction.

Original function definitions are loaded without running their
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
    # Official calculate_metrics and the metric functions it calls, from the authors' eval.py.
    # Its eval_saliconeval import is missing from the deposit; the one saliconeval value it computes
    # for these inputs, 'CC (saliconeval)', is not used here.
    return load_definitions(repository, 'Code/scripts/saliency/eval.py',
                            ('calculate_metrics', 'r2', 'rmse', 'cc_npy', 'nss_npy', 'kl_npy', 'sim_npy'),
                            {'np': np, 'cc_saliconeval': lambda gt, p: float('nan')})


def score_saliency(repository, prediction, heatmap, fixations):
    """Table 2 metrics through the official calculate_metrics (its own min-max normalization)."""
    metrics = saliency_functions(str(repository))['calculate_metrics'](
        prediction, gt_map=heatmap, gt_fix_map=fixations, sAUC=False)
    return {name: float(metrics[name][0]) for name in ('NSS', 'CC', 'KL', 'SIM')}


@lru_cache(maxsize=None)
def _fixmap_function(repository):
    # Official make_fixmap_and_heatmap from the authors' saliency_saltinet_csv.py.
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



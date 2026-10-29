"""UMSS scanpath sampling (stage 2) on three temporal saliency maps.

The maps cover 0-0.5 s, 0.5-3 s and 3-5 s on a 480x640 grid. Fixations are placed by the official
``sample_slice`` from UMSS-S2.ipynb. Two protocols decide how many fixations each map gets:

* ``n_fix=None``, official UMSS-S2: the official ``sample_volume`` draws the length and durations.
* ``n_fix=k``, ours: ``k`` fixations with ex-Gaussian durations; each one uses the map of the time it starts.

The official functions draw from the global ``np.random``, so each scanpath seeds it.
"""
from functools import lru_cache

import numpy as np

GRID = (480, 640)  # rows, columns
BOUNDARIES = (.5, 3.)  # seconds where the next map starts
SLICE_LABELS = ('0-0.5s', '0.5-3s', '3-5s')
DURATION_MS = (124.06, 17.49, 89.37)  # ours: normal mean, normal std and exponential mean of the ex-Gaussian
S2_FIXATION_COUNTS = tuple(range(2, 36))  # lengths the official get_number_fixations can return


@lru_cache(maxsize=None)
def official_s2():
    """The official UMSS-S2 functions, by name.

    ``sample_slice`` and ``sample_volume`` are nested in ``predict`` of UMSS-S2.ipynb; ``get_number_fixations``
    and ``get_duration_fixation`` come from Code/scripts/UMSS/saltinet_util.py, which that notebook imports.
    """
    from scipy import ndimage
    from .reference.setup_umss import DEFAULT_DESTINATION, load_definitions
    functions = load_definitions(DEFAULT_DESTINATION, 'Code/scripts/UMSS/saltinet_util.py',
                                 ('get_number_fixations', 'get_duration_fixation'), {'np': np})
    functions.update(ndimage=ndimage, array_shape=GRID)  # `array_shape` is a local of the official `predict`
    return load_definitions(DEFAULT_DESTINATION, 'Code/scripts/UMSS/UMSS-S2.ipynb',
                            ('sample_slice', 'sample_volume'), functions, inside='predict')


def validate_volume(volume):
    """Three finite, nonnegative 480x640 maps, each with some mass, as floats."""
    volume = np.asarray(volume, dtype=float)
    if volume.shape != (3,) + GRID or not np.isfinite(volume).all() or (volume < 0).any():
        raise ValueError('UMSS requires three finite nonnegative 480x640 maps in image coordinates')
    if (volume.sum(axis=(1, 2)) <= 0).any():
        raise ValueError('Every UMSS temporal map must have positive mass')
    return volume


def sample_scanpath(volume, *, n_fix=8, seed=0, sigma=200.):
    """One scanpath: ``grid_points`` (x, y) on the 480x640 grid, ``durations`` in seconds and the map
    (``slices``, 0-2) each fixation was drawn from. ``n_fix=None`` selects the official UMSS-S2 protocol."""
    if sigma != 200:
        raise ValueError('The official UMSS-S2 sampler uses sigma 200')
    # The official sampler thresholds the maps in place: give it its own copy.
    volume = validate_volume(volume).copy()
    np.random.seed(seed)
    if n_fix is None:
        points, durations, slices = _sample_s2(volume)
    else:
        points, durations, slices = _sample_fixed_length(volume, n_fix)
    return dict(grid_points=points, durations=durations, slices=slices)


def _sample_s2(volume):
    """Official UMSS-S2: ``sample_volume`` draws durations, splits them between maps and samples positions."""
    official = official_s2()
    sample_slice = official['sample_slice']
    per_map = []  # fixations sampled from each map, in order

    def counting_sample_slice(pred, size, n_samples=1, use_heuristic=True, samples=[]):
        per_map.append(int(n_samples))
        return sample_slice(pred, size, n_samples=n_samples, use_heuristic=use_heuristic, samples=samples)

    # sample_volume looks sample_slice up by name; count its calls to know each fixation's map.
    official['sample_slice'] = counting_sample_slice
    try:
        samples = official['sample_volume'](volume, n_samples=official['get_number_fixations'](),
                                            size=GRID, use_heuristic=True)
    finally:
        official['sample_slice'] = sample_slice
    points = [(x, y) for x, y, _ in samples]
    durations = [float(duration) for _, _, duration in samples]
    slices = [index for index, count in enumerate(per_map) for _ in range(count)]
    return points, durations, slices


def _sample_fixed_length(volume, n_fix):
    """Ours: ``n_fix`` ex-Gaussian durations; each fixation uses the map of its start time.
    Positions come from the official ``sample_slice``, map by map as ``sample_volume`` does."""
    durations = _exgaussian_durations(n_fix)
    starts = np.r_[0., np.cumsum(durations)[:-1]]
    slices = np.searchsorted(BOUNDARIES, starts, side='right').tolist()
    samples = []
    for index in range(3):
        samples = official_s2()['sample_slice'](volume[index], GRID, n_samples=slices.count(index),
                                                use_heuristic=True, samples=samples)
    # Integer grid positions, as sample_volume returns them.
    points = [(int(x), int(y)) for x, y in samples]
    return points, durations, slices


def _exgaussian_durations(n_fix):
    """Positive ex-Gaussian durations in seconds (published MASSVIS fit); nonpositive draws are redrawn."""
    mu, std, tau = DURATION_MS
    durations = []
    while len(durations) < n_fix:
        duration = (np.random.normal(mu, std) + np.random.exponential(tau)) / 1000
        if duration > 0:
            durations.append(float(duration))
    return durations

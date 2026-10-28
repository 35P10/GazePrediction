"""UMSS temporal-map validation and scanpath sampling."""
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
OFFICIAL_MODEL = PACKAGE_ROOT / 'training' / 'official_reference'
DATASET = 'doi:10.18419/DARUS-3361'
VERSION = '2.0'
GRID = (480, 640)
BOUNDARIES = (.5, 3.)
SLICE_LABELS = ('0-0.5s', '0.5-3s', '3-5s')
DURATION_MS = (124.06, 17.49, 89.37)  # normal mean, normal std, exponential mean
# UMSS-S2 (n_fix=None): verbatim from DaRUS Code/scripts/UMSS/saltinet_util.py, including its 0.55 s bin.
S2_FIXATION_COUNTS = tuple(range(2, 36))
S2_FIXATION_COUNT_P = (0.000458, 0.000458, 0.000153, 0.000305, 0.000458, 0.000763, 0.001069, 0.001527, 0.001832,
                       0.002290, 0.002748, 0.005038, 0.007328, 0.011298, 0.015878, 0.029008, 0.043511, 0.062290,
                       0.085344, 0.093130, 0.109160, 0.119542, 0.105038, 0.095878, 0.076947, 0.056947, 0.032366,
                       0.020458, 0.010534, 0.004275, 0.002595, 0.000763, 0.000458, 0.000153)
S2_DURATIONS_S = (0.11, 0.125, 0.55, 0.185, 0.215, 0.245, 0.28, 0.325, 0.375, 0.45, 0.6, 0.85, 1.25, 1.75, 2.25)
S2_DURATION_P = (0.022423, 0.141273, 0.207362, 0.181547, 0.132462, 0.097051, 0.081960, 0.056275, 0.031451,
                 0.029280, 0.015654, 0.002686, 0.000522, 0.000027, 0.000027)


def validate_volume(volume):
    import numpy as np
    volume = np.asarray(volume, dtype=float)
    if volume.shape != (3,) + GRID or not np.isfinite(volume).all() or (volume < 0).any():
        raise ValueError('UMSS requires three finite nonnegative 480x640 maps in image coordinates')
    if (volume.sum(axis=(1, 2)) <= 0).any():
        raise ValueError('Every UMSS temporal map must have positive mass')
    return volume


def sample_scanpath(volume, *, n_fix=8, seed=0, sigma=200.):
    """Temporal allocation, half-maximum threshold and local Gaussian attention.

    ``n_fix`` fixations: positive ex-Gaussian durations (published MASSVIS fit), slice chosen by the elapsed time
    before each fixation. ``n_fix=None``: UMSS-S2's protocol, with length and durations drawn from the released
    discrete distributions and the slice switching at the first fixation whose end passes 0.5 s / 3 s.
    No artificial center is inserted.
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    if (n_fix is not None and (type(n_fix) is not int or n_fix < 1)) or type(seed) is not int or seed < 0:
        raise ValueError('Positive integer (or None) n_fix and nonnegative integer seed required')
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('Positive finite sigma required')
    volume = validate_volume(volume).copy()
    volume[volume < volume.max(axis=(1, 2), keepdims=True) / 2] = 0
    rng = np.random.default_rng(seed)
    if n_fix is None:
        count = int(rng.choice(S2_FIXATION_COUNTS, p=S2_FIXATION_COUNT_P))
        durations = [float(d) for d in rng.choice(S2_DURATIONS_S, size=count, p=S2_DURATION_P)]
        slices = np.searchsorted(BOUNDARIES, np.cumsum(durations), side='left')
    else:
        mu, std, tau = DURATION_MS
        durations = []
        for _ in range(n_fix):
            duration = -1.
            while duration <= 0:
                duration = (rng.normal(mu, std) + rng.exponential(tau)) / 1000
            durations.append(float(duration))
        starts = np.r_[0., np.cumsum(durations)[:-1]]
        slices = np.searchsorted(BOUNDARIES, starts, side='right')
    points = []
    for index, slice_index in enumerate(slices):
        probability = volume[slice_index].copy()
        if index:
            x, y = points[-1]
            # Separable equivalent of gaussian_filter(2-D impulse, sigma).
            row, column = np.zeros(GRID[0]), np.zeros(GRID[1])
            row[y], column[x] = 1, 1
            probability *= np.outer(gaussian_filter1d(row, sigma), gaussian_filter1d(column, sigma))
        total = probability.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError('UMSS foveal mask has no overlap with the temporal map')
        choice = int(rng.choice(probability.size, p=(probability / total).ravel()))
        y, x = divmod(choice, GRID[1])
        points.append((x, y))
    return dict(grid_points=points, durations=durations, slices=slices.tolist())

"""Scanpath metrics of Chartist's Table 1 (Shi et al., CHI 2025).

The paper releases no evaluation code. These functions follow the scanpath metrics of the
authors' group (UMSS repository, Code/scripts/metrics/) and were fitted to reproduce the
paper's Human column on the 183 TaskVis scanpaths:

- Fixations, LEV mean/best and DTW mean match to the published rounding; DTW best too,
  except retrieve_value (17692.5 vs 17692, 0.003%).
- DTW mean uses the compiled fastdtw (the package default) and DTW best its pure-Python
  implementation; each one reproduces only its own column, so they were computed separately.
- Human means include each scanpath compared with itself (DTW 0, LEV 0, SS 1); human best
  values exclude that pair.
- Sequence Score uses the official chart labels (title/axis/mark, Z outside). The paper's
  AOIs are not published and no tested AOI set reproduces it exactly (1-3% below).
"""
import json

import numpy as np
import pandas as pd
from fastdtw._fastdtw import fastdtw as fastdtw_compiled
from fastdtw.fastdtw import fastdtw as fastdtw_python
from scipy.spatial.distance import euclidean

METRICS = ['Fixations', 'SS mean', 'SS best', 'LEV mean', 'LEV best', 'DTW mean', 'DTW best']
OUTSIDE = 'Z'


def dtw_for_mean(p, q):
    return fastdtw_compiled(p, q, dist=euclidean)[0]


def dtw_for_best(p, q):
    return fastdtw_python(p, q, dist=euclidean)[0]


def levenshtein(a, b):
    """editdistance.eval."""
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def grid_string(path, height, width, xbins=12, ybins=8):
    """UMSS scanpath_to_string with Tbins=0: two characters per fixation."""
    height_step, width_step = height // ybins, width // xbins
    return ''.join(chr(97 + (height - int(y)) // height_step) + chr(65 + int(x) // width_step) for x, y in path)


def nw_matching(a, b):
    """UMSS nw_matching: Needleman-Wunsch, zero-one similarity, gap 0, divided by the longer string."""
    previous = [0.0] * (len(b) + 1)
    for ca in a:
        current = [0.0]
        for j, cb in enumerate(b, 1):
            current.append(max(previous[j - 1] + (ca == cb), previous[j], current[j - 1]))
        previous = current
    return previous[-1] / max(len(a), len(b))


def load_aois(labels_json):
    """Rectangles of an official labelme chart file, as (letter, left, top, right, bottom)."""
    boxes = []
    for shape in json.loads(open(labels_json).read())['shapes']:
        (x0, y0), (x1, y1) = shape['points']
        boxes.append((shape['label'][0].upper(), min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
    return boxes


def aoi_string(path, aois):
    """One letter per fixation: the first AOI that contains it, or Z."""
    return ''.join(next((letter for letter, x0, y0, x1, y1 in aois if x0 <= int(x) <= x1 and y0 <= int(y) <= y1),
                        OUTSIDE) for x, y in path)


def evaluate(predictions, references, cases, aois, *, self_pairs=False):
    """Table 1 metrics per task, in original-image pixels.

    predictions, references: {case_id: [array (n, 2) of x, y]}.
    cases: DataFrame with id, task, width, height and image; aois: {image: load_aois(...)}.
    self_pairs=True when predictions are the references (human column): prediction i and
    reference i are the same scanpath, counted in the means and left out of best.
    """
    pairs, counts = [], []
    for case in cases.itertuples():
        refs = references[case.id]
        ref_grid = [grid_string(r, case.height, case.width) for r in refs]
        ref_aoi = [aoi_string(r, aois[case.image]) for r in refs]
        for i, path in enumerate(predictions.get(case.id, [])):
            if len(path) == 0:
                continue
            counts.append((case.task, len(path)))
            grid, aoi = grid_string(path, case.height, case.width), aoi_string(path, aois[case.image])
            for j, ref in enumerate(refs):
                same = self_pairs and i == j
                pairs.append(dict(task=case.task, pred=(case.id, i), same=same,
                                  SS=1.0 if same else nw_matching(aoi, ref_aoi[j]),
                                  LEV=0 if same else levenshtein(grid, ref_grid[j]),
                                  DTW_mean=0.0 if same else dtw_for_mean(path, ref),
                                  DTW_best=np.nan if same else dtw_for_best(path, ref)))
    pairs = pd.DataFrame(pairs)
    others = pairs[~pairs.same]
    best = others.groupby(['task', 'pred']).agg(SS=('SS', 'max'), LEV=('LEV', 'min'), DTW=('DTW_best', 'min'))
    best = best.groupby('task').mean()
    mean = pairs.groupby('task')[['SS', 'LEV', 'DTW_mean']].mean()
    fixations = pd.DataFrame(counts, columns=['task', 'n']).groupby('task').n.mean()
    table = pd.DataFrame({'Fixations': fixations, 'SS mean': mean.SS, 'SS best': best.SS,
                          'LEV mean': mean.LEV, 'LEV best': best.LEV,
                          'DTW mean': mean.DTW_mean, 'DTW best': best.DTW})
    return table[METRICS].stack().rename_axis(['task', 'metric'])

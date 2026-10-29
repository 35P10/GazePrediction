"""UMSS on MASSVIS: do S1 (Table 2, saliency) and S2 (Table 1, scanpaths) give the paper's results?

Each table compares the paper, the authors' released outputs and ours, scored with the same evaluation.
The evaluation uses the authors' code from ``official_repository/`` (the DaRUS V2 deposit). Two references
the deposit does not contain are rebuilt from MASSVIS: the 5 s human scanpaths and the accumulated
human saliency maps. ScanMatch (MATLAB) is not computed.
"""
import contextlib
import io
import json
import os
import sys
import types
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from PIL import Image

from models.umss import UMSS
from models.umss.evaluation import saliency as S

EXPERIMENT = Path(__file__).resolve().parent
DATA = EXPERIMENT / 'data'
REFERENCE = EXPERIMENT / 'official_repository'
RELEASED = REFERENCE / 'Data/UMSS/UMSS'
IMAGE_ROOT = REFERENCE / 'Data/UMSS/GT_EFD_500_2000_5000'
OUR_MAPS = DATA / 'ours/npz'
SOURCE = json.loads((EXPERIMENT / 'source.json').read_text())
PAPER_TABLE2 = SOURCE['paper_values']['table2']
PAPER_TABLE1 = SOURCE['paper_values']['table1']
SALIENCY_METRICS = ('NSS', 'CC', 'KL', 'SIM')
SCANPATH_METRICS = ('SS_mean', 'SS_best', 'sTDE_mean', 'sTDE_best', 'DTW_mean', 'DTW_best',
                    'MM_shape', 'MM_direction', 'MM_length', 'MM_position', 'MM_duration')


# ---------------------------------------------------------------- data

def download():
    """Unzip the evaluation images and released outputs from the mirror and fetch MASSVIS fixations and
    element labels. Restore the mirror first with ``sync_official_repository.py``."""
    _unzip(REFERENCE / 'Data/UMSS/GT_EFD_500_2000_5000.zip', REFERENCE / 'Data/UMSS')
    _unzip(REFERENCE / 'Data/UMSS/UMSS.zip', REFERENCE / 'Data/UMSS')
    for item in SOURCE['github']:
        archive = DATA / 'massvis' / Path(item['url']).name
        if not archive.exists():
            archive.parent.mkdir(parents=True, exist_ok=True)
            urlretrieve(item['url'], archive)
        _unzip(archive, DATA / 'massvis')


def _unzip(archive, target):
    done = target / ('.extracted_' + Path(archive).name)
    if not done.exists():
        with zipfile.ZipFile(archive) as z:
            z.extractall(target, [name for name in z.namelist() if not name.startswith('__MACOSX/')
                                  and not os.path.basename(name).startswith('._')])
        done.touch()


def images():
    """All 393 MASSVIS charts and the 66 evaluation ones, in the authors' order (natural sort, every sixth)."""
    from natsort import natsorted
    paths = {p.stem: p for split in ('train', 'val') for p in (IMAGE_ROOT / split / 'raw_img').glob('*.png')}
    names = natsorted(paths)
    return names, names[::6], paths


def image_size(path):
    with Image.open(path) as image:
        return image.size


def encoding_files(name):
    """One csv per viewer: index, x, y, duration (ms) of each fixation while encoding the chart."""
    return sorted((DATA / 'massvis/fixationsByVis' / name / 'enc').glob('*.csv'))


def read_fixations(path):
    return pd.read_csv(path, header=None, names=['index', 'x', 'y', 'time'])


def truncate(fixations, limit_ms=5000):
    """5 s human scanpath (not in the deposit): fixations whose cumulative duration, including their own,
    stays below ``limit_ms``, the rule of the authors' time-bucketing code."""
    return fixations[np.cumsum(fixations.time.astype(float).astype(int).to_numpy()) < limit_ms]


def official():
    """The authors' scanpath metric modules, imported from the mirror.

    ``metrics.py`` imports ``scipy.misc.imresize`` (removed from SciPy) and ``editdistance``, and ``multimatch.py``
    calls ``np.asscalar`` (removed from NumPy); none matters for the functions used here, so they are stubbed.
    """
    import scipy.misc
    if not hasattr(scipy.misc, 'imresize'):
        scipy.misc.imresize = None
    sys.modules.setdefault('editdistance', types.ModuleType('editdistance'))
    if not hasattr(np, 'asscalar'):
        np.asscalar = lambda a: a.item()
    code = REFERENCE / 'Code/scripts'
    # metrics.py must come before the deposit's metrics/ package.
    for folder in ('', 'multimatch', 'metrics'):
        if str(code / folder) not in sys.path:
            sys.path.insert(0, str(code / folder))
    with contextlib.redirect_stdout(io.StringIO()):  # metrics.py prints a MATLAB notice on import
        import metrics, multimatch, hungarian, preprocess_utils
    preprocess_utils.eledir = str(DATA / 'massvis/elementLabels') + '/'
    return types.SimpleNamespace(metrics=metrics, multimatch=multimatch, hungarian=hungarian,
                                 preprocess=preprocess_utils)


# ---------------------------------------------------------------- S1: Table 2

def s1(maps_dir=OUR_MAPS):
    """Table 2: NSS, CC, KL and SIM at 3 s and 5 s for the paper, the released maps and ours.

    As in the authors' evaluation, the 0.5-3 s and 3-5 s maps are scored against all human fixations
    up to 3 s and 5 s (Gaussian sigma 19 px). Ours are the chart-size PNGs the export writes with the
    authors' predict_and_save_md (``<maps_dir>/png/``), like the released maps.
    """
    _, names, paths = images()
    rows = []
    for name in names:
        width, height = image_size(paths[name])
        for time in (3000, 5000):
            fixations, heatmap = S.make_fixmap_and_heatmap(
                REFERENCE, width, height, S.accumulated_coords(encoding_files(name), time))
            predictions = {'Released maps': read_png(RELEASED / 'salmaps' / f'{name}_{time}.png'),
                           'Ours': our_chart_map(maps_dir, name, paths[name], time)}
            for source, prediction in predictions.items():
                scores = S.score_saliency(REFERENCE, prediction, np.asarray(heatmap), np.asarray(fixations))
                rows.append(dict(source=source, duration=f'{time // 1000}s', **scores))
    means = pd.DataFrame(rows).groupby(['duration', 'source'])[list(SALIENCY_METRICS)].mean().unstack()
    paper = pd.DataFrame(PAPER_TABLE2).T[list(SALIENCY_METRICS)]
    table = {}
    for duration in ('3s', '5s'):
        for metric in SALIENCY_METRICS:
            table[(duration, metric)] = {'Paper': paper.loc[duration, metric],
                                         'Released maps': means.loc[duration, (metric, 'Released maps')],
                                         'Ours': means.loc[duration, (metric, 'Ours')]}
    return pd.DataFrame(table).T.rename_axis(['duration', 'metric'])


def published_saliency():
    """The authors' per-image MD-EAM scores (DaRUS saliency/csv), in evaluation-image order."""
    folder = REFERENCE / 'Code/scripts/saliency/csv'
    return {duration: pd.read_csv(folder / f'mdeam_{duration}.tab') for duration in ('3s', '5s')}


def read_png(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('L'))


def our_chart_map(maps_dir, name, image_path, time):
    """Our map for ``time`` at the chart size: the exported PNG, or, for older exports without ``png/``,
    the grid map resized (bilinear) to the chart and clipped to 8 bits."""
    png = Path(maps_dir) / 'png' / f'{name}_{time}.png'
    if png.exists():
        return read_png(png)
    grid_map = UMSS(maps_dir=maps_dir).predict_maps(image_path)['maps'][{3000: 1, 5000: 2}[time]]
    resized = np.asarray(Image.fromarray(grid_map.astype(np.float32)).resize(image_size(image_path), Image.BILINEAR))
    return np.uint8(np.clip(resized, 0, 1) * 255)


# ---------------------------------------------------------------- S2: Table 1

BUCKETS = ['A', 'X', 'G', 'L', 'O', 'T', 'S', 'D', 'Z']
STRING_SUBJECTS = ['abl', 'cmb', 'lcm', 'ls', 'mg', 'pi', 'rb', 'rk', 'sok', 'wab']
EXGAUSS_MS = (5.114058378644653, 124.0602219896999, 17.48864970980731)  # mm_massvis_*: K, loc, scale


def s2(maps_dir=OUR_MAPS, *, seed=0, workers=None):
    """Table 1 (without ScanMatch) for the paper, the released scanpaths and ours.

    Ours are 40 scanpaths per image from ``UMSS.run(n_fix=None)`` (official UMSS-S2 sampler); run r uses
    seed + r. Images are processed in parallel; sampling with the official filter takes a while.
    """
    from concurrent.futures import ProcessPoolExecutor
    _, names, paths = images()
    jobs = [(name, paths[name], maps_dir, seed) for name in names]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_s2_image, jobs))
    released = summarize_scanpaths([result['released'] for result in results])
    ours = summarize_scanpaths([result['ours'] for result in results])
    return pd.DataFrame({'Paper': {metric: PAPER_TABLE1[metric] for metric in SCANPATH_METRICS},
                         'Released scanpaths': released, 'Ours': ours}).rename_axis('metric')


def _s2_image(job):
    name, path, maps_dir, seed = job
    o = official()
    size = image_size(path)
    released = read_scanpaths(RELEASED / 'scanpaths_5s/csv' / f'{name}.png.csv')
    ours = UMSS(maps_dir=maps_dir).run([{'image_id': name, 'image_path': path}],
                                       n_runs=40, n_fix=None, seed=seed)['scanpaths']
    # Same layout as the released csv: user, index, time (s), x, y on the 640x480 grid.
    ours = pd.DataFrame({'user': ours.run + 1, 'index': ours.fix_idx + 1, 'time': ours.duration,
                         'x': ours.grid_x, 'y': ours.grid_y})
    return dict(released=scanpath_metrics(o, name, size, released, seed=seed),
                ours=scanpath_metrics(o, name, size, ours, seed=seed))


def read_scanpaths(path):
    """Released UMSS-S2 csv: user, index, time (s), x, y on the 640x480 grid."""
    frame = pd.read_csv(path)
    frame.columns = ['user', 'index', 'time', 'x', 'y']
    return frame


def scanpath_metrics(o, name, size, scanpaths, *, seed=0):
    """Table 1 metrics of one image: per-pair values and per-image bests.

    As in the authors' scripts: the n human (5 s) scanpaths are compared with predicted users 1..n; DTW and
    MultiMatch keep predictions on the 640x480 grid, sTDE rescales them to image pixels, and the Sequence
    Score compares element strings of all 40 predictions with the human ones.
    """
    from scipy.stats import exponnorm
    width, height = size
    humans = [truncate(read_fixations(path)) for path in encoding_files(name)]
    n = len(humans)
    image = np.zeros((height, width))  # scaled_TDE only reads its size
    grid = {user: group[['x', 'y']].to_numpy(float) for user, group in scanpaths.groupby('user')}
    scale = np.array([width / 640, height / 480])
    rng = np.random.RandomState(seed)
    dtw, tde, mm_pairs, cost = [], [], [[] for _ in range(n)], np.zeros((n, n))
    for g, human in enumerate(humans):
        xy = human[['x', 'y']].to_numpy(float)
        xyt = human[['x', 'y', 'time']].to_numpy(float)
        for user in range(1, n + 1):
            predicted = grid[user]
            dtw.append(o.metrics.DTW(predicted, xy))
            tde.append(o.metrics.scaled_TDE(predicted * scale, xy, image))
            # Predicted durations for MultiMatch are drawn from the authors' ex-Gaussian fit.
            times = exponnorm.rvs(EXGAUSS_MS[0], loc=EXGAUSS_MS[1], scale=EXGAUSS_MS[2], size=len(predicted),
                                  random_state=rng)
            mm = o.multimatch.docomparison(np.c_[predicted, times], xyt, sz=[width, height])
            cost[user - 1][g] = 1 - np.sum(mm[0]) / 5
            mm_pairs[g].append(mm[0])
    dtw = np.array(dtw).reshape(n, n)
    tde = np.array(tde).reshape(n, n)
    # MultiMatch pairs are matched one to one (Hungarian); NaN (human path under 3 fixations) gets the worst cost.
    hungarian = o.hungarian.Hungarian(np.nan_to_num(cost, nan=1.0))
    hungarian.calculate()
    multimatch = [mm_pairs[a][b] for a, b in hungarian.get_results()]
    humans_strings = human_strings(o, name, size)
    ss = [o.metrics.nw_matching(gt, string) for string in scanpath_strings(o, name, size, scanpaths).values()
          for gt in humans_strings]
    return dict(dtw=dtw.ravel().tolist(), dtw_best=dtw.min(axis=0).tolist(),
                stde=tde.ravel().tolist(), stde_best=tde.max(axis=0).tolist(),
                multimatch=np.asarray(multimatch, float).tolist(), ss=ss, ss_best=[max(ss)] if ss else [])


def summarize_scanpaths(per_image):
    """Table 1 row: mean over all pairs; best: mean of per-image bests. MultiMatch pairs with NaN are dropped."""
    pairs = lambda key: np.concatenate([np.asarray(r[key], float).reshape(-1) for r in per_image])
    mm = np.concatenate([np.asarray(r['multimatch'], float).reshape(-1, 5) for r in per_image])
    mm = mm[~np.isnan(mm).any(axis=1)]
    return dict(SS_mean=pairs('ss').mean(), SS_best=pairs('ss_best').mean(),
                sTDE_mean=pairs('stde').mean(), sTDE_best=pairs('stde_best').mean(),
                DTW_mean=pairs('dtw').mean(), DTW_best=pairs('dtw_best').mean(),
                MM_shape=mm[:, 0].mean(), MM_direction=mm[:, 1].mean(), MM_length=mm[:, 2].mean(),
                MM_position=mm[:, 3].mean(), MM_duration=mm[:, 4].mean())


def element_map(o, name, size, *, human):
    """Element-class map (width x height, 9 classes), as the authors build it.

    Human strings map every element with the 9-class names; prediction strings (3_tsv2str.py) use
    get_gt_elements(simple=True), whose last element keeps its 14-class id.
    """
    _, _, _, boxes = o.preprocess.get_gt_elements(name, simple=True)
    if human:
        for box in boxes:
            box.id = o.preprocess.get_bboxid_by_name_9(box.name)
    return o.preprocess.get_idmap(boxes, size[0], size[1], 9)


def _accum_bucket(accum_time):
    """get_accum_bucket of fixation_individual_analysis.ipynb: 1 s buckets, -1 from 9 s on."""
    for index, limit in enumerate(range(1000, 10000, 1000)):
        if int(accum_time) < limit:
            return index
    return -1


def human_strings(o, name, size):
    """5 s element strings of the ten string viewers (fixation_individual_analysis.ipynb): skip the first
    two fixations and keep buckets below 5 (as written, this also keeps fixations after 9 s)."""
    id_map = element_map(o, name, size, human=True)
    strings = []
    for subject in STRING_SUBJECTS:
        path = DATA / 'massvis/fixationsByVis' / name / 'enc' / f'{subject}.csv'
        if not path.exists():
            continue
        text, accum = '', 0
        for index, row in read_fixations(path).iterrows():
            accum += row['time']
            if index >= 2 and _accum_bucket(accum) < 5:
                text += BUCKETS[id_map[int(row['x'])][int(row['y'])]]
        strings.append(text)
    return strings


def scanpath_strings(o, name, size, scanpaths):
    """3_tsv2str.py: per predicted scanpath, cut at EOS, keep the first 60 % of fixations, grid to pixels."""
    id_map = element_map(o, name, size, human=False)
    width, height = size
    strings = {}
    for user, path in scanpaths.groupby('user'):
        fixations = o.preprocess.truncate_EOS(path[['x', 'y', 'time']])
        strings[user] = ''.join(
            BUCKETS[id_map[int(int(fixations[i][0]) / 640 * width)][int(int(fixations[i][1]) / 480 * height)]]
            for i in range(int(fixations.shape[0] * 0.6)))
    return strings

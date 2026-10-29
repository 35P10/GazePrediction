"""Build the UMSS stage-1 training set (MASSVIS-MDEFD) from the pinned downloads.

Targets per branch, as in the paper (Sec. 5.2): the 0-0.5 s branch keeps an ordinary
saliency map (Gaussian-blurred fixations) to learn the first-fixation centre bias; the
0.5-3 s and 3-5 s branches use element fixation density (EFD) maps.

``training/data_preparation.py`` loads EFD definitions from the original
``generate_Massvis-EFD.ipynb``. Shared ``evaluation/saliency.py`` provides the
original fixation-map and heatmap function. Only directory orchestration, time bucketing and
compatibility adaptations remain here.

The train/val split is the one of the released `GT_EFD_500_2000_5000` (327 / 66 images),
which is the paper's rule: natural sort by name, every sixth image to evaluation.

`--check` regenerates the released 500/2000/5000/10000 maps and compares them pixel by pixel,
which verifies the official functions and input data before training.
"""
import argparse
import csv
import os
from glob import glob

import numpy as np
from PIL import Image
from pathlib import Path
import sys

if __package__:
    from models.umss.evaluation import saliency as S
    from models.umss.training import data_preparation as D
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from evaluation import saliency as S
    from training import data_preparation as D

TIMES = [500, 3000, 5000]
CHECK_TIMES = [500, 2000, 5000, 10000]
SPLITS = ['train', 'val']


def bucket_coords(csv_files, times, t):
    """Official: same loop as generate_Massvis-EFD.ipynb; a fixation belongs to the bucket where its
    cumulative end time falls. Returns the bucket's coordinates and the accumulated ones."""
    allcoords, allcoords_accum = [], []
    mintime = 0 if t == 0 else times[t - 1]
    for subcsv in csv_files:
        with open(subcsv) as subcsvfile:
            timestamp = 0
            for row in csv.reader(subcsvfile, delimiter=','):
                timestamp += int(float(row[3]))
                if mintime <= timestamp < times[t]:
                    allcoords.append([int(float(row[1])), int(float(row[2]))])
                if timestamp < times[t]:
                    allcoords_accum.append([int(float(row[1])), int(float(row[2]))])
    return allcoords, allcoords_accum


def build(efd_root, fixations, annots, savedir, times, with_fixmaps=True, repository=None):
    """``with_fixmaps=False`` (--check) writes only the released EFD layout."""
    repository = repository or S.DEFAULT_DESTINATION
    for split in SPLITS:
        raw_src = os.path.join(efd_root, split, 'raw_img')
        names = sorted(os.path.splitext(f)[0] for f in os.listdir(raw_src) if f.endswith('.png'))
        out = os.path.join(savedir, split)
        kinds = ['heatmaps', 'heatmaps_accum'] + (['fix_maps', 'fix_maps_accum'] if with_fixmaps else [])
        for kind in kinds:
            for time in times:
                os.makedirs(os.path.join(out, kind, str(time)), exist_ok=True)
        os.makedirs(os.path.join(out, 'raw_img'), exist_ok=True)
        for count, imname in enumerate(names):
            if count % 50 == 0:
                print('%s: image %d/%d' % (split, count, len(names)), flush=True)
            with Image.open(os.path.join(raw_src, imname + '.png')) as im:
                w, h = im.size
                im.save(os.path.join(out, 'raw_img', imname + '.png'))
            csv_files = glob(os.path.join(fixations, imname, 'enc', '*.csv'))
            if not csv_files:
                raise FileNotFoundError('No MASSVIS encoding fixations for ' + imname)

            def save(image, kind, time):
                image.save(os.path.join(out, kind, str(time), imname + '.png'))

            for t, time in enumerate(times):
                coords, coords_accum = bucket_coords(csv_files, times, t)
                if with_fixmaps:
                    fix, sal = S.make_fixmap_and_heatmap(repository, w, h, coords)
                    fix_accum, _ = S.make_fixmap_and_heatmap(repository, w, h, coords_accum)
                    save(fix, 'fix_maps', time)
                    save(fix_accum, 'fix_maps_accum', time)
                if with_fixmaps and t == 0:
                    # The 0-0.5 s branch learns the first-fixation centre bias from an ordinary saliency map.
                    save(sal, 'heatmaps', time)
                    save(sal, 'heatmaps_accum', time)
                else:
                    save(D.make_efdmap(repository, w, h, coords, imname, annots), 'heatmaps', time)
                    save(D.make_efdmap(repository, w, h, coords_accum, imname, annots), 'heatmaps_accum', time)
        # load_datasets_multiduration lists `<dataset>/test`; the S1 notebook never uses it.
        os.makedirs(os.path.join(savedir, 'test'), exist_ok=True)


def check(efd_root, regenerated):
    total = exact = 0
    worst = 0
    for split in SPLITS:
        for kind in ['heatmaps', 'heatmaps_accum']:
            for time in CHECK_TIMES:
                folder = os.path.join(efd_root, split, kind, str(time))
                for f in sorted(os.listdir(folder)):
                    a = np.asarray(Image.open(os.path.join(folder, f)), dtype=np.int16)
                    b = np.asarray(Image.open(os.path.join(regenerated, split, kind, str(time), f)), dtype=np.int16)
                    total += 1
                    exact += a.shape == b.shape and np.array_equal(a, b)
                    if a.shape == b.shape:
                        worst = max(worst, int(np.abs(a - b).max()))
    print('Released maps reproduced exactly: %d / %d (max abs diff %d)' % (exact, total, worst))
    if exact != total:
        raise SystemExit('The regenerated EFD maps differ from the released ones; do not train on them.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--work', default='/work')
    parser.add_argument('--check', action='store_true', help='verify the official functions against the released maps')
    parser.add_argument('--times', type=int, nargs=3, default=TIMES, metavar='MS',
                        help='interval ends in ms (default 500 3000 5000, as in the S1 notebook)')
    args = parser.parse_args()
    repository = os.path.join(args.work, 'upstream')
    downloads = os.path.join(args.work, 'downloads')
    efd_root = os.path.join(downloads, 'GT_EFD_500_2000_5000')
    fixations = os.path.join(downloads, 'massvis', 'fixationsByVis')
    annots = os.path.join(downloads, 'massvis', 'elementLabels')
    if args.check:
        regenerated = os.path.join(args.work, 'datasets', 'check_500_2000_5000_10000')
        build(efd_root, fixations, annots, regenerated, CHECK_TIMES, with_fixmaps=False, repository=repository)
        check(efd_root, regenerated)
    else:
        savedir = os.path.join(args.work, 'datasets', 'massvis_bucket_' + '_'.join(map(str, args.times)))
        build(efd_root, fixations, annots, savedir, args.times, repository=repository)
        print('Dataset ready:', savedir)

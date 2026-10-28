"""Evaluate the trained MD-EAM against the authors' released UMSS maps, and export maps for new charts.

evaluate: reports the paper's Table 2 metrics (NSS, CC, KL, SIM) at 3 and 5 seconds
          against accumulated human fixations, for both our checkpoint and the
          released maps. Uses the shared verified official functions.
export:   writes `<stem>.npz` files in the format `models.umss.UMSS(maps_dir=...)` reads,
          so sampling can run on any machine without TensorFlow.

Inference mirrors models/umss/model.py: md_sem with one output, upstream pre/post-processing.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

if __package__:
    from models.umss.evaluation import saliency as S
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from evaluation import saliency as S

TIMES = [500, 3000, 5000]
GRID = (480, 640)
SALIENCY_METRICS = ('NSS', 'CC', 'KL', 'SIM')


def load_model(work, checkpoint):
    code = os.path.join(work, 'upstream', 'Code', 'scripts')
    sys.path[:0] = [os.path.join(code, 'src'), os.path.join(code, 'saliency')]
    from multiduration_models import md_sem
    from sal_imp_utilities import preprocess_images, postprocess_predictions
    model = md_sem(input_shape=(240, 320, 3), nb_timestep=3, n_outs=1, verbose=False, print_shapes=False)
    model.load_weights(checkpoint)
    return model, preprocess_images, postprocess_predictions


def predict(model, preprocess, postprocess, path, height, width):
    predictions = model.predict(preprocess([str(path)], 240, 320))
    if isinstance(predictions, list):
        predictions = predictions[0]
    return [postprocess(predictions[0, t, :, :, 0], height, width) for t in range(len(TIMES))]


def evaluate(args):
    """Score saved 8-bit predictions with the same protocol as the reproduction."""
    from PIL import Image
    model, pre, post = load_model(args.work, args.checkpoint)
    work = Path(args.work)
    repository = work / 'upstream'
    downloads = work / 'downloads'
    images = sorted((downloads / 'GT_EFD_500_2000_5000/val/raw_img').glob('*.png'))
    if not images:
        raise FileNotFoundError('No validation images; run the download stage first')
    humans = downloads / 'massvis/fixationsByVis'
    official = downloads / 'official/UMSS/salmaps'
    out = work / 'predictions/val'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for image in images:
        with Image.open(image) as im:
            width, height = im.size
        maps = predict(model, pre, post, image, height, width)
        png_maps = {}
        for time, prediction in zip(TIMES, maps):
            png_maps[time] = np.uint8(np.clip(prediction, 0, 1) * 255)
            Image.fromarray(png_maps[time]).save(out / f'{image.stem}_{time}.png')
        files = sorted((humans / image.stem / 'enc').glob('*.csv'))
        if not files:
            raise FileNotFoundError('No human encoding fixations for ' + image.stem)
        for time in (3000, 5000):
            coords = S.accumulated_coords(files, time)
            fix, heat = S.make_fixmap_and_heatmap(repository, width, height, coords)
            with Image.open(official / f'{image.stem}_{time}.png') as ref:
                author_map = np.asarray(ref.convert('L'))
            if author_map.shape != (height, width):
                raise ValueError('Released map dimensions differ for ' + image.stem)
            for source, prediction in [('ours', png_maps[time]), ('official', author_map)]:
                scores = S.score_saliency(repository, prediction, np.asarray(heat), np.asarray(fix))
                rows.append(dict(image=image.stem, time=time, source=source, **scores))
    report = work / 'predictions/paper_saliency_per_image.csv'
    with report.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        str(time): {
            source: {metric: float(np.mean([row[metric] for row in rows
                                           if row['time'] == time and row['source'] == source]))
                     for metric in SALIENCY_METRICS}
            for source in ('ours', 'official')}
        for time in (3000, 5000)}
    metadata = dict(protocol='umss-table2-accumulated-human-3s-5s-v1',
                    images=len(images), sigma_px=S.SIGMA, prediction_format='8-bit grayscale PNG',
                    checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                    source_manifest_sha256=hashlib.sha256((repository / 'sources.json').read_bytes()).hexdigest(),
                    means=summary)
    (work / 'predictions/paper_saliency_summary.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('Paper Table 2 protocol: accumulated human references, sigma=19 px')
    print('%-6s %-10s %9s %9s %9s %9s' % ('t(ms)', 'source', *SALIENCY_METRICS))
    for time, sources in summary.items():
        for source, scores in sources.items():
            print('%-6s %-10s %9.6f %9.6f %9.6f %9.6f' %
                  (time, source, *(scores[metric] for metric in SALIENCY_METRICS)))
    print('Details:', report)


def export(args):
    import cv2
    from PIL import Image
    model, pre, post = load_model(args.work, args.checkpoint)
    out = Path(args.out or os.path.join(args.work, 'predictions', 'npz'))
    out.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg'))
    for image in images:
        with Image.open(image) as im:
            width, height = im.size
        maps = predict(model, pre, post, image, height, width)
        volume = np.stack([cv2.resize(m, (GRID[1], GRID[0])) for m in maps]).astype(np.float32)
        np.savez_compressed(out / (image.stem + '.npz'), maps=volume, width=width, height=height,
                            image_sha256=hashlib.sha256(image.read_bytes()).hexdigest())
    import keras
    import tensorflow as tf
    import platform
    metadata = dict(checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                    source_manifest_sha256=hashlib.sha256((Path(args.work)/'upstream/sources.json').read_bytes()).hexdigest(),
                    tensorflow=tf.__version__, keras=keras.__version__, numpy=np.__version__,
                    python=platform.python_version(), platform=platform.platform(),
                    image_sha256={image.stem:hashlib.sha256(image.read_bytes()).hexdigest() for image in images},
                    grid=list(GRID), input_shape=[240,320,3], preprocessing='upstream preprocess_images; postprocess_predictions; cv2 grid resize')
    (out/'export_metadata.json').write_text(__import__('json').dumps(metadata,indent=2)+'\n')
    print('Exported %d map volumes to %s' % (len(images), out))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--work', default='/work')
    parser.add_argument('--checkpoint', default=None, help='default: <work>/ckpt/md_eam.hdf5')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('evaluate')
    p = sub.add_parser('export')
    p.add_argument('--images', required=True, help='folder with chart images')
    p.add_argument('--out', default=None, help='default: <work>/predictions/npz')
    args = parser.parse_args()
    args.checkpoint = args.checkpoint or os.path.join(args.work, 'ckpt', 'md_eam.hdf5')
    {'evaluate': evaluate, 'export': export}[args.command](args)

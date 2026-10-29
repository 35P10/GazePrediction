"""Evaluate the trained MD-EAM against the authors' released UMSS maps, and export maps for new charts.

evaluate: reports the paper's Table 2 metrics (NSS, CC, KL, SIM) at 3 and 5 seconds
          against accumulated human fixations, for both our checkpoint and the
          released maps. Uses the shared official functions.
export:   writes `<stem>.npz` files in the format `models.umss.UMSS(maps_dir=...)` reads,
          so sampling can run on any machine without TensorFlow.

Inference uses the official md_sem network and pre/post-processing, as models/umss/model.py does.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

if __package__:
    from models.umss.evaluation import saliency as S
    from models.umss.reference.setup_umss import load_md_eam, sha256
else:  # Standalone training Docker image.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from evaluation import saliency as S
    from reference.setup_umss import load_md_eam, sha256

TIMES = [500, 3000, 5000]
GRID = (480, 640)
SALIENCY_METRICS = ('NSS', 'CC', 'KL', 'SIM')


def load_model(work, checkpoint):
    return load_md_eam(Path(work) / 'upstream', checkpoint)


def predict(model, preprocess, postprocess, path, height, width):
    """Official pre/post-processing around the network: three maps at the image resolution."""
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
                    checkpoint_sha256=sha256(args.checkpoint),
                    source_manifest_sha256=sha256(repository / 'sources.json'),
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
    import platform
    import keras
    import tensorflow as tf
    from PIL import Image
    model, pre, post = load_model(args.work, args.checkpoint)
    out = Path(args.out or os.path.join(args.work, 'predictions', 'npz'))
    out.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg'))
    if not images:
        raise FileNotFoundError('No images to export: ' + str(args.images))
    checkpoint_hash = sha256(args.checkpoint)
    source_hash = sha256(Path(args.work) / 'upstream/sources.json')
    image_hashes, npz_hashes = {}, {}
    for image in images:
        image_hashes[image.stem] = sha256(image)
        with Image.open(image) as im:
            width, height = im.size
        maps = predict(model, pre, post, image, height, width)
        volume = np.stack([cv2.resize(m, GRID[::-1]) for m in maps]).astype(np.float32)
        npz = out / (image.stem + '.npz')
        np.savez_compressed(npz, maps=volume, width=width, height=height, image_sha256=image_hashes[image.stem],
                            checkpoint_sha256=checkpoint_hash, source_manifest_sha256=source_hash)
        npz_hashes[image.stem] = sha256(npz)
    metadata = dict(export_schema='umss-certified-maps-v1', checkpoint_sha256=checkpoint_hash,
                    source_manifest_sha256=source_hash, npz_sha256=npz_hashes, image_sha256=image_hashes,
                    tensorflow=tf.__version__, keras=keras.__version__, numpy=np.__version__,
                    python=platform.python_version(), platform=platform.platform(),
                    grid=list(GRID), input_shape=[240, 320, 3],
                    preprocessing='upstream preprocess_images; postprocess_predictions; cv2 grid resize')
    temporary = out / 'export_metadata.json.part'
    temporary.write_text(json.dumps(metadata, indent=2) + '\n')
    temporary.replace(out / 'export_metadata.json')
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

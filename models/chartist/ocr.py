"""Recompute and validate Chartist text_positions as the authors' data_gen.py builds them.

data_gen.py runs PaddleOCR on the original chart, scales each box to the 320x320 model image
and scores each text against the question with stsb-roberta-base-v2. With the versions in
requirements-ocr.txt this reproduces the released meta.json for all 40 conditions: same boxes,
texts and positions; ocr_score within 3e-6 and semantic_similarity within 1e-6.

PaddlePaddle does not run in the Python 3.8 model environment on Apple Silicon, so this module
uses its own environment (see README). Usage:

    python -m models.chartist.ocr META_JSON IMAGES_DIR [--ids 1 6 9]
"""
import argparse
import json
from pathlib import Path

IMAGE_SIZE = 320
SCORE_TOLERANCE = 1e-5


def load_ocr():
    from paddleocr import PaddleOCR
    return PaddleOCR(use_angle_cls=True, lang='en', show_log=False)


def load_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('stsb-roberta-base-v2')


def ocr_text_positions(origin_image, question, *, ocr=None, encoder=None):
    """text_positions for one condition, computed as in data_gen.py.

    origin_image: path of the original chart (read with cv2, alpha dropped, as data_gen.py).
    """
    import cv2
    from sklearn.metrics.pairwise import cosine_similarity

    ocr, encoder = ocr or load_ocr(), encoder or load_encoder()
    image = cv2.imread(str(origin_image))
    if image is None:
        raise FileNotFoundError(origin_image)
    height, width = image.shape[:2]
    positions = []
    for box, (text, score) in ocr.ocr(image)[0] or []:
        embeddings = encoder.encode([question, text])
        similarity = max(float(cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))[0][0]), 0)
        positions.append({'position': [int(box[0][0] * IMAGE_SIZE / width), int(box[0][1] * IMAGE_SIZE / height),
                                       int(box[2][0] * IMAGE_SIZE / width), int(box[2][1] * IMAGE_SIZE / height)],
                          'text': text, 'ocr_score': score, 'semantic_similarity': similarity})
    return positions


def compare_text_positions(given, expected, *, tolerance=SCORE_TOLERANCE):
    """Differences between two text_positions lists, compared box by box in order."""
    problems = []
    if len(given) != len(expected):
        problems.append(f'{len(given)} boxes given, PaddleOCR finds {len(expected)}')
    for i, (a, b) in enumerate(zip(given, expected)):
        for field in ('text', 'position'):
            if a.get(field) != b[field]:
                problems.append(f'box {i} {field}: given {a.get(field)!r}, PaddleOCR {b[field]!r}')
        for field in ('ocr_score', 'semantic_similarity'):
            if field not in a:
                problems.append(f'box {i} {field}: missing')
            elif abs(a[field] - b[field]) > tolerance:
                problems.append(f'box {i} {field}: given {a[field]:.6f}, PaddleOCR {b[field]:.6f}')
    return problems


def validate_text_positions(meta, origin_image, *, ocr=None, encoder=None, tolerance=SCORE_TOLERANCE):
    """Check that meta['text_positions'] equals the PaddleOCR result for its chart and question.

    Returns the list of differences; an empty list means the record is valid.
    """
    expected = ocr_text_positions(origin_image, meta['question'], ocr=ocr, encoder=encoder)
    return compare_text_positions(meta['text_positions'], expected, tolerance=tolerance)


def validate_meta(meta_json, images, *, ids=None):
    """Validate every record of a meta.json against images/chart_<id>_origin.png."""
    ocr, encoder = load_ocr(), load_encoder()
    report = {}
    for meta in json.loads(Path(meta_json).read_text()):
        if ids is None or meta['id'] in ids:
            report[meta['id']] = validate_text_positions(meta, Path(images) / f"chart_{meta['id']}_origin.png",
                                                         ocr=ocr, encoder=encoder)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('meta_json')
    parser.add_argument('images')
    parser.add_argument('--ids', type=int, nargs='+')
    args = parser.parse_args()
    report = validate_meta(args.meta_json, args.images, ids=args.ids)
    for case_id, problems in report.items():
        print(f'{case_id}: ' + ('OK' if not problems else f'{len(problems)} differences'))
        for problem in problems[:10]:
            print('   ', problem)
    invalid = [case_id for case_id, problems in report.items() if problems]
    print(f'{len(report) - len(invalid)}/{len(report)} records match PaddleOCR')
    raise SystemExit(1 if invalid else 0)


if __name__ == '__main__':
    main()

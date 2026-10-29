# UMSS · image-only maps and scanpaths

UMSS predicts three MD-EAM saliency maps and samples free-viewing scanpaths. Inputs contain only an image;
questions and AOIs belong to subsequent experiment analysis. Neural imports are lazy.

Official sources: [paper/project](https://www.collaborative-ai.org/publications/wang23_tvcg/),
[DaRUS V2](https://doi.org/10.18419/DARUS-3361). The authors' GPLv3 sources are downloaded separately;
`reference/umss_sources.json` pins the published file identifiers and checksums.
The repository contains a **retrained MD-EAM checkpoint**, not an author-released trained UMSS checkpoint.
The existing [MASSVIS evaluation](../../reproductions/umss_massvis/README.md) documents reconstruction and validation limits.

## Official vs. own code

- **Official (authors' GPLv3, never edited):** everything under [training/official_reference/](training/official_reference/)
  except `sources.json`. It must stay byte-identical to the DaRUS files pinned (with MD5s) in `reference/umss_sources.json`; our code loads definitions from it instead of copying them.
- **Official functions are called, not reimplemented:** `md_sem`, `preprocess_images`, `postprocess_predictions`,
  `reverse_preprocess`, `calculate_metrics` and its metrics, `make_fixmap_and_heatmap`, the EFD notebook functions,
  the S1 training cells, and the UMSS-S2 sampler (`sample_slice`/`sample_volume` from `UMSS-S2.ipynb`,
  `get_number_fixations`/`get_duration_fixation` from `Code/scripts/UMSS/saltinet_util.py`). Each use is marked
  with an `official` comment. The only copied logic is the fixation time-bucketing loop in
  `training/pipeline/prepare_data.py`, which the EFD notebook writes inline rather than as a function.
- **Own:** everything else (`model.py`, the fixed-length ex-Gaussian profile, tools, adapters, training pipeline).

## Integrity checks (separate flow)

The main flow loads and runs the official code without checking it. Integrity is verified on its own:

```bash
python models/umss/reference/verify.py               # bundled official sources (MD5s)
python models/umss/reference/verify.py --work DIR    # a training run: upstream/ and downloads/ (SHA-256s)
./models/umss/training/run.sh verify                  # the same inside the training container
```

`reproductions/umss_massvis/validation` keeps its own checks of exported maps (`verify_maps`).

## Files and readiness

- `model.py`: `UMSS`, `ScanpathInput`, inference and the batch `run()` API.
- `sampling.py`: temporal-map validation, distributions and S2 scanpath sampling.
- `tools/inspection.py`: `UMSSUtils` preprocessing inspection.
- `tools/export_maps.py`: host-side staging and Docker map export.
- `evaluation/saliency.py`: shared saliency metrics and human-reference preparation for training evaluation and reproduction.
- `training/data_preparation.py`: training-only EFD generation using the original notebook functions.
- `training/s1.py`: our training-only adapter for executing the original S1 notebook cells.
- `__init__.py`: public imports, preserving `from models.umss import UMSS`.
- `reference/verify.py`: separate integrity checks (official sources, training downloads).
- `reference/setup_umss.py`, `reference/umss_sources.json`: download and loading of the official
  code into [training/official_reference](training/official_reference/), including the `md_sem` network shared by
  inference and the Docker pipeline, plus the file `sha256` helper.
- The training reference preserves `Code/scripts/src/`, `Code/scripts/saliency/` and `Code/scripts/UMSS/` from the original repository. Training and inference import modules from those paths.
- Official evaluation sources live in [official_repository/Code](../../reproductions/umss_massvis/official_repository/Code/); our adapter and evaluation helpers remain outside that folder.
- `artifacts/checkpoints/umss/md_eam.hdf5`: retrained checkpoint via Git LFS (`git lfs pull` after cloning).
- `training/`: existing Linux/NVIDIA Docker pipeline. See [setup](training/README.md).
- Working files and run outputs: `artifacts/runs/umss/massvis_md_eam_validation/` by default; override with `UMSS_WORK`.
- `tests/`: checks of sources/distributions, preprocessing, typed inputs, seeds and coordinate conversion.

The current [pilot notebook](../../pilot_gaze_biases_01/06_umss.ipynb) is **prepared, not executed**.
It exports the maps of the 21 images of the image-only dataset (pilot v4) on the Linux GPU machine
(`EXPORT_MAPS = True`), then samples and analyzes them in the regular Jupyter environment, there or on another
machine after copying `predicciones/umss/image_only_v4/npz/`. It does not retrain or download MASSVIS.

## Typed batch API

```python
from pathlib import Path
from typing import List
from models.umss import UMSS, ScanpathInput, UMSSUtils

inputs: List[ScanpathInput] = [
    {'image_id': 'bar_neutral', 'image_path': Path('dataset/images/bar_neutral.png')},
]
model = UMSS(maps_dir='/path/to/exact-image-npz')
preview = UMSSUtils.inputs(inputs[0]['image_path'])  # no TensorFlow import
folder = model.run(inputs, 'predictions/umss', n_runs=100, n_fix=10, seed=17300)
```

Ordinary dictionaries work; no experiment dataset class is required.
`run()` writes the three native maps to `maps/<image_id>.npy` and paths to `scanpaths.csv`:
`image_id, run, seed, fix_idx, x, y, grid_x, grid_y, duration, slice`.
`fix_idx` is zero-based; durations are seconds; `slice` is 0/1/2.
Image coordinates are `x=grid_x/640*width`, `y=grid_y/480*height`.
Run `r` uses **seed + r**, separately for each image. Selected outputs are replaced;
experiment selection, configuration and reuse decisions stay in the notebook.

Without an output directory, `run()` returns `{'maps': ..., 'scanpaths': ...}` in memory:

```python
outputs = model.run(inputs, n_runs=40, n_fix=None, seed=0)
maps = outputs['maps']              # image_id -> three temporal maps
scanpaths = outputs['scanpaths']    # DataFrame
```

The MASSVIS reproduction uses this mode and evaluates released author outputs alongside our exports.
Legacy task-region summaries are in
`pilot_exploratory_01.metrics.summarize_umss`.

## Real MD-EAM inputs

Each `<image-stem>.npz` contains:

| Field | Content |
|---|---|
| `maps` | Three finite nonnegative maps `(3, 480, 640)`, each with positive mass |
| `width`, `height` | Original image dimensions |
| `image_sha256` | Hash of the exact input PNG |

The order is **0–0.5 s, 0.5–3 s, 3–5 s**. Upstream postprocessing removes the letterbox before
resizing to the 640 × 480 sampling grid; the grid corresponds to the whole image.
`predict_maps()` reads the maps and dimensions as stored. Docker exports additionally write
`export_metadata.json` with checkpoint/source/image hashes and actual TensorFlow/Keras versions.
Transfer that file with the NPZ files to preserve provenance.

```python
from models.umss import export_maps
# <work>/ckpt/md_eam.hdf5 and <work>/upstream/ must already exist.
npz = export_maps([item['image_path'] for item in inputs], 'pilot_v3_image_only', work='/path/to/work')
```

For the Linux GPU export use the existing Docker runner (Keras 2.3.1 / NVIDIA TF 1.15.5).
The pilot notebook supplies the staging and export steps. For maps-only sampling/analysis, TensorFlow is unnecessary.
Direct checkpoint inference remains available as `UMSS(checkpoint=...)` in a compatible legacy environment;
the constructor's default checkpoint is `artifacts/checkpoints/umss/md_eam.hdf5`.
Do not install legacy TF1 dependencies into the host analysis environment.

`UMSSUtils.inputs()` reads the upstream `padding` and `preprocess_images` functions without
importing their Keras-dependent module. It returns the actual `(1,240,320,3)` BGR/mean-subtracted tensor
and an RGB preview that retains the padding. The preview uses `training/official_reference/Code/scripts/saliency/sal_imp_utilities.py`. Neural source setup: `python models/umss/reference/setup_umss.py` (defaults to `models/umss/training/official_reference/`). Docker training keeps its separate `<work>/upstream/` staging directory.

## Sampling conventions

Both protocols place fixations with the official `sample_slice` (half-maximum map threshold and Gaussian
proximity weighting with sigma 200 grid pixels, the official 2-D `gaussian_filter`, about 0.3 s per fixation).
This is **not IOR**. Each scanpath seeds the global `np.random`, as the official functions draw from it. The first fixation is sampled;
no artificial center is inserted. After 3 seconds the third map is retained.

- **Fixed length** (`n_fix=10` by default in `run()`; `predict()` retains its legacy default of 8):
  adapter profile using positive ex-Gaussian durations (μ=124.06, σ=17.49, τ=89.37 ms).
  The map is selected by elapsed time **before** each fixation. This is not the original variable-length benchmark.
- **UMSS-S2** (`n_fix=None`): the official `sample_volume` with the official length and duration functions
  (2–35 fixations, including the 0.55-second bin) and its own map switching.

Do not interpret Monte Carlo draws as independent human participants or the three temporal maps as cumulative
attention. Ten fixations may not reach the late interval; the pilot records actual slice coverage.
AOI projection/normalization, visit summaries and interpretation belong to `pilot_gaze_biases_01/umss_analysis.py`
and the notebook, outside the reusable model.

## Local checks

```bash
.venv-chartist/bin/python -m unittest discover -s models/umss/tests -v
```

These checks do not establish neural runtime readiness for the pilot on a different Linux GPU.
Run the prepared notebook there and retain its runtime metadata, exported maps and complete outputs.

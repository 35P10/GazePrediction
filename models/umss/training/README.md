# UMSS stage 1 training (MD-EAM)

Fine-tune MD-SEM on MASSVIS to produce the three temporal saliency maps used by
UMSS. The pipeline executes the original training cells of `UMSS-S1.ipynb` and
saves a selected checkpoint as `md_eam.hdf5`. Stage 2 samples scanpaths and does
not require a separate training run.

## Requirements

- Linux with an NVIDIA GPU.
- Docker with NVIDIA Container Toolkit configured for GPU access.
- Network access for the initial data and weight downloads.

The Docker image supplies the legacy TensorFlow 1.15.5 / Keras 2.3.1 environment.
Do not install these training dependencies into the host analysis environment.

## Run training

From the repository root:

```bash
cd models/umss/training
export UMSS_WORK="$(cd ../../.. && pwd)/artifacts/runs/umss/massvis_s1_run01"
./run.sh download
./run.sh check
./run.sh prepare
./run.sh train
```

Use a different `UMSS_WORK` directory for each experiment. If unset, the pipeline
uses `artifacts/runs/umss/massvis_md_eam_validation/` under the repository root.
All paths below are relative to `UMSS_WORK`.

| Stage | Purpose | Output |
|---|---|---|
| `download` | Fetch sources, MASSVIS inputs and initial weights. | `upstream/`, `downloads/` |
| `verify` | Optional, separate: check sources and downloads against their pinned checksums. | — |
| `check` | Regenerate the released EFD maps and require pixel-for-pixel agreement. | `datasets/check_500_2000_5000_10000/` |
| `prepare` | Build the temporal training targets and fixation maps. | `datasets/massvis_bucket_500_3000_5000/` |
| `train` | Fine-tune the model and keep the last epoch, as the authors' notebook does. | `ckpt/` |

Downloads already present are reused; run `./run.sh verify` to check them. Run `check` before training to validate the data
preparation. To check the training setup with one epoch after preparation:

```bash
./run.sh train --epochs 1
```

Use a fresh run directory when changing training settings.

## Training settings

| Setting | Value |
|---|---|
| Dataset | MASSVIS encoding fixations; 327 training and 66 validation images |
| Initialization | MD-SEM CodeCharts weights |
| Model | `md_sem`, input 240 × 320, output 480 × 640 |
| Temporal intervals | 0–0.5 s, 0.5–3 s and 3–5 s |
| Targets | Gaussian fixation heatmap for the first interval; EFD maps for the later intervals; fixation maps for NSS |
| Heatmap sigma | 19 pixels at the original image resolution |
| Loss | 10 × KL − 5 × CC − NSS + 3 × CC-match |
| Optimizer | Adam, initial learning rate 1e-4, multiplied by 0.1 every three epochs |
| Batch size | 8 for training; 1 for validation |
| Augmentation | Disabled |
| Epochs | 6 by default, following the paper; the released notebook uses 9 |
| Seed | 0 by default; `--seed -1` disables explicit seeding |

To use the notebook's epoch count or change the seed:

```bash
./run.sh train --epochs 9 --seed 42
```

Training targets are regenerated at the required temporal boundaries. The
pipeline uses the original model, loss and generator implementations, replaces
the interactive training plot with a CSV log, and makes epochs, seeds and
verbosity configurable. GPU operations may remain nondeterministic even with a
fixed seed.

## Limited GPU memory

If batch 8 does not fit, use micro-batches of 4 or 2:

```bash
./run.sh train --micro-batch 4
```

Gradient accumulation retains an effective batch size of 8. A numerical
self-check runs before training. BatchNorm statistics are computed per
micro-batch, so this mode is not identical to a full batch of 8. The chosen
micro-batch size is recorded in `ckpt/md_eam.json`.

## Training outputs

| File under `ckpt/` | Content |
|---|---|
| `*_epXX_valloss*.hdf5` | Weights saved after each epoch |
| `training_log.csv` | Training and validation losses by epoch |
| `md_eam.hdf5` | Weights after the last epoch |
| `md_eam.json` | Last epoch, run settings and the saved epoch files |

All epoch checkpoints are retained. The paper does not say which epoch it kept; the
authors' notebook predicts with the model left after the last epoch, so `md_eam.hdf5` is that one.
Record each kept checkpoint and its settings in
[artifacts/checkpoints/umss](../../../artifacts/checkpoints/umss/README.md).
It is not chosen on the validation images, which are also the Table 2 evaluation images.

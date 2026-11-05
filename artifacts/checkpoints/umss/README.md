# UMSS MD-EAM weights

Two retrainings of MD-EAM (UMSS stage 1) with the authors' recipe (`UMSS-S1.ipynb`, DaRUS V2) and
`models/umss/training/`. The authors did not release trained weights; the first author later sent two by
email ([Authors' weights](#authors-weights)). `UMSS(checkpoint=...)` loads any of them; the default is
`md_eam.hdf5`.

Settings shared by both, taken from the authors' notebook: MASSVIS encoding fixations, 327 training and
66 validation images (the Table 2 evaluation images); MD-SEM CodeCharts initial weights; `md_sem` with a
240 × 320 input and 480 × 640 output; intervals 0–0.5, 0.5–3 and 3–5 s; loss 10 KL − 5 CC − NSS + 3 CC-match;
Adam, learning rate 1e-4 multiplied by 0.1 every three epochs; batch 8 (validation 1); no augmentation.

| | `md_eam.hdf5` (old) | `md_eam_last_epoch.hdf5` (new) |
|---|---|---|
| Status | Default: pilots and the MASSVIS reproduction | Trained 2026-10-29; MASSVIS maps not exported yet |
| Epoch kept | Lowest validation loss (rule of the pipeline at the time) | Last epoch (6 of 6), as the authors' notebook predicts with it |
| Chosen on the evaluation images | Yes: the validation images are the Table 2 images | No |
| Epochs | Not recorded (pipeline default 6) | 6 |
| Seed | Not recorded (pipeline default 0) | 0 |
| Batch | Not recorded (pipeline default 8) | 8, as 2 accumulated micro-batches of 4 (10 GB GPU); the self-test matched batch-8 updates to 3e-08, BatchNorm statistics use 4 images |
| Hardware and environment | Not recorded (pipeline: NVIDIA TF 1.15.5, Keras 2.3.1) | RTX 3080 10 GB; `nvcr.io/nvidia/tensorflow:23.03-tf1-py3` (TF 1.15.5, Keras 2.3.1); about 21 s per epoch |
| Training record | Not kept | `md_eam_last_epoch.json`, `md_eam_last_epoch_training_log.csv` |
| SHA-256 | `1b7b259b97e48e87199ad6e07784dc8a2dd88e8eda643bd39cc9ce235edbbd0a` | `262aff266531172874217c6a4113b77b1d8f39f0c233ff24fad513df04247fc9` |
| Date | File dated 2026-10-20 | 2026-10-29 |
| Exported MASSVIS maps | `reproductions/umss_massvis/data/ours/npz/` (2026-10-21, no `png/`, no checkpoint hash) | Not yet |

The old weights' settings were not saved with them; "not recorded" entries cannot be recovered from the file.
The pipeline defaults are what a plain `./run.sh train` used, not a confirmed record.

New run, validation loss by epoch: 3.841, 2.615, 1.163, 0.927, 0.773, 0.777 (the kept epoch 6 is not the lowest).

## Results on MASSVIS (66 charts)

Table 2 protocol (accumulated human maps, sigma 19 px). The authors' released maps give the paper's numbers
exactly with the same evaluation.

| Table 2 | Paper | Old | New |
|---|---:|---:|---:|
| 3 s NSS / CC / KL / SIM | 1.406 / 0.586 / 0.754 / 0.516 | 1.577 / 0.663 / 0.677 / 0.543 | 1.579 / 0.664 / 0.684 / 0.538 |
| 5 s NSS / CC / KL / SIM | 1.024 / 0.514 / 0.689 / 0.530 | 1.071 / 0.540 / 0.687 / 0.533 | 1.071 / 0.540 / 0.690 / 0.527 |

- Old: from [reproductions/umss_massvis](../../../reproductions/umss_massvis/README.md), grid maps resized to each chart.
- New: `./run.sh evaluate`, chart-size maps written by the authors' `predict_and_save_md`; per image in
  `md_eam_last_epoch_saliency_per_image.csv` (our NSS is higher than the released maps' on 60 of 66 charts at 3 s,
  43 of 66 at 5 s).

The two retrainings agree to within 0.01 although they differ in epoch selection, micro-batch and map export, so
choosing the epoch on the evaluation charts did not inflate the old scores noticeably. Both are above the paper
at 3 s (NSS +0.17, CC +0.08) and close to it at 5 s: the published recipe gives a model that matches or exceeds
the paper's, not the paper's numbers. The authors released neither weights nor a training log
(`UMSS-S1.ipynb` is saved without outputs); the weights they later sent show the gap is in their training
([Authors' weights](#authors-weights)).

Their released maps are almost the same as the new weights' (CC 0.93 per chart) but sit about 10 px further
right on every chart (median 10 px, range 5–25 px): moved 10 px left they score 1.448 / 0.602 (3 s NSS / CC)
and 1.063 / 0.532 (5 s), so the 5 s gap closes and about a quarter of the 3 s gap does. The authors' weights
give their released maps through our export, so the shift comes from their model, not from how the maps were
made (all 198 exported PNGs differ from the released ones by at most one grey level). Blurring our maps
barely changes the scores.

### Seeds and epochs

`models/umss/training/sweep.sh` (2026-10-29, micro-batch 4): seeds 0–3 with 6 epochs and seed 0 with the
notebook's 9, every epoch evaluated. Seed 0, epoch 6 is `md_eam_last_epoch.hdf5` (same scores; the epoch file
has another hash because Keras saves it as a full model). Records in [sweep/](sweep/), without weights:
`<run>/eval_epNN.json` (Table 2 means and checkpoint hash), `<run>/ckpt/training_log.csv` and `md_eam.json`.
`sweep/seed0_6ep_2000/` is the 0.5–2 s training run (`--times 500 2000 5000`).

| Checkpoints | 3 s NSS / CC / KL / SIM | 5 s NSS / CC / KL / SIM |
|---|---|---|
| Paper | 1.406 / 0.586 / 0.754 / 0.516 | 1.024 / 0.514 / 0.689 / 0.530 |
| Last epoch, seeds 0–3 (6 epochs) | 1.572–1.584 / 0.661–0.665 / 0.684–0.686 / 0.537–0.539 | 1.049–1.078 / 0.528–0.545 / 0.686–0.701 / 0.522–0.528 |
| Last epoch, seed 0, 9 epochs | 1.579 / 0.663 / 0.685 / 0.537 | 1.065 / 0.536 / 0.694 / 0.526 |
| Epochs 3–9, all runs | 1.569–1.591 NSS | 1.042–1.081 NSS |
| Trained on 0.5–2 s, seed 0, last epoch | 1.592 / 0.669 / 0.663 / 0.552 | 1.156 / 0.586 / 0.647 / 0.542 |
| Epoch 1, all runs | 1.469–1.487 / 0.625–0.631 / 0.754–0.774 / 0.493–0.502 | 1.239–1.263 / 0.636–0.646 / 0.575–0.593 / 0.559–0.570 |

From epoch 3 on, seeds and epoch count move the scores by about 0.01, far less than the 0.17 gap at 3 s; 9
epochs give the same as 6. Only epoch 1 comes near the paper at 3 s, but it is then far above it at 5 s, so no
checkpoint of the published recipe gives Table 2. Training on 0.5–2 s, the paper model's interval, moves our
scores further from the paper (3 s NSS 1.59 at epochs 3–6, 5 s NSS 1.15–1.16), so our training differs from
the authors' in something besides the interval.

## Authors' weights

`authors/` holds the two MD-EAM checkpoints Yao Wang (first author) sent by email on 2026-11-05, from her
Google Drive, with their original file names. According to her, the paper's model is the 6-epoch one with
intervals 0–0.5, 0.5–2 and 2–5 s, although the paper says 0.5–3 s. She could not answer the other questions
(seed, versions, how the released maps were made) because the training server no longer exists.

| File | Intervals | Epoch | SHA-256 |
|---|---|---|---|
| `massvis_bucket_500_2000_5000_kl10cc-5nss-1ccmatch3_ep06_valloss1.1899.hdf5` (paper) | 0.5 / 2 / 5 s | 6 of 6 | `d092744b0606d05d20f7d9ddbb5e71f043cb880200b2dd386eb66d7a6a902285` |
| `massvis_bucket_500_3000_5000_kl10cc-5nss-1ccmatch3_ep09_valloss1.4642.hdf5` | 0.5 / 3 / 5 s | 9 of 9 | `6b3119148c78320a69e22a963b3e12379863206cef6e9668fb97080ee4899b9b` |

Their validation losses are about twice ours with the same settings (1.190 vs 0.596 for 0.5–2 s, 6 epochs;
1.464 vs 0.763 for 0.5–3 s, 9 epochs), so their training differed in some way the published code does not
show.

Evaluated on the Linux GPU machine (2026-11-05) with `./run.sh evaluate --checkpoint ...`, then exported for
the 66 MASSVIS charts. Records in `artifacts/runs/umss/massvis_md_eam_validation/`: `authors/eval_ep06.json`,
`authors/eval_9ep_3000.json` (Table 2 means and checkpoint hash), the per-chart CSVs next to them, and the maps
in `predictions/massvis_authors/` and `predictions/massvis_authors_9ep/`.

| Table 2 | 3 s NSS / CC / KL / SIM | 5 s NSS / CC / KL / SIM |
|---|---|---|
| Paper | 1.406 / 0.586 / 0.754 / 0.516 | 1.024 / 0.514 / 0.689 / 0.530 |
| Authors, 6 epochs, 0.5–2 s (paper model) | 1.406 / 0.586 / 0.754 / 0.516 | 1.024 / 0.514 / 0.689 / 0.530 |
| Authors, 9 epochs, 0.5–3 s | 1.428 / 0.595 / 0.752 / 0.515 | 1.032 / 0.518 / 0.671 / 0.536 |
| Ours, 6 epochs, 0.5–2 s (`sweep/seed0_6ep_2000`) | 1.592 / 0.669 / 0.663 / 0.552 | 1.156 / 0.586 / 0.647 / 0.542 |
| Ours, 9 epochs, 0.5–3 s (`sweep/seed0_9ep`) | 1.579 / 0.663 / 0.685 / 0.537 | 1.065 / 0.536 / 0.694 / 0.526 |

The paper model gives Table 2 to within 1e-4 (8-bit PNGs), so our preprocessing, export and evaluation match
the authors' end to end. With the same known settings our retrainings are 0.15–0.19 NSS above theirs at 3 s,
with every seed and epoch count we tried, so the gap comes from their training, which the published code does
not fully describe (their validation loss is also twice ours).

## Adding weights

On the Linux GPU machine, with a fresh `UMSS_WORK` folder, run all stages (download, check, prepare, train,
evaluate; see [models/umss/training](../../../models/umss/training/README.md)); add `--micro-batch 4` on GPUs
with 10 GB:

```bash
./models/umss/training/run.sh
```

Copy `<work>/ckpt/md_eam.hdf5`, `md_eam.json`, `training_log.csv` and `predictions/paper_saliency_per_image.csv`
here under a new name prefix, add a column above and a Git LFS rule in `.gitattributes`. Export the 66 MASSVIS
charts with them before comparing scanpaths (S2).

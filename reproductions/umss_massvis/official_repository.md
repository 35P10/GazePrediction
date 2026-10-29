# Official UMSS repository mirror

`official_repository/` mirrors all 138 published files from
[DaRUS V2](https://doi.org/10.18419/DARUS-3361), preserving the original paths and
file contents. The pinned inventory is `official_repository_manifest.json`,
including file IDs, original byte sizes and MD5 checksums. The original root README is
preserved inside the mirror; this document describes our local setup.

- `Code/`: original model, training, sampling, analysis and evaluation sources,
  plus published score tables.
- `Data/`: original UMSS, baseline and prepared-dataset archives.
- `Study/`: original qualitative study data, analysis scripts and GIF stimuli.

Download missing files or verify existing files from the project root:

```bash
python reproductions/umss_massvis/sync_official_repository.py
python reproductions/umss_massvis/sync_official_repository.py --verify-only
```

DaRUS labels ingested tables with `.tab` filenames, but publishes MD5 checksums
for the originally uploaded CSV files. The mirror keeps those published paths
and downloads `format=original`, using `originalFileSize` for verification.
The manifest also records the uploaded CSV filenames. This preserves the same
original table bytes used by our existing evaluation.

The deposit is approximately 1.86 GB before extraction. Large archives are
excluded from Git and restored using the pinned download script. Archives for
baselines and study stimuli are retained in their original ZIP format.

For the evaluation, the original `Data/UMSS/UMSS.zip` and
`Data/UMSS/GT_EFD_500_2000_5000.zip` also have extracted inputs beside them:

- `official_repository/Data/UMSS/UMSS/`: released maps, scanpaths and strings.
- `official_repository/Data/UMSS/GT_EFD_500_2000_5000/`: the images required by
  the published split. Only the image members required here are extracted.

These extracted directories are local conveniences, not additional files in
the published deposit. `replication.download()` verifies/downloads the pinned
evaluation subset, extracts its required inputs and prepares external MASSVIS
fixations and annotations in `data/massvis/`.

Our predictions remain in `data/ours/`; results remain in `results/`. External
MASSVIS inputs and our outputs are not part of the official mirror.

The full original neural code is now included in the mirror under
`Code/scripts/src/` and `Code/scripts/UMSS/`. The separate verified copy in
`models/umss/training/official_reference/` remains available for training and
direct inference, with the same original paths under `Code/scripts/`.
`UMSSUtils.inputs()` reads the mirror's original
`Code/scripts/saliency/sal_imp_utilities.py` to preview preprocessing without
loading TensorFlow/Keras.

Keep all adaptations in our own modules. Do not edit the mirrored originals.

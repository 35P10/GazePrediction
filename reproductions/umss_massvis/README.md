# UMSS on MASSVIS

Do our two UMSS stages give the paper's results? Open [01_reproduce_umss.ipynb](01_reproduce_umss.ipynb)
with the `.venv-chartist` kernel.

- **S1**: the three temporal saliency maps, compared with Table 2.
- **S2**: scanpaths sampled from those maps with `UMSS.run(n_fix=None)`, compared with Table 1.

Each table shows the paper, the authors' released outputs and ours, scored with the same evaluation: the
authors' code in `official_repository/`, on the 66 evaluation charts of the published split. The released
outputs check the evaluation (they should give the paper's numbers). "Ours" comes from one of the two
retrained weights (settings: [artifacts/checkpoints/umss](../../artifacts/checkpoints/umss/README.md)),
chosen with `WEIGHTS` in the notebook:

| `WEIGHTS` | Weights | Maps | Results |
|---|---|---|---|
| `'last_epoch'` | `md_eam_last_epoch.hdf5` (new) | `data/ours_last_epoch/npz/` | `results/s1_last_epoch.csv`, `results/s2_last_epoch.csv` |
| `'old'` | `md_eam.hdf5` | `data/ours/npz/` | `results/s1_old.csv`, `results/s2_old.csv` |

S1 scores the chart-size PNGs the export writes in `png/` with the authors' `predict_and_save_md`; the old
maps predate that, so S1 resizes their grid maps instead.

## Results

Old weights; the new weights' S1 is in
[artifacts/checkpoints/umss](../../artifacts/checkpoints/umss/README.md) and their S2 is still to run.

S1, Table 2 (higher is better except KL):

| Duration | Metric | Paper | Released maps | Ours |
|---|---|---:|---:|---:|
| 3 s | NSS | 1.406 | 1.406 | 1.577 |
| 3 s | CC | 0.586 | 0.586 | 0.663 |
| 3 s | KL | 0.754 | 0.754 | 0.677 |
| 3 s | SIM | 0.516 | 0.516 | 0.543 |
| 5 s | NSS | 1.024 | 1.024 | 1.071 |
| 5 s | CC | 0.514 | 0.514 | 0.540 |
| 5 s | KL | 0.689 | 0.689 | 0.687 |
| 5 s | SIM | 0.530 | 0.530 | 0.533 |

S2, Table 1 without ScanMatch (40 scanpaths per chart, seed 0):

| Metric | Paper | Released scanpaths | Ours |
|---|---:|---:|---:|
| Sequence Score, mean | 0.446 | 0.364 | 0.382 |
| Sequence Score, best | 0.724 | 0.654 | 0.699 |
| sTDE, mean | 0.906 | 0.905 | 0.913 |
| sTDE, best | 0.925 | 0.922 | 0.943 |
| DTW, mean (lower is better) | 6529.1 | 6531.5 | 6590.3 |
| DTW, best (lower is better) | 4683.4 | 4369.4 | 3775.1 |
| MultiMatch shape | 0.943 | 0.943 | 0.958 |
| MultiMatch direction | 0.728 | 0.730 | 0.767 |
| MultiMatch length | 0.935 | 0.936 | 0.950 |
| MultiMatch position | 0.771 | 0.772 | 0.774 |
| MultiMatch duration | 0.712 | 0.711 | 0.710 |

The released maps give Table 2 exactly. The released scanpaths give Table 1 for sTDE, DTW mean and
MultiMatch, but not for Sequence Score and DTW best: these depend on the human references the deposit
does not include and that are rebuilt here.

Tables: [results/s1_old.csv](results/s1_old.csv), [results/s2_old.csv](results/s2_old.csv).

## What to keep in mind

- The paper's 5 s human scanpaths and accumulated human maps are not in the deposit; `replication.py` rebuilds
  them from MASSVIS with the rules of the authors' code.
- ScanMatch needs MATLAB and is not computed.
- The old checkpoint's epoch was chosen on these same 66 charts. A second retraining that keeps the last epoch
  (`md_eam_last_epoch.hdf5`) gives the same S1 scores within 0.01, so ours being above the paper at 3 s does
  not come from that choice: the published recipe does not give the paper's numbers
  ([details](../../artifacts/checkpoints/umss/README.md)). S2 has not been run with the new weights yet.
- S2 is random: another seed gives slightly different numbers.

## Files

| Path | Content |
|---|---|
| `01_reproduce_umss.ipynb` | The comparison |
| `replication.py` | Data loading, the authors' evaluation, `s1()` and `s2()` |
| `source.json` | Paper, MASSVIS downloads and published table values |
| `official_repository/` | The authors' DaRUS V2 deposit, unchanged ([details](official_repository.md)) |
| `data/massvis/` | MASSVIS fixations and element labels (research use, not redistributed) |
| `data/ours/npz/`, `data/ours_last_epoch/npz/` | Our exported maps, old and new weights (`.npz` for sampling, `png/` for S1) |
| `results/` | `s1_<weights>.csv` and `s2_<weights>.csv` written by the notebook |

First setup: restore the deposit with `python reproductions/umss_massvis/sync_official_repository.py`
(`--verify-only` checks it against the pinned MD5s), then run `replication.download()`.

Exporting maps for new weights, on the Linux GPU machine (the default checkpoint is `<work>/ckpt/md_eam.hdf5`;
the 66 evaluation charts are the validation images the download stage unpacks):

```bash
./models/umss/training/run.sh export --images /work/downloads/GT_EFD_500_2000_5000/val/raw_img --out /work/predictions/massvis_last_epoch
```

Then copy `<work>/predictions/massvis_last_epoch/` (66 `.npz`, `png/` and `export_metadata.json`) to
`data/ours_last_epoch/npz/`.

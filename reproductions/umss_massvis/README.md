# UMSS on MASSVIS

Do our two UMSS stages give the paper's results? Open [01_reproduce_umss.ipynb](01_reproduce_umss.ipynb)
with the `.venv-chartist` kernel.

- **S1**: the three temporal saliency maps, compared with Table 2.
- **S2**: scanpaths sampled from those maps with `UMSS.run(n_fix=None)`, compared with Table 1.

Each table shows the paper, the authors' released outputs and ours, scored with the same evaluation: the
authors' code in `official_repository/`, on the 66 evaluation charts of the published split. The released
outputs check the evaluation (they should give the paper's numbers). "Ours" comes from the weights chosen
with `WEIGHTS` in the notebook: one of our two retrainings, or one of the two the first author sent later
(settings: [artifacts/checkpoints/umss](../../artifacts/checkpoints/umss/README.md)):

| `WEIGHTS` | Weights | Maps | Results |
|---|---|---|---|
| `'last_epoch'` | `md_eam_last_epoch.hdf5` (new) | `data/ours_last_epoch/npz/` | `results/s1_last_epoch.csv`, `results/s2_last_epoch.csv` |
| `'old'` | `md_eam.hdf5` | `data/ours/npz/` | `results/s1_old.csv`, `results/s2_old.csv` |
| `'authors'` | authors', 6 epochs, 0.5–2 s (paper model) | `artifacts/runs/umss/massvis_md_eam_validation/predictions/massvis_authors/` | `results/s2_authors.csv` |
| `'authors_9ep'` | authors', 9 epochs, 0.5–3 s | `…/predictions/massvis_authors_9ep/` | `results/s2_authors_9ep.csv` |

S1 scores the chart-size PNGs the export writes in `png/` with the authors' `predict_and_save_md`; the old
maps predate that, so S1 resizes their grid maps instead.

## Results

"Old" and "New" are our two retrained weights (`WEIGHTS = 'old'` / `'last_epoch'`); "Authors" are the
first author's two checkpoints (`'authors'` / `'authors_9ep'`; S1 from `run.sh evaluate` on the Linux GPU machine).

S1, Table 2 (higher is better except KL):

| Duration | Metric | Paper | Released maps | Ours, old | Ours, new | Authors, paper model | Authors, 9 epochs |
|---|---|---:|---:|---:|---:|---:|---:|
| 3 s | NSS | 1.406 | 1.406 | 1.577 | 1.579 | 1.406 | 1.428 |
| 3 s | CC | 0.586 | 0.586 | 0.663 | 0.664 | 0.586 | 0.595 |
| 3 s | KL | 0.754 | 0.754 | 0.677 | 0.684 | 0.754 | 0.752 |
| 3 s | SIM | 0.516 | 0.516 | 0.543 | 0.538 | 0.516 | 0.515 |
| 5 s | NSS | 1.024 | 1.024 | 1.071 | 1.071 | 1.024 | 1.032 |
| 5 s | CC | 0.514 | 0.514 | 0.540 | 0.540 | 0.514 | 0.518 |
| 5 s | KL | 0.689 | 0.689 | 0.687 | 0.690 | 0.689 | 0.671 |
| 5 s | SIM | 0.530 | 0.530 | 0.533 | 0.527 | 0.530 | 0.536 |

S2, Table 1 without ScanMatch (40 scanpaths per chart, seed 0):

| Metric | Paper | Released scanpaths | Ours, old | Ours, new | Authors, paper model | Authors, 9 epochs |
|---|---:|---:|---:|---:|---:|---:|
| Sequence Score, mean | 0.446 | 0.364 | 0.382 | 0.381 | 0.352 | 0.367 |
| Sequence Score, best | 0.724 | 0.654 | 0.699 | 0.704 | 0.663 | 0.683 |
| sTDE, mean | 0.906 | 0.905 | 0.913 | 0.913 | 0.906 | 0.909 |
| sTDE, best | 0.925 | 0.922 | 0.943 | 0.943 | 0.937 | 0.939 |
| DTW, mean (lower is better) | 6529.1 | 6531.5 | 6590.3 | 6583.5 | 6610.2 | 6485.3 |
| DTW, best (lower is better) | 4683.4 | 4369.4 | 3775.1 | 3771.1 | 3947.7 | 3865.7 |
| MultiMatch shape | 0.943 | 0.943 | 0.958 | 0.959 | 0.960 | 0.958 |
| MultiMatch direction | 0.728 | 0.730 | 0.767 | 0.761 | 0.766 | 0.759 |
| MultiMatch length | 0.935 | 0.936 | 0.950 | 0.951 | 0.951 | 0.949 |
| MultiMatch position | 0.771 | 0.772 | 0.774 | 0.775 | 0.772 | 0.777 |
| MultiMatch duration | 0.712 | 0.711 | 0.710 | 0.711 | 0.714 | 0.712 |

The released maps give Table 2 exactly. The released scanpaths give Table 1 for sTDE, DTW mean and
MultiMatch, but not for Sequence Score and DTW best: these depend on the human references the deposit
does not include and that are rebuilt here. Our two weights give the same results within run-to-run noise.
With the authors' paper model our maps are the released ones, yet our scanpaths differ from the released
scanpaths as they do with our own maps (MultiMatch direction 0.77 vs 0.73): the S2 difference is in the
sampling, not in S1.

Tables: `results/s1_<weights>.csv`, `results/s2_<weights>.csv`.

## What to keep in mind

- The paper's 5 s human scanpaths and accumulated human maps are not in the deposit; `replication.py` rebuilds
  them from MASSVIS with the rules of the authors' code.
- ScanMatch needs MATLAB and is not computed.
- The authors' paper model (6 epochs, 0.5–2 s, not the 0.5–3 s the paper states) gives Table 2 exactly with our
  export and evaluation, and its maps are the released ones to within one grey level. Our retrainings with the
  published recipe stay about 0.17 NSS above it at 3 s whatever the seed, epochs or epoch kept, so the gap is in
  the authors' training ([details](../../artifacts/checkpoints/umss/README.md)).
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

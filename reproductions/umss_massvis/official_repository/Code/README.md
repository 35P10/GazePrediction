# UMSS toolbox

This repository contains the scripts for analysis of UMSS and other baseline methods on MASSVIS dataset. MASSVIS dataset is downloaded from https://github.com/massvis.

ENV: `pip install -r scripts/requirements.txt`

## Code Map

```
$this Directory
│
│─ README.md —— this file
│
└─ scripts —— Scripts for processing MASSVIS dataset
   │
   │─ metrics —— computing scanpath metrics (ScanMatch, DTW, SS) (Table 1)
   │
   │─ multimatch —— computing MultiMatch metrics (Table 2)
   │
   │─ preprocess —— some preprocessing scipts for formatting etc.
   │
   │─ saliency —— scripts for analysis of MASSVIS data
   │   │
   │   │─ eval_dvs/mdeam.py —— evaluating saliency maps with 3s/5s gt (Table 2)
   │   │
   │   │─ fixation_all_analysis.ipynb —— element-wise gaze dynamics (Figure 2)
   │   │
   │   │─ fixation_individual_analysis.ipynb —— inv. transition matrices (Figure 3)
   │   │
   │   └─ saliency_toolbox.ipynb —— computing some dataset stats and saliency metrics (suppl.)
   │
   │── UMSS
   │    │
   │    │─ generate_Massvis-MDEFD —— generating MASSVIS-MDEFD (training data for UMSS stage 1)
   │    │
   │    │─ UMSS-S1.ipynb —— fine-tuning MDSEM model with MASSVIS-MDEFD data (UMSS stage 1)
   │    │
   │    └─ UMSS-S2.ipynb —— sampling scanpaths from saliency maps (UMSS stage 2)
   │
   │── src ── dependency scripts
   │
   └── visualization ── plotting scanpaths, and some figures in the manuscript

```

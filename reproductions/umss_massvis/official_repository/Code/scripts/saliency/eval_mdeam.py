# -*- coding: utf-8 -*-
"""
Created on 20220428

@author: Yao Wang
@purpose: to compute the saliency metrics of MDEAM model in MASSVIS dataset
@output : saliency metrics of MDEAM model
"""

import pandas as pd
import numpy as np
from glob import glob
import os, sys
from PIL import Image
from natsort import natsorted
sys.path.append("..") 
from eval import calculate_metrics
import tqdm

ROOT_PATH = '/netpool/homes/wangyo/Dataset/Massvis/targets393/'
FIX_PATH = ROOT_PATH + 'GT_new/all/fix_maps_accum/3000'
SAL_PATH = ROOT_PATH + 'GT_new/all/heatmaps_accum/3000'

PRED_PATH = ROOT_PATH + 'MDSEM_bucket_500_2000_5000/salmaps/'

FIXCOORD_PATH = ROOT_PATH + 'GT_new/all/fix_coords/5000'

print('evaluating on: ', PRED_PATH)

# your suffix
pred_suffix = '.png'

list_img_targets = natsorted(glob(PRED_PATH+ f'*_3000{pred_suffix}'))
metrics = ('R2', 'RMSE', 'CC', 'KL', 'SIM', 'NSS', 'AUC')
RESULT = {vt: [] for vt in metrics}


for idx in tqdm.trange(0, len(list_img_targets)):
    imgfullpath = list_img_targets[idx]
    #print(imgfullpath)
    with Image.open(imgfullpath) as predmap:
        imgname = os.path.basename(imgfullpath)
        imgname = imgname.replace(f'_3000{pred_suffix}','')
        salmap = Image.open(os.path.join(SAL_PATH, imgname + '.png'))
        fixmap = Image.open(os.path.join(FIX_PATH, imgname +'.png'))
        fixcoord = np.load(os.path.join(FIXCOORD_PATH, imgname + '.npy'))

        sm_idx = calculate_metrics(np.array(predmap), gt_map = np.array(salmap), gt_fix_map = np.array(fixmap), gt_fix_points = fixcoord, sAUC = False)
        for metric in metrics:
            RESULT[metric].append(sm_idx[metric][0])

for metric in metrics:
    print(metric, ' :', np.round(np.mean(RESULT[metric]), 3))


df = pd.DataFrame(list(RESULT[metric] for metric in metrics)).T
df.columns = ['R2', 'RMSE', 'CC', 'KL', 'SIM', 'NSS', 'AUC']
df.to_csv('csv/mdeam_3s.csv', index=False)

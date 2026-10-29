# -*- coding: utf-8 -*-
"""
Created on 20210316

@author: Yao Wang
@purpose: to compute the scanpath metrics of Saltinet-postprocess predictions one by one to the GT scanpaths in MASSVIS dataset
@output : the final avg. Metrics
"""

import pandas as pd
import numpy as np
import glob,os, sys
from PIL import Image
from natsort import natsorted
pd.options.mode.chained_assignment = None
from metrics import euclidean_distance, scaled_TDE, TDE, DTW, eyenalysis
sys.path.append("..") 
from preprocess_utils import get_gt_files, truncate_EOS

ROOT_PATH = '/netpool/homes/wangyo/Dataset/Massvis/targets393/'

imgpath = ROOT_PATH + 'targets/'
predpath = ROOT_PATH + 'Scanpath_baselines/PathGAN_finetuned/scanpaths/csv/'

list_img_targets = natsorted(glob.glob(imgpath+'*.png'))
ED= []
#len(list_img_targets)
for idx in range(0, len(list_img_targets), 6):
    imgfullpath = list_img_targets[idx]
    with Image.open(imgfullpath) as im:
        width, height = im.size # original image size
        print('working on %s'% list_img_targets[idx])

        imgname = imgfullpath.replace(imgpath,'')
        imgname = imgname.replace('.png','')
        predCsv = predpath + '%s.png.csv'%imgname
        df_pred = pd.read_csv(predCsv)

        gtpath, gt_files = get_gt_files(imgname, 'csv', truncate=True)

        for pp in range(1, len(gt_files) + 1):
            df_pred.columns = ['user', 'x', 'y', 'time']
            df_predI = df_pred[df_pred['user'] == pp]
            df_predI = df_predI.drop(['user','time'] ,axis=1)
            #df_predI = df_predI.drop(['user'] ,axis=1)

            # Best metrics for each Pred
            tmp = 0

            for gg in range(len(gt_files)):
                df_gt = pd.read_csv(os.path.join(gtpath, gt_files[gg]), header=None)
                df_gt.columns = ['index', 'x', 'y', 'time']
                df_gt = df_gt.drop(['index','time'],axis=1)
                #df_gt = df_gt.drop(['index'],axis=1)

                #ed_idx = euclidean_distance(truncate_EOS(df_predI),df_gt.to_numpy())
                #ed_idx = DTW(truncate_EOS(df_predI),df_gt)
                #ed_idx = TDE(truncate_EOS(df_predI),df_gt, distance_mode='Mean')
                ed_idx = scaled_TDE(truncate_EOS(df_predI),df_gt, im)

                if (ed_idx > tmp):
                    tmp = ed_idx

            ED.append(tmp)

ED = np.asarray(ED, dtype=float)
print(np.round(np.mean(ED), 3))
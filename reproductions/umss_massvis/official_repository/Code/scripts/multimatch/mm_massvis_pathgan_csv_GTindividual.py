# -*- coding: utf-8 -*-
"""
Created on 20210316

@author: Yao Wang
@purpose: to compute the Multimatch metrics of Saltinet-postprocess predictions one by one to the GT scanpaths in MASSVIS dataset
@output : MM full results in _full.npz, and the m*m shape cost matrix for the input of Hungarian matrix

"""

import pandas as pd
import numpy as np
import glob, os, sys
from PIL import Image
from natsort import natsorted
pd.options.mode.chained_assignment = None
from multimatch import docomparison
sys.path.append("..") 
from preprocess_utils import get_gt_files

"""
Please replace the following lines with your file locations
"""
ROOT_PATH = '/netpool/homes/wangyo/Dataset/Massvis/targets393/'
imgpath = ROOT_PATH + 'targets/'
predpath = ROOT_PATH + 'Scanpath_baselines/PathGAN_finetuned/scanpaths/csv/'
outpath = ROOT_PATH + 'Scanpath_baselines/PathGAN_finetuned/MM_npz_5s_new/'

list_img_targets = natsorted(glob.glob(imgpath+'*.png'))
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

        # 5s: truncate=True, 10s: truncate=False
        gtpath, gt_files = get_gt_files(imgname, 'csv', truncate=True)
        np_mm = []
        np_mm_hung = np.zeros((len(gt_files), len(gt_files)))
        for gg in range(len(gt_files)):
            np_mm.append([])
            df_gt = pd.read_csv(os.path.join(gtpath, gt_files[gg]), header=None)
            df_gt.columns = ['index', 'x', 'y', 'time']
            df_gt = df_gt.drop(['index'],axis=1)

            for pp in range(1, len(gt_files) + 1):
                df_pred.columns = ['user', 'x', 'y', 'time']
                df_predI = df_pred[df_pred['user'] == pp]
                df_predI = df_predI.drop(['user'] ,axis=1)
                df_predI = df_predI[0:int(df_predI.shape[0] * 0.6)]

                MM_idx = docomparison(df_predI.to_numpy(),df_gt.to_numpy(), sz = [width,height])
                np_mm_hung[pp-1][gg] = 1 - np.sum(MM_idx[0])/5
                np_mm[-1].append(MM_idx[0])
    
        outpath_full = outpath + 'VAL0_%s_full'%imgname
        outpath_hung = outpath + 'VAL0_%s_hung'%imgname
        np.savez_compressed(outpath_full, np_mm)
        np.savez_compressed(outpath_hung, np_mm_hung)

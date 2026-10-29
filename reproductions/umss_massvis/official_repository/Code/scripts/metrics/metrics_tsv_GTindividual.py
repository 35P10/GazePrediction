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
from metrics import euclidean_distance, TDE, DTW, scaled_TDE
sys.path.append("..") 
from preprocess_utils import get_gt_files, truncate_EOS

ROOT_PATH = '/netpool/homes/wangyo/Dataset/Massvis/targets393/'

imgpath = ROOT_PATH + 'targets/'
#predpath = ROOT_PATH + 'PathGAN_finetuned_10/tsv/'
predpath = ROOT_PATH + 'MDSGAN/Salicon_MDSGAN_ms36_wloss_reduce0_dp0.1_rdp0.1_20210401-172339gen-epoch:97.h5/tsv/'


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
        predCsv = predpath + '%s.png.tsv'%imgname
        df_pred = pd.read_csv(predCsv, header=None, sep='\t')
        df_pred.columns = ['x', 'y', 'time']

        ########## For 3D metrics, comment this line out
        #df_pred = df_pred.drop(['time'] ,axis=1)

        gtpath, gt_files = get_gt_files(imgname, 'csv', truncate=True)

        for gg in range(len(gt_files)):
            df_gt = pd.read_csv(os.path.join(gtpath, gt_files[gg]), header=None)
            df_gt.columns = ['index', 'x', 'y', 'time']
            ########## 2D Metrics
            #df_gt = df_gt.drop(['index','time'],axis=1)
            ########## 3D Metrics
            df_gt = df_gt.drop(['index'],axis=1)

            #ed_idx = euclidean_distance(df_pred.to_numpy(),df_gt.to_numpy())
            #ed_idx = TDE(df_pred.to_numpy(),df_gt.to_numpy(), distance_mode='Mean')
            #ed_idx = scaled_TDE(df_pred.to_numpy(),df_gt.to_numpy(), im)
            ed_idx = DTW(truncate_EOS(df_pred),df_gt.to_numpy())

            ED.append(ed_idx)

ED = np.asarray(ED, dtype=float)
print(np.round(np.mean(ED), 3))
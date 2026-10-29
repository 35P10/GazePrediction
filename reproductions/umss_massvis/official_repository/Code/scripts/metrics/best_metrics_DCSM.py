# -*- coding: utf-8 -*-
"""
Created on 20210320

@author: Yao Wang
@purpose: to compute the scanpath metrics of nc2020 predictions to the individual GT scanpaths in MASSVIS dataset
@output : the final avg. Metrics
"""

import pandas as pd
import numpy as np
import glob, os, sys
sys.path.append("..")
sys.path.append("../..")

from PIL import Image
from natsort import natsorted
pd.options.mode.chained_assignment = None
from metrics import euclidean_distance, scaled_TDE, TDE, DTW, eyenalysis
from preprocess_utils import get_gt_files

ROOT_PATH = '/netpool/homes/wangyo/Dataset/Massvis/targets393/'
imgpath = ROOT_PATH + 'targets/'
predpath = ROOT_PATH + 'Scanpath_baselines/nc2020/scanpaths/'


list_img_targets = natsorted(glob.glob(imgpath+'*.png'))

LL = []
ED = []
for idx in range(0, len(list_img_targets), 6):
    imgfullpath = list_img_targets[idx]
    with Image.open(imgfullpath) as im:
        width, height = im.size # original image size
        print('working on %s'% list_img_targets[idx])
        imgname = imgfullpath.replace(imgpath,'')
        imgname = imgname.replace('.png','')

        f = open('%s%s.txt'%(predpath, imgname), "r")
        line = f.readline().split()

        gtpath, gt_files = get_gt_files(imgname, 'csv', truncate=True)

        ########## 2D Metrics
        pred_idx = np.zeros((int(len(line) / 3), 2))
        ########## 3D Metrics
        #pred_idx = np.zeros((int(len(line) / 3), 3))

        for ll in range(int(len(line) / 3)):
            if(int(line[3 * ll]) > 0):
                pred_idx[ll][0] = line[3 * ll]
                pred_idx[ll][1] = line[3 * ll + 1]
                #pred_idx[ll][2] = line[3 * ll + 2] # 2D Metrics, comment this out

        LL.append(int(len(line) / 3))

        tmp = 0
        for gg in range(len(gt_files)):
            df_gt = pd.read_csv(os.path.join(gtpath, gt_files[gg]), header=None)
            df_gt.columns = ['index', 'x', 'y', 'time']
            ########## 2D Metrics
            df_gt = df_gt.drop(['index','time'],axis=1)
            ########## 3D Metrics
            #df_gt = df_gt.drop(['index'],axis=1)


            #ed_idx = euclidean_distance(pred_idx,df_gt.to_numpy())
            ed_idx = scaled_TDE(pred_idx, df_gt.to_numpy(), im)
            #ed_idx = TDE(pred_idx, df_gt.to_numpy(),  distance_mode='Mean')
            #ed_idx = DTW(pred_idx, df_gt.to_numpy())
            #print(ed_idx)
            if (ed_idx > tmp):
                tmp = ed_idx

        ED.append(tmp)

print(np.std(np.array(LL)))
ED = np.asarray(ED, dtype=float)
print(np.round(np.mean(ED), 3))
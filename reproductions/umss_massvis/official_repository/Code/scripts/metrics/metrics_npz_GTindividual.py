# -*- coding: utf-8 -*-
"""
Created on 20210320

@author: Yao Wang
@purpose: to compute the Euclidean metrics of Meanshifted GT to the individual GT scanpaths in MASSVIS dataset
@output : the final avg. Metrics
"""

import pandas as pd
import numpy as np
import glob, os, sys
from sklearn.cluster import MeanShift, estimate_bandwidth
from PIL import Image
from natsort import natsorted
pd.options.mode.chained_assignment = None
from metrics import euclidean_distance, scaled_TDE, DTW, TDE
sys.path.append("..") 
from preprocess_utils import get_gt_files

ROOT_PATH = '/netpool/homes/wangyo/Dataset/Massvis/targets393/'

imgpath = ROOT_PATH + 'targets/'
mit393_pred_path = '../../massvis393/massvis_full_raw_55_qv0.0285.npz'
#mit393_pred_path = '../../massvis393/massvis_full_raw_36_qv0.0400.npz'
#mit393_pred_path = '../..//massvis393/massvis_full_raw_12_qv0.0800.npz'
#mit393_pred_path = '../..//massvis393/massvis_full_raw_10_qv0.0800.npz'
mit393_pred = np.load(mit393_pred_path,allow_pickle='True')
mit393_pred = mit393_pred['arr_0']
#print(np.shape(mit393_pred))

list_img_targets = natsorted(glob.glob(imgpath+'*.png'))

ED= []
for idx in range(0, np.shape(mit393_pred)[0], 6):
    imgname = mit393_pred[idx][0][6]
    width = mit393_pred[idx][0][7]
    height = mit393_pred[idx][0][8]

    print('working on %s'% imgname)

    gtpath, gt_files = get_gt_files(imgname, 'csv')

    ########## 2D Metrics
    pred_idx = mit393_pred[idx,:,[0,1]]
    ########## 3D Metrics
    #pred_idx = mit393_pred[idx,:,[0,1,3]]

    pred_idx = pred_idx.swapaxes(0,1)

    imgfullpath = imgpath + mit393_pred[idx,0,6] + '.png'
    with Image.open(imgfullpath) as im:
        for gg in range(len(gt_files)):
            df_gt = pd.read_csv(os.path.join(gtpath, gt_files[gg]), header=None)
            df_gt.columns = ['index', 'x', 'y', 'time']
            ########## 2D Metrics
            df_gt = df_gt.drop(['index', 'time'],axis=1)
            ########## 3D Metrics
            #df_gt = df_gt.drop(['index'],axis=1)

            #ed_idx = euclidean_distance(pred_idx,df_gt.to_numpy())
            ed_idx = DTW(pred_idx,df_gt.to_numpy())
            #ed_idx = TDE(pred_idx,df_gt.to_numpy(), distance_mode='Mean')
            #ed_idx = scaled_TDE(pred_idx,df_gt.to_numpy(), im)


            ED.append(ed_idx)
 
ED = np.asarray(ED, dtype=float)
print(np.round(np.mean(ED), 3))
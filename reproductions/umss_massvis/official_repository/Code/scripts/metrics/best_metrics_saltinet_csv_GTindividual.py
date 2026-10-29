# -*- coding: utf-8 -*-
"""
Created on 20210316

@author: Yao Wang
@purpose: to compute the scanpath metrics of Saltinet-postprocess predictions one by one to the GT scanpaths in MASSVIS dataset
@output : the final avg. Metrics
"""

import pandas as pd
import numpy as np
import glob,os,sys
from PIL import Image
from natsort import natsorted
pd.options.mode.chained_assignment = None
from metrics import euclidean_distance, DTW, TDE, scaled_TDE, eyenalysis
sys.path.append("..") 
from preprocess_utils import get_gt_files, truncate_EOS

"""
Please replace the following lines with your file locations
"""
ROOT_PATH = '/netpool/homes/wangyo/Dataset/Massvis/targets393/'
imgpath = ROOT_PATH + 'targets/'


#predpath = ROOT_PATH + 'GT/val/saltinet/10s/csv/'
#predpath = ROOT_PATH + 'Scanpath_baselines/DVS/scanpaths/'
#predpath = ROOT_PATH + 'Scanpath_baselines/DVS/scanpaths_5s/'
#predpath = ROOT_PATH + 'Scanpath_baselines/UMSI/scanpaths/'
#predpath = ROOT_PATH + 'Scanpath_baselines/UMSI/scanpaths_5s/'
#predpath = ROOT_PATH + 'Scanpath_baselines/Saltinet/scanpaths/csv/'
#predpath = ROOT_PATH + 'MDSEM_finetuned/scanpaths/'
#predpath = ROOT_PATH + 'MDSEM_bucket_500_2000_5000/scanpaths/'
predpath = ROOT_PATH + 'MDSEM_bucket_500_2000_5000/scanpaths_5s/csv/'
#predpath = ROOT_PATH + 'MDSEM_bucket_500_2000_5000/scanpaths_5s_nsample_foveal_right/csv/'
#predpath = ROOT_PATH + 'MDSEM_bucket_3000_5000_10000/scanpaths_10s_nsamples/csv/'

list_img_targets = natsorted(glob.glob(imgpath+'*.png'))

ED = []
for idx in range(0, len(list_img_targets), 6):
    imgfullpath = list_img_targets[idx]
    with Image.open(imgfullpath) as im:
        width, height = im.size # original image size
        print('working on %s'% list_img_targets[idx])

        imgname = imgfullpath.replace(imgpath,'')
        imgname = imgname.replace('.png','')
        predCsv = predpath + '%s.png.csv'%imgname

        if not os.path.exists(predCsv): continue

        df_pred = pd.read_csv(predCsv)

        # 10s / 5s GT
        #gtpath, gt_files = get_gt_files(imgname, 'csv', truncate=False)
        gtpath, gt_files = get_gt_files(imgname, 'csv', truncate=True)

        for pp in range(1, len(gt_files) + 1):
            df_pred.columns = ['user', 'index', 'time', 'x', 'y']
            df_predI = df_pred[df_pred['user'] == pp]
            ########## 2D Metrics
            df_predI = df_predI.drop(['user', 'index', 'time'] ,axis=1)
            ########## 3D Metrics
            #df_predI = df_predI.drop(['user', 'index'] ,axis=1)

            #df_predI = df_predI[0:int(df_predI.shape[0]/2)]


            # Normalization required
            df_predI['x'] /= 640
            df_predI['y'] /= 480
            df_predI['x'] *= width
            df_predI['y'] *= height

            # Best metrics for each Pred
            tmp = 0

            for gg in range(len(gt_files)):
                df_gt = pd.read_csv(os.path.join(gtpath, gt_files[gg]), header=None)
                df_gt.columns = ['index', 'x', 'y', 'time']
                ########## 2D Metrics
                df_gt = df_gt.drop(['index','time'],axis=1)
                ########## 3D Metrics
                #df_gt = df_gt.drop(['index'],axis=1)
            
                #ed_idx = euclidean_distance(df_predI,df_gt)
                #ed_idx = DTW(df_predI, df_gt.to_numpy())
                #ed_idx = TDE(df_predI,df_gt.to_numpy(), distance_mode='Mean')
                ed_idx = scaled_TDE(df_predI,df_gt.to_numpy(), im)

                if (ed_idx > tmp):
                    tmp = ed_idx

            #print(tmp)
            ED.append(tmp)

ED = np.asarray(ED, dtype=float)
print(predpath, np.round(np.mean(ED), 3))
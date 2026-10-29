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
from scipy.stats import exponnorm
from PIL import Image
from natsort import natsorted
pd.options.mode.chained_assignment = None
from multimatch import docomparison
sys.path.append("..") 
from preprocess_utils import get_gt_files

expon_params = (5.114058378644653, 124.0602219896999, 17.48864970980731)

"""
Please replace the following lines with your file locations
"""
ROOT_PATH = '/netpool/homes/wangyo/Dataset/Massvis/targets393/'
imgpath = ROOT_PATH + 'targets/'
#predpath = ROOT_PATH + 'DVS/scanpaths/'
#predpath = ROOT_PATH + 'Scanpath_baselines/Saltinet/scanpaths/'
#predpath = ROOT_PATH + 'Scanpath_baselines/MDSEM_finetuned/scanpaths_saltinet/'
#predpath = ROOT_PATH + '/MDSEM_bucket_500_2000_5000/scanpaths_5s_nsamples/csv/'
#npzpath = ROOT_PATH + '/MDSEM_bucket_500_2000_5000/scanpaths_5s_nsamples/MM_npz/'
predpath = ROOT_PATH + '/MDSEM_bucket_500_2000_5000/scanpaths_5s_nsamples_new/csv/'
npzpath = ROOT_PATH + '/MDSEM_bucket_500_2000_5000/scanpaths_5s_nsamples_new/MM_npz_new/'


list_img_targets = natsorted(glob.glob(imgpath+'*.png'))
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

        np_mm = []
        np_mm_hung = np.zeros((len(gt_files), len(gt_files)))
        for gg in range(len(gt_files)):
            np_mm.append([])
            df_gt = pd.read_csv(os.path.join(gtpath, gt_files[gg]), header=None)
            df_gt.columns = ['index', 'x', 'y', 'time']
            df_gt = df_gt.drop(['index'],axis=1)
            #print(df_gt)

            for pp in range(1, len(gt_files) + 1):
                df_pred.columns = ['user', 'index', 'time', 'x', 'y']
                df_predI = df_pred[df_pred['user'] == pp]
                df_predI = df_predI.drop(['user', 'index'] ,axis=1)
                df_predI = df_predI[['x', 'y', 'time']]
                #df_predI['time'] *= 1000
                rrr = exponnorm.rvs(expon_params[0], loc=expon_params[1], scale=expon_params[2], size=df_predI.shape[0])
                df_predI['time'] = exponnorm.rvs(expon_params[0], loc=expon_params[1], scale=expon_params[2], size=df_predI.shape[0])

                #print(df_predI)

                #re_pred_sp       =  rescale_data(df_predI.to_numpy(),640/width,480/height)
                #re_np_gt_sp      =  rescale_data(df_gt.to_numpy(),640/width,480/height)

                #print(re_pred_sp)
                #print(re_np_gt_sp)

                MM_idx = docomparison(df_predI.to_numpy(),df_gt.to_numpy(), sz = [width,height])
                #print(MM_idx)
                np_mm_hung[pp-1][gg] = 1 - np.sum(MM_idx[0])/5
                np_mm[-1].append(MM_idx[0])
        outpath_full = npzpath + 'VAL0_%s_full'%imgname
        outpath_hung = npzpath + 'VAL0_%s_hung'%imgname
        np.savez_compressed(outpath_full, np_mm)
        np.savez_compressed(outpath_hung, np_mm_hung)

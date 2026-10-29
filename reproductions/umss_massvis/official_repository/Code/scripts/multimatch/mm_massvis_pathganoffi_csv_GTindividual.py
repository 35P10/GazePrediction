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
predpath = ROOT_PATH + 'PathGAN_official/scanpaths/'

list_img_targets = natsorted(glob.glob(imgpath+'*.png'))
#len(list_img_targets)
MM = []
for idx in range(0, 200, 6):
    imgfullpath = list_img_targets[idx]
    with Image.open(imgfullpath) as im:
        width, height = im.size # original image size
        print('working on %s'% list_img_targets[idx])

        imgname = imgfullpath.replace(imgpath,'')
        imgname = imgname.replace('.png','')
        predCsv = predpath + '%s.csv'%imgname
        if not os.path.exists(predCsv): continue

        df_pred = pd.read_csv(predCsv)
        df_pred.columns = ['x', 'y', 'timestamp', 'nan']

        df_pred['x'] *= width
        df_pred['y'] *= height
        df_pred['time'] = df_pred['timestamp'] - df_pred['timestamp'].shift(1)
        df_pred['time'].iloc[0] = df_pred['timestamp'].iloc[0]
        df_pred['x'] = df_pred['x'].astype('int')
        df_pred['y'] = df_pred['y'].astype('int')
        df_pred['time'] = df_pred['time'].astype('int')

        df_pred = df_pred.drop(['timestamp','nan'] ,axis=1)
        #print(df_pred)
        
        gtpath, gt_files = get_gt_files(imgname, 'csv')

        for gg in gt_files:
            df_gt = pd.read_csv(os.path.join(gtpath, gg), header=None)
            df_gt.columns = ['index', 'x', 'y', 'time']
            df_gt = df_gt.drop(['index'],axis=1)
            #print(df_gt)

            MM_idx = docomparison(df_pred.to_numpy(),df_gt.to_numpy(), sz = [width,height])
            #print(MM_idx)
            MM.append(MM_idx[0])

mm=np.asarray(MM)
print(np.shape(mm))

print('Vector:',np.round(np.mean(mm[:,:,0]),3),'Direction:',np.round(np.mean(mm[:,:,1]),3), \
      'Length:',np.round(np.mean(mm[:,:,2]),3),'Position:',np.round(np.mean(mm[:,:,3]),3), \
      'Duration:',np.round(np.mean(mm[:,:,4]),3),'Average:',np.round(np.mean(mm),3))

max_index_col = np.argmax(mm[:,:,1], axis=0)

print(max_index_col)
print(mm[max_index_col,:,:])
print(np.shape(mm))
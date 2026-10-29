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
#predpath = ROOT_PATH + 'PathGAN_finetuned_10/tsv/'
predpath = ROOT_PATH + 'MDSGAN/Salicon_MDSGAN_ms36_wloss_reduce0_dp0.1_rdp0.1_20210401-172339gen-epoch:97.h5/tsv/'

MM_tmp = []
MM = []
list_img_targets = natsorted(glob.glob(imgpath+'*.png'))
#len(list_img_targets)
for idx in range(0, len(list_img_targets), 6):
    MM_tmp.append([])
    imgfullpath = list_img_targets[idx]
    with Image.open(imgfullpath) as im:
        width, height = im.size # original image size
        print('working on %s'% list_img_targets[idx])

        imgname = imgfullpath.replace(imgpath,'')
        imgname = imgname.replace('.png','')
        predCsv = predpath + '%s.png.tsv'%imgname
        df_pred = pd.read_csv(predCsv, header=None, sep='\t')
        df_pred.columns = ['x', 'y', 'time']
        #print(df_predI)

        gtpath, gt_files = get_gt_files(imgname, 'csv', truncate=True)
        for gg in range(len(gt_files)):
            df_gt = pd.read_csv(os.path.join(gtpath, gt_files[gg]), header=None)
            df_gt.columns = ['index', 'x', 'y', 'time']
            df_gt = df_gt.drop(['index'],axis=1)
            #print(df_gt)

            MM_idx = docomparison(df_pred.to_numpy(),df_gt.to_numpy(), sz = [width,height])
            MM_tmp[-1].append(MM_idx[0])
        mm = np.asarray(MM_tmp[-1], dtype=float)
        print('Vector:',np.round(np.mean(mm[:,0]),3),'Direction:',np.round(np.mean(mm[:,1]),3), \
            'Length:',np.round(np.mean(mm[:,2]),3),'Position:',np.round(np.mean(mm[:,3]),3), \
            'Duration:',np.round(np.mean(mm[:,4]),3),'Average:',np.round(np.mean(mm),3))
        MM.append([np.mean(mm[:,0]), np.mean(mm[:,1]), np.mean(mm[:,2]), np.mean(mm[:,3]), np.mean(mm[:,4]) ])


mm=np.asarray(MM, dtype=float)
print(np.shape(mm))
print(mm[0])
print('Vector:',np.round(np.mean(mm[:,0]),3),'Direction:',np.round(np.mean(mm[:,1]),3), \
    'Length:',np.round(np.mean(mm[:,2]),3),'Position:',np.round(np.mean(mm[:,3]),3), \
    'Duration:',np.round(np.mean(mm[:,4]),3),'Average:',np.round(np.mean(mm),3))

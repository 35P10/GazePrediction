# -*- coding: utf-8 -*-
"""
Created on 20210316

@author: Yao Wang
@purpose: to compute the Multimatch metrics between GT scanpaths in MASSVIS dataset
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

MM_tmp = []
MM = []
list_img_targets = natsorted(glob.glob(imgpath+'*.png'))

#for idx in range(30, 31, 1):
#for idx in range(72, 73, 1):
#for idx in range(96, 97, 1):
#for idx in range(204, 205, 1):
#for idx in range(258, 259, 1):
for idx in range(0, len(list_img_targets), 6):
    MM_tmp.append([])
    imgfullpath = list_img_targets[idx]
    with Image.open(imgfullpath) as im:
        width, height = im.size # original image size
        print('working on %s'% list_img_targets[idx])

        imgname = imgfullpath.replace(imgpath,'')
        imgname = imgname.replace('.png','')

        gtpath, gt_files = get_gt_files(imgname, 'csv', truncate=True)

        #for pp in range(5,6):
        for pp in range(len(gt_files)):
            #print(gt_files[pp])
            df_pred = pd.read_csv(os.path.join(gtpath, gt_files[pp]), header=None)
            df_pred.columns = ['index','x', 'y', 'time']
            df_pred = df_pred.drop(['index'],axis=1)

            for gg in range(len(gt_files)):
                df_gt = pd.read_csv(os.path.join(gtpath, gt_files[gg]), header=None)
                df_gt.columns = ['index', 'x', 'y', 'time']
                df_gt = df_gt.drop(['index'],axis=1)

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

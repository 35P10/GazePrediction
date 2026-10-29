# -*- coding: utf-8 -*-
"""
Created on 20210316

@author: Yao Wang
@purpose: to compute the Multimatch metrics of meanshifted GT to the individual GT scanpaths in MASSVIS dataset
@output : MM score of each image and the final avg. MM score

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

ROOT_PATH = '/netpool/homes/wangyo/Dataset/Massvis/targets393/'


"""
function input: csv file containing all users data for a image e.g (tj.csv) and the corresponding image e.g(tj.png)
function output: a dataframe which is similar to the input of pathgan
 df_massivis
 x :  logitude
 y : latitude
 EOS : End of sequence neuron
 td : time duration
 st : start time
 et : end time
 image name: name of the image
 The height and width of the image is also taken into account as the each image has a different dimension
 image width : width of the image
 image height : height of the image
 
"""

"""
Please replace the following lines with your file locations
"""

imgpath = ROOT_PATH + 'targets/'
#mit393_pred_path = '/netpool/homes/wangyo/Projects/Massvis_code/massvis393/massvis_full_raw_55_qv0.0285.npz'
mit393_pred_path = '/netpool/homes/wangyo/Projects/Massvis_code/massvis393/massvis_full_raw_10_qv0.0800.npz'

mit393_pred = np.load(mit393_pred_path,allow_pickle='True')
mit393_pred = mit393_pred['arr_0']
print(np.shape(mit393_pred))

list_img_targets = natsorted(glob.glob(imgpath+'*.png'))


MM_tmp = []
MM = []
for idx in range(0, np.shape(mit393_pred)[0], 6):
    MM_tmp.append([])
    imgname = mit393_pred[idx][0][6]
    width = mit393_pred[idx][0][7]
    height = mit393_pred[idx][0][8]
    
    print('working on %s'% imgname)

    gtpath, gt_files = get_gt_files(imgname, 'csv')

    pred_idx = mit393_pred[idx,:,[0,1,3]]
    for gg in range(len(gt_files)):
        df_gt = pd.read_csv(os.path.join(gtpath, gt_files[gg]), header=None)
        df_gt.columns = ['index', 'x', 'y', 'time']
        df_gt = df_gt.drop(['index'],axis=1)
        #print(df_gt)

        MM_idx = docomparison(pred_idx,df_gt.to_numpy(), sz = [width,height])
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

# -*- coding: utf-8 -*-
"""
Created on 20210316

@author: Yao Wang
@process:
    1. run hungarian to make correspondence between between GT and preds
    2. load precomputed Multimatch metrics
    3. generate final avg.MM score
@output : The final avg. MM score

"""

from hungarian import Hungarian
import numpy as np
import glob,os
from natsort import natsorted

#ROOT_PATH = '/netpool/homes/wangyo/Dataset/Massvis/targets393/Scanpath_baselines/Saltinet/MM_npz_5s/'
#ROOT_PATH = '/netpool/homes/wangyo/Dataset/Massvis/targets393/Scanpath_baselines/MDSEM_finetuned/MM_npz/'
#ROOT_PATH = '/netpool/homes/wangyo/Dataset/Massvis/targets393/Scanpath_baselines/PathGAN_finetuned/MM_npz_5s_new/'
ROOT_PATH = '/netpool/homes/wangyo/Dataset/Massvis/targets393/MDSEM_bucket_500_2000_5000/scanpaths_5s_nsamples_new/MM_npz_new/'
PATH_HUNGARIAN_MATRIXs = natsorted(glob.glob(ROOT_PATH + '*hung.npz'))

#print(len(PATH_HUNGARIAN_MATRIXs))
#PATH_HUNGARIAN_MATRIXs = [ROOT_PATH + 'VAL0_economist_daily_chart_258_hung.npz']
#PATH_HUNGARIAN_MATRIXs = [ROOT_PATH + 'VAL0_treasuryK03_3_hung.npz']
#PATH_HUNGARIAN_MATRIXs = [ROOT_PATH + 'VAL0_v484_n7393_6_f4_hung.npz']
#PATH_HUNGARIAN_MATRIXs = [ROOT_PATH + 'VAL0_visMost93_hung.npz']
#PATH_HUNGARIAN_MATRIXs = [ROOT_PATH + 'VAL0_whoB22_1_hung.npz']
MM=[]

for PATH_HUNGARIAN_MATRIX in PATH_HUNGARIAN_MATRIXs:
    print(PATH_HUNGARIAN_MATRIX)
    hungarian_matrix = np.load(PATH_HUNGARIAN_MATRIX,allow_pickle='True')
    hungarian_matrix = hungarian_matrix['arr_0']
    PATH_MM = PATH_HUNGARIAN_MATRIX.replace('hung', 'full')
    np_mm = np.load(PATH_MM,allow_pickle='True')
    np_mm = np_mm['arr_0']

    # No Hungarian
    #ll = np.shape(np_mm)[0]
    #for j in range(ll):
    #  MM.append(np_mm[3][j])
    #    for j in range(ll):
    #        MM.append(np_mm[i][j])

    # Hungarian
    hungarian = Hungarian(hungarian_matrix)
    hungarian.calculate()
    res = hungarian.get_results()
    for pair in res:
        MM.append(np_mm[pair[0]][pair[1]])

mm=np.asarray(MM)
print('Vector:',np.round(np.mean(mm[:,0]),3),'Direction:',np.round(np.mean(mm[:,1]),3), \
    'Length:',np.round(np.mean(mm[:,2]),3),'Position:',np.round(np.mean(mm[:,3]),3), \
    'Duration:',np.round(np.mean(mm[:,4]),3),'Average:',np.round(np.mean(mm),3))

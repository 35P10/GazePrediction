# -*- coding: utf-8 -*-
"""
Created on 20210320

@author: Yao Wang
@purpose: to convert csv scanpaths to .tsv format as the input of scanmatch
@output : .tsv scanpaths
"""

import os
import pandas as pd

SUBJECTS=40

#path = '/netpool/homes/wangyo/Dataset/Massvis/targets393/Saltinet/scanpaths/'
#outpath = '/netpool/homes/wangyo/Dataset/Massvis/targets393/Saltinet/tsv/'
#outpath = '/netpool/homes/wangyo/Projects/ws2020_sruthisaiteja/Code/metrics/ScanMatch/Model_data/all_mit393_saltinet_raw/'

#path = '/netpool/homes/wangyo/Dataset/Massvis/targets393/Scanpath_baselines/UMSI/scanpaths_5s/csv/'
#outpath = '/netpool/homes/wangyo/Dataset/Massvis/targets393/Scanpath_baselines/UMSI/scanpaths_5s/tsv/'

#path = '/netpool/homes/wangyo/Dataset/Massvis/targets393/GT/val/saltinet/5s/csv/'
#outpath = '/netpool/homes/wangyo/Dataset/Massvis/targets393/GT/val/saltinet/5s/tsv/'

#path = '/netpool/homes/wangyo/Dataset/Massvis/targets393/Scanpath_baselines/MDSEM_finetuned/scanpaths/'
#outpath = '/netpool/homes/wangyo/Dataset/Massvis/targets393/Scanpath_baselines/MDSEM_finetuned/tsv/'

#path = '/netpool/homes/wangyo/Dataset/Massvis/targets393/PathGAN_finetuned/scanpaths/'
#outpath = '/netpool/homes/wangyo/Dataset/Massvis/targets393/PathGAN_finetuned/tsv/'

path = '/netpool/homes/wangyo/Dataset/Massvis/targets393/MDSEM_bucket_500_2000_5000/scanpaths_5s_nsample_foveal_right/csv/'
outpath = '/netpool/homes/wangyo/Dataset/Massvis/targets393/MDSEM_bucket_500_2000_5000/scanpaths_5s_nsample_foveal_right/tsv/'

#path = '/netpool/homes/wangyo/Dataset/Massvis/targets393/MDSEM_bucket_3000_5000_10000/scanpaths_10s_nsamples/csv/'
#outpath = '/netpool/homes/wangyo/Dataset/Massvis/targets393/MDSEM_bucket_3000_5000_10000/scanpaths_10s_nsamples/tsv/'



for csvfile in os.listdir(path):
    csvpath = '%s%s'%(path, csvfile)

    if os.path.exists(csvpath):
        f = open(csvpath, "r")
    else:
        continue;
    df_full = pd.read_csv(csvpath)
    df_full.columns = ['user', 'index', 'duration', 'start_x', 'start_y']
    print('working on %s'%(csvpath))
    imgname = csvpath.replace(path,'')
    imgname = imgname.replace('.png.csv','')

    for i in range(1,SUBJECTS+1):
        sp = df_full.loc[df_full['user']==i]
        sp = sp.drop(['user', 'index'],axis=1)
        sp = sp.reindex(columns=['start_x', 'start_y', 'duration'])

        os.makedirs(outpath+imgname, exist_ok=True)

        w_filenameTSV = '%s%s/%s_%d.tsv'%(outpath, imgname, imgname, i)
        with open(w_filenameTSV,'w') as write_tsv:
            write_tsv.write(sp.to_csv(sep='\t', index=False))

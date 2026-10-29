# -*- coding: utf-8 -*-
"""
Created on 20210320

@author: Yao Wang
@purpose: to convert GT scanpaths to .tsv format as the input of scanmatch
@output : .tsv scanpaths
"""

import pandas as pd
import numpy as np
import glob, os, sys
from PIL import Image
from natsort import natsorted
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
sys.path.append("..") 
from preprocess_utils import get_gt_files

"""
Please replace the following lines with your file locations
"""
ROOT_PATH = '/netpool/homes/wangyo/Dataset/Massvis/targets393/'
imgpath = ROOT_PATH + 'targets/'
outpath = ROOT_PATH + 'gt0/5s/tsv/'

list_img_targets = natsorted(glob.glob(imgpath+'*.png'))
#len(list_img_targets)
for idx in range(0, len(list_img_targets)):
#for idx in range(0, len(list_img_targets), 6):
    imgfullpath = list_img_targets[idx]
    with Image.open(imgfullpath) as im:
        width, height = im.size # original image size
        print('working on %s'% list_img_targets[idx])

        imgname = imgfullpath.replace(imgpath,'')
        imgname = imgname.replace('.png','')

        gtpath, gt_files = get_gt_files(imgname, 'csv', truncate=True)

        os.makedirs(outpath+imgname, exist_ok=True)

        for gg in range(len(gt_files)):
            df_gt = pd.read_csv(os.path.join(gtpath, gt_files[gg]), header=None)
            df_gt.columns = ['index', 'start_x', 'start_y', 'duration']
            df_gt = df_gt.drop(['index'],axis=1)
            #print(df_gt)

            w_filenameTSV = '%s%s/%s_%s'%(outpath, imgname, imgname, gt_files[gg].replace('csv','tsv'))
            with open(w_filenameTSV,'w') as write_tsv:
                write_tsv.write(df_gt.to_csv(sep='\t', index=False))

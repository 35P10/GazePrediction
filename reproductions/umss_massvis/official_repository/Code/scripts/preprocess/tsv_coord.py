# -*- coding: utf-8 -*-
"""
Created on 20210403

@author: Yao Wang
@purpose: to convert tsv scanpaths to a ROI string
@output : .txt string
"""

import os, sys
from PIL import Image
import numpy as np
import pandas as pd
sys.path.append("..") 
from preprocess_utils import load_img_list, get_gt_elements, get_gt_strings, get_idmap, truncate_EOS

ROOT_PATH = '/netpool/homes/wangyo/Dataset/Massvis/targets393'
#ROOT_PATH2 = ROOT_PATH + '/MDSGAN/Salicon_MDSGAN_ms36_wloss_reduce0_dp0.1_rdp0.1_20210401-172339gen-epoch:99.h5'
#ROOT_PATH2 = ROOT_PATH + '/nc2020'
#ROOT_PATH2 = ROOT_PATH + '/Scanpath_baselines/Saltinet/scanpaths/'
#ROOT_PATH2 = ROOT_PATH + '/Scanpath_baselines/DVS/scanpaths_5s_new/'
#ROOT_PATH2 = ROOT_PATH + '/Scanpath_baselines/UMSI/scanpaths_5s/'
#ROOT_PATH2 = ROOT_PATH + '/Scanpath_baselines/MDSEM_finetuned/scanpaths/'
ROOT_PATH2 = ROOT_PATH + '/MDSEM_bucket_500_2000_5000/scanpaths_5s_nsamples_new/'
#ROOT_PATH2 = ROOT_PATH + '/MDSEM_bucket_3000_5000_10000/scanpaths_10s_nsamples/'
#ROOT_PATH2 = ROOT_PATH + '/GT/val/saltinet/5s/'


tsvpath = ROOT_PATH2 + 'tsv/'
outpath = ROOT_PATH2 + 'tsv_new/'

list_img_targets = load_img_list(ROOT_PATH) 

for idx in range(0, len(list_img_targets), 6):
    print('working on %s'% list_img_targets[idx])
    imgfullpath = list_img_targets[idx]
    imgname = imgfullpath.split('/')[-1]
    imname = imgname.replace('.png','')

    os.makedirs(outpath+imname, exist_ok=True)

    for pp in range(1, 41):
        predTsv = '%s%s/%s_%d.tsv'%(tsvpath,imname,imname, pp)

        #fixations = pd.read_csv(predTsv, header=None, sep='\t')
        fixations = pd.read_csv(predTsv, sep='\t')

        fixations.columns = ['start_x', 'start_y', 'duration']

        # load the raw image
        with Image.open(list_img_targets[idx]) as im:
            width, height = im.size # original image size

            fixations['start_x'] /= 640
            fixations['start_y'] /= 480
            fixations['start_x'] *= width
            fixations['start_y'] *= height

            fixations = fixations.astype({'start_x': int, 'start_y': int})

            w_filenameTSV = '%s%s/%s_%d.tsv'%(outpath, imname, imname, pp)
            with open(w_filenameTSV,'w') as write_tsv:
                write_tsv.write(fixations.to_csv(sep='\t', index=False))

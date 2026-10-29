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

BucketStr=['A','X','G','L','O','T','S','D','Z']

tsvpath = ROOT_PATH2 + 'tsv/'
strpath = ROOT_PATH2 + 'str/'

list_img_targets = load_img_list(ROOT_PATH) 

for idx in range(0, len(list_img_targets), 6):
    print('working on %s'% list_img_targets[idx])
    imgfullpath = list_img_targets[idx]
    imgname = imgfullpath.split('/')[-1]
    imname = imgname.replace('.png','')
    elementCoords, elementX, elementY, boxes = get_gt_elements(imname, simple=True)
    for pp in range(1, 41):
        #predTsv = tsvpath + '%s.png.tsv'%imname
        #predTsv = tsvpath + 'ncnc_%s.tsv'%imname
        predTsv = '%s%s/%s_%d.tsv'%(tsvpath,imname,imname, pp)

        #fixations = pd.read_csv(predTsv, header=None, sep='\t')
        fixations = pd.read_csv(predTsv, sep='\t')

        fixations.columns = ['x', 'y', 'time']

        # Truncation
        fixations = truncate_EOS(fixations)

        scanpath_str = ""

        # load the raw image
        with Image.open(list_img_targets[idx]) as im:
            width, height = im.size # original image size
            #Handle Overlap problem
            id_map = get_idmap(boxes, width, height, 9)

            #for row2 in fixations.iterrows():
            #    xx = int(row2[1][0])
            #    yy = int(row2[1][1])
            for i in range(int(fixations.shape[0]*0.6)):
                #xx = int(fixations[i][0])
                #yy = int(fixations[i][1])
                xx = int(int(fixations[i][0]) / 640 * width)
                yy = int(int(fixations[i][1]) / 480 * height)
                s_roi = BucketStr[id_map[xx][yy]]
                scanpath_str += s_roi


            # save scanpath string
            outpath = '%s/%d/'%(strpath, pp)
            if not os.path.exists(outpath):
        	    os.makedirs(outpath)
            file = open('%s%s.txt'%(outpath, imname),'w');
            file.write(str(scanpath_str));
            file.close();

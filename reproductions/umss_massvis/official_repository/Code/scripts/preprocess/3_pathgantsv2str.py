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
ROOT_PATH2 = ROOT_PATH + '/PathGAN_unfinetuned'

BucketStr=['A','X','G','L','O','T','S','D','Z']

tsvpath = ROOT_PATH2 + '/tsv/'
strpath = ROOT_PATH2 + '/str/'
#strpath = ROOT_PATH2 + '/truncate_str/'

list_img_targets = load_img_list(ROOT_PATH) 

for idx in range(0, len(list_img_targets), 6):
    imgfullpath = list_img_targets[idx]
    imgname = imgfullpath.split('/')[-1]
    imname = imgname.replace('.png','')
    elementCoords, elementX, elementY, boxes = get_gt_elements(imname, simple=True)

    predTsv = tsvpath + '%s/%s_2.tsv'%(imname,imname)
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

        for i in range(fixations.shape[0]):
            xx = int(fixations[i][0])
            yy = int(fixations[i][1])
            s_roi = BucketStr[id_map[xx][yy]]
            #print(s_roi)
            scanpath_str += s_roi


        #for row2 in fixations.iterrows():
        #    xx = int(row2[1][0])
        #    yy = int(row2[1][1])
        #    s_roi = BucketStr[id_map[xx][yy]]
        #    scanpath_str += s_roi

        # save scanpath string
        file = open('%s/%s.txt'%(strpath, imname),'w');
        file.write(str(scanpath_str));
        file.close();

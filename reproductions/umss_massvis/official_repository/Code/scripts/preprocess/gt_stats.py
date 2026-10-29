# -*- coding: utf-8 -*-
"""
Created on 20210403

@author: Yao Wang
@purpose: to convert tsv scanpaths to a ROI string
@output : .txt string
"""

import os, sys
from glob import glob
from PIL import Image
import numpy as np
import pandas as pd
sys.path.append("..") 
from preprocess_utils import load_img_list, get_gt_elements, get_gt_strings, get_idmap, truncate_EOS

ROOT_PATH = '/netpool/homes/wangyo/Dataset/Massvis/targets393'
ROOT_PATH2 = ROOT_PATH + '/gt0/5s/'

BucketStr=['A','X','G','L','O','T','S','D','Z']

HIT=0
OFF=0

tsvpath = ROOT_PATH2 + 'tsv/'

list_img_targets = load_img_list(ROOT_PATH) 

for idx in range(0, len(list_img_targets)):
    print('working on %s'% list_img_targets[idx])
    imgfullpath = list_img_targets[idx]
    imgname = imgfullpath.split('/')[-1]
    imname = imgname.replace('.png','')
    elementCoords, elementX, elementY, boxes = get_gt_elements(imname, simple=True)
    curdir = '%s%s'%(tsvpath,imname)

    print(curdir+'/*.tsv')
    curfiles = glob(curdir+'/*.tsv')
    for predTsv in curfiles:
        #print(predTsv)
        #fixations = pd.read_csv(predTsv, header=None, sep='\t')
        fixations = pd.read_csv(predTsv, sep='\t')

        fixations.columns = ['x', 'y', 'time']

        # load the raw image
        with Image.open(list_img_targets[idx]) as im:
            width, height = im.size # original image size
            #Handle Overlap problem
            id_map = get_idmap(boxes, width, height, 9)

            for row2 in fixations.iterrows():
                xx = int(row2[1][0])
                yy = int(row2[1][1])
                s_roi = BucketStr[id_map[xx][yy]]
                if s_roi == 'Z': OFF+=1
                else: HIT+=1
    print(HIT / (HIT+OFF))

print(HIT / (HIT+OFF))


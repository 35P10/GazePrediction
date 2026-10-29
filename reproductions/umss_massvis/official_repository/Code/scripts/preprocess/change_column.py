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

ROOT_PATH = '/netpool/homes/wangyo/Dataset/Massvis/targets393/targets'
ROOT_PATH2 = '/netpool/homes/wangyo/Dataset/Massvis/Selected-40/tsv/Ours/'

tsvpath = ROOT_PATH2
outpath = ROOT_PATH2

list_img_targets = load_img_list(ROOT_PATH) 

for predTsv in os.listdir(tsvpath):
    print('working on %s'% predTsv)
    #fixations = pd.read_csv(predTsv, header=None, sep='\t')
    fixations = pd.read_csv(tsvpath + predTsv, sep='\t')

    fixations.columns = ['start_x', 'start_y', 'duration']

    with open(tsvpath + predTsv,'w') as write_tsv:
        write_tsv.write(fixations.to_csv(sep='\t', index=False))

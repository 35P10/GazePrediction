# -*- coding: utf-8 -*-
"""
Created on 20210405

@author: Yao Wang
@purpose: to convert GT scanpaths to .npz format for dataloader
@output : .npz
"""

import pandas as pd
import numpy as np
import glob,re,os
from PIL import Image
from natsort import natsorted
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt

NUM_FIX = 36

"""
function input : time_duration as a list
function output: a list of start timestamps
"""
def accumulate_times(times):
    new_times = np.copy(times)
    for i in range(len(times)):
        new_times[i] = np.sum(times[:i+1])
    new_times = new_times - new_times[0]
    return new_times


def convert_npz(files,imname, width, height):
    if not os.path.exists(files): return []
    df_full = pd.read_csv(files, header=None)
    df_full.columns = ['id', 'x', 'y', 'td']
    df_full['image_width'] = width
    df_full['image_height'] = height
    df_full['image_name'] = imname

    st = accumulate_times(df_full['td'])
    df_full['st'] = st.tolist()
    df_full = df_full.drop(index=[0])

    df_full = df_full.head(NUM_FIX)
    EOS = np.zeros(len(df_full)-1).tolist()
    EOS.append(1.0)
    df_full['EOS'] = EOS
    df_full = df_full.append( df_full.iloc[[-1]*(NUM_FIX-len(df_full['x']))] )
    df_full['et'] = df_full['st'] + df_full['td']
    df_massvis = df_full[['x','y','EOS','td','st','et','image_name','image_width','image_height']]
    return df_massvis


"""
Please replace the following lines with your file locations
"""
ROOT_PATH = '/netpool/homes/wangyo/Dataset/Massvis/targets393/'
imgpath = ROOT_PATH + 'targets/'
outpath = ROOT_PATH + 'zips/'

indiv_names = ['abl','cmb','lcm','ls','mg','pi','rb','rk','sok', 'wab']
indiv_name = indiv_names[1]
massvis_list = []

list_img_targets = natsorted(glob.glob(imgpath+'*.png'))
#len(list_img_targets)
for idx in range(0, len(list_img_targets), 1):
    if idx%6 == 0: continue
    imgfullpath = list_img_targets[idx]
    with Image.open(imgfullpath) as im:
        width, height = im.size # original image size
        print('working on %s'% list_img_targets[idx])

        imgname = imgfullpath.replace(imgpath,'')
        imgname = imgname.replace('.png','')

        gtpath = '../../eyetracking/csv_files/fixationsByVis/%s/enc/%s.csv'%(imgname, indiv_name)
        df_gt = convert_npz(gtpath, imgname, width, height)

        if len(df_gt) > 0:
            massvis_list.append(np.array(df_gt))

massvis_list.append(np.array(df_gt))

massvis_npz_full = np.array(massvis_list)
np.savez_compressed('%smassvis_train0_%s_len%d'%(outpath, indiv_name, NUM_FIX),massvis_npz_full)
#np.savez_compressed('%smassvis_%s_len%d'%(outpath, indiv_name, NUM_FIX),massvis_npz_full)
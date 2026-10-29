# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 01:06:04 2020

@author: Sruthi, Saiteja and Yao
@purpose: to visualize PathGAN scanpaths overlay on original images of MASSVIS dataset
@output : Visualizations in .png format in one directory
"""

import pandas as pd
import numpy as np
import glob,re,os
from PIL import Image
from natsort import natsorted
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt

"""
Please replace the following lines with your file locations
"""

ROOT_PATH = '/netpool/homes/wangyo/Dataset/Massvis/targets393'
ROOT_PATH2 = ROOT_PATH + '/MDSGAN/Salicon_MDSGAN_ms36_wloss_reduce0_dp0.1_rdp0.1_20210401-172339gen-epoch:99.h5'

predpath = ROOT_PATH2 + '/tsv/'
figpath = ROOT_PATH2 + '/scanpaths_vis/'

imgpath = ROOT_PATH + '/targets/'

list_img_targets = natsorted(glob.glob(imgpath+'*.png'))

for idx in range(0, len(list_img_targets), 6):
    imgfullpath = list_img_targets[idx]
    imgname = imgfullpath.replace(imgpath,'')
    imgname = imgname.replace('.png','')

    csvpath = '%s%s.png.tsv'%(predpath, imgname)

    if os.path.exists(csvpath):
        df_pred = pd.read_csv(csvpath, header=None, sep='\t')
    else:
        continue;
    df_pred.columns = ['x', 'y', 'time']

    img = Image.open(imgpath+imgname+'.png')
    width, height = img.size # original image size

    #Add the image to the path
    fig = plt.figure()
    plt.plot(df_pred['x'],df_pred['y'], '-ro', color='yellow',lw=2)
    #visualize the plot on image
    for i in range(len(df_pred['x'].tolist()) - 1):
        dur = (float(df_pred['time'].iloc[i+1]) - float(df_pred['time'].iloc[i])) / 10
        dur = dur if dur>15 else 15
        dur = dur if dur<35 else 35
        xx = df_pred['x'].iloc[i]
        yy = df_pred['y'].iloc[i]
        if(i == 0):
            plt.plot(xx, yy, '-o', ms = dur, mfc='blue') # 
        elif(i == len(df_pred['x'].tolist()) - 2):
            plt.plot(xx, yy, '-o', ms = dur, mfc='black', mec='#e9b96e')
        else:
            plt.plot(xx, yy, '-o', ms = dur, mfc='red', mec='#e9b96e') 
        plt.annotate(i, (xx, yy), color='yellow',weight='bold',fontsize=15)
    plt.imshow(img)
    plt.title("MDSGAN salicon")
    fig.savefig('%s%s.png'%(figpath,imgname))
    plt.close()
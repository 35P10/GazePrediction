# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 14:45:49 2020

@author: Yao Wang
@usage: plot element bounding boxes and fixations from all subjects onto orignal image
"""
import numpy as np
import sys,os
sys.path.append("..") 
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
from shutil import rmtree
import pandas as pd
from natsort import natsorted
from visualization_utils import plot_visualization, plot_img
from preprocess_utils import get_gt_elements

bp = "/netpool/homes/wangyo/Dataset/Massvis/targets393" # change base path to the directory where you have downloaded the salicon data
curdir = '../../eyetracking/csv_files/fixationsByVis'
origimdir = os.path.join(bp, 'targets')
DPI=144

i = 0
for curfile in os.listdir(curdir):
    i+=1
    if(i<340): continue
    os.makedirs(os.path.join(bp, 'plot_individual'), exist_ok=True)
    outpath = os.path.join(bp, 'plot_individual', curfile)
    os.makedirs(outpath, exist_ok=True)
    basepath = os.path.basename(curfile)
    imname, ext = os.path.splitext(basepath)
    print('imname',imname)

    elementCoords, elementX, elementY, boxes = get_gt_elements(imname)

    # load the raw image
    with Image.open(os.path.join(origimdir, curfile+'.png')) as im:
        width, height = im.size # original image size

        allfiles = natsorted(glob(os.path.join(curdir, curfile, 'enc' ,'*.csv')))

        for subcsv in allfiles:
            fixations = pd.read_csv(subcsv, header=None)
            x=[]
            y=[]
            duration=[]
            for row in fixations.iterrows():
                x.append(row[1][1])
                y.append(row[1][2])
                duration.append(row[1][3])

            fig = plot_img(im, DPI)

            # Visualize bounding boxes
            #for i in range(len(elementX)):
            #    plt.fill(elementX[i], elementY[i], color=(0.1, 0.7, 0.7, 0.3))

            plot_visualization(x,y,duration)

            #plt.title(imname+os.path.basename(subcsv))
            fig.savefig('%s/%s%s.png'%(outpath, imname, os.path.basename(subcsv)), dpi=DPI)
            plt.close()
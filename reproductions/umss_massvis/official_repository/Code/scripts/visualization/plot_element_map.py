# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 14:45:49 2020

@author: Yao Wang
@usage: plot element bounding boxes and fixations from all subjects onto orignal image
"""
import numpy as np
import os, sys
sys.path.append("..") 
sys.path.append("../MDSEM") 
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
from natsort import natsorted
from preprocess_utils import get_gt_elements
from polygon_util import bucketColors

bp = "/netpool/homes/wangyo/Dataset/Massvis/targets393" # change base path to the directory where you have downloaded the salicon data
curdir = '../../eyetracking/csv_files/fixationsByVis'
origimdir = os.path.join(bp, 'targets')
outpath = os.path.join(bp, 'plot_element')
os.makedirs(outpath, exist_ok=True)


for curfile in os.listdir(curdir):
    basepath = os.path.basename(curfile)
    imname, ext = os.path.splitext(basepath)
    print('imname',imname)

    elementCoords, elementX, elementY, boxes = get_gt_elements(imname)

    # load the raw image
    with Image.open(os.path.join(origimdir, imname+'.png')) as im:
        width, height = im.size # original image size
        
        #Add the image to the path
        fig = plt.figure(figsize=(16,16))
        for i in range(len(elementX)):
            plt.fill(elementX[i], elementY[i], color=bucketColors(boxes[i].id))

        plt.imshow(im)
        plt.axis('off')

        fig.savefig('%s/%s_element.png'%(outpath, imname), dpi=fig.dpi)
        plt.close()

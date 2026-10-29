# -*- coding: utf-8 -*-
import numpy as np
import os, sys
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
from eval import calculate_metrics
from natsort import natsorted
from glob import glob
import pandas as pd
from tqdm import trange
from skimage.draw import polygon
from polygon_util import bucketColors
sys.path.append("..") 
from preprocess_utils import get_gt_elements, get_accum_bucket
np.set_printoptions(suppress=True)

# change base path to the directory where you have downloaded the salicon data
bp = "/netpool/homes/wangyo/Dataset/Massvis/targets393"
curdir = '../../eyetracking/csv_files/fixationsByVis'
origimdir = os.path.join(bp, 'targets')
curfiles = natsorted(glob(curdir + '/*'))


mds_bucket = np.zeros((14, 20))
outpath = os.path.join(bp, 'plot_overall')
os.makedirs(outpath, exist_ok=True)

mds_bucket = np.zeros((14, 20))
areas = np.zeros((14))

for id in trange(0,len(curfiles), 1):
    basepath = os.path.basename(curfiles[id])
    imname, ext = os.path.splitext(basepath)
    #print('imname',imname)    
    elementCoords, elementX, elementY, boxes = get_gt_elements(imname)

    # load the raw image
    with Image.open(os.path.join(origimdir, imname+'.png')) as im:
        width, height = im.size # original image size

        #Handle bbox Overlap
        #id_map = np.full([width,height], 13)

        #Preserve bbox Overlap
        id_map = np.zeros((13, width,height))
        id_other = np.ones((width,height))

        for tt in range(12, -1, -1):
            for box in boxes:
                if box.id == tt:
                    rr, cc = polygon(np.array(box.coords)[:,0], np.array(box.coords)[:,1], (width, height))
                    #id_map[rr, cc] = tt
                    id_map[tt, rr, cc] = 1
                    id_other[rr, cc] = 0

        for tt in range(13):
            areas[tt] += np.sum(id_map[tt] > 0)
        areas[13] = np.sum(id_other > 0)

print(areas)
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

length =[]


mds_bucket = np.zeros((14, 20))
outpath = os.path.join(bp, 'plot_overall')
os.makedirs(outpath, exist_ok=True)

mds_bucket = np.zeros((14, 20))
area = np.zeros((14))

for id in trange(0,len(curfiles), 1):
    basepath = os.path.basename(curfiles[id])
    imname, ext = os.path.splitext(basepath)
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
            area[tt] += np.sum(id_map > 0)
        area[13] = np.sum(id_other > 0)

        allfiles = glob(os.path.join(curfiles[id], 'enc' ,'*.csv'))

        #Add the image to the path
        #fig = plt.figure(figsize=(16,16))

        #plt.imshow(im)

        for subcsv in allfiles:
            fixations = pd.read_csv(subcsv, header=None, names=['index','x','y','time'])

            x = fixations['x'].tolist()
            y = fixations['y'].tolist()
            duration = fixations['time'].tolist()
            length.append(len(x))

            accum_time = 0
            for row2 in fixations.iterrows():
                accum_time += row2[1][3]
                if (int(row2[0]) < 2): continue
                # If the fixation is inside the box, add one to the bucket counter
                id2 = get_accum_bucket(accum_time)   
                for mm in range(13):
                    mds_bucket[mm][id2] += id_map[mm,int(row2[1][1]),int(row2[1][2])]
                #id1 = id_map[int(row2[1][1]),int(row2[1][2])]
                #mds_bucket[id1][id2] += id
                mds_bucket[13][id2] += id_other[int(row2[1][1]),int(row2[1][2])]

            # expell the first fixation
            #x.pop(0)
            #y.pop(0)
            #duration.pop(0)

            #plt.plot(x,y, '-ro', color='yellow',lw=1)
            #visualize the plot on image

            #for i in range(len(x)):
            #    dur = duration[i]/10 if duration[i]/10>10 else 10
            #    dur = dur if dur<20 else 20
            #    if(i == 0):
            #        plt.plot(x[i], y[i], '-o', ms = dur, mfc=(0.1, 0.1, 1.0, 0.5) ) # 
            #    elif(i == len(x) - 1):
            #        plt.plot(x[i], y[i], '-o', ms = dur, mfc=(0.5, 0.5, 0.5, 0.2), mec='#e9b96e')
            #    else:
            #        plt.plot(x[i], y[i], '-o', ms = dur, mfc=(1.0, 0.0, 0.0, 0.5), mec='#e9b96e') 
            #    plt.annotate(i, (x[i], y[i]), color=(1, 1, 0, 0.7),weight='bold',fontsize=5)

        #for i in range(len(elementX)):
        #    plt.fill(elementX[i], elementY[i], color=bucketColors(boxes[i].id))

        #plt.title(imname+'_total')
        #fig.savefig('%s/%s_sum.png'%(outpath, imname), dpi=fig.dpi)
        #plt.close()

print(mds_bucket)

res = mds_bucket.copy()

for i in range(np.shape(res)[1]):
    res[:,i] /= np.sum(res[:,i])
    res[:,i] = np.round(100*res[:,i], 2)

print(res)
print(np.std(np.array(length), ddof=1))
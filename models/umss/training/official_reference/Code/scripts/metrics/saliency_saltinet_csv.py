# -*- coding: utf-8 -*-
"""
Created on 20210504

@author: Yao Wang
@purpose: to compute the saliency metrics of the generated saliency map in MASSVIS dataset
@output : the final avg. Metrics
"""

import pandas as pd
import numpy as np
import glob,os,sys
from PIL import Image
from natsort import natsorted
pd.options.mode.chained_assignment = None
from metrics import euclidean_distance, DTW, TDE, scaled_TDE, eyenalysis
sys.path.append("..") 
from preprocess_utils import get_gt_files, truncate_EOS
from scipy import ndimage

"""
Please replace the following lines with your file locations
"""
ROOT_PATH = '/netpool/homes/wangyo/Dataset/Massvis/targets393/'
imgpath = ROOT_PATH + 'targets/'

#ROOT_PATH2 = ROOT_PATH + 'Scanpath_baselines/Saltinet/'
ROOT_PATH2 = ROOT_PATH + 'Scanpath_baselines/DVS/'
#ROOT_PATH2 = ROOT_PATH + 'Scanpath_baselines/UMSI/'
#ROOT_PATH2 = ROOT_PATH + 'MDSEM_finetuned/scanpaths/'
#ROOT_PATH2 = ROOT_PATH + 'MDSEM_bucket_500_2000_5000/'
#ROOT_PATH2 = ROOT_PATH + 'MDSEM_bucket_global_500_2000_5000/scanpaths/'

predpath = ROOT_PATH2 + 'scanpaths_5s_new/csv/'
salmappath = ROOT_PATH2 + 'scanpaths_5s_new/sample_salmaps/'
fixmappath = ROOT_PATH2 + 'scanpaths_5s_new/sample_fixmaps/'

#predpath = ROOT_PATH2 + 'scanpaths_5s_nsamples_test2/csv/'
#salmappath = ROOT_PATH2 + 'scanpaths_5s_nsamples_test2/sample_salmaps'
#fixmappath = ROOT_PATH2 + 'scanpaths_5s_nsamples_test2/sample_fixmaps/'


list_img_targets = natsorted(glob.glob(imgpath+'*.png'))
#len(list_img_targets)

SIGMA = 19

def make_fixmap_and_heatmap(w, h, coords, sigma=50): 
    xs = tuple([elt[0] for elt in coords])
    ys = tuple([elt[1] for elt in coords])
    bitmap = np.zeros((w, h))
    fixations = np.zeros((w, h))
    for c in coords: 
        x,y = int(c[0]),int(c[1])
        if x<w and y<h:
            fixations[x,y] +=1    
            bitmap[x,y] = 1
    heatmap = ndimage.filters.gaussian_filter(fixations, [sigma, sigma])
    heatmap = 255*heatmap/float(np.max(heatmap))

    fix_img = Image.fromarray(np.uint8(np.transpose(bitmap)), "L")
    heatmap_img = Image.fromarray(np.uint8(np.transpose(heatmap)), "L")
    
    return fix_img, heatmap_img

for idx in range(0, len(list_img_targets), 6):
    imgfullpath = list_img_targets[idx]
    with Image.open(imgfullpath) as im:
        width, height = im.size # original image size
        print('working on %s'% list_img_targets[idx])

        imgname = imgfullpath.replace(imgpath,'')
        imgname = imgname.replace('.png','')
        predCsv = predpath + '%s.png.csv'%imgname
        df_pred = pd.read_csv(predCsv)

        data_list = []
        for pp in range(1, 20):
            df_pred.columns = ['user', 'index', 'time', 'x', 'y']
            df_predI = df_pred[df_pred['user'] == pp]
            ########## 2D Metrics
            df_predI = df_predI.drop(['user', 'index', 'time'] ,axis=1)
            df_predI['x'] /= 640
            df_predI['y'] /= 480
            df_predI['x'] *= width
            df_predI['y'] *= height
            tmp_list = df_predI.values.tolist()
            data_list.extend(tmp_list)

        fm, hm = make_fixmap_and_heatmap(width, height, data_list, sigma=SIGMA)
        fm.save(os.path.join(fixmappath, imgname + '.png'))
        hm.save(os.path.join(salmappath, imgname + '.png'))


# -*- coding: utf-8 -*-
"""
Created on 20211103

@author: Yao Wang
@purpose: to estimate the exponnorm parameters of durations in MASSVIS dataset
@output : estimated durations
"""

import pandas as pd
import numpy as np
import glob,os,sys
from PIL import Image
from scipy.stats import exponnorm
import matplotlib.pyplot as plt
from natsort import natsorted
pd.options.mode.chained_assignment = None
sys.path.append("..") 
from preprocess_utils import get_gt_files, truncate_EOS

fig, ax = plt.subplots(1, 1)


ROOT_PATH = '/netpool/homes/wangyo/Dataset/Massvis/targets393/'

imgpath = ROOT_PATH + 'targets/'

list_img_targets = natsorted(glob.glob(imgpath+'*.png'))
DUR= []


for idx in range(0, len(list_img_targets)):
    # Skip the val set
    if idx%6==0: continue

    imgfullpath = list_img_targets[idx]
    with Image.open(imgfullpath) as im:
        width, height = im.size # original image size
        print('working on %s'% list_img_targets[idx])

        imgname = imgfullpath.replace(imgpath,'')
        imgname = imgname.replace('.png','')

        gtpath, gt_files = get_gt_files(imgname, 'csv', truncate=True)
        #gtpath, gt_files = get_gt_files(imgname, 'csv', truncate=False)

        for pp in range(len(gt_files)):
            df_pred = pd.read_csv(os.path.join(gtpath, gt_files[pp]), header=None)
            df_pred.columns = ['index', 'x', 'y', 'time']
            df_pred = df_pred.drop(['index','x', 'y'],axis=1)
            df_pred = df_pred.to_numpy()
            df_pred = np.swapaxes(df_pred, 0, 1)
            df_pred = df_pred.tolist()
            #print(df_pred[0])
            DUR = DUR + df_pred[0]

#mm = exponnorm.fit(np.array(DUR)/1000)
mm = exponnorm.fit(np.array(DUR))
print(mm)

# train (5.418, 122.954, 16.944)
# val (5.114, 124.060, 17.489)

x = np.linspace(exponnorm.ppf(0.01, mm[0], loc=mm[1], scale=mm[2]),
                exponnorm.ppf(0.99, mm[0], loc=mm[1], scale=mm[2]), 100)
ax.plot(x, exponnorm.pdf(x, mm[0], loc=mm[1], scale=mm[2]),
       'k-', lw=4, alpha=0.9)
       #label='Estimated Duration Distribution'

#rrr = exponnorm.rvs(mm[0], loc=mm[1], scale=mm[2], size=1000)
#ax.hist(rrr, density=True, histtype='stepfilled', alpha=0.2, bins=range(50,800,20))
ax.legend(loc='best', frameon=False)

ax.set_xlabel('Fixation duration (ms)')

plt.show()

import numpy as np
import os, sys
sys.path.append("..") 
from preprocess_utils import load_img_list, get_gt_strings
from metrics import nw_matching

ROOT_PATH = '/netpool/homes/wangyo/Dataset/Massvis/targets393'
list_img_targets = load_img_list(ROOT_PATH) 

a = []
for idx in range(0, len(list_img_targets), 1):
    imgfullpath = list_img_targets[idx]
    imgname = imgfullpath.split('/')[-1]
    imgname = imgname.replace('.png','')

    strings = get_gt_strings(imgname, truncate=False)
    #print(strings)
    for i in range(len(strings)):
        for j in range(len(strings)):
            a.append(nw_matching(strings[i], strings[j]))

print(np.mean(np.array(a)))
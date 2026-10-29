import numpy as np
import os, sys
sys.path.append("..") 
from preprocess_utils import load_img_list, get_gtindiv_string
from metrics import nw_matching

ROOT_PATH = '/netpool/homes/wangyo/Dataset/Massvis/targets393'
indiv_names = ['abl','cmb','lcm','ls','mg','pi','rb','rk','sok', 'wab']

SIMM = np.zeros((10,10))

list_img_targets = load_img_list(ROOT_PATH) 


for i in range(10):
    for j in range(10):
        a = []
        for idx in range(0, len(list_img_targets), 1):
            imgfullpath = list_img_targets[idx]
            imgname = imgfullpath.split('/')[-1]
            imgname = imgname.replace('.png','')

            STR1 = get_gtindiv_string(imgname, indiv_names[i], truncate=False)
            STR2 = get_gtindiv_string(imgname, indiv_names[j], truncate=False)
            if STR1 and STR2:
                a.append(nw_matching(STR1, STR2))
        SIMM[i,j] = np.mean(np.array(a))

print(SIMM)
import numpy as np
import os, sys
import pandas as pd
sys.path.append("..") 
from preprocess_utils import load_img_list, get_gt_strings
from metrics import nw_matching

ROOT_PATH = '/netpool/homes/wangyo/Dataset/Massvis/targets393'
#ROOT_PATH2 = ROOT_PATH + '/GT/val/saltinet/5s'
#ROOT_PATH2 = ROOT_PATH + '/MDSGAN/Salicon_MDSGAN_ms36_wloss_reduce0_dp0.1_rdp0.1_20210401-172339gen-epoch:99.h5'
#ROOT_PATH2 = ROOT_PATH + '/Scanpath_baselines/PathGAN_finetuned/scanpaths'
ROOT_PATH2 = ROOT_PATH + '/Scanpath_baselines/nc2020'
#ROOT_PATH2 = ROOT_PATH + '/Scanpath_baselines/Saltinet/scanpaths'
#ROOT_PATH2 = ROOT_PATH + '/Scanpath_baselines/DVS/scanpaths_5s'
#ROOT_PATH2 = ROOT_PATH + '/Scanpath_baselines/UMSI/scanpaths_5s'
#ROOT_PATH2 = ROOT_PATH + '/Scanpath_baselines/MDSEM_finetuned/scanpaths'

#ROOT_PATH2 = ROOT_PATH + '/MDSEM_bucket_500_2000_5000/scanpaths_5s'
#ROOT_PATH2 = ROOT_PATH + '/MDSEM_bucket_500_2000_5000/scanpaths_5s_nsamples'
#ROOT_PATH2 = ROOT_PATH + '/MDSEM_bucket_500_2000_5000/scanpaths_5s_nsample_foveal_new'
#ROOT_PATH2 = ROOT_PATH + '/MDSEM_bucket_3000_5000_10000/scanpaths_10s_nsamples'

strpath = ROOT_PATH2 + '/str/'

#logfile = open('strsim/best/saltinet_best.txt', "w")
#logfile = open('strsim/best/UMSS_best.txt', "w")
logfile = open('strsim/best/nc2020_best.txt', "w")
#logfile = open('strsim/best/pathgan_best.txt', "w")
#logfile = open('stringsim_GTindiv_baseline_5s.txt', "w")

list_img_targets = load_img_list(ROOT_PATH) 

STRSIM = []
for idx in range(0, len(list_img_targets), 6):
    tmp = 0
    print('working on %s'% list_img_targets[idx])
    imgfullpath = list_img_targets[idx]
    imgname = imgfullpath.split('/')[-1]
    imgname = imgname.replace('.png','')

    # 5seconds
    strings = get_gt_strings(imgname, truncate=True)
    # 10 seconds
    #strings = get_gt_strings(imgname)

    # DCSM, PathGAN, nc2020
    strfile = '%s/%s.txt'%(strpath, imgname)

    #Saltinet and UMSS
    #for pp in range(1, 41):
    #    strfile = '%s%d/%s.txt'%(strpath, pp, imgname)
    #    if not os.path.exists(strfile): continue

    f = open(strfile,'r')
    line = f.readline()

    for i in range(len(strings)):
        res = nw_matching(strings[i], line)
        if res > tmp:
            tmp = res


    STRSIM.append(tmp)


df = pd.DataFrame(list(zip(STRSIM)))
df.columns = ['SS']

#df.to_csv('strsim/best/UMSS_best.csv', index=False)
df.to_csv('strsim/best/nc2020_best.csv', index=False)
#df.to_csv('strsim/best/pathgan_best.csv', index=False)
#df.to_csv('strsim/best/saltinet_best.csv', index=False)

final_score = np.mean(np.array(STRSIM))
print(final_score)
logfile.write("sum: %.3f\n"%(final_score))
logfile.close()
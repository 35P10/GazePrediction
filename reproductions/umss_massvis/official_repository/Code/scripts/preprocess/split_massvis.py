import os
import numpy as np
from PIL import Image

bp = '/datasets/external/yao_data/MASSVIS/Massvis393'
out_path = '/datasets/external/yao_data/MASSVIS/massviz_split'
img_path = os.path.join(bp, 'raw_img')
imp_path = os.path.join(bp, 'heatmaps_accum', '10000')
fix_path = os.path.join(bp, 'fix_maps_accum', '10000')

out_train_img_path = os.path.join(out_path, 'train', 'targets')
out_train_imp_path = os.path.join(out_path, 'train', 'attmap_gt')
out_train_fix_path = os.path.join(out_path, 'train', 'fix_maps')
out_val_img_path = os.path.join(out_path, 'val', 'targets')
out_val_imp_path = os.path.join(out_path, 'val', 'attmap_gt')
out_val_fix_path = os.path.join(out_path, 'val', 'fix_maps')

os.makedirs(out_train_img_path, exist_ok=True)
os.makedirs(out_train_imp_path, exist_ok=True)
os.makedirs(out_train_fix_path, exist_ok=True)
os.makedirs(out_val_img_path, exist_ok=True)
os.makedirs(out_val_imp_path, exist_ok=True)
os.makedirs(out_val_fix_path, exist_ok=True)

imgs = sorted([os.path.join(img_path, f) for f in os.listdir(img_path) if f.endswith('.png')])
maps = sorted([os.path.join(imp_path, f) for f in os.listdir(imp_path) if f.endswith('.png')])
fixmaps = sorted([os.path.join(fix_path, f) for f in os.listdir(fix_path) if f.endswith('.png')])

#Randomly shuffling massviz
np.random.seed(42)
idxs = list(range(len(imgs)))
np.random.shuffle(idxs)
imgs = np.array(imgs)[idxs]
maps = np.array(maps)[idxs]
fixmaps = np.array(fixmaps)[idxs]

#ES: need to verfiy if I should implement a sorting functions, not sure. I dont think so bc the file names are the same

img_filenames_train = imgs[:354] #ES taking 90% for train and 10% for test -- TODO might change this so it is more like codecharts, that is 20-25% for val/test
imp_filenames_train = maps[:354]
fix_filenames_train = fixmaps[:354]

img_filenames_val = imgs[354:]
imp_filenames_val = maps[354:]
fix_filenames_val = fixmaps[354:]

for filename in img_filenames_train:
    tmp_img = Image.open(filename)
    basename = os.path.basename(filename)
    tmp_img.save(os.path.join(out_train_img_path, basename))

for filename in imp_filenames_train:
    tmp_img = Image.open(filename)
    basename = os.path.basename(filename)
    tmp_img.save(os.path.join(out_train_imp_path, basename))

for filename in fix_filenames_train:
    tmp_img = Image.open(filename)
    basename = os.path.basename(filename)
    tmp_img.save(os.path.join(out_train_fix_path, basename))


for filename in img_filenames_val:
    basename = os.path.basename(filename)
    tmp_img = Image.open(filename)
    tmp_img.save(os.path.join(out_val_img_path, basename))

for filename in imp_filenames_val:
    basename = os.path.basename(filename)
    tmp_img = Image.open(filename)
    tmp_img.save(os.path.join(out_val_imp_path, basename))

for filename in fix_filenames_val:
    basename = os.path.basename(filename)
    tmp_img = Image.open(filename)
    tmp_img.save(os.path.join(out_val_fix_path, basename))

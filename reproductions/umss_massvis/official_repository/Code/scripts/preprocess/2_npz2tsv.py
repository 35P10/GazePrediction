import os
import glob
from natsort import natsorted
import numpy as np
from PIL import Image


ROOT_DIR = '/netpool/work/gpu-3/users/wangyo/datasets/massvis393'

imgpath = os.path.join(ROOT_DIR, 'target')
npz_path = '/netpool/work/gpu-3/users/wangyo/projects/ws2020_sruthisaiteja/sample_test/whole_val_results/SaliconResnetLSTM_dp0.5_rdp0.5_20210303-182103MIT393.npz'

outpath = '/netpool/work/gpu-3/users/wangyo/projects/ws2020_sruthisaiteja/Code/metrics/ScanMatch/Model_data/SaliconResnetLSTM_dp0.5_rdp0.5/'

npz_file = np.load(npz_path, allow_pickle='True')
npz_file = npz_file['arr_0']

file_names = natsorted(glob.glob(imgpath+"/*.png"))


for i in range(np.shape(npz_file)[0]):
    im = Image.open(file_names[i])
    im = im.convert('RGB')
    width, height = im.size
    tmp = np.delete(npz_file[i], 2, axis=1)
    tmp[:,0] *= width
    tmp[:,1] *= height
    tmp[:,2] *= 1000
    w_filenameTSV = outpath + file_names[i].replace(imgpath,'')+'.tsv'
    with open(w_filenameTSV,'w') as write_tsv:
        np.savetxt(w_filenameTSV, tmp, delimiter="\t")
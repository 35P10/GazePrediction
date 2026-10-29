import os, glob
import numpy as np
import pandas as pd
from skimage.draw import polygon
from natsort import natsorted

eledir = '../../dataset/csv_files/targets393_elementLabels/elementLabels/'


class BBox:
    name = ''
    coords = None
    id = -1
    durations = None
    def __init__(self, name, coords, id):
        self.name = name
        self.coords = coords
        self.id = id

# Bucket 1-D index for semantic regions:
# 0: Annotation
# 1: Axis
# 2: Graphical Element
# 3: Legend
# 4: Object [Photograph]
# 5: Object [Pictogram]
# 6: Text [Title]
# 7: Text [Header Row]
# 8: Text [Label]
# 9: Text [Paragraph]
# 10: Text [Source]
# 11: Data [Type]
# 12: Data (Region)
# 13: Other 

# Bucket 2-D index for Durations:
# 0: <500ms
# 1: 500-3000ms
# 2: 3000-5000ms
# 3: 5000-7500ms
# 4: 7500ms+


def get_bboxid_by_name(name):
    id1 = -1
    if 'annotation' in name:
        id1 = 0
    elif 'axis' in name:
        id1 = 1
    elif 'graphical element' in name:
        id1 = 2
    elif 'legend' in name:
        id1 = 3
    elif 'object' in name:
        if 'photograph' in name:
            id1 = 4
        else: # 'pictogram'
            id1 = 5
    elif 'text' in name:
        if '(title)' in name:
            id1 = 6
        elif '(header row)' in name:
            id1 = 7
        elif '(label)' in name:
            id1 = 8
        elif '(paragraph)' in name:
            id1 = 9
        else:
            id1 = 10
    elif 'data (' in name:
        id1 = 11
    elif name == 'data':
        id1 = 12
    else:
        id1 = 13
    return id1

def get_accum_bucket(accum_time):
    accum_time = int(accum_time)
    if accum_time < 500:
        id2 = 0
    elif accum_time < 1000:
        id2 = 1
    elif accum_time < 1500:
        id2 = 2
    elif accum_time < 2000:
        id2 = 3
    elif accum_time < 2500:
        id2 = 4
    elif accum_time < 3000:
        id2 = 5
    elif accum_time < 3500:
        id2 = 6
    elif accum_time < 4000:
        id2 = 7
    elif accum_time < 4500:
        id2 = 8
    elif accum_time < 5000:
        id2 = 9
    elif accum_time < 5500:
        id2 = 10
    elif accum_time < 6000:
        id2 = 11
    elif accum_time < 6500:
        id2 = 12
    elif accum_time < 7000:
        id2 = 13
    elif accum_time < 7500:
        id2 = 14
    elif accum_time < 8000:
        id2 = 15
    elif accum_time < 8500:
        id2 = 16
    elif accum_time < 9000:
        id2 = 17
    elif accum_time < 9500:
        id2 = 18
    else:
        id2 = 19
    return id2


def get_bboxid_by_name_9(name):
    id1 = -1
    if 'annotation' in name:
        id1 = 0
    elif 'axis' in name:
        id1 = 1
    elif 'graphical element' in name:
        id1 = 2
    elif 'legend' in name:
        id1 = 3
    elif 'object' in name:
        id1 = 4
    elif 'text' in name:
        if '(title)' in name or '(header row)' in name or '(paragraph)' in name:
            id1 = 5
        else:
            id1 = 6
    elif 'data' in name:
        id1 = 7
    else:
        id1 = 8
    return id1

def get_accum_bucket_9(accum_time):
    accum_time = int(accum_time)
    if accum_time < 1000:
        id2 = 0
    elif accum_time < 2000:
        id2 = 1
    elif accum_time < 3000:
        id2 = 2
    elif accum_time < 4000:
        id2 = 3
    elif accum_time < 5000:
        id2 = 4
    elif accum_time < 6000:
        id2 = 5
    elif accum_time < 7000:
        id2 = 6
    elif accum_time < 8000:
        id2 = 7
    elif accum_time < 9000:
        id2 = 8
    else:
        id2 = -1
    return id2


def get_gt_files(imgname, extension, truncate=False):
    if truncate:
        gtdir = '../../eyetracking/csv_files/fixationsByVis_5000/%s/enc/'%imgname
    else:
        gtdir = '../../eyetracking/csv_files/fixationsByVis/%s/enc/'%imgname
    for path, subdir, files in os.walk(gtdir,extension):
        gt_files = files
    return path, gt_files

def get_gtindiv_string(imname, subject, truncate=False):
    if truncate:
        gtstring = '../../eyetracking/csv_files/strings_5000/%s/%s_%s.txt'%(subject,imname,subject)
    else:
        gtstring = '../../eyetracking/csv_files/strings/%s/%s_%s.txt'%(subject,imname,subject)
    if not os.path.exists(gtstring):
        return ""
    f = open(gtstring,'r')
    STR = f.readline()
    return STR

def get_gt_strings(imname, truncate=False):
    names = ['abl', 'cmb', 'lcm', 'ls', 'mg', 'pi', 'rb', 'rk', 'sok', 'wab']
    STRS=[]
    for subject in names:
        if truncate:
            gtstring = '../../eyetracking/csv_files/strings_5000/%s/%s_%s.txt'%(subject,imname,subject)
        else:
            gtstring = '../../eyetracking/csv_files/strings/%s/%s_%s.txt'%(subject,imname,subject)
        if not os.path.exists(gtstring): continue
        f = open(gtstring,'r')
        line = f.readline()
        STRS.append(line)
    return STRS

def get_gt_elements(imname, simple = False):
    elementLabel = pd.read_csv(os.path.join(eledir, imname), names=['bboxID','category','x','y'])
    elementCoords = []
    elementX = []
    elementY = []
    tmp = 0
    curName = ""

    boxes = []

    for row in elementLabel.iterrows():
        # row[1][0]: id
        # row[1][1]: category name
        # row[1][2]: x
        # row[1][3]: y

        # a new bbox
        if int(row[1][0]) > tmp:
            # Store the last bbox
            if tmp > 0:
                if (simple): 
                    boxid = get_bboxid_by_name_9(curName)
                else:
                    boxid = get_bboxid_by_name(curName)
                box = BBox(curName, elementCoords[-1], boxid)
                boxes.append(box)
            tmp = int(row[1][0])
            curName = row[1][1].strip()

            elementCoords.append([])
            elementX.append([])
            elementY.append([])

        elementCoords[-1].append([int(row[1][2]), int(row[1][3])])
        if(curName != 'data'):
            elementX[-1].append(int(row[1][2]))
            elementY[-1].append(int(row[1][3]))

    boxid = get_bboxid_by_name(curName)
    box = BBox(curName, elementCoords[-1], boxid)
    boxes.append(box)
    return elementCoords, elementX, elementY, boxes

def get_idmap(boxes, width, height, classn):
    id_map = np.full([width,height], classn - 1)
    for tt in range(classn - 2, -1, -1):
        for box in boxes:
            if box.id == tt:
                rr, cc = polygon(np.array(box.coords)[:,0], np.array(box.coords)[:,1], (width, height))
                id_map[rr, cc] = tt
    return id_map

def load_img_list(ROOT_PATH):
    imgpath = ROOT_PATH + '/targets/'
    return natsorted(glob.glob(imgpath+'*.png'))

def truncate_EOS(P):
    if not isinstance(P, np.ndarray):
        P = np.array(P, dtype=np.float32)
    elif P.dtype != np.float32:
	    P = P.astype(np.float32)

    x = -1
    y = -1
    endId = P.shape[0]
    for i in range(P.shape[0]):
        if abs(P[i][0] - x) + abs(P[i][1] - y) < 4:
            endId = i
            break
        x = P[i][0]
        y = P[i][1]
    return P[:endId,:]

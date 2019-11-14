"""
    To use the code, users should to config detpath, annopath and imagesetfile
    detpath is the path for 15 result files, for the format, you can refer to "http://captain.whu.edu.cn/DOTAweb/tasks.html"
    search for PATH_TO_BE_CONFIGURED to config the paths
    Note, the evaluation is on the large scale images
"""
import os
import numpy as np
import re
import time
import sys
sys.path.insert(0,'..')
try:
    import dota_utils as util
except:
    import dota_kit.dota_utils as util

from PIL import Image
from multiprocessing import Pool
from functools import partial
from draw_box_in_img import draw_boxes_with_label_and_scores

ODAI_LABEL_MAP = {
        'back-ground': 0,
        'plane': 1,
        'baseball-diamond': 2,
        'bridge': 3,
        'ground-track-field': 4,
        'small-vehicle': 5,
        'large-vehicle': 6,
        'ship': 7,
        'tennis-court': 8,
        'basketball-court': 9,
        'storage-tank': 10,
        'soccer-ball-field': 11,
        'roundabout': 12,
        'harbor': 13,
        'swimming-pool': 14,
        'helicopter': 15,
    }

def osp(savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)

def drawBoxsingle(imgpath, dstpath, nameboxdict, imgname):
    det = np.array(nameboxdict[imgname])
    confidence = list(map(float, det[:, -2]))
    bbox = det[:, 0:-2]

    imgnamepath = os.path.join(imgpath, imgname + '.png')
    outimg = Image.open(imgnamepath).convert('RGB')
    outimg = np.array(outimg)
    label = det[:, -1]
    image = draw_boxes_with_label_and_scores(outimg, bbox, confidence, label, 0)
    image.save(os.path.join(dstpath, imgname + '.png'))

def drwaBox_parallel(srcpath, imgpath, dstpath):
    pool = Pool(16)
    filelist = util.GetFileFromThisRootDir(srcpath)
    nameboxdict = {}
    for file in filelist:
        name = util.custombasename(file)
        with open(file, 'r') as f_in:
            lines = f_in.readlines()
            splitlines = [x.strip().split(' ') for x in lines]
            for splitline in splitlines:
                oriname = splitline[0]
                confidence = float(splitline[1])
                bbox = list(map(float, splitline[2:]))
                bbox.append(confidence)
                bbox.append(name.split('_')[-1])
                if (oriname not in nameboxdict):
                    nameboxdict[oriname] = []
                nameboxdict[oriname].append(bbox)

    drawBoxsingle_fn = partial(drawBoxsingle, imgpath, dstpath, nameboxdict)
    # pdb.set_trace()
    pool.map(drawBoxsingle_fn, nameboxdict)

def drwaBox(srcpath, imgpath, dstpath):
    """
    srcpath: result files before merge and nms
    dstpath: result files after merge and nms
    """
    osp(srcpath)
    osp(dstpath)
    drwaBox_parallel(srcpath, imgpath, dstpath)
if __name__ == '__main__':
    drwaBox(r'/mnt/lustre/yanhongchang/project/one-rpn/mmdetection/demo/work_dirs/out_img/faster_x101/tr_merge3/txt2',
                r'/mnt/lustre/yanhongchang/project/one-rpn/mmdetection/data/rpn15/test',
                r'/mnt/lustre/yanhongchang/project/one-rpn/mmdetection/demo/work_dirs/out_img/faster_x101/tr_merge3/img2')

    # mergebyrec()
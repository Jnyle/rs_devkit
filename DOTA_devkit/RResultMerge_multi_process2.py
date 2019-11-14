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
import cv2
sys.path.insert(0,'..')
try:
    import dota_utils as util
except:
    import dota_kit.dota_utils as util
import polyiou
import pdb
import math
from multiprocessing import Pool
from functools import partial

## the thresh for nms when merge image
nms_thresh = 0.5

def osp(savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)

def nms_rotate_cpu(boxes, iou_threshold):
    keep = []
    scores = boxes[:, -1]
    order = scores.argsort()[::-1]
    num = boxes.shape[0]

    suppressed = np.zeros((num), dtype=np.int)

    for _i in range(num):

        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        r1 = ((boxes[i, 0], boxes[i, 1]), (boxes[i, 2], boxes[i, 3]), boxes[i, 4])
        area_r1 = boxes[i, 2] * boxes[i, 3]
        for _j in range(_i + 1, num):
            j = order[_j]
            if suppressed[i] == 1:
                continue
            r2 = ((boxes[j, 0], boxes[j, 1]), (boxes[j, 2], boxes[j, 3]), boxes[j, 4])
            area_r2 = boxes[j, 2] * boxes[j, 3]
            inter = 0.0
            inter_iof = 0.0
            distance = np.sqrt((r1[0][0] - r2[0][0]) ** 2 + (r1[0][1] - r2[0][1]) ** 2)
            try:
                int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]

                if int_pts is not None:
                    order_pts = cv2.convexHull(int_pts, returnPoints=True)

                    int_area = cv2.contourArea(order_pts)

                    inter = int_area * 1.0 / (area_r1 + area_r2 - int_area + 0.00001)
                    inter_iof = int_area * 1.0 / (area_r2 + 0.00001)

            except:
                """
                  cv2.error: /io/opencv/modules/imgproc/src/intersection.cpp:247:
                  error: (-215) intersection.size() <= 8 in function rotatedRectangleIntersection
                """
                # print(r1)
                # print(r2)
                inter = 0.9999

            if inter >= iou_threshold or inter_iof >= 0.8 or distance <= min(r1[1][0], r1[1][1]):
                suppressed[j] = 1

    return np.array(keep, np.int64)

def py_cpu_nms_poly(dets, thresh):
    scores = dets[:, 8]
    polys = []
    areas = []
    for i in range(len(dets)):
        tm_polygon = polyiou.VectorDouble([dets[i][0], dets[i][1],
                                            dets[i][2], dets[i][3],
                                            dets[i][4], dets[i][5],
                                            dets[i][6], dets[i][7]])
        polys.append(tm_polygon)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]
        keep.append(i)
        for j in range(order.size - 1):
            iou = polyiou.iou_poly(polys[i], polys[order[j + 1]])
            ovr.append(iou)
        ovr = np.array(ovr)

        # print('ovr: ', ovr)
        # print('thresh: ', thresh)
        try:
            if math.isnan(ovr[0]):
                pdb.set_trace()
        except:
            pass
        inds = np.where(ovr <= thresh)[0]
        # print('inds: ', inds)

        order = order[inds + 1]

    return keep


def py_cpu_nms_poly_fast(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    # obbs = dets[:, 0:-1]
    # x1 = np.min(obbs[:, 0::2], axis=1)
    # y1 = np.min(obbs[:, 1::2], axis=1)
    # x2 = np.max(obbs[:, 0::2], axis=1)
    # y2 = np.max(obbs[:, 1::2], axis=1)
    # scores = dets[:, 8]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    polys = []
    for i in range(len(dets)):
        # tm_polygon = polyiou.VectorDouble([dets[i][0], dets[i][1],
        #                                     dets[i][2], dets[i][3],
        #                                     dets[i][4], dets[i][5],
        #                                     dets[i][6], dets[i][7]])
        tm_polygon = polyiou.VectorDouble([x1[i], y1[i],
                                           x2[i], y1[i],
                                           x2[i], y2[i],
                                           x1[i], y2[i]])
        polys.append(tm_polygon)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]
        keep.append(i)
        # if order.size == 0:
        #     break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # w = np.maximum(0.0, xx2 - xx1 + 1)
        # h = np.maximum(0.0, yy2 - yy1 + 1)
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        hbb_inter = w * h
        hbb_ovr = hbb_inter / (areas[i] + areas[order[1:]] - hbb_inter)
        # h_keep_inds = np.where(hbb_ovr == 0)[0]
        h_inds = np.where(hbb_ovr > 0)[0]
        tmp_order = order[h_inds + 1]
        for j in range(tmp_order.size):
            iou = polyiou.iou_poly(polys[i], polys[tmp_order[j]])
            hbb_ovr[h_inds[j]] = iou
            # ovr.append(iou)
            # ovr_index.append(tmp_order[j])

        # ovr = np.array(ovr)
        # ovr_index = np.array(ovr_index)
        # print('ovr: ', ovr)
        # print('thresh: ', thresh)
        try:
            if math.isnan(ovr[0]):
                pdb.set_trace()
        except:
            pass
        inds = np.where(hbb_ovr <= thresh)[0]

        # order_obb = ovr_index[inds]
        # print('inds: ', inds)
        # order_hbb = order[h_keep_inds + 1]
        order = order[inds + 1]
        # pdb.set_trace()
        # order = np.concatenate((order_obb, order_hbb), axis=0).astype(np.int)
    return keep

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    #print('dets:', dets)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    ## index for dets
    order = scores.argsort()[::-1]


    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ovr_iof = inter / (areas[order[1:]])

        # inds = np.where(ovr <= thresh)[0]
        inds = np.where((ovr <= thresh) & (ovr_iof <= 0.8))[0]
        order = order[inds + 1]

    return keep

def nmsbynamedict(nameboxdict, nms, thresh):
    nameboxnmsdict = {x: [] for x in nameboxdict}
    for imgname in nameboxdict:
        #print('imgname:', imgname)
        #keep = py_cpu_nms(np.array(nameboxdict[imgname]), thresh)
        #print('type nameboxdict:', type(nameboxnmsdict))
        #print('type imgname:', type(imgname))
        #print('type nms:', type(nms))
        keep = nms(np.array(nameboxdict[imgname]), thresh)
        #print('keep:', keep)
        outdets = []
        #print('nameboxdict[imgname]: ', nameboxnmsdict[imgname])
        for index in keep:
            # print('index:', index)
            outdets.append(nameboxdict[imgname][index])
        nameboxnmsdict[imgname] = outdets
    return nameboxnmsdict
def poly2origpoly(poly, x, y, rate):
    origpoly = []
    for i in range(int(len(poly)/2)):
        tmp_x = float(poly[i * 2] + x) / float(rate)
        tmp_y = float(poly[i * 2 + 1] + y) / float(rate)
        origpoly.append(tmp_x)
        origpoly.append(tmp_y)
    return origpoly

def mergesingle(dstpath, nms, fullname):
    name = util.custombasename(fullname)
    #print('name:', name)
    dstname = os.path.join(dstpath, name + '.txt')
    with open(fullname, 'r') as f_in:
        nameboxdict = {}
        lines = f_in.readlines()
        splitlines = [x.strip().split(' ') for x in lines]
        for splitline in splitlines:
            oriname = splitline[0]
            confidence = splitline[1]
            if float(confidence) < 0.5:
                continue
            points = list(map(float, splitline[2:]))
            rect = cv2.minAreaRect(np.float32([[points[0], points[1]], [points[2], points[3]],
                                               [points[4], points[5]], [points[6], points[7]]]))
            x_ctr, y_ctr, width, height, angle = rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]
            if angle == -90:
                if width <= height:
                    angle = 0
                    width, height = height, width
                else:
                    angle = 90
            elif width >= height:
                angle = -angle
            else:
                angle = -(90 + angle)
                width, height = height, width

            det = list()
            det.extend([x_ctr, y_ctr, width, height, angle])
            det.append(confidence)
            det = list(map(float, det))
            if (oriname not in nameboxdict):
                nameboxdict[oriname] = []
            nameboxdict[oriname].append(det)
        nameboxnmsdict = nmsbynamedict(nameboxdict, nms, nms_thresh)
        with open(dstname, 'w') as f_out:
            print('dstname {}'.format(dstname))
            for imgname in nameboxnmsdict:
                for det in nameboxnmsdict[imgname]:
                    #print('det:', det)
                    confidence = det[-1]
                    rect = det[0:-1]

                    box = cv2.boxPoints(((rect[0], rect[1]), (rect[2], rect[3]), -rect[4]))
                    box = np.reshape(box, [-1, ])
                    bbox = [box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]]
                    outline = imgname + ' ' + str(confidence) + ' ' + ' '.join(map(str, bbox))
                    #print('outline:', outline)
                    f_out.write(outline + '\n')

def mergebase_parallel(srcpath, dstpath, nms):
    pool = Pool(16)
    filelist = util.GetFileFromThisRootDir(srcpath)

    mergesingle_fn = partial(mergesingle, dstpath, nms)
    # pdb.set_trace()
    pool.map(mergesingle_fn, filelist)

def mergebase(srcpath, dstpath, nms):
    filelist = util.GetFileFromThisRootDir(srcpath)
    for filename in filelist:
        mergesingle(dstpath, nms, filename)

def mergebyrec(srcpath, dstpath):
    """
    srcpath: result files before merge and nms
    dstpath: result files after merge and nms
    """
    # srcpath = r'E:\bod-dataset\results\bod-v3_rfcn_2000000'
    # dstpath = r'E:\bod-dataset\results\bod-v3_rfcn_2000000_nms'

    mergebase(srcpath,
              dstpath,
              py_cpu_nms)
def mergebypoly(srcpath, dstpath):
    """
    srcpath: result files before merge and nms
    dstpath: result files after merge and nms
    """
    # srcpath = r'/home/dingjian/evaluation_task1/result/faster-rcnn-59/comp4_test_results'
    # dstpath = r'/home/dingjian/evaluation_task1/result/faster-rcnn-59/testtime'

    # mergebase(srcpath,
    #           dstpath,
    #           py_cpu_nms_poly)
    osp(srcpath)
    osp(dstpath)
    mergebase_parallel(srcpath,
              dstpath,
              nms_rotate_cpu)
if __name__ == '__main__':
    mergebypoly(r'/mnt/lustre/yanhongchang/project/mmdetection/demo/work_dirs/out_img/faster_r50/tr_merge3/txt2',
                r'/mnt/lustre/yanhongchang/project/mmdetection/demo/work_dirs/out_img/faster_r50/tr_merge3/txt3')
    # mergebyrec()
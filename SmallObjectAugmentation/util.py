import os
import cv2
import numpy as np
from PIL import Image
from os.path import join, split
import random
import copy


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def issmallobject(bbox, thresh):
    if bbox[0] * bbox[1] <= thresh:
        return True
    else:
        return False


def read_label_txt(label_dir):
    labels = []
    with open(label_dir) as fp:
        for f in fp.readlines():
            labels.append(f.strip().split(' '))
    return labels


def load_txt_label(label_dir):
    return np.loadtxt(label_dir, dtype=str)


def load_txt_labels(label_dir):
    labels = []
    for l in label_dir:
        la = load_txt_label(l)
        labels.append(la)
    return labels


def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def rescale_yolo_labels(labels, img_shape):
    height, width, nchannel = img_shape
    rescale_boxes = []
    for obj in list(labels):
        box = obj[:8]
        xmin, ymin, xmax, ymax = min(box[0::2]), min(box[1::2]), max(box[0::2]), max(box[1::2])
        rescale_boxes.append([xmin, ymin, xmax, ymax, obj[8], obj[9]])
    return rescale_boxes


def draw_annotation_to_image(img, annotation, save_img_dir):
    for anno in annotation:
        x1, y1, x2, y2 = anno
        cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 0))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, (int((x1 + x2) / 2), y1 - 5), font, fontScale=0.8, color=(0, 0, 255))
    cv2.imwrite(save_img_dir, img)


def bbox_iou(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b1_x1, b1_y1, b1_x2, b1_y2 = int(float(b1_x1)), int(float(b1_y1)), int(float(b1_x2)), int(float(b1_y2))
    b2_x1, b2_y1, b2_x2, b2_y2 = box2
    b2_x1, b2_y1, b2_x2, b2_y2 = int(float(b2_x1)), int(float(b2_y1)), int(float(b2_x2)), int(float(b2_y2))
    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)
    # Intersection area
    inter_width = inter_rect_x2 - inter_rect_x1 + 1
    inter_height = inter_rect_y2 - inter_rect_y1 + 1
    if inter_width > 0 and inter_height > 0:  # strong condition
        inter_area = inter_width * inter_height
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
        iou = inter_area / (b1_area + b2_area - inter_area)
    else:
        iou = 0
    return iou


def swap(x1, x2):
    if (x1 > x2):
        temp = x1
        x1 = x2
        x2 = temp
    return x1, x2


def norm_sampling(search_space):
    # 随机生成点
    search_x_left, search_y_left, search_x_right, search_y_right = search_space
    new_bbox_x_center = random.randint(int(search_x_left), int(search_x_right))
    new_bbox_y_center = random.randint(int(search_y_left), int(search_y_right))
    return [new_bbox_x_center, new_bbox_y_center]


def flip_bbox(roi):
    roi = roi[:, ::-1, :]
    return roi


def sampling_new_bbox_center_point(img_shape, bbox):
    #### sampling space ####
    height, width, nc = img_shape
    x_left, y_left, x_right, y_right = bbox
    bbox_w, bbox_h = x_right - x_left, y_right - y_left
    ### left top ###
    if x_left <= width / 2:
        search_x_left, search_y_left, search_x_right, search_y_right = width * 0.6, height / 2, width * 0.75, height * 0.75
    if x_left > width / 2:
        search_x_left, search_y_left, search_x_right, search_y_right = width * 0.25, height / 2, width * 0.5, height * 0.75
    return [search_x_left, search_y_left, search_x_right, search_y_right]


def random_add_patches(bbox, rescale_boxes, shape, paste_number, iou_thresh):
    temp = []
    for rescale_bbox in rescale_boxes:
        temp.append(rescale_bbox)
    x_left, y_left, x_right, y_right = bbox
    bbox_w, bbox_h = x_right - x_left, y_right - y_left
    center_search_space = sampling_new_bbox_center_point(shape, bbox)
    success_num = 0
    new_bboxes = []
    while success_num < paste_number:
        new_bbox_x_center, new_bbox_y_center = norm_sampling(center_search_space)
        print(norm_sampling(center_search_space))
        new_bbox_x_left, new_bbox_y_left, new_bbox_x_right, new_bbox_y_right = new_bbox_x_center - 0.5 * bbox_w, \
                                                                               new_bbox_y_center - 0.5 * bbox_h, \
                                                                               new_bbox_x_center + 0.5 * bbox_w, \
                                                                               new_bbox_y_center + 0.5 * bbox_h
        new_bbox = [int(new_bbox_x_left), int(new_bbox_y_left), int(new_bbox_x_right), int(new_bbox_y_right)]
        ious = [bbox_iou(new_bbox, bbox_t) for bbox_t in rescale_boxes]
        if max(ious) <= iou_thresh:
            # for bbox_t in rescale_boxes:
            # iou =  bbox_iou(new_bbox[1:],bbox_t[1:])
            # if(iou <= iou_thresh):
            success_num += 1
            temp.append(new_bbox)
            new_bboxes.append(new_bbox)
        else:
            continue
    return new_bboxes


def sampling_new_bbox_center_point2(img_shape, bbox):
    #### sampling space ####
    height, width, nc = img_shape
    bbox_h, bbox_w, bbox_c = bbox
    ### left top ###
    '''
    search_x_left, search_y_left, search_x_right, search_y_right = width * 0.55 , height * 0.5 , \
                                                                   width * 0.9 , height * 0.95
    '''
    search_x_left, search_y_left, search_x_right, search_y_right = width * 0.35 , height * 0.6 , \
                                                                   width * 1 , height * 0.95

    return [search_x_left, search_y_left, search_x_right, search_y_right]


def random_add_patches2(bbox_img, rescale_boxes, shape, paste_number, iou_thresh):
    temp = []
    for rescale_bbox in rescale_boxes:
        temp.append(rescale_bbox)
    bbox_h, bbox_w, bbox_c = bbox_img
    img_h, img_w, img_c = shape
    # center_search_space = sampling_new_bbox_center_point2(shape, bbox_img)  # 选取生成随机点区域
    center_search_space = 5, 5, 505, 505
    success_num = 0
    new_bboxes = []
    cl = 1
    while success_num < paste_number:
        new_bbox_x_center, new_bbox_y_center = norm_sampling(center_search_space)   # 随机生成点坐标
        if new_bbox_x_center-0.5*bbox_w < 0 or new_bbox_x_center+0.5*bbox_w > img_w:
            continue
        if new_bbox_y_center-0.5*bbox_h < 0 or new_bbox_y_center+0.5*bbox_h > img_h:
            continue
        new_bbox_x_left, new_bbox_y_left, new_bbox_x_right, new_bbox_y_right = new_bbox_x_center - 0.5 * bbox_w, \
                                                                               new_bbox_y_center - 0.5 * bbox_h, \
                                                                               new_bbox_x_center + 0.5 * bbox_w, \
                                                                               new_bbox_y_center + 0.5 * bbox_h

        new_bbox = [int(new_bbox_x_left), int(new_bbox_y_left), int(new_bbox_x_right), int(new_bbox_y_right)]
        ious = []
        if len(rescale_boxes) > 0:
            ious = [bbox_iou(new_bbox, bbox_t[:4]) for bbox_t in rescale_boxes]
        if ious == []:
            ious.append(0)
        ious2 = [bbox_iou(new_bbox, bbox_t1) for bbox_t1 in new_bboxes]
        if ious2 == []:
            ious2.append(0)
        if max(ious) <= iou_thresh and max(ious2) <= iou_thresh:
            success_num += 1
            temp.append(new_bbox)
            new_bboxes.append(new_bbox)
        else:
            continue
    return new_bboxes

def savepatch(image, labels, save, write_crop, allowed_border = 3):
    basename = os.path.basename(os.path.splitext(image)[0])
    cnt = 0
    img = Image.open(image)
    img = img.convert('RGB')
    img = np.array(img).astype(np.uint8)
    objects = parse_txt_poly(labels)
    for obj in objects:
        cnt += 1
        xmin, ymin, xmax, ymax = min(obj['poly'][0::2]), min(obj['poly'][1::2]), max(obj['poly'][0::2]), max(obj['poly'][1::2])
        xmin = xmin - allowed_border if (xmin - allowed_border) > 0 else xmin
        ymin = ymin - allowed_border if (ymin - allowed_border) > 0 else ymin
        xmax = xmax + allowed_border if (xmax + allowed_border) < img.shape[1] else xmax
        ymax = ymax + allowed_border if (ymax + allowed_border) < img.shape[0] else ymax
        subimg = copy.deepcopy(img[ymin:ymax, xmin:xmax, :])
        img_obj = Image.fromarray(subimg)
        imgname = join(save, basename+'_'+str(cnt)+'.png')
        img_obj.save(imgname)
        write_crop.write(imgname+'\n')


def parse_poly(filename):
    fd = open(filename, 'r')
    objects = []
    while True:
        line = fd.readline()
        if line:
            splitlines = line.strip().split(' ')
            if splitlines[8] != 'swimming-pool':
                continue
            object_struct = {}
            object_struct['poly'] = [(float(splitlines[0]), float(splitlines[1])),
                                     (float(splitlines[2]), float(splitlines[3])),
                                     (float(splitlines[4]), float(splitlines[5])),
                                     (float(splitlines[6]), float(splitlines[7]))]
            objects.append(object_struct)
        else:
            break
    return objects

def TuplePoly2Poly(poly):
    outpoly = [poly[0][0], poly[0][1],poly[1][0], poly[1][1],poly[2][0], poly[2][1],poly[3][0], poly[3][1]]
    return outpoly

def parse_txt_poly(filename):
    objects = parse_poly(filename)
    for obj in objects:
        obj['poly'] = TuplePoly2Poly(obj['poly'])
        obj['poly'] = list(map(int, obj['poly']))
    return objects

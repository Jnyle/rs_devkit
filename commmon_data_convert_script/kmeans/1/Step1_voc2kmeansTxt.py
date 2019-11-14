import xml.etree.ElementTree as ET
from os import getcwd

#classes = ['AB']
#classes = ['1', '2', '3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']
# classes = [
#         'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
#         'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
#         'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
#     ]
classes = ['coarse', 'crease', 'hole', 'dusty']

#img_path = '/media/clwclw/data/2019tianchi/VOC/train/images/'
#ann_path = '/media/clwclw/data/2019tianchi/VOC/train/Annotations/'
# img_path = '/media/clwclw/data/VOCdevkit/VOC2007/JPEGImages/'
# ann_path = '/media/clwclw/data/VOCdevkit/VOC2007/Annotations/'
img_path = 'C:/Users/Administrator/Desktop/train_crop/train/images/'
ann_path = 'C:/Users/Administrator/Desktop/train_crop/train/Annotations/'

count = 0
def convert_annotation(image_id, list_file):
    #in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    in_file = open(ann_path + image_id + '.xml') #clw modify
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

import os
image_names = os.listdir(img_path) 
image_ids = [image_id.split('.')[0] for image_id in image_names ]
list_file = open('train.txt', 'w')
for image_id in image_ids:
    count = count + 1
    print('clw: image nums = ', count)
    #list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(wd, year, image_id))
    list_file.write(img_path + image_id + '.jpg')
    convert_annotation(image_id, list_file)
    list_file.write('\n')
list_file.close()


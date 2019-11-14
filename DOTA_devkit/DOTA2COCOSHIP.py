import os
import cv2
import json
import numpy as np

wordname_15 = ['ship']


def txtCvtCocoRotated(srcpath, destfile):
    img_folder = os.path.join(srcpath, 'images')
    txt_folder = os.path.join(srcpath, 'labelTxt')
    data_dict = {}
    info = {'contributor': 'sensetime remote sensing group',
            'data_created': '2019',
            'description': 'This is 1.0 version of DOTA dataset.',
            'url': 'http://captain.whu.edu.cn/DOTAweb/',
            'version': '1.0',
            'year': 2018}
    data_dict['info'] = info
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    for idex, name in enumerate(wordname_15):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1
    with open(destfile, 'w') as f_out:
        filenames = os.listdir(txt_folder)
        for filename in filenames:
            basename = os.path.basename(os.path.splitext(filename)[0])
            imagepath = os.path.join(img_folder, basename + '.png')
            labelpath = os.path.join(txt_folder, filename)
            img = cv2.imread(imagepath)
            height, width, c = img.shape

            single_image = {}
            single_image['file_name'] = basename + '.png'
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

            # annotations
            with open(labelpath, 'r') as labelread:
                line = labelread.readlines()
                for str in line:
                    if str == '' or str == '\n':
                        continue
                    else:
                        res = str.split(' ')
                        if len(res) < 8:
                            continue
                        points = np.array(res[:8]).astype('float').tolist()

                        rect = cv2.minAreaRect(np.float32([[points[0], points[1]], [points[2], points[3]],
                                                           [points[4], points[5]], [points[6], points[7]]]))
                        x_ctr, y_ctr, width, height, angle = rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]

                        if angle == -90:
                            #angle = 0 if width >= height else 90
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

                        category = res[8]
                        # if height * width < 150:  # clw note: for DOTA
                        #    print('area < 150 -> delete,count = ', count_small)
                        #    count_small = count_small + 1
                        #    continue
                    single_obj = {}
                    single_obj['area'] = int(width * height)
                    single_obj['category_id'] = wordname_15.index(category) + 1
                    single_obj['segmentation'] = [
                        [points[0], points[1], points[2], points[3], points[4], points[5], points[6], points[7]]]
                    single_obj['iscrowd'] = 0
                    single_obj['bbox'] = [int(x_ctr), int(y_ctr), int(width), int(height), int(angle)]
                    single_obj['image_id'] = image_id
                    data_dict['annotations'].append(single_obj)
                    single_obj['id'] = inst_count
                    inst_count = inst_count + 1
                image_id = image_id + 1
        json.dump(data_dict, f_out)

def main():
    srcpath = r'/mnt/lustre/yanhongchang/data/DOTA/rotated_ship/trainval'
    destfile = r'/mnt/lustre/yanhongchang/data/DOTA/rotated_ship/trainval/trainval.json'
    txtCvtCocoRotated(srcpath, destfile)

if __name__ == '__main__':
    main()

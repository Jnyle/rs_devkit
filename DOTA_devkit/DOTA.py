#The code is used for visulization, inspired from cocoapi
#  Licensed under the Simplified BSD License [see bsd.txt]

import os
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Circle
plt.switch_backend('agg')
import numpy as np
import dota_utils as util
from collections import defaultdict
import cv2

def _isArrayLike(obj):
    if type(obj) == str:
        return False
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

class DOTA:
    def __init__(self, basepath):
        self.basepath = basepath
        self.labelpath = os.path.join(basepath, 'labelTxt')
        self.imagepath = os.path.join(basepath, 'images')
        self.imgpaths = util.GetFileFromThisRootDir(self.labelpath)
        self.imglist = [util.custombasename(x) for x in self.imgpaths]
        self.catToImgs = defaultdict(list)
        self.ImgToAnns = defaultdict(list)
        self.createIndex()
        self.areas = list()

    def createIndex(self):
        for filename in self.imgpaths:
            objects = util.parse_dota_poly(filename)
            imgid = util.custombasename(filename)
            self.ImgToAnns[imgid] = objects
            for obj in objects:
                cat = obj['name']
                self.catToImgs[cat].append(imgid)

    def getImgIds(self, catNms=[]):
        """
        :param catNms: category names
        :return: all the image ids contain the categories
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        if len(catNms) == 0:
            return self.imglist
        else:
            imgids = []
            for i, cat in enumerate(catNms):
                if i == 0:
                    imgids = set(self.catToImgs[cat])
                else:
                    imgids &= set(self.catToImgs[cat])
        return list(imgids)

    def loadAnns(self, catNms=[], imgId = None, difficult=None):
        """
        :param catNms: category names
        :param imgId: the img to load anns
        :return: objects
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        objects = self.ImgToAnns[imgId]
        if len(catNms) == 0:
            return objects
        outobjects = [obj for obj in objects if (obj['name'] in catNms)]
        return outobjects
    def showAnns(self, objects, imgId, range):
        """
        :param catNms: category names
        :param objects: objects to show
        :param imgId: img to show
        :param range: display range in the img
        :return:
        """
        img = self.loadImgs(imgId)[0]
        #plt.imshow(img)
        plt.axis('off')

        ax = plt.gca()
        ax.set_autoscale_on(False)
        polygons = []
        color = []
        circles = []
        r = 5
        c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
        for obj in objects:
            poly = obj['poly']
            polygons.append(Polygon(poly))
            color.append(c)
            point = poly[0]
            circle = Circle((point[0], point[1]), r)
            circles.append(circle)
        p = PatchCollection(polygons, facecolors=color, linewidths=0, alpha=0.4)
        ax.add_collection(p)
        p = PatchCollection(polygons, facecolors='none', edgecolors=color, linewidths=2)
        ax.add_collection(p)
        p = PatchCollection(circles, facecolors='red')
        ax.add_collection(p)
        #plt.show()
        plt.savefig(os.path.join(self.basepath, 'swimming-pool', 'show', imgId+'.png').replace('\\','/'))
    def loadImgs(self, imgids=[]):
        """
        :param imgids: integer ids specifying img
        :return: loaded img objects
        """
        print('isarralike:', _isArrayLike(imgids))
        imgids = imgids if _isArrayLike(imgids) else [imgids]
        print('imgids:', imgids)
        imgs = []
        for imgid in imgids:
            filename = os.path.join(self.imagepath, imgid + '.png')
            print('filename:', filename)
            img = cv2.imread(filename)
            imgs.append(img)
        return imgs

    def saveImgAndlabel(self, imgId,anns):
        img = self.loadImgs(imgId)[0]
        imgpath = os.path.join(self.basepath, 'swimming-pool', 'images', imgId+'.png').replace('\\','/')
        cv2.imwrite(imgpath, img)
        outline = list()
        for obj in anns:
            self.areas.append(float(obj['area']))
            tmp = list()
            for x, y in obj['poly']:
                tmp.append(x)
                tmp.append(y)
            outline.append(' '.join(list(map(str, tmp)))+' swimming-pool\n')
        filepath = os.path.join(self.basepath, 'swimming-pool', 'labelTxt', imgId+'.txt').replace('\\','/')
        print(filepath)
        with open(filepath, 'w') as f_out:
            f_out.writelines(outline)

    def histogram(self, data, x_label, y_label, title):
        _, ax = plt.subplots()
        bins = np.arange(0, 3200, 20)
        res = ax.hist(data, color='#539caf', bins=bins) 
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        ax.set_title(title)
        return res

if __name__ == '__main__':
    examplesplit = DOTA('/mnt/lustre/yanhongchang/data/DOTA/ValSet/hv1.0')
    imgids = examplesplit.getImgIds(catNms=['swimming-pool'])
    img = examplesplit.loadImgs(imgids)
    for imgid in imgids:
        anns = examplesplit.loadAnns(catNms=['swimming-pool'],imgId=imgid)
        #examplesplit.showAnns(anns, imgid, 2)
        examplesplit.saveImgAndlabel(imgid,anns)
        #for obj in anns:
        #    examplesplit.areas.append(float(obj['area']))
    #res = examplesplit.histogram(data=examplesplit.areas
    #                     , x_label='area'
    #                     , y_label='Frequency'
    #                     , title='Distribution of train object area {}'.format(len(examplesplit.areas)))
    #plt.savefig('/mnt/lustre/yanhongchang/data/DOTA/TrainSet/ship_train/area.png')
    # plt.show()

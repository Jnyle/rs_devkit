import aug as am
import Helpers as hp
from util import *
import os
from os.path import join
from tqdm import tqdm
import random

base_dir = os.getcwd()

save_base_dir = join(base_dir, 'data', 'save1')

check_dir(save_base_dir)

imgs_dir = [f.strip() for f in open(join(base_dir, 'data', 'train.txt')).readlines()]
labels_dir = hp.replace_labels(imgs_dir)

small_imgs_dir = [f.strip() for f in open(join(base_dir, 'data', 'crop.txt')).readlines()]
random.shuffle(small_imgs_dir)

for i in tqdm(range(2000)):
    small_img = []
    for x in range(10):
        if small_imgs_dir == []:
            #exit()
            small_imgs_dir = [f.strip() for f in open(join(base_dir, 'data', 'crop.txt')).readlines()]
            random.shuffle(small_imgs_dir)
        small_img.append(small_imgs_dir.pop())
    am.copysmallobjects2(imgs_dir[0], labels_dir[0], save_base_dir, small_img, i+1)

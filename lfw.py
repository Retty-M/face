import os
import cv2
import numpy as np
from os.path import join as pjoin

data_dir = '/home/ubuntu/Datasets/lfw'
pics_ctr = 1


def preproc(im):
    im = cv2.resize(im, (96, 96))
    return im.astype(np.float32)/255


for guy in os.listdir(data_dir):
    person_dir = pjoin(data_dir, guy)
    for f in os.listdir(person_dir):
        img_dir = pjoin(person_dir, f)
        # img = cv2.resize(cv2.imread(img_dir), (96, 96), interpolation=cv2.INTER_CUBIC)
        # cv2.imwrite('/home/ubuntu/Code/face/train_data/pic_others/other%d.jpg' % pics_ctr, img)
        os.system('cp %s /home/ubuntu/Code/face/train_data/others/other%d.jpg' % (img_dir, pics_ctr))
        break
    if pics_ctr > 200:
        break
    pics_ctr += 1
# return data, pics_ctr

print pics_ctr

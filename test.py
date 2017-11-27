import cv2
import os
from os.path import join as pjoin


def load_data(data_dir):
    data = {}
    pics_ctr = 0
    for guy in os.listdir(data_dir):
        curr_pics = []
        person_dir = pjoin(data_dir, guy)
        for f in os.listdir(person_dir):
            img_dir = pjoin(person_dir, f)
            img = cv2.imread(img_dir)
            curr_pics.append(img)
        data[guy] = curr_pics
    return data


# data = load_data('./train_data')
# keys = []
# for key in data.iterkeys():
#     keys.append(key)
#     print('folder:{} image numbers:{}'.format(key, len(data[key])))
#
# print keys[0]
# print keys[1]
# print keys[2]
# print data['wym']
# print len(data[keys[0]])
# print len(data[keys[1]])
# print len(data[keys[2]])

# hah = False
# if hah is False:
#     print 'sds'
# print ~hah, hah
keys = [1, 2, 3, 4]
print 'No{}:'.format(i for i in range(len(keys)))

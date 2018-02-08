# coding=utf-8

import os
import cv2
import numpy as np
import face as Face

from os.path import join as pjoin

data_dir = './train_data'

# Number of frames after which to run face detection


face_capture = Face.Capture()

for guy in os.listdir(data_dir):

    image_count = 0
    encoder = []
    person_dir = pjoin(data_dir, guy)
    encoder_file = pjoin(person_dir, 'encoder.npy')

    if os.path.exists(encoder_file):
        os.remove(encoder_file)

    for f in os.listdir(person_dir):
        img_dir = pjoin(person_dir, f)
        img = cv2.imread(img_dir)
        face = face_capture.capture_encode(img)
        if face is not None:
            image_count += 1
            encoder.append(face.embedding)
    print('INFO: {}\'s valuable images number is {}'.format(guy, image_count))

    np.save('./train_data/%s/encoder.npy' % guy, encoder)
    print('INFO: {}\'s encoder file is generated'.format(guy))



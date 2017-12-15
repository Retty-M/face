# coding=utf-8

# MIT License
#
# Copyright (c) 2017 François Gervais
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import sys
import cv2
import pickle
import argparse

import numpy as np
import face as Face

from os.path import join as pjoin
from sklearn.svm import SVC
from sklearn.svm import OneClassSVM
from sklearn.externals import joblib


def main(args):
    train_x = []
    train_y = []
    face_recognition = Face.Recognition()
    data, keys = load_data('./train_data')

    if args.debug:
        print("Debug enabled")
        Face.debug = True

    temp = 0
    for index, name in enumerate(keys):
        # name = keys[index]
        for img in data[name]:
            face = face_recognition.add_identity(img, person_name=name)
            if face is not None:
                # print face.embedding
                # print face.name
                train_x.append(face.embedding)
                train_y.append(keys.index(face.name))
                # train_x = np.concatenate([train_x, face.embedding]) if train_x is not None else face.embedding
                # train_y = np.concatenate([train_y, face.name]) if train_y is not None else face.name
        print('INFO: {} has {} images available'.format(name, len(train_x) - temp))
        temp = len(train_x)
    train_x = np.array(train_x).reshape(-1, 128)
    train_y = np.array(train_y)
    # print(train_x.shape)
    # print(train_y.shape)
    train_boundary(train_x, 'boundary.model')
    train_classifier(train_x, train_y, keys, '1208.pkl')


def load_data(data_dir):
    keys = []
    data = {}
    for guy in os.listdir(data_dir):
        curr_pics = []
        keys.append(guy)
        person_dir = pjoin(data_dir, guy)
        for f in os.listdir(person_dir):
            img_dir = pjoin(person_dir, f)
            img = cv2.imread(img_dir)
            curr_pics.append(img)
        data[guy] = curr_pics
        print('INFO: {} images number is {}'.format(guy, len(curr_pics)))
    return data, keys


def train_boundary(emb_array, boundary_filename):
    print '------- Training Boundary -------'
    model = OneClassSVM(nu=0.005, gamma=0.5)
    model.fit(emb_array)

    joblib.dump(model, boundary_filename)
    print 'Saved classifier model to file "%s"' % boundary_filename


def train_classifier(emb_array, label_array, class_names, classifier_filename):
    print '------- Training Classifier -------'
    model = SVC(kernel='linear', probability=True, verbose=False)
    model.fit(emb_array, label_array)

    with open(classifier_filename, 'wb') as outfile:
        pickle.dump((model, class_names), outfile)
    print 'Saved classifier model to file "%s"' % classifier_filename


def parse_arguments(argv):
        parser = argparse.ArgumentParser()
        parser.add_argument('--debug', action='store_true',
                            help='Enable some debug outputs.')
        return parser.parse_args(argv)


if __name__ == '__main__':
        main(parse_arguments(sys.argv[1:]))

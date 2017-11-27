import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from os.path import join as pjoin
import sys
import copy
import facenet2 as facenet
import detect_face
import nn4 as network
import random


import sklearn

from sklearn.externals import joblib

# face detection parameters
# minimum size of face
minsize = 20
# three steps's threshold
threshold = [0.6, 0.7, 0.7]
# scale factor
factor = 0.709
# frame intervals
frame_interval = 1


def to_rgb(img):
  w, h = img.shape
  ret = np.empty((w, h, 3), dtype=np.uint8)
  ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
  return ret


def update_dir(data_dir):
    keys = []
    for guy in os.listdir(data_dir):
        keys.append(guy)
    print 'Detect: %s' % str(keys)
    return keys


# restore mtcnn model
print 'Creating networks and loading parameters'
gpu_memory_fraction = 0.8
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, './data/')

# restore facenet model
print('Restore facenet embedding model')
with tf.Graph().as_default():
    with tf.Session() as sess:
        facenet.load_model('/home/ubuntu/Code/face/models/20170511-185253/')
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        print('Facenet embedding success')

        model = joblib.load('/home/ubuntu/Code/face/knn_20170512-110547_2.model')

        video_capture = cv2.VideoCapture(2)
        # c = 0
        find_result = 'other'
        keys = update_dir('./train_data/')
        sess.run(tf.global_variables_initializer())
        while True:
            
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if gray.ndim == 2:
                img = to_rgb(gray)

            bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
            # number of faces
            # nrof_faces = bounding_boxes.shape[0]

            for face_position in bounding_boxes:

                face_position=face_position.astype(int)
                cv2.rectangle(frame, (face_position[0], face_position[1]), (face_position[2], face_position[3]),
                              (0, 255, 0), 2)
                crop = img[face_position[1]:face_position[3], face_position[0]:face_position[2], ]
                try:
                    crop = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_CUBIC)
                    data = crop.reshape(-1, 160, 160, 3)
                    emb_data = sess.run([embeddings], feed_dict={images_placeholder: np.array(data), phase_train_placeholder: False})[0]
                    predict = model.predict(emb_data)
                    for i in range(len(keys)):
                        if predict == i:
                            find_result = keys[i]
                    cv2.putText(frame, find_result, (face_position[0], face_position[1]-10),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.8, (255, 0, 0), thickness=2, lineType=2)
                except:
                    pass
            # Display the resulting frame
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()

import cv2
import numpy as np
import tensorflow as tf
import re
from tensorflow.python.platform import gfile
import os
from os.path import join as pjoin
import sys
import copy
import detect_face
import facenet2 as facenet
import nn4 as network
import matplotlib.pyplot as plt

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.externals import joblib

# Face detection parameters
# minimum size of face
minsize = 20
# three steps's threshold
threshold = [0.6, 0.7, 0.7]
# scale factor
factor = 0.709

# Facenet embedding parameters
# Directory containing the graph definition and checkpoint files
model_dir = './data/model.ckpt-500000'
# Points to a module containing the definition of the inference graph
model_def = 'models.nn4'
# Image size (height, width) in pixels
image_size = 96
# The type of pooling to use for some of the inception layers {'MAX', 'L2'}
pool_type = 'MAX'
# Enables Local Response Normalization after the first layers of the inception network
use_lrn = False
# Random seed
seed = 42
# Number of images to process in a batch
batch_size = None

# Train folder
data_dir = './train_data/'


def to_rgb(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def read_img(person_dir, f):
    print pjoin(person_dir, f)
    img = cv2.imread(pjoin(person_dir, f))
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # if gray.ndim == 2:
    #     img = to_rgb(gray)
    return img


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


def load_model(model):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if os.path.isfile(model_exp):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


data = load_data('./train_data')
keys = []
for key in data.iterkeys():
    keys.append(key)
for i in range(len(keys)):
    print('No{}: {} image numbers is {}'.format(i, keys[i], len(data[keys[i]])))

print 'Creating networks and loading parameters'
gpu_memory_fraction = 0.8
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, './data/')


print('Restore facenet embedding model')

# FLAGS = utils.load_tf_flags()
# tf.Graph().as_default()
# sess = tf.Session()
# images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3), name='input')
# phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
# embeddings = network.inference(images_placeholder, FLAGS.pool_type, FLAGS.use_lrn, 1.0, phase_train=phase_train_placeholder)
# ema = tf.train.ExponentialMovingAverage(1.0)
# saver = tf.train.Saver(ema.variables_to_restore())
# ckpt = tf.train.get_checkpoint_state(os.path.expanduser(FLAGS.model_dir))
# saver.restore(sess, ckpt.model_checkpoint_path)

# tf.Graph().as_default()
# sess = tf.Session()
# images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
# embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
# phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
# images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, 3), name='input')
# phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
# embeddings = network.inference(images_placeholder, pool_type, use_lrn, 1.0, phase_train=phase_train_placeholder)
# embeddings = network.inference(images_placeholder, 1.0, phase_train=phase_train_placeholder)
# ema = tf.train.ExponentialMovingAverage(1.0)
# saver = tf.train.Saver(ema.variables_to_restore())
# print saver.saver_def.filename_tensor_name
    # ckpt = tf.train.get_checkpoint_state('./data/model-20170511-185253.ckpt-80000.data-00000-of-00001')
    # saver.restore(sess, ckpt.model_checkpoint_path)
# with sess.as_default():
# load_model('./models/20170511-185253')
# model_checkpoint_path = './models/model-20160506.ckpt-500000'
# saver.restore(sess, model_checkpoint_path)

# saver = tf.train.import_meta_graph('./models/20170511-185253/model-20170511-185253.meta')
# saver.restore(sess, './models/20170511-185253/model-20170511-185253.ckpt-80000')
# print('Facenet embedding restore success')

# load_model('./models/20170511-185253/')
with tf.Graph().as_default():
    with tf.Session() as sess:
        facenet.load_model('./models/20170512-110547/')
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        # images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, 3), name='input')
        # phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        # embeddings = network.inference(images_placeholder, pool_type, use_lrn, 1.0, phase_train=phase_train_placeholder)

        train_x = []
        train_y = []
        sess.run(tf.global_variables_initializer())
        for index in range(len(keys)):
            for x in data[keys[index]]:
                x_color = to_rgb(x)
                bounding_boxes, _ = detect_face.detect_face(x_color, minsize, pnet, rnet, onet, threshold, factor)
                # nrof_faces = bounding_boxes.shape[0]  # number of faces

                for face_position in bounding_boxes:
                    face_position = face_position.astype(int)
                    # print(face_position[0:4])
                    cv2.rectangle(x_color, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 2)
                    crop = x_color[face_position[1]:face_position[3], face_position[0]:face_position[2], ]
                    try:
                        crop = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_CUBIC)
                        crop_data = crop.reshape(-1, 160, 160, 3)

                        # print(crop_data.shape)
                        # images = facenet.load_data(['./train_data/wym/my_photo-1.jpg'], False, False, 96)
                        feed_dict = {images_placeholder: np.array(crop_data), phase_train_placeholder: False}
                        emb_data = sess.run(embeddings, feed_dict=feed_dict)[0]
                        # emb_data = sess.run([embeddings], feed_dict={images_placeholder: np.array(crop_data), phase_train_placeholder: False})[0]
                        # print emb_data
                        # print len(emb_data[0])
                        train_x.append(emb_data)
                        train_y.append(index)
                    except:
                        print 'error'
            print len(train_x)
        print('Down number is:{}'.format(len(train_x)))

# train/test split
train_x = np.array(train_x)
train_x = train_x.reshape(-1, 128)
train_y = np.array(train_y)
print(train_x.shape)
print(train_y.shape)

x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=.1, random_state=42)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


# KNN Classifier
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model


def svm_classifier(train_x, train_y):
    from sklearn import svm
    model = svm.SVC(kernel='rbf')
    model.fit(train_x, train_y)
    return model


classifiers = knn_classifier
# classifiers = svm_classifier
model = classifiers(x_train, y_train)
predict = model.predict(x_test)

accuracy = metrics.accuracy_score(y_test, predict)
print ('accuracy: %.2f%%' % (100 * accuracy))

# test = x_train[0]
# predict_test = model.predict(test)
# print y_train[0], predict

# Save model
# joblib.dump(model, 'knn_20170511-185253_1.model')
# joblib.dump(model, 'svm_20170511-185253_1.model')
joblib.dump(model, 'knn_20170512-110547_2.model')
# joblib.dump(model, 'svm_20170512-110547_1.model')

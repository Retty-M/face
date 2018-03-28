# coding=utf-8
import os
import cv2
import sys
import pickle
import argparse
import collections
import numpy as np
import face as Face
from os.path import join as pjoin

from database import DB
from sklearn.svm import SVC


def add_overlays(frame, face, image_rate):
    if face is not None:
        face_bb = face.bounding_box.astype(int)
        cv2.rectangle(frame,
                      (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                      (0, 255, 0), 2)

    cv2.putText(frame, str(image_rate) + "%", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                thickness=2, lineType=2)


def capture(device_num, face_capture):
    # Number of frames after which to run face detection
    frame_interval = 2
    frame_count = 0
    image_count = 0
    encoder = []

    db = DB('./data/info.db')
    user = info_capture()

    db.add_data(user)
    name = user['name']
    person_dir = pjoin('./train_data', name)
    if os.path.exists(person_dir):
        os.removedirs(person_dir)
    else:
        os.makedirs(person_dir)

    video_capture = cv2.VideoCapture(device_num)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # face_capture = Face.Capture()

    while True:
        # Capture frame-by-frame
        face = None
        ret, frame = video_capture.read()
        if frame_count > 20:
            if (frame_count % frame_interval) == 0:
                face = face_capture.capture_encode(frame)
                if face is not None:
                    image_count += 1
                    encoder.append(face.embedding)
                    cv2.imwrite('./train_data/%s/%s_%d.jpg' % (name, name, image_count), frame)
        add_overlays(frame, face, image_count)

        frame_count += 1
        cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            os.system('rm -rf ./train_data/%s' % name)
            break

        if image_count == 100:
            break

    # encoder = np.array(encoder).reshape(-1, 128)
    # print(encoder.shape)
    np.save('./train_data/%s/encoder.npy' % name, encoder)

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


def info_capture():
    print('**********************************************************************************')
    user = collections.defaultdict(str)
    name = raw_input('姓名:')
    user['name'] = name
    age = raw_input('年龄:')
    while not age.isdigit() or int(age) not in range(1, 100):
        age = raw_input('输入有误，请正确输入数字1~99:')
    user['age'] = int(age)
    duty = raw_input('职务:')
    user['duty'] = duty
    department_array = ('综合管理办', '研发中心', '工艺中心', '测试中心', '装联中心')
    department = raw_input('部门(1=综合管理办 2=研发中心 3=工艺中心 4=测试中心 5=装联中心):')
    while not department.isdigit() or int(department) not in range(1, 6):
        department = raw_input('输入有误，请正确输入数字1~5:')
    user['department'] = department_array[int(department) - 1]
    secret_array = ('绝密', '机密', '秘密')
    secret = raw_input('密级(1=绝密 2=机密 3=秘密):')
    while not secret.isdigit() or int(secret) not in range(1, 4):
        secret = raw_input('输入有误，请正确输入数字1~3:')
    user['secret'] = secret_array[int(secret) - 1]
    print('**********************************************************************************')
    return user


def encoder(data_dir, face_capture):
    for guy in os.listdir(data_dir):
        encode = []
        person_dir = pjoin(data_dir, guy)
        encoder_file = pjoin(person_dir, 'encoder.npy')
        if os.path.exists(encoder_file):
            os.system('rm %s' % encoder_file)
        for image in os.listdir(person_dir):
            person_img = pjoin(person_dir, image)
            frame = cv2.imread(person_img)
            # cv2.imshow('image', frame)
            face = face_capture.capture_encode(frame)
            if face is not None:
                encode.append(face.embedding)
        np.save(encoder_file, encode)


def train(data_dir, classifier_filename):
    keys = []
    train_x = []
    train_y = []
    for guy in os.listdir(data_dir):
        person_dir = pjoin(data_dir, guy)
        encoder_file = pjoin(person_dir, 'encoder.npy')
        if os.path.exists(encoder_file):
            keys.append(guy)
            encoder = np.load(encoder_file)
            if len(train_x) == 0:
                train_x = encoder
            else:
                train_x = np.append(train_x, encoder, axis=0)
            for i in range(0, encoder.shape[0]):
                train_y.append(keys.index(guy))
            print('INFO: {}\'s encoder file is loaded'.format(guy))
        else:
            print('ERROR: cannot find {}\'s encoder file'.format(guy))
    train_x = np.array(train_x).reshape(-1, 128)
    train_y = np.array(train_y)
    # print(train_x.shape)
    # print(train_y.shape)

    print '------- Training Classifier -------'
    model = SVC(kernel='linear', probability=True)
    model.fit(train_x, train_y)

    with open(classifier_filename, 'wb') as outfile:
        pickle.dump((model, keys), outfile)
    print 'Saved classifier model to file "%s"' % classifier_filename


def main(args):
    if args.encode or args.capture:
    # if args.train is False:
        face_capture = Face.Capture()
        if args.encode:
            encoder('./train_data', face_capture)
        if args.capture:
            while True:
                capture(0, face_capture)
                key = raw_input('继续（任意键）退出（N）:')
                if str(key).upper() == 'N':
                    break
    train('./train_data', '0308.pkl')


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Enable some debug outputs.')
    parser.add_argument('--capture', action='store_true', help='Enable some debug outputs.')
    parser.add_argument('--encode', action='store_true', help='Enable some debug outputs.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    # main(parse_arguments(sys.argv[1:]))
    db = DB('./data/info.db')
    user = info_capture()
    db.add_data(user)

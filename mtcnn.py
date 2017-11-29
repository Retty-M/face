# coding=utf-8
import cv2
import detect_face
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Face detection parameters
# minimum size of face
minsize = 20
# three steps's threshold
threshold = [0.6, 0.7, 0.7]
# scale factor
factor = 0.709

# cap = cv2.VideoCapture(2)
frame = cv2.imread('./train_data/lr/my_photo-1.jpg')

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

# Restore mtcnn model
print('Creating networks and loading parameters')
gpu_memory_fraction = 0.8
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, './data/')

while True:
    # ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.ndim == 2:
        img = to_rgb(gray)

    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

    for face_position in bounding_boxes:

        face_position = face_position.astype(int)

        # print((int(face_position[0]), int( face_position[1])))
        # word_position.append((int(face_position[0]), int( face_position[1])))

        cv2.rectangle(frame, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 2)
        crop = img[face_position[1]:face_position[3], face_position[0]:face_position[2], ]
        cv2.putText(frame, 'Good', (face_position[0], face_position[1]-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.8,
                    (255, 0, 0), thickness=2, lineType=2)
        try:
            crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_CUBIC)
            data = crop.reshape(-1, 96, 96, 3)
        except:
            pass

        # emb_data = sess.run([embeddings], feed_dict={images_placeholder: np.array(data), phase_train_placeholder: False})[0]

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

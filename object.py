# coding: utf-8

import os
import face
import collections
import numpy as np
import tensorflow as tf
from sort import Sort

from utils import label_map_util
from utils import visualization_utils as vis_util

# What model to download.

# ssd_mobilenet_v1_coco
# faster_rcnn_resnet50_coco
# faster_rcnn_inception_v2_coco

MODEL_NAME = 'ssd_mobilenet_v1_coco'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './models/' + MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


class Detection:

    def __init__(self):
        self.persons = collections.defaultdict(str)
        self.tracker = Sort()
        self.face_recognition = face.Recognition()

        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            with tf.Session(graph=detection_graph) as self.sess:
                # Definite input and output Tensors for detection_graph
                self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    def find_objects(self, image):
        # with self.detection_graph.as_default():
        #     with tf.Session(graph = self.detection_graph) as sess:

        # image_np = load_image_into_numpy_array(frame)
        image_np_expanded = np.expand_dims(image, axis=0)

        # Actual detection.
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})

        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
        )

    def track_person(self, image):
        image_np_expanded = np.expand_dims(image, axis=0)

        # Actual detection.
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})

        locations = vis_util.find_person_custom(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores)
        )

        trackers = self.tracker.update(locations, image)
        persons_temp = collections.defaultdict(str)
        for d in trackers:
            d = d.astype(np.int32)
            if d[4] in self.persons:
                persons_temp[d[4]] = self.persons[d[4]]
                if self.persons[str(d[4])] != '1111':
                    faces = self.face_recognition.identify(image[d[1]:d[3], d[0]:d[2]])
                    if faces is not None:
                        for face in faces:
                            if persons_temp[d[4]] == face.name:
                                persons_temp[str(d[4])] += '1'
            else:
                persons_temp[d[4]] = '未知'
                persons_temp[str(d[4])] = '1'
            if persons_temp[d[4]] == '未知':
                faces = self.face_recognition.identify(image[d[1]:d[3], d[0]:d[2]])
                if faces is not None:
                    for face in faces:
                        persons_temp[str(d[4])] += '1'
                        persons_temp[d[4]] = face.name
            color = vis_util.STANDARD_COLORS[d[4] % len(vis_util.STANDARD_COLORS)]
            vis_util.draw_bounding_box_on_image_array_custom(image, d[1], d[0], d[3], d[2],
                                                             color=color,
                                                             display_str_list=unicode(persons_temp[d[4]], 'utf-8'),
                                                             use_normalized_coordinates=False)
        self.persons = persons_temp


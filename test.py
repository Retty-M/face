# coding:utf-8
import cv2
# import os
# from os.path import join as pjoin
#
#
# def load_data(data_dir):
#     data = {}
#     pics_ctr = 0
#     for guy in os.listdir(data_dir):
#         curr_pics = []
#         person_dir = pjoin(data_dir, guy)
#         for f in os.listdir(person_dir):
#             img_dir = pjoin(person_dir, f)
#             img = cv2.imread(img_dir)
#             curr_pics.append(img)
#         data[guy] = curr_pics
#     return data


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
# keys = [1, 2, 3, 4]
# print keys.index(3)
# print 'No{}:'.format(i for i in range(len(keys)))


import os
# import pyttsx

# reload(sys)
# sys.setdefaultencoding('utf-8')
# print sys.getdefaultencoding()
#
# engine = pyttsx.init(driverName='espeak')
# voices = engine.getProperty('voices')
# for voice in voices:
#     engine.setProperty('voice', voice.id)
#     engine.say(u'哈哈哈')
# engine.runAndWait()
# engine.say('hello world')
# engine.say(u'哈哈哈')
# engine.runAndWait()

# os.system('espeak -v zh+f2 -b 1 腾讯')

# import cv2
# import numpy as np
# from PIL import Image, ImageDraw, ImageFont
#
# cap = cv2.VideoCapture(0)
# while True:
#
#     ret, im = cap.read()
#     cv2_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#     pil_im = Image.fromarray(cv2_im)
#
#     draw = ImageDraw.Draw(pil_im)
#     font = ImageFont.truetype('simhei', 20, encoding='utf-8')
#     draw.rectangle((100, 100, 300, 300), outline=(0, 255, 0))
#     draw.text((90, 90), u"{}%".format(unicode('收到', 'utf-8')), (0, 0, 255), font=font)
#
#     cv2_text_im = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
#     cv2.imshow("Video", cv2_text_im)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

# import pyaudio
# import wave
#
# # define stream chunk
# chunk = 1024
#
# # open a wav format music
# f = wave.open('./audio/wym.wav', 'rb')
# # instantiate PyAudio
# p = pyaudio.PyAudio()
# # open stream
# stream = p.open(format=p.get_format_from_width(f.getsampwidth()),
#                 channels=f.getnchannels(),
#                 rate=f.getframerate(),
#                 output=True)
# # read data
# data = f.readframes(chunk)
#
# # paly stream
# while data != '':
#     stream.write(data)
#     data = f.readframes(chunk)
#
# # stop stream
# stream.stop_stream()
# stream.close()
#
# # close PyAudio
# p.terminate()

# def visualize_boxes_and_labels_custom(image,
#                                       boxes,
#                                       classes,
#                                       scores,
#                                       category_index,
#                                       instance_masks=None,
#                                       keypoints=None,
#                                       use_normalized_coordinates=False,
#                                       max_boxes_to_draw=20,
#                                       min_score_thresh=.5,
#                                       agnostic_mode=False,
#                                       line_thickness=4):
#   """Overlay labeled boxes on an image with formatted scores and label names.
#
#   This function groups boxes that correspond to the same location
#   and creates a display string for each detection and overlays these
#   on the image. Note that this function modifies the image in place, and returns
#   that same image.
#
#   Args:
#     image: uint8 numpy array with shape (img_height, img_width, 3)
#     boxes: a numpy array of shape [N, 4]
#     classes: a numpy array of shape [N]. Note that class indices are 1-based,
#       and match the keys in the label map.
#     scores: a numpy array of shape [N] or None.  If scores=None, then
#       this function assumes that the boxes to be plotted are groundtruth
#       boxes and plot all boxes as black with no classes or scores.
#     category_index: a dict containing category dictionaries (each holding
#       category index `id` and category name `name`) keyed by category indices.
#     instance_masks: a numpy array of shape [N, image_height, image_width], can
#       be None
#     keypoints: a numpy array of shape [N, num_keypoints, 2], can
#       be None
#     use_normalized_coordinates: whether boxes is to be interpreted as
#       normalized coordinates or not.
#     max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
#       all boxes.
#     min_score_thresh: minimum score threshold for a box to be visualized
#     agnostic_mode: boolean (default: False) controlling whether to evaluate in
#       class-agnostic mode or not.  This mode will display scores but ignore
#       classes.
#     line_thickness: integer (default: 4) controlling line width of the boxes.
#
#   Returns:
#     uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
#   """
#   # Create a display string (and color) for every box location, group any boxes
#   # that correspond to the same location.
#   box_to_display_str_map = collections.defaultdict(list)
#   box_to_color_map = collections.defaultdict(str)
#   box_to_instance_masks_map = {}
#   box_to_keypoints_map = collections.defaultdict(list)
#   if not max_boxes_to_draw:
#     max_boxes_to_draw = boxes.shape[0]
#   for i in range(min(max_boxes_to_draw, boxes.shape[0])):
#     if scores is None or scores[i] > min_score_thresh:
#       box = tuple(boxes[i].tolist())
#       if instance_masks is not None:
#         box_to_instance_masks_map[box] = instance_masks[i]
#       if keypoints is not None:
#         box_to_keypoints_map[box].extend(keypoints[i])
#       if scores is None:
#         box_to_color_map[box] = 'black'
#       else:
#         if not agnostic_mode:
#           if classes[i] in category_index.keys():
#             class_name = category_index[classes[i]]['name']
#           else:
#             class_name = 'N/A'
#           display_str = '{}: {}%'.format(
#               class_name,
#               int(100*scores[i]))
#         else:
#           display_str = 'score: {}%'.format(int(100 * scores[i]))
#         box_to_display_str_map[box].append(display_str)
#         if agnostic_mode:
#           box_to_color_map[box] = 'DarkOrange'
#         else:
#           box_to_color_map[box] = STANDARD_COLORS[
#               classes[i] % len(STANDARD_COLORS)]
#
#   # Draw all boxes onto image.
#   for box, color in box_to_color_map.items():
#     print len(box_to_color_map)
#     ymin, xmin, ymax, xmax = box
#     if instance_masks is not None:
#       draw_mask_on_image_array(
#           image,
#           box_to_instance_masks_map[box],
#           color=color
#       )
#     draw_bounding_box_on_image_array(
#         image,
#         ymin,
#         xmin,
#         ymax,
#         xmax,
#         color=color,
#         thickness=line_thickness,
#         display_str_list=box_to_display_str_map[box],
#         use_normalized_coordinates=use_normalized_coordinates)
#     if keypoints is not None:
#       draw_keypoints_on_image_array(
#           image,
#           box_to_keypoints_map[box],
#           color=color,
#           radius=line_thickness / 2,
#           use_normalized_coordinates=use_normalized_coordinates)
#
#   return image

# import numpy as np
# a = (0.35919106006622314, 0.2792535126209259, 0.9985294342041016, 0.6948285102844238)
# # a = list([i*100 for i in a])
# b = np.array(a)
# print a, b

# import numpy
#
# c = []
# a = numpy.load('./train_data/王艺谋/encoder.npy')
# print(a is None)
# print(len(c))
# if c:
#     print('xx')
# b = []
# c = a
# print(a.shape)
# for _ in range(0, a.shape[0]):
#     b.append(2)
# b = np.array(b)
# print(b)
# print(c.shape)
# print(b.shape)

# import collections

import gi
import numpy as np
# gi.require_version('Gst', '1.0')
# from gi.repository import Gst
# import cv2
#
# Gst.init(None)
#
# # gst-launch-1.0 udpsrc uri="udp://192.168.1.105:3221" caps="application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H265,sprop-parameter-sets=(string)\"Z0JAMpWgHgCJ+VA\\=\\,aM48gA\\=\\=\", payload=(int)96" !
# # rtph265depay ! decodebin ! autovideosink sync=false
#
# # Create elements
# # udpsrc = Gst.ElementFactory.make('udpsrc')
# # rtph265depay = Gst.ElementFactory.make('rtph265depay')
# # autovideosink = Gst.ElementFactory.make('autovideosink')
# #
# # caps = Gst.caps_from_string('application/x-rtp, media=(string)video, clock-rate=(int)90000, '
# #                             'encoding-name=(string)H265, payload=(int)96')
#
# # udpsrc.set_property('port', 3221)
# # udpsrc.set_property('uri', "udp://192.168.1.105:3221")
# # udpsrc.set_property('caps', caps)
#
# # autovideosink.set_property('sync', False)
#
# # filesrc = Gst.ElementFactory.make("videotestsrc", "source")
# # appsink = Gst.ElementFactory.make("autovideosink", "sink")
#
# filesrc = Gst.ElementFactory.make("filesrc", "filesrc")
# avidemux = Gst.ElementFactory.make("avidemux", "avidemux")
# decodebin = Gst.ElementFactory.make("decodebin", "decodebin")
# appsink = Gst.ElementFactory.make("autovideosink", "appsink")
#
# pipeline = Gst.Pipeline.new('pipeline')
#
# if not filesrc or not appsink or not pipeline:
#     print("Not all elements could be created.")
#     exit(-1)
#
# pipeline.add(filesrc)
# pipeline.add(avidemux)
# pipeline.add(decodebin)
# pipeline.add(appsink)
#
# # if not Gst.Element.link(filesrc, decodebin):
# #     print("Elements could not be linked 0.")
# if not Gst.Element.link(filesrc, avidemux):
#     print("Elements could not be linked 1.")
# if not Gst.Element.link(filesrc, appsink):
#     print("Elements could not be linked 2.")
#
# # filesrc.link(avidemux)
# # avidemux.link(decodebin)
# # decodebin.link(appsink)
#
# # filesrc.set_property('location', '/home/id/TownCentreXVID.avi')
#
# # Start playing
# ret = pipeline.set_state(Gst.State.PLAYING)
# if ret == Gst.StateChangeReturn.FAILURE:
#     print("Unable to set the pipeline to the playing state.")
#
# # Wait until error or EOS
# bus = pipeline.get_bus()
# msg = bus.timed_pop_filtered(
#     Gst.CLOCK_TIME_NONE, Gst.MessageType.ERROR | Gst.MessageType.EOS)
#
# if (msg):
#     if msg.type == Gst.MessageType.ERROR:
#         err, debug = msg.parse_error()
#         print("Error received from element %s: %s" % (
#             msg.src.get_name(), err))
#         print("Debugging information: %s" % debug)
#     elif msg.type == Gst.MessageType.EOS:
#         print("End-Of-Stream reached.")
#     else:
#         print("Unexpected message received.")
#
# # Free resources
# pipeline.set_state(Gst.State.NULL)


# # cap = cv2.VideoCapture('/home/id/LG.OLED.4K.DEMO_NASA.Two/LG.OLED.4K.DEMO_NASA.Two.ts')
# # cap = cv2.VideoCapture('rtsp://184.72.239.149/vod/mp4://BigBuckBunny_175k.mov')
# # cap = cv2.VideoCapture('rtsp://192.168.1.168:8554/test')
# cap = cv2.VideoCapture('gst-launch-1.0 filesrc location=/home/id/TownCentreXVID.avi ! avidemux ! decodebin ! autovideosink', cv2.CAP_GSTREAMER)
# # cap = cv2.VideoCapture('udpsrc uri=udp://192.168.1.105:3221  ! application/x-rtp,media=video,clock-rate=90000,encoding-name=H265,payload=96 ! rtph265depay ! decodebin ! appsink', cv2.CAP_GSTREAMER)
# print cap.isOpened()
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# while True:
#
#     ret, im = cap.read()
#
#     cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
#     cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#     cv2.imshow("Video", im)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

# ******************************************** Thread ************************************************* #

# import gi
# import cv2
# import time
# import numpy as np
# gi.require_version('Gst', '1.0')
# from gi.repository import Gst
# from threading import Thread
#
#
# class MyThread(Thread):
#
#     def __init__(self, url):
#         Thread.__init__(self)
#         Gst.init(None)
#         self.frame = None
#         self.stopped = False
#         self.pipeline = Gst.parse_launch(
#             "filesrc location=/home/id/TownCentreXVID.avi ! avidemux ! decodebin ! appsink name=sink")
#         self.appsink = self.pipeline.get_by_name('sink')
#
#     def run(self):
#         print('ss')
#         while True:
#             self.pipeline.set_state(Gst.State.PLAYING)
#             self.pipeline.seek_simple(Gst.Format.BUFFERS, Gst.SeekFlags.FLUSH, 1)
#             smp = self.appsink.emit('pull-preroll')
#             buf = smp.get_buffer()
#             self.pipeline.set_state(Gst.State.PAUSED)
#             self.frame = buf.extract_dup(0, buf.get_size())[:3110400]
#             # self.frame = np.fromstring(data, dtype='uint8').reshape((1620, 1920))
#             if self.stopped:
#                 self.pipeline.set_state(Gst.State.NULL)
#                 break
#
#     def stop(self):
#         self.stopped = True
#
#     def get_frame(self):
#         return self.frame
#
#
# t = MyThread('ss')
# t.start()
# # t.join()
#
# while True:
#     data = t.get_frame()
#     if data is not None:
#         frame = np.fromstring(data, dtype='uint8').reshape((1620, 1920))
#         frame_new = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
#         cv2.imshow("Video", frame_new)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             t.stop()
#             cv2.destroyAllWindows()
#             break

# ******************************************** Thread ************************************************* #

# import gi
# import cv2
# import time
# import numpy as np
# from multiprocessing import Queue, Process, Pipe
#
# gi.require_version('Gst', '1.0')
# from gi.repository import Gst


# def get_frame(q):
#     Gst.init(None)
#     # pipeline = Gst.parse_launch(
#     #     "filesrc location=/home/id/TownCentreXVID.avi ! avidemux ! decodebin ! appsink name=sink")
#     # udpsrc port = 3221 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, " \
#     # "encoding-name=(string)H264,sprop-parameter-sets=(string)\"Z0JAMpWgHgCJ+VA\\=\\,aM48gA\\=\\=\", payload=(int)96" ! rtph264depay ! decodebin ! autovideosink sync = false
#     print(r'"\\=')
#     pipeline = Gst.parse_launch(r'udpsrc port=3221 caps="application/x-rtp, media=(string)video, '
#                                 r'clock-rate=(int)90000, encoding-name=(string)H264, '
#                                 r'sprop-parameter-sets=(string)\"Z0JAMpWgHgCJ+VA\\=\\,aM48gA\\=\\=\", '
#                                 r'payload=(int)96\" ! rtph264depay ! decodebin ! appsink name=sink')
#
#     # pipeline = Gst.parse_launch("udpsrc uri=\"udp://192.168.1.120:3220\" caps=\"application/x-rtp, media=(string)video, "
#     #                             "clock-rate=(int)90000, encoding-name=(string)H265, sprop-parameter-sets=(string)1, "
#     #                             "payload=(int)96\" ! rtph265depay ! decodebin ! appsink name=sink")
#     appsink = pipeline.get_by_name('sink')
#     # print(q.full())
#     while True:
#         time.sleep(0.02)
#         if q.qsize() > 100:
#             q.get()
#         else:
#             pipeline.set_state(Gst.State.PLAYING)
#             pipeline.seek_simple(Gst.Format.BUFFERS, Gst.SeekFlags.FLUSH, 1)
#             smp = appsink.emit('pull-preroll')
#             buf = smp.get_buffer()
#             pipeline.set_state(Gst.State.PAUSED)
#             data = buf.extract_dup(0, buf.get_size())[:3110400]
#             frame = np.fromstring(data, dtype='uint8').reshape((1620, 1920))
#             frame_new = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
#             q.put(frame_new)
#             # q.send(frame_new)
#
#
# # (con1, con2) = Pipe()
# q = Queue()
# p = Process(target=get_frame, args=(q,))
# p.start()
# # time.sleep(5)
# # print(q.qsize())
# # print(q.get())
# start = time.clock()
# count = 0
# while True:
#     # print(time.clock() - start)
#     frame = q.get()
#     # frame = con2.recv()
#     # time.sleep(0.05)
#     # print(q.qsize())
#     if frame is not None:
#     # if time.clock() - start < 3:
#         # frame = np.fromstring(data, dtype='uint8').reshape((1620, 1920))
#         # frame_new = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
#         # count += 1
#         cv2.imshow("Video", frame)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             p.terminate()
#             cv2.destroyAllWindows()
#             break
#     else:
#         print('sd%d' % count)
#         p.terminate()
#         cv2.destroyAllWindows()
#         break
# p.terminate()

# ******************************************** OpenCV ************************************************* #

# import gi
# import cv2
# import time
# import numpy as np
#
# gi.require_version('Gst', '1.0')
# from gi.repository import Gst
#
#
# Gst.init(None)
# # pipeline = Gst.parse_launch("filesrc location=/home/id/TownCentreXVID.avi ! avidemux ! decodebin ! appsink name=sink")
#
# # udpsrc port = 3221 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, " \
# # "encoding-name=(string)H264,sprop-parameter-sets=(string)\"Z0JAMpWgHgCJ+VA\\=\\,aM48gA\\=\\=\", payload=(int)96" ! rtph264depay ! decodebin ! autovideosink sync = false
#
# # pipeline = Gst.parse_launch(r'udpsrc port=3221 caps="application/x-rtp, media=(string)video, '
# #                             r'clock-rate=(int)90000, encoding-name=(string)H264, '
# #                             r'sprop-parameter-sets=(string)\"Z0JAMpWgHgCJ+VA\\=\\,aM48gA\\=\\=\", '
# #                             r'payload=(int)96\" ! rtph264depay ! decodebin ! appsink name=sink')
#
# pipeline = Gst.parse_launch("udpsrc uri=\"udp://192.168.1.109:3220\" caps=\"application/x-rtp, media=(string)video, "
#                             "clock-rate=(int)90000, encoding-name=(string)H265, sprop-parameter-sets=(string)1, "
#                             "payload=(int)96\" ! rtph265depay ! decodebin ! appsink name=sink")
#
# appsink = pipeline.get_by_name('sink')
# appsink.set_property('emit-signals', True)
# appsink.set_property('drop', True)
# appsink.set_property('max-buffers', 1)
#
# pipeline.set_state(Gst.State.PLAYING)
# # smp = appsink.emit('pull-sample')
#
# while True:
#
#     # pipeline.seek_simple(Gst.Format.BUFFERS, Gst.SeekFlags.FLUSH, 1)
#     time.sleep(0.1)
#     smp = appsink.emit('pull-sample')
#     buf = smp.get_buffer()
#     # pipeline.set_state(Gst.State.PAUSED)
#     data = buf.extract_dup(0, buf.get_size())[:3110400]
#     frame = np.fromstring(data, dtype='uint8').reshape((1620, 1920))
#     frame_new = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
#
#     if frame is not None:
#         cv2.imshow("Video", frame_new)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             cv2.destroyAllWindows()
#             break



# Gst.init(None)
#
# #Build the pipeline
# pipeline = Gst.parse_launch("udpsrc uri=\"udp://192.168.1.120:3224\" caps=\"application/x-rtp, media=(string)video, "
#                             "clock-rate=(int)90000, encoding-name=(string)H265, sprop-parameter-sets=(string)1, "
#                             "payload=(int)96\" ! rtph265depay ! decodebin ! video/x-raw, format=RGB ! videoconvert ! appsink name=sink")
#
# pipeline = Gst.parse_launch("filesrc location=/home/ubuntu/TownCentreXVID.avi ! avidemux ! decodebin ! appsink name=sink")
#
# appsink = pipeline.get_by_name('sink')
# appsink.set_property("max-buffers", 1)
# appsink.set_property('emit-signals', True)
# appsink.set_property('sync', False)
#
# videoconvert = pipeline.get_by_name('convert')
# videoconvert.set_property('format', 'RGBA')
#
# while True:
#     pipeline.set_state(Gst.State.PLAYING)
#     pipeline.seek_simple(Gst.Format.BUFFERS, Gst.SeekFlags.FLUSH, 1)
#     print (pipeline.query_duration(Gst.Format.BUFFERS))
#     smp = appsink.emit('pull-preroll')
#     buf = smp.get_buffer()
#     pipeline.set_state(Gst.State.PAUSED)
#     data = buf.extract_dup(0, buf.get_size())[:3110400]
#     frame = np.fromstring(data, dtype='uint8').reshape((1620, 1920))
#
#     frame_new = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
#     cv2.imshow("Video", frame_new)
#
#     while True:
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             # cv2.destroyAllWindows()
#             break


# # Start playing
# pipeline.set_state(Gst.State.PLAYING)
#
# # Wait until error or EOS
# bus = pipeline.get_bus()
# msg = bus.timed_pop_filtered(
#     Gst.CLOCK_TIME_NONE, Gst.MessageType.ERROR | Gst.MessageType.EOS)
#
# # Free resources
# pipeline.set_state(Gst.State.NULL)

# import numpy
# import subprocess as sp
#
# command = "gst-launch-1.0 udpsrc uri=\"udp://192.168.1.113:3221\" caps=\"application/x-rtp, media=(string)video, " \
#           "clock-rate=(int)90000, encoding-name=(string)H265, sprop-parameter-sets=(string)1, " \
#           "payload=(int)96\" ! rtph265depay ! decodebin ! autovideosink sync=false"
# pipe = sp.Popen(command, shell=True, stdout=sp.PIPE, bufsize=10**8)
#
# while True:
#     # Capture frame-by-frame
#     raw_image = pipe.stdout.read(1920*1080*3)
#     # transform the byte read into a numpy array
#     image = numpy.fromstring(raw_image, dtype='uint8')
#     image = image.reshape((1920, 1080, 3))          # Notice how height is specified first and then width
#     if image is not None:
#         cv2.imshow('Video', image)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#     pipe.stdout.flush()
#
# cv2.destroyAllWindows()

# cap = cv2.VideoCapture('/dev/video0')
# print cap.isOpened()
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#
# while True:
#
#     ret, im = cap.read()
#
#     cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
#     cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#     cv2.imshow("Video", im)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

import numpy as np
a = np.array([[1, 2], [3, 4]])
b = np.mean(a, axis=0)
b = b.reshape(-1, 2)
c = np.linalg.norm(a[0]-a[1])
print(b)
print(c)
print(a.shape)
print(b.shape)

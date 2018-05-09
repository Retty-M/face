# coding=utf-8
"""Performs face detection in realtime.
Based on code from https://github.com/shanren7/real_time_face_recognition
"""
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

import sys
import time
import argparse

import gi
import cv2
import face
import object
from database import DB

import numpy as np
from multiprocessing import Queue, Process
from PIL import Image, ImageDraw, ImageFont

gi.require_version('Gst', '1.0')
from gi.repository import Gst


def add_overlays(frame, db, faces_T, faces_F):
    if faces_T is not None or faces_F is not None:

        cv2_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(cv2_frame)
        draw = ImageDraw.Draw(pil_frame)
        font = ImageFont.truetype('simhei', 20, encoding='utf-8')

        for face in faces_T:
            face_bb = face.bounding_box.astype(int)

            draw.rectangle((face_bb[0], face_bb[1], face_bb[2], face_bb[3]), outline=(0, 255, 0))
            if face.name is not None:
                info = db.fetchone(face.name)
                if info is not None:
                    str = u'%s %d%%\n%d\n%s\n%s\n%s' % (info[0], face.score*100, info[1], info[2], info[3], info[4])
                else:
                    str = u'%s %d%%' % (unicode(face.name, 'utf-8'), face.score*100)
                draw.text((face_bb[2]+2, face_bb[1]), str, (0, 255, 0), font=font)

        for face in faces_F:
            face_bb = face.bounding_box.astype(int)

            draw.rectangle((face_bb[0], face_bb[1], face_bb[2], face_bb[3]), outline=(255, 0, 0))
            draw.text((face_bb[0], face_bb[3]), u'未授权', (255, 0, 0), font=font)

        return cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)


def get_frame(q, source):
    Gst.init(None)
    command = "udpsrc uri=\"udp://%s\" caps=\"application/x-rtp, media=(string)video, " \
              "clock-rate=(int)90000, encoding-name=(string)H265, sprop-parameter-sets=(string)1, " \
              "payload=(int)96\" ! rtph265depay !  decodebin ! appsink name=sink" % source
    pipeline = Gst.parse_launch(command)
    # pipeline = Gst.parse_launch(r'udpsrc port=3221 caps="application/x-rtp, media=(string)video, '
    #                             r'clock-rate=(int)90000, encoding-name=(string)H264, '
    #                             r'sprop-parameter-sets=(string)\"Z0JAMpWgHgCJ+VA\\=\\,aM48gA\\=\\=\", '
    #                             r'payload=(int)96\" ! rtph264depay ! decodebin ! appsink name=sink')
    appsink = pipeline.get_by_name('sink')
    appsink.set_property('emit-signals', True)
    appsink.set_property('drop', True)
    appsink.set_property('max-buffers', 1)

    pipeline.set_state(Gst.State.PLAYING)

    while True:
        # time.sleep(0.05)
        if q.qsize() > 3:
            q.get()
        else:
            # pipeline.set_state(Gst.State.PLAYING)
            # pipeline.seek_simple(Gst.Format.BUFFERS, Gst.SeekFlags.FLUSH, 1)
            sample = appsink.emit('pull-sample')
            buf = sample.get_buffer()
            # pipeline.set_state(Gst.State.PAUSED)
            data = buf.extract_dup(0, buf.get_size())[:3110400]
            frame = np.fromstring(data, dtype='uint8').reshape((1620, 1920))
            frame_new = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
            q.put(frame_new)


def main(args):
    frame_interval = 3  # Number of frames after which to run face detection
    fps_display_interval = 5  # seconds
    frame_rate = 0
    frame_count = 0

    db = DB('./data/info')

    if args.debug:
        print("Debug enabled")
        face.debug = True

    if args.face or args.track:
        face_recognition = face.Recognition()
    if args.object or args.track:
        object_detection = object.Detection()
    start_time = time.time()

    if args.remote:
        # command = "gst-launch-1.0 udpsrc uri=\"udp://%s\" caps=\"application/x-rtp, media=(string)video, " \
        #           "clock-rate=(int)90000, encoding-name=(string)H265, sprop-parameter-sets=(string)1, " \
        #           "payload=(int)96\" ! rtph265depay !  decodebin ! fdsink" % args.source
        # command = "gst-launch-1.0 udpsrc uri=\"udp://%s\" caps = \"application/x-rtp, media=(string)video, " \
        #           "clock-rate=(int)90000, encoding-name=(string)H265,sprop-parameter-sets=(string)1, " \
        #           "payload=(int)96\" ! rtph265depay ! decodebin ! videoconvert ! " \
        #           "\"video/x-raw, format=(string)RGBA\" ! fdsink" % args.source
        # pipe = sp.Popen(command, shell=True, stdout=sp.PIPE)
        # Gst.init(None)
        # command = "udpsrc uri=\"udp://%s\" caps = \"application/x-rtp, media=(string)video, " \
        #           "clock-rate=(int)90000, encoding-name=(string)H265,sprop-parameter-sets=(string)1, " \
        #           "payload=(int)96\" ! rtph265depay ! decodebin ! appsink name=sink" % args.source
        # pipeline = Gst.parse_launch(command)
        # appsink = pipeline.get_by_name('sink')
        # appsink.set_property('emit-signals', True)
        # appsink.set_property('drop', True)
        # appsink.set_property('max-buffers', 1)
        #
        # pipeline.set_state(Gst.State.PLAYING)
        q = Queue()
        p = Process(target=get_frame, args=(q, args.source,))
        p.start()
    else:
        video_capture = cv2.VideoCapture('%s' % args.source)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        # Capture frame-by-frame
        faces_T = None
        faces_F = None
        if args.remote:
            # frame = get_frame(pipeline, appsink)
            frame = q.get()
            # cv2.imwrite('./audio/i.jpg', frame)
            # raw_image = pipe.stdout.read(3110400)
            # smp = appsink.emit('pull-sample')
            # buf = smp.get_buffer()
            # data = buf.extract_dup(0, buf.get_size())[:3110400]
            # frame = np.fromstring(data, dtype='uint8').reshape((1620, 1920))
            # frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
        else:
            ret, frame = video_capture.read()

        if args.object:
            object_detection.find_objects(frame)
        if args.track:
            object_detection.track_person(frame, face_recognition)
        if (frame_count % frame_interval) == 0:
            if args.face:
                faces_T, faces_F = face_recognition.identify(frame)

            # Check our current fps
            # end_time = time.time()
            # if (end_time - start_time) > fps_display_interval:
            #     frame_rate = int(frame_count / (end_time - start_time))
            #     start_time = time.time()
            #     frame_count = 0
        # print(len(faces_T), len(faces_F))
        frame_tmp = add_overlays(frame, db, faces_T, faces_F)
        if frame_tmp is not None:
            frame = frame_tmp

        # frame_count += 1
        # cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # if args.remote:
            # pipe.stdout.flush()
            # pipeline.set_state(Gst.State.PLAYING)

    # When everything is done, release the capture
    if args.remote:
        p.terminate()
    else:
        video_capture.release()
        cv2.destroyAllWindows()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true', help='Enable some debug outputs.')
    parser.add_argument('--object', action='store_true',  help='Enable detect objects.')
    parser.add_argument('--face', action='store_true', help='Enable detect faces.')
    parser.add_argument('--track', action='store_true', help='Enable visual tracker.')
    parser.add_argument('-r', '--remote', action='store_true', help='Get video from remote server')
    parser.add_argument('-s', '--source', type=str, default='/dev/video0', help='The video source, default: /dev/video1')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

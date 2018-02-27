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
import argparse
import sys
import time

import cv2
import face
import object

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def add_overlays(frame, faces, frame_rate):
    if faces is not None:

        cv2_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(cv2_frame)

        for face in faces:
            face_bb = face.bounding_box.astype(int)

            draw = ImageDraw.Draw(pil_frame)
            draw.rectangle((face_bb[0], face_bb[1], face_bb[2], face_bb[3]), outline=(0, 255, 0))

            # cv2.rectangle(frame,
            #               (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
            #               (0, 255, 0), 2)
            if face.name is not None:
                font = ImageFont.truetype('simhei', 20, encoding='utf-8')
                draw.text((face_bb[0], face_bb[3]),
                          u'%s %d%%' % (unicode(face.name, 'utf-8'), face.score*100),
                          (0, 255, 0), font=font)
                # cv2.putText(frame, '%s %d%%' % (face.name, face.score*100), (face_bb[0], face_bb[3]),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                #             thickness=2, lineType=2)
        return cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)

    # cv2.putText(frame, str(frame_rate) + " fps", (10, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
    #             thickness=2, lineType=2)


def main(args):
    frame_interval = 3  # Number of frames after which to run face detection
    fps_display_interval = 5  # seconds
    frame_rate = 0
    frame_count = 0

    # video_capture = cv2.VideoCapture(0)
    video_capture = cv2.VideoCapture('rtsp://192.168.1.110:8554/test')

    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if args.face or args.track:
        face_recognition = face.Recognition()
    if args.object or args.track:
        object_detection = object.Detection()
    start_time = time.time()

    if args.debug:
        print("Debug enabled")
        face.debug = True

    while True:
        # Capture frame-by-frame
        faces = None
        ret, frame = video_capture.read()

        if args.object:
            object_detection.find_objects(frame)
        if args.track:
            object_detection.track_person(frame, face_recognition)
        if (frame_count % frame_interval) == 0:
            if args.face:
                faces = face_recognition.identify(frame)

            # Check our current fps
            # end_time = time.time()
            # if (end_time - start_time) > fps_display_interval:
            #     frame_rate = int(frame_count / (end_time - start_time))
            #     start_time = time.time()
            #     frame_count = 0

        frame_tmp = add_overlays(frame, faces, frame_rate)
        if frame_tmp is not None:
            frame = frame_tmp

        # frame_count += 1
        cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true',
                        help='Enable some debug outputs.')
    parser.add_argument('--object', action='store_true',
                        help='Enable detect objects.')
    parser.add_argument('--face', action='store_true',
                        help='Enable detect faces.')
    parser.add_argument('--track', action='store_true',
                        help='Enable visual tracker.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

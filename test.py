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

# name = raw_input('>')
# print name

import os
import pyttsx

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

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

cap = cv2.VideoCapture(0)
while True:

    ret, im = cap.read()
    cv2_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)

    draw = ImageDraw.Draw(pil_im)
    font = ImageFont.truetype('simhei', 20, encoding='utf-8')
    draw.rectangle((100, 100, 300, 300), outline=(0, 255, 0))
    draw.text((90, 90), u"{}%".format(unicode('收到', 'utf-8')), (0, 0, 255), font=font)

    cv2_text_im = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
    cv2.imshow("Video", cv2_text_im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

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

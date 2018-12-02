import numpy as np  
import cv2  
from collections import deque
from PIL import Image
import tensorflow as tf
from config import *
import sys

data_names = ["train", "test"]
for dn in data_names:
    list_f = open('data_video/' + dn + '_list', 'rw+')
    temp = list_f.read()
    video_list = temp.split('\n')
    writer = tf.python_io.TFRecordWriter("data/" + dn + ".tfrecords")

    for video_name in video_list:
        if (video_name == ""):
            break
        stable_cap = cv2.VideoCapture('data_video/stable/' + video_name)  
        unstable_cap = cv2.VideoCapture('data_video/unstable/' + video_name)  
        unstable_frames = []
        stable_frames = []
        print('data_video/stable/' + video_name)
        for i in range(tot_ch):
            ret, frame = stable_cap.read()  
            stable_frames.append(cvt_img2train(frame, crop_rate))
            ret, frame = unstable_cap.read()
            unstable_frames.append(cvt_img2train(frame, 1))
        len = 0
        while(True):
            len += 1
            if (len % 10 == 0):
                print(len)
            x = stable_frames[0]
            for i in range(1, before_ch):
                x = np.concatenate((x, stable_frames[i]), axis=3)
            for i in range(before_ch, tot_ch):
                x = np.concatenate((x, unstable_frames[i]), axis=3)
            y = stable_frames[before_ch]
            x = x.flatten().tolist()
            y = y.flatten().tolist()
            example = tf.train.Example(features=tf.train.Features(feature={
                "x": tf.train.Feature(float_list=tf.train.FloatList(value=x)),
                "y": tf.train.Feature(float_list=tf.train.FloatList(value=y))
            }))
            writer.write(example.SerializeToString())


            ret, frame_stable = stable_cap.read()  
            ret, frame_unstable = unstable_cap.read()
            if (not ret):
                break
            stable_frames.append(cvt_img2train(frame_stable, crop_rate))
            unstable_frames.append(cvt_img2train(frame_unstable, 1))
            stable_frames.pop(0)
            unstable_frames.pop(0)
            '''
            stable_frames.append(cvt_img(frame_stable))
            frame_unstable = cvt_img(frame_unstable)
            example = tf.train.Example(features=tf.train.Features(feature={
                "unstable": tf.train.Feature(bytes_list=tf.train.BytesList(value=[frame_unstable.tobytes()])),
                "s0": tf.train.Feature(bytes_list=tf.train.BytesList(value=[stable_frames[0].tobytes()])),
                "s1": tf.train.Feature(bytes_list=tf.train.BytesList(value=[stable_frames[1].tobytes()])),
                "s2": tf.train.Feature(bytes_list=tf.train.BytesList(value=[stable_frames[2].tobytes()])),
                "s3": tf.train.Feature(bytes_list=tf.train.BytesList(value=[stable_frames[3].tobytes()])),
                "s4": tf.train.Feature(bytes_list=tf.train.BytesList(value=[stable_frames[4].tobytes()])),
            }))
            writer.write(example.SerializeToString())
            stable_frames.pop(0)
            '''
        stable_cap.release()  
        unstable_cap.release()  
    writer.close()

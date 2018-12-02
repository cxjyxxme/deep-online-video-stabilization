# pylint: disable=E1101
import numpy as np  
import cv2  
from collections import deque
from PIL import Image
import tensorflow as tf
from config import *
import sys
import struct
import feature_fetcher
import os
import re
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--start', type=str)
parser.add_argument('--start-file-num', type=int, default=0)
args = parser.parse_args()

#data_names = ["test", "train"]
data_names = ["train","test","valid"]
#data_names = ["temp"]
data_path = "data10/"
before_ch = 64
after_ch = 64
for dn in data_names:
    output_data_path = os.path.join(data_path + dn)
    if not os.path.exists(output_data_path):
        os.makedirs(output_data_path)
    print('Output data to ', output_data_path)
    list_f = open('data_video/' + dn + '_list_', 'r')
    temp = list_f.read()
    video_list = temp.split('\n')

    record_num = 0
    file_num = args.start_file_num
    file_list = str(file_num) + ".tfrecords"
    writer = tf.python_io.TFRecordWriter(data_path + dn + "/" + str(file_num) + ".tfrecords")
    
    start = 0
    if args.start is not None:
        for idx, video_name in enumerate(video_list):
            if video_name == args.start:
                start = idx
                break
    for idx in range(start, len(video_list)):
        video_name = video_list[idx]
        if (video_name == ""):
            break
        print('processing video '+video_name)
        #flowfile = open('data_video/flow/' + video_name[:-4] + '.bin', 'rb')
        #flowdata = flowfile.read()
        float_cnt = 4
        cnt = 2 * height * width * float_cnt * before_ch

        stable_cap = cv2.VideoCapture('data_video/stable/' + video_name)  
        unstable_cap = cv2.VideoCapture('data_video/unstable/' + video_name)  
        assert(os.path.isfile('data_video/stable/' + video_name))
        assert(os.path.isfile('data_video/unstable/' + video_name))

        video_len = stable_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        stable_cap.release()  
        for i in range(before_ch + 1, int(video_len - after_ch - 2)):
            flow = np.zeros((height, width, 2), dtype=np.float32)
            for xx in range(height):
                for yy in range(width):
                    #bit = float(struct.unpack('f', flowdata[cnt:cnt+float_cnt])[0])
                    bit = 0
                    cnt += float_cnt
                    flow[xx, yy, 0]=bit + yy
            flow[:, :, 0] = flow[:, :, 0] / width * 2 - 1
            #calc flow_y
            for xx in range(height):
                for yy in range(width):
                    #bit = float(struct.unpack('f', flowdata[cnt:cnt+float_cnt])[0])
                    bit = 0
                    cnt += float_cnt
                    flow[xx, yy, 1]=bit + xx
            flow[:, :, 1] = flow[:, :, 1] / height * 2 - 1

            stable_path = 'data_video/stable/' + video_name[:-4] + '/'
            unstable_path = 'data_video/unstable/' + video_name[:-4] + '/'

            flow = flow.flatten().tolist()

            example = tf.train.Example(features=tf.train.Features(feature={
                "stable_path": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(stable_path)])),
                "unstable_path": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(unstable_path)])),
                "pos":tf.train.Feature(int64_list = tf.train.Int64List(value=[i])),
                "flow": tf.train.Feature(float_list=tf.train.FloatList(value=flow)),
                "feature_matches1": tf.train.Feature(float_list=tf.train.FloatList(value=\
                    feature_fetcher.fetch(video_name, i - 1).flatten().tolist())),
                "feature_matches2": tf.train.Feature(float_list=tf.train.FloatList(value=\
                    feature_fetcher.fetch(video_name, i).flatten().tolist())),
            }))

            writer.write(example.SerializeToString())
            record_num += 1
            if (record_num == tfrecord_item_num):
                record_num = 0
                file_num += 1
                writer.close()
                file_list += " " + str(file_num) + ".tfrecords"
                writer = tf.python_io.TFRecordWriter(data_path + dn + "/" + str(file_num) + ".tfrecords")
                print('new record:', file_num)
    writer.close()
    file_object = open(data_path + dn + "/list.txt", 'w')
    file_object.write(file_list)
    file_object.close( )

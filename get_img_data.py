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
from os.path import isfile
parser = argparse.ArgumentParser()
parser.add_argument('--start', type=str)
parser.add_argument('--start-file-num', type=int, default=0)
args = parser.parse_args()

data_path = "data_video/"
data_names = ["stable", "unstable"]

for dn in data_names:
    video_data_path = os.path.join(data_path + dn)
    videos = os.listdir(video_data_path)

    for video_name in videos:
        if not isfile(os.path.join(video_data_path, video_name)):
            continue
        output_data_path = os.path.join(video_data_path, video_name[:-4])
        if not os.path.exists(output_data_path):
            os.makedirs(output_data_path)
        print(output_data_path)
        video_path = os.path.join(video_data_path, video_name)
        
        cap = cv2.VideoCapture(video_path)
        i = 0
        while (True):
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(os.path.join(output_data_path, str(i) + '.jpg'), frame)
            i += 1
        cap.release()  

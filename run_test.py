import tensorflow as tf
import numpy as np
from config import *
from PIL import Image
import cv2
import time
import os
import traceback
import math
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument('--GPU_id')
args = parser.parse_args()

file_object = open('spatial-transformer-tensorflow/configs/run_list.txt')
configs = file_object.read( ).split('\n')
file_object.close()

file_object2 = open('spatial-transformer-tensorflow/config.py')
config_file_txt = file_object2.read()
temp = config_file_txt.find('\n')
out_txt = config_file_txt[temp:]
file_object2.close()
#cases = ["Regular", "watermark", "GaussianNoise", "MultipleExposure", "NightScene", "QuickRotation", "Zooming", "Parallax", "Running", "Crowd"]
cases = ["Night2"]
#cases = ["Regular", "Crowd"]

n = int(configs[0])
for i in range(n):
    for case in cases:
        now = configs[i + 1].split(' ')
        now_id = int(now[0][3:])
        print(now_id)
        temp_out_txt = 'from configs.' + now[0] + ' import *' + out_txt
        file_object = open('spatial-transformer-tensorflow/config.py', 'w')
        file_object.write(temp_out_txt)
        file_object.close( )

        if (now_id >= 65):
            "+case+"
            cmd1 = 'CUDA_VISIBLE_DEVICES="' + args.GPU_id + '" python -u spatial-transformer-tensorflow/deploy_bundle.py --model-dir ./models/' + now[0] + "/ --model-name model-" + now[1] + " --before-ch 32 --gpu_memory_fraction 0.7 --output-dir ./dataset/" + now[0] + "-" + now[1] + "/"+case+"  --test-list /home/ubuntu/"+case+"/"+case+"/list.txt --indices 1 2 4 8 16 32 --prefix /home/ubuntu/"+case+"/"+case+";"
        else:
            cmd1 = 'CUDA_VISIBLE_DEVICES="' + args.GPU_id + '" python -u spatial-transformer-tensorflow/deploy_bundle.py --model-dir ./models/' + now[0] + "/ --model-name model-" + now[1] + " --before-ch 32 --gpu_memory_fraction 0.7 --output-dir ./dataset/" + now[0] + "-" + now[1] + "/"+case+"  --test-list /home/ubuntu/"+case+"/"+case+"/list.txt --indices 1 2 4 8 16 32 --no_bm 0 --prefix /home/ubuntu/"+case+"/"+case+";"
 
        print(cmd1)
        os.system(cmd1)

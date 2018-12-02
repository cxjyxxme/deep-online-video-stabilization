import os
import scipy.io
import re
import numpy as np

BaseDir = './data9/'
regexp = re.compile(r'(\d+)(\.mp4)?\.avi')

ori_width, ori_height = 1280, 720

def fetch(video_name, frame_id):
    video_name = regexp.match(video_name).group(1)
    path = os.path.join(BaseDir, video_name, '{:04d}'.format(frame_id))
    mat = scipy.io.loadmat(path)
    print('Read {}. Shape={}'.format(path, mat['res'].shape))
    return (mat['res'].astype(np.float32) / [ori_width, ori_height, ori_width, ori_height] - 0.5) * 2\
            if mat['res'].shape != (0,0) else mat['res'].astype(np.float32)

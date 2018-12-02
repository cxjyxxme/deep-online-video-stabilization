import tensorflow as tf
import numpy as np
from config import *
from PIL import Image
import cv2
import time
import os
import traceback
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model-dir')
parser.add_argument('--model-name')
parser.add_argument('--before-ch', type=int)
parser.add_argument('--after-ch', type=int)
parser.add_argument('--output-dir', default='data_video_local')
parser.add_argument('--infer-with-stable', action='store_true')
parser.add_argument('--infer-with-last', action='store_true')
parser.add_argument('--test-list', nargs='+', default=['data_video/test_list', 'data_video/train_list_deploy'])
#parser.add_argument('--train-list', default='data_video/train_list_deploy')
parser.add_argument('--prefix', default='data_video')
parser.add_argument('--max-span', type=int, default=1)
parser.add_argument('--random-black', type=int, default=None)
args = parser.parse_args()

MaxSpan = args.max_span

start_with_stable = True

sess = tf.Session()

model_dir = args.model_dir#'models/vbeta-1.1.0/'
model_name = args.model_name#'model-5000'
before_ch = args.before_ch
after_ch = args.after_ch
new_saver = tf.train.import_meta_graph(model_dir + model_name + '.meta')
new_saver.restore(sess, model_dir + model_name)
graph = tf.get_default_graph()
x_tensor = graph.get_tensor_by_name('stable_net/input/x_tensor:0')
#output = graph.get_tensor_by_name('stable_net/SpatialTransformer/_transform/Reshape_7:0')
#black_pix = graph.get_tensor_by_name('stable_net/SpatialTransformer/_transform/Reshape_6:0')
output = graph.get_tensor_by_name('stable_net/inference/SpatialTransformer/_transform/Reshape_7:0')
black_pix = graph.get_tensor_by_name('stable_net/inference/SpatialTransformer/_transform/Reshape_6:0')
#black_pix = graph.get_tensor_by_name('stable_net/img_loss/StopGradient:0')

#list_f = open('data_video/test_list_deploy', 'r')

video_list = []

for list_path in args.test_list:
    if os.path.isfile(list_path):
        print('adding '+list_path)
        list_f = open(list_path, 'r')
        temp = list_f.read()
        video_list.extend(temp.split('\n'))

def make_dirs(path):
    if not os.path.exists(path): os.makedirs(path)

cvt_train2img = lambda x: ((np.reshape(x, (height, width)) + 0.5) * 255).astype(np.uint8)

def draw_imgs(net_output, stable_frame, unstable_frame, inputs):
    cvt2int32 = lambda x: x.astype(np.int32)
    assert(net_output.ndim == 2)
    assert(stable_frame.ndim == 2)
    assert(unstable_frame.ndim == 2)

    net_output = cvt2int32(net_output)
    stable_frame = cvt2int32(stable_frame)
    unstable_frame = cvt2int32(unstable_frame)
    last_frame = cvt2int32(cvt_train2img(inputs[..., before_ch - 1]))
    output_minus_input  = abs(net_output - unstable_frame)
    output_minus_stable = abs(net_output - stable_frame)
    output_minus_last   = abs(net_output - last_frame)
    img_top    = np.concatenate([net_output,         output_minus_stable], axis=1)
    img_bottom = np.concatenate([output_minus_input, output_minus_last], axis=1)
    img = np.concatenate([img_top, img_bottom], axis=0).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

def getNext(delta, bound, speed = 5):
    tmp = delta + speed
    if tmp >= bound or tmp < 0: speed *= -1
    return delta + speed, speed
    # return np.random.randint(0, bound), 5

production_dir = os.path.join(args.output_dir, 'output')
visual_dir = os.path.join(args.output_dir, 'output-vis')
make_dirs(production_dir)
make_dirs(visual_dir)

for video_name in video_list:
    if (video_name == ""):
        continue
    print(video_name)
    unstable_cap = cv2.VideoCapture(os.path.join(args.prefix,'unstable', video_name))
    fps = unstable_cap.get(cv2.CAP_PROP_FPS)
    print(os.path.join(args.prefix,'unstable', video_name))
    videoWriter = cv2.VideoWriter(os.path.join(production_dir, video_name), 
            cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width, height))
    videoWriterVis = cv2.VideoWriter(os.path.join(visual_dir, video_name), 
            cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width * 2, height * 2))
    before_frames = []
    after_frames = []
    print(video_name)
    if (not start_with_stable):
        ret, frame = unstable_cap.read()
        for i in range(before_ch):
            before_frames.append(cvt_img2train(frame, crop_rate))
        for i in range(after_ch + 1):
            ret, frame = unstable_cap.read()
            after_frames.append(cvt_img2train(frame, 1))

        temp = before_frames[0]
        temp = ((np.reshape(temp, (height, width)) + 0.5) * 255).astype(np.uint8)
        videoWriter.write(cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR))
    else:
        stable_cap = cv2.VideoCapture(os.path.join(args.prefix,'stable', video_name)) 
        for i in range(before_ch):
            ret, frame = unstable_cap.read()
            ret, frame = stable_cap.read()
            before_frames.append(cvt_img2train(frame, crop_rate))
             
            temp = before_frames[i]
            temp = ((np.reshape(temp, (height, width)) + 0.5) * 255).astype(np.uint8)
            videoWriter.write(cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR))
            temp = np.concatenate([temp, np.zeros_like(temp)], axis=1)
            temp = np.concatenate([temp, np.zeros_like(temp)], axis=0)
            videoWriterVis.write(cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR))
        for i in range(after_ch + 1):
            ret, frame = unstable_cap.read()
            after_frames.append(cvt_img2train(frame, 1))
    length = 0
    in_xs = []
    delta = 0
    speed = args.random_black
    try:
        while(True):
            _, stable_cap_frame = stable_cap.read()
            stable_train_frame = cvt_img2train(stable_cap_frame, crop_rate)
            if args.random_black is not None:
                delta, speed = getNext(delta, 50, speed)
                print(delta, speed)
                stable_train_frame[:, :, delta:width, ...] = stable_train_frame[:, :, 0:width-delta, ...]
                stable_train_frame[:, :, :delta, ...] = -1
            stable_frame = cvt_train2img(stable_train_frame)
            unstable_frame = cvt_train2img(after_frames[0])
            in_x = before_frames[0]
            for i in range(1, before_ch):
                in_x = np.concatenate((in_x, before_frames[i]), axis = 3)
            for i in range(after_ch + 1):
                in_x = np.concatenate((in_x, after_frames[i]), axis = 3)
            '''
            in_x_t = in_x
            for i in range(batch_size - 1):
                in_x_t = np.concatenate((in_x_t, in_x), axis = 0)
            '''
            # for max span
            if MaxSpan != 1:
                in_xs.append(in_x)
                if len(in_xs) > MaxSpan: 
                    in_xs = in_xs[-1:]
                    print('cut')
                in_x = in_xs[0].copy()
                in_x[0, ..., before_ch] = after_frames[0][..., 0]

            img, black = sess.run([output, black_pix], feed_dict={x_tensor:in_x})
            black = black[0, :, :]
            img = img[0, :, :, :].reshape(height, width)
            frame = img + black * (-1)
            frame = frame.reshape(1, height, width, 1)
            img = ((np.reshape(img, (height, width)) + 0.5) * 255).astype(np.uint8)
            net_output = img
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
            videoWriter.write(img)
            videoWriterVis.write(draw_imgs(net_output, stable_frame, unstable_frame, in_x))

            ret, frame_unstable = unstable_cap.read() 
            if (not ret):
                break
            length = length + 1
            if (length % 10 == 0):
                print("length: " + str(length))       
            if args.infer_with_stable:
                before_frames.append(stable_train_frame)
            else:
                before_frames.append(frame)
            if args.infer_with_last:
                for i in range(len(before_frames)):
                    before_frames[i] = before_frames[-1]
            before_frames.pop(0)
            after_frames.append(cvt_img2train(frame_unstable, 1))
            after_frames.pop(0)

            #if (len == 100):
            #    break
    except Exception as e:
        traceback.print_exc()
    finally:
        videoWriter.release()
        unstable_cap.release()

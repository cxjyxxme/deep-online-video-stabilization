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
parser.add_argument('--output-dir', default='data_video_local/outputs/')
parser.add_argument('--input', default='/Users/lazycal/workspace/lab/3.1/qudou/frames/stable/6/image-0001.jpg')
parser.add_argument('--indices', nargs='+', type=int, required=True)
# parser.add_argument('--infer-with-stable', action='store_true')
# parser.add_argument('--infer-with-last', action='store_true')
# parser.add_argument('--test-list', nargs='+', default=['data_video/test_list', 'data_video/train_list_deploy'])
# #parser.add_argument('--train-list', default='data_video/train_list_deploy')
# parser.add_argument('--prefix', default='data_video')
# parser.add_argument('--max-span', type=int, default=1)
# parser.add_argument('--random-black', type=int, default=5)
args = parser.parse_args()

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

def make_dirs(path):
    if not os.path.exists(path): os.makedirs(path)

cvt_train2img = lambda x: ((np.reshape(x, (height, width)) + 0.5) * 255).astype(np.uint8)
cvt_train2img3 = lambda x: cv2.cvtColor(cvt_train2img(x), cv2.COLOR_GRAY2BGR)
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

def forward(in_x):
    img, black = sess.run([output, black_pix], feed_dict={x_tensor:in_x})
    black = black[0, :, :]
    img = img[0, :, :, :].reshape(height, width)
    return img, black

def applyDelta(frame, delta):
    frame = frame.copy()
    if delta >= 0:
        frame[:, :, delta:width, ...] = frame[:, :, 0:width-delta, ...]
        frame[:, :, :delta, ...] = -1
    else:
        delta *= -1
        frame[:, :, 0:width-delta, ...] = frame[:, :, delta:width, ...]
        frame[:, :, width-delta:, ...] = -1

    return frame

def getDelta(i, step):
    return i * step
    #return (before_ch - 1 - i) * step

production_dir = os.path.join(args.output_dir)
make_dirs(production_dir)

img = cv2.imread(os.path.join(args.input))
img = cvt_img2train(img)



for step in [0, 2, 4, 6, 8]:
    print('step={}'.format(step))
    in_x_list = []
    fps = 60
    fourcc = cv2.cv.CV_FOURCC(*'MJPG') if cv2.__version__.startswith('2') else cv2.VideoWriter_fourcc('M','J','P','G')
    videoWriter = cv2.VideoWriter(os.path.join(production_dir, "%d.avi"%step), 
            fourcc, fps, (width, height))
    for i in range(before_ch - 1, -1, -1):
        delta = getDelta(args.indices[i] - 7, step)
        frame = applyDelta(img, delta)
        in_x_list.append(frame)
        videoWriter.write(cvt_train2img3(frame))
    for i in range(after_ch):
        delta = getDelta(i + before_ch + 1, step)
        frame = applyDelta(img, delta)
        in_x_list.append(img)
    delta = getDelta(0, step)
    frame = applyDelta(img, delta)
    videoWriter.write(cvt_train2img3(frame))
    in_x_list.append(frame)
    in_x = np.concatenate(in_x_list, axis=3)
    res, _black = forward(in_x)
    res = cvt_train2img3(res)

    gt = cvt_train2img3(applyDelta(img, getDelta(before_ch, step)))
    output_minus_gt = abs(res.astype(np.int32) - gt.astype(np.int32)).astype(np.uint8)
    cv2.imwrite(os.path.join(args.output_dir, 'diff-%d-%d.jpg'%(step,-1)), output_minus_gt)

    for j in range(0, before_ch):
        gt = cvt_train2img3(in_x_list[before_ch - j])
        output_minus_gt = abs(res.astype(np.int32) - gt.astype(np.int32)).astype(np.uint8)
        cv2.imwrite(os.path.join(args.output_dir, 'diff-%d-%d.jpg'%(step,j)), output_minus_gt)
    
    cv2.imwrite(os.path.join(args.output_dir, 'output-%d.jpg'%step), res)
    videoWriter.write(res)
    for i in range(after_ch):
        delta = getDelta(i + before_ch + 1, step)
        frame = applyDelta(img, delta)
        videoWriter.write(cvt_train2img3(frame))
    videoWriter.release()
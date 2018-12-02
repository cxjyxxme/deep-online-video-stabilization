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
#parser.add_argument('--after-ch', type=int)
parser.add_argument('--output-dir', default='data_video_local')
# parser.add_argument('--infer-with-stable', action='store_true')
# parser.add_argument('--infer-with-last', action='store_true')
parser.add_argument('--test-list', nargs='+', default=['data_video/test_list', 'data_video/train_list_deploy'])
#parser.add_argument('--train-list', default='data_video/train_list_deploy')
parser.add_argument('--prefix', default='data_video')
# parser.add_argument('--max-span', type=int, default=1)
# parser.add_argument('--random-black', type=int, default=None)
parser.add_argument('--indices', type=int, nargs='+', required=True)
parser.add_argument('--start-with-stable', action='store_true')
# parser.add_argument('--refine', type=int, default=1)
parser.add_argument('--gpu_memory_fraction', type=float, default=0.9)
# parser.add_argument('--deploy-vis', action='store_true')
args = parser.parse_args()


sess = tf.Session(config=tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)))

model_dir = args.model_dir#'models/vbeta-1.1.0/'
model_name = args.model_name#'model-5000'
before_ch = args.before_ch
#after_ch = args.after_ch
#after_ch = 0
new_saver = tf.train.import_meta_graph(model_dir + model_name + '.meta')
new_saver.restore(sess, model_dir + model_name)
graph = tf.get_default_graph()
x_tensor = graph.get_tensor_by_name('stable_net/input/x_tensor:0')
#output = graph.get_tensor_by_name('stable_net/SpatialTransformer/_transform/Reshape_7:0')
#black_pix = graph.get_tensor_by_name('stable_net/SpatialTransformer/_transform/Reshape_6:0')
output = graph.get_tensor_by_name('stable_net/inference/SpatialTransformer/_transform/Reshape_7:0')
black_pix = graph.get_tensor_by_name('stable_net/inference/SpatialTransformer/_transform/Reshape_6:0')
theta_mat_tensor = graph.get_tensor_by_name('stable_net/feature_loss/Reshape:0')
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

# def draw_imgs(net_output, stable_frame, unstable_frame, inputs):
#     cvt2int32 = lambda x: x.astype(np.int32)
#     assert(net_output.ndim == 2)
#     assert(stable_frame.ndim == 2)
#     assert(unstable_frame.ndim == 2)

#     net_output = cvt2int32(net_output)
#     stable_frame = cvt2int32(stable_frame)
#     unstable_frame = cvt2int32(unstable_frame)
#     last_frame = cvt2int32(cvt_train2img(inputs[..., 0]))
#     output_minus_input  = abs(net_output - unstable_frame)
#     output_minus_stable = abs(net_output - stable_frame)
#     output_minus_last   = abs(net_output - last_frame)
#     img_top    = np.concatenate([net_output,         output_minus_stable], axis=1)
#     img_bottom = np.concatenate([output_minus_input, output_minus_last], axis=1)
#     img = np.concatenate([img_top, img_bottom], axis=0).astype(np.uint8)
#     return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# def getNext(delta, bound, speed = 5):
#     tmp = delta + speed
#     if tmp >= bound or tmp < 0: speed *= -1
#     return delta + speed, speed
#     # return np.random.randint(0, bound), 5

# def cvt_theta_mat(theta_mat):
#     # theta_mat * x = x'
#     # ret * scale_mat * x = scale_mat * x'
#     # ret = scale_mat * theta_mat * scale_mat^-1
#     scale_mat = np.eye(3)
#     scale_mat[0, 0] = width / 2.
#     scale_mat[0, 2] = width / 2.
#     scale_mat[1, 1] = height / 2.
#     scale_mat[1, 2] = height / 2.
#     assert(theta_mat.shape == (3, 3))
#     from numpy.linalg import inv
#     return np.matmul(np.matmul(scale_mat, theta_mat), inv(scale_mat))

def warpRev(img, theta):
    assert(img.ndim == 3)
    assert(img.shape[-1] == 3)
    theta_mat_cvt = cvt_theta_mat(theta)
    return cv2.warpPerspective(img, theta_mat_cvt, dsize=(width, height), flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR)

production_dir = os.path.join(args.output_dir, 'output')
visual_dir = os.path.join(args.output_dir, 'output-vis')
make_dirs(production_dir)
make_dirs(visual_dir)

print('inference with {}'.format(args.indices))
for video_name in video_list:
    start = time.time()
    if (video_name == ""):
        continue
    print(video_name)
    unstable_cap = cv2.VideoCapture(os.path.join(args.prefix,'unstable', video_name))
    fps = unstable_cap.get(cv2.CAP_PROP_FPS)
    videoWriter = cv2.VideoWriter(os.path.join(production_dir, video_name), 
            cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width, height))
    before_frames = []
    after_frames = []
    ret, frame = unstable_cap.read()
    videoWriter.write(cv2.resize(frame, (width, height)))
    for i in range(before_ch):
        before_frames.append(cvt_img2train(frame, crop_rate))
        temp = before_frames[i]
        temp = ((np.reshape(temp, (height, width)) + 0.5) * 255).astype(np.uint8)
        #videoWriter.write(cv2.resize(stable_cap_frame, (width, height)))
        temp = np.concatenate([temp, np.zeros_like(temp)], axis=1)
        temp = np.concatenate([temp, np.zeros_like(temp)], axis=0)
    # for i in range(after_ch + 1):
    ret, frame = unstable_cap.read()
    frame_unstable = frame
    after_frames.append(cvt_img2train(frame, 1))

    length = 0
    in_xs = []
    try:
        s = [0, 0, 0, 0, 0]
        t = [0, 0, 0, 0]
        while(True):
            s[0] = time.time()
            unstable_frame = cvt_train2img(after_frames[0])
            in_x = []
            assert(len(after_frames) == 1)
            for i in args.indices:
                in_x.append(before_frames[-i])
            in_x.append(after_frames[0])
            in_x = np.concatenate(in_x, axis = 3)
            s[1] = time.time()
            img, black = sess.run([output, black_pix], feed_dict={x_tensor:in_x})
            s[2] = time.time()
            black = black[0, :, :]
            img = img[0, :, :, :].reshape(height, width)
            frame = img + black * (-1)
            frame = frame.reshape(1, height, width, 1)
            img = ((np.reshape(img, (height, width)) + 0.5) * 255).astype(np.uint8)
            
            net_output = img
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
            s[3] = time.time()
            videoWriter.write(img)
            ret, frame_unstable = unstable_cap.read() 
            if (not ret):
                break
            length = length + 1
            if (length % 10 == 0):
                print("length: " + str(length))       
            before_frames.append(frame)
            before_frames.pop(0)
            after_frames.append(cvt_img2train(frame_unstable, 1))
            after_frames.pop(0)
            s[4] = time.time()
            for i in range(4):
                t[i] += s[i + 1] - s[i]

    except Exception as e:
        traceback.print_exc()
    finally:
        print('total length={}'.format(length + 2))
        print('fps={}'.format(length / (time.time() - start)))
        print('fps1={}'.format(length / (t[1])))
        print('time={}'.format(t))
        videoWriter.release()
        unstable_cap.release()

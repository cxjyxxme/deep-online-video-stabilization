# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
import tensorflow as tf
from spatial_transformer import *
import numpy as np
from tf_utils import weight_variable, bias_variable, dense_to_one_hot
import cv2
from resnet import *
#import get_data_flow
import get_data_mini as get_data_flow
from config import *
import time
#import s_net
import s_net_bundle as s_net
from tensorflow.python.client import timeline
import utils
from collections import namedtuple
import argparse
import os
import random
slim = tf.contrib.slim

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_memory_fraction', type=float, default=0.95)
parser.add_argument('--restore', action='store_true')
args = parser.parse_args()
cnt = 0

logger = utils.get_logger()
def save_warpped_features(input_data, stable_warpped_pts_batch, theta_mat, output, name):
    output_prefix = os.path.join(log_dir, name)
    if not os.path.exists(output_prefix):
        os.makedirs(output_prefix)
    def draw(img, pts, mask, color=None):
        res = img.copy()
        assert(pts.shape[0] == max_matches)
        assert(mask.shape[0] == max_matches)
        pts = (pts / 2 + .5) * img.shape[1::-1]
        #logger.info('pts={}'.format(pts))
        pts = pts.astype(np.int32)
        for i in range(pts.shape[0]):
            if not mask[i]: continue
            cv2.circle(res, tuple(pts[i]), 10, tuple(np.random.rand(3) * 255) if color is None else color)
        return res
    def cvt_theta_mat(theta_mat):
        # theta_mat * x = x'
        # ret * scale_mat * x = scale_mat * x'
        # ret = scale_mat * theta_mat * scale_mat^-1
        scale_mat = np.eye(3)
        scale_mat[0, 0] = width / 2.
        scale_mat[0, 2] = width / 2.
        scale_mat[1, 1] = height / 2.
        scale_mat[1, 2] = height / 2.
        assert(theta_mat.shape == (3, 3))
        from numpy.linalg import inv
        return np.matmul(np.matmul(scale_mat, theta_mat), inv(scale_mat))
    cvt = lambda x: (np.tile(x, [1, 1, 3]) + .5) * 255
    matches = input_data.feature_matches1
    stable = input_data.y1
    mask = input_data.mask1
    for i in range(matches.shape[0]):
        global cnt
        sta = cvt(stable[i])
        out = cvt(output[i])
        error = abs(sta - out)
        unstable = cvt(input_data.x1[i, :, :, before_ch, None])
        theta_mat_cvt = cvt_theta_mat(theta_mat[i])
        np.savetxt(os.path.join(output_prefix, 'theta-%04d.txt'%cnt), theta_mat_cvt)

        img = draw(sta, matches[i, :, :2], mask[i])
        unstable = draw(unstable, stable_warpped_pts_batch[i], mask[i], (255, 0, 0))
        unstable = draw(unstable, input_data.feature_matches1[i, :, 2:], mask[i], (0, 255, 0))
        res = np.concatenate([img, out], axis=1)
        res1 = np.concatenate([error, unstable], axis=1)
        res = np.concatenate([res, res1], axis=0)
        cv2.imwrite(os.path.join(output_prefix, 'img-%04d.jpg'%cnt), res)
        cnt = (cnt + 1) % 20
    x1 = input_data.x1
    x2 = input_data.x2
    for i in range(x1.shape[3]):
        cv2.imwrite(os.path.join(output_prefix, 'x1-%d.jpg'%i), cvt(x1[0, ..., i, None]))
    for i in range(x2.shape[3]):
        cv2.imwrite(os.path.join(output_prefix, 'x2-%d.jpg'%i), cvt(x2[0, ..., i, None]))

def show_image(name, img, min_v = 0, max_v = 1):
    #img_ = tf.pad(img, [[0, 0], [1, 1], [1, 1], [0, 0]], constant_values = max_v)
    #img_ = tf.pad(img_, [[0, 0], [1, 1], [1, 1], [0, 0]], constant_values = min_v)
    tf.summary.image(name, img)

def name_in_checkpoint(var):
    return var.op.name[18:]

def rand_crop():
    return random.random() * (1 - max_crop_rate) + max_crop_rate

ret1 = s_net.inference_stable_net(False)
ret2 = s_net.inference_stable_net(True)

with tf.name_scope('data_flow'):
    flow = tf.placeholder(tf.float32, [None, height, width, 2])
    x_flow = tf.slice(flow, [0, 0, 0, 0], [-1, -1, -1, 1])
    y_flow = tf.slice(flow, [0, 0, 0, 1], [-1, -1, -1, 1])

with tf.name_scope('temp_loss'):
    use_temp_loss = tf.placeholder(tf.float32)
    output2_aft_flow = interpolate(ret2['output'], x_flow, y_flow, (height, width))
    noblack_pix2_aft_flow = interpolate(1 - ret2['black_pix'], x_flow, y_flow, (height, width))
    #output2_aft_flow = ret2['output']#28
    temp_err = ret1['output'] - output2_aft_flow
    noblack = (1 - ret1['black_pix']) * noblack_pix2_aft_flow
    temp_err = temp_err * noblack
    show_image('err_temp', temp_err * temp_err)
    temp_loss = tf.reduce_sum(tf.reduce_sum(temp_err * temp_err, [1, 2, 3]) / 
            (tf.reduce_sum(noblack, [1, 2, 3]) + 1e-8), [0]) / batch_size * use_temp_loss
    #temp_loss = tf.nn.l2_loss(temp_err) / batch_size * use_temp_loss

with tf.name_scope('errors'):
    show_image('error_temp', tf.abs(ret1['output'] - output2_aft_flow))
    show_image('error_1', ret1['error'])
    show_image('error_2', ret2['error'])

with tf.name_scope('test_flow'):
    warped_y2 = interpolate(ret2['y'], x_flow, y_flow, (height, width))
    show_image('error_black_wy2', tf.abs(ret1['y'] - warped_y2))
    show_image('error_black_nowarp', tf.abs(ret2['y'] - ret1['y']))

loss_displayer = tf.placeholder(tf.float32)
with tf.name_scope('test_loss'):
    tf.summary.scalar('test_loss', loss_displayer, collections=['test'])

total_loss = ret1['total_loss'] + ret2['total_loss'] + temp_loss * temp_mul
with tf.name_scope('train_loss'):
    tf.summary.scalar('black_loss', ret1['black_loss'] + ret2['black_loss'])
    tf.summary.scalar('theta_loss', ret1['theta_loss'] + ret2['theta_loss'])
    tf.summary.scalar('grid_theta_loss', ret1['grid_theta_loss'] + ret2['grid_theta_loss'])
    tf.summary.scalar('img_loss', ret1['img_loss'] + ret2['img_loss'])
    tf.summary.scalar('regu_loss', ret1['regu_loss'] + ret2['regu_loss'])
    tf.summary.scalar('temp_loss', temp_loss * temp_mul)
    tf.summary.scalar('feature_loss', (ret1['feature_loss'] + ret2['feature_loss']))
    tf.summary.scalar('distortion_loss', (ret1['distortion_loss'] + ret2['distortion_loss']))
    tf.summary.scalar('consistency_loss', (ret1['consistency_loss'] + ret2['consistency_loss']))
    tf.summary.scalar('total_loss', total_loss)

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                           global_step=global_step,
                                           decay_steps=step_size,decay_rate=0.1, staircase=True)
opt = tf.train.AdamOptimizer(learning_rate)
optimizer = opt.minimize(total_loss, global_step=global_step)


with tf.name_scope('datas'):
    data_x1, data_y1, data_x2, data_y2, data_flow, \
        data_feature_matches1, data_mask1, data_feature_matches2, data_mask2 = get_data_flow.read_and_decode(
            data_dir + "train/", int(training_iter * batch_size / train_data_size) + 2)
    test_x1, test_y1, test_x2, test_y2, test_flow, \
        test_feature_matches1, test_mask1, test_feature_matches2, test_mask2 = get_data_flow.read_and_decode(
            data_dir + "test/", int(training_iter * batch_size * test_batches / test_data_size / test_freq) + 2)

    x1_batch, y1_batch, x2_batch, y2_batch, flow_batch,\
        feature_matches1_batch, mask1_batch, feature_matches2_batch, mask2_batch = tf.train.shuffle_batch(
                                                [data_x1, data_y1, data_x2, data_y2, data_flow,
                                                data_feature_matches1, data_mask1, data_feature_matches2, data_mask2],
                                                batch_size=batch_size, capacity=120,
                                                min_after_dequeue=80, num_threads=10)
    test_x1_batch, test_y1_batch, test_x2_batch, test_y2_batch, test_flow_batch,\
        test_feature_matches1_batch, test_mask1_batch, test_feature_matches2_batch, test_mask2_batch = tf.train.shuffle_batch(
                                                [test_x1, test_y1, test_x2, test_y2, test_flow,
                                                test_feature_matches1, test_mask1, test_feature_matches2, test_mask2],
                                                batch_size=batch_size, capacity=120,
                                                min_after_dequeue=80, num_threads=10)

checkpoint_file = 'data_video/resnet_v2_50.ckpt'
vtr = slim.get_variables_to_restore(exclude=['stable_net/resnet/resnet_v2_50/conv1', 'stable_net/resnet/fc'])
vtr = [v for v in vtr if ((not (('Adam' in v.op.name) or ('gen_theta' in v.op.name))) and (len(v.op.name) > 18))]
vtr = {name_in_checkpoint(var):var for var in vtr}
#print (vtr)
#variables_to_restore = slim.get_model_variables()
#variables_to_restore = {name_in_checkpoint(var):var for var in variables_to_restore}
restorer = tf.train.Saver(vtr)

merged = tf.summary.merge_all()
test_merged = tf.summary.merge_all("test")
saver = tf.train.Saver()
#init_all = tf.initialize_all_variables()
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()
sv = tf.train.Supervisor(logdir=log_dir, save_summaries_secs=0, saver=None)
Data = namedtuple('Data', ['x1', 'y1', 'x2', 'y2', 'flow', 'feature_matches1', 'mask1', 'feature_matches2', 'mask2'])
with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True,gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction))) as sess:
    #sess.run(init_all)
    #threads = tf.train.start_queue_runners(sess=sess)
    if args.restore: 
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        logger.info('restoring {}'.format(tf.train.latest_checkpoint(model_dir)))
    else:
        restorer.restore(sess, checkpoint_file)

    st_step = max(0,sess.run(global_step))
    sv.summary_writer.add_session_log(tf.SessionLog(status=tf.SessionLog.START), global_step=st_step-1)
    time_start = time.time()
    tot_time = 0
    tot_train_time = 0

    for i in range(st_step, training_iter):
        batch_x1s, batch_y1s, batch_x2s, batch_y2s, batch_flows, batch_feature_matches1, batch_mask1, batch_feature_matches2, batch_mask2 = sess.run(
            [x1_batch, y1_batch, x2_batch, y2_batch, flow_batch, feature_matches1_batch, mask1_batch, feature_matches2_batch, mask2_batch])
        if (i > no_theta_iter):
            use_theta = 0
        else:
            use_theta = 1
        if (i >= do_temp_loss_iter):
            use_temp = 1
        else:
            use_temp = 0
        if (i <= do_theta_10_iter):
            use_theta = 10
        if (i >= do_black_loss_iter):
            use_black = 1
        else:
            use_black = 0
        if (i <= do_theta_only_iter):
            theta_only = 1
        else:
            theta_only = 0
        if i % disp_freq == 0 or i == training_iter - 1:
            print('==========================')
            print('read data time:' + str(tot_time / disp_freq) + 's')
            print('train time:' + str(tot_train_time / disp_freq) + 's')
            tot_train_time = 0
            tot_time = 0
            time_start = time.time()
            c_rate = rand_crop()
            loss, summary = sess.run([total_loss, merged],
                            feed_dict={
                                ret1['x_tensor']: batch_x1s,
                                ret1['y']: batch_y1s,
                                ret1['mask']: batch_mask1,
                                ret1['matches']: batch_feature_matches1,
                                ret2['x_tensor']: batch_x2s,
                                ret2['y']: batch_y2s,
                                ret2['mask']: batch_mask2,
                                ret2['matches']: batch_feature_matches2,
                                flow: batch_flows,
                                ret1['use_theta_loss']: use_theta,
                                ret2['use_theta_loss']: use_theta,
                                use_temp_loss: use_temp,
                                ret1['use_black_loss']: use_black,
                                ret2['use_black_loss']: use_black,
                                ret1['use_theta_only']: theta_only,
                                ret2['use_theta_only']: theta_only
                            })
            sv.summary_writer.add_summary(summary, i)
            print('Iteration: ' + str(i) + ' Loss: ' + str(loss))
            lr = sess.run(learning_rate)
            print(lr)
            time_end = time.time()
            print('disp time:' + str(time_end - time_start) + 's')

        if i % save_freq == 0 or i == training_iter - 1:
            saver.save(sess, model_dir + 'model', global_step=i)
        if i % test_freq == 0 or i == training_iter - 1:
            sum_test_loss = 0.0
            for j in range(test_batches):
                # test_batch_x1s, test_batch_y1s, test_batch_x2s, test_batch_y2s, test_batch_flows, \
                #     test_batch_feature_matches1, test_batch_mask1, test_batch_feature_matches2, test_batch_mask2 = sess.run(
                    # [test_x1_batch, test_y1_batch, test_x2_batch, test_y2_batch, test_flow_batch, 
                    # test_feature_matches1_batch, test_mask1_batch, test_feature_matches2_batch, test_mask2_batch])
                def fetch_test_data(input_data):
                     c_rate = rand_crop()
                     return sess.run([total_loss, ret1['stable_warpped'], ret1['output']],
                                feed_dict={
                                    ret1['x_tensor']: input_data.x1,
                                    ret1['y']: input_data.y1,
                                    ret1['mask']: input_data.mask1,
                                    ret1['matches']: input_data.feature_matches1,
                                    ret2['x_tensor']: input_data.x2,
                                    ret2['y']: input_data.y2,
                                    ret2['mask']: input_data.mask2,
                                    ret2['matches']: input_data.feature_matches2,
                                    flow: input_data.flow,
                                    ret1['use_theta_loss']: use_theta,
                                    ret2['use_theta_loss']: use_theta,
                                    use_temp_loss: use_temp,
                                    ret1['use_black_loss']: use_black,
                                    ret2['use_black_loss']: use_black,
                                    ret1['use_theta_only']: theta_only,
                                    ret2['use_theta_only']: theta_only

                                })
                input_tensor = Data(test_x1_batch, test_y1_batch, test_x2_batch, test_y2_batch, test_flow_batch, 
                    test_feature_matches1_batch, test_mask1_batch, test_feature_matches2_batch, test_mask2_batch)
                input_data = Data(**sess.run(input_tensor._asdict()))
                loss, stable_warpped_pts_batch, output = fetch_test_data(input_data)
                #save_warpped_features(input_data, stable_warpped_pts_batch, theta_mat, output, name='test')
                sum_test_loss += loss

            sum_test_loss /= test_batches
            print("Test Loss: " + str(sum_test_loss))
            summary = sess.run(test_merged,
                    feed_dict={
                        loss_displayer: sum_test_loss
                    })
            sv.summary_writer.add_summary(summary, i)

            input_tensor = Data(x1_batch, y1_batch, x2_batch, y2_batch, flow_batch, 
                feature_matches1_batch, mask1_batch, feature_matches2_batch, mask2_batch)
            input_data = Data(**sess.run(input_tensor._asdict()))
            loss, stable_warpped_pts_batch, output = fetch_test_data(input_data)
            #save_warpped_features(input_data, stable_warpped_pts_batch, theta_mat, output, name='train')
            
        time_end = time.time()
        tot_time += time_end - time_start
        t_s = time.time()
        c_rate = rand_crop()
        sess.run(optimizer,
                    feed_dict={
                        ret1['x_tensor']: batch_x1s,
                        ret1['y']: batch_y1s,
                        ret1['mask']: batch_mask1,
                        ret1['matches']: batch_feature_matches1,
                        ret2['x_tensor']: batch_x2s,
                        ret2['y']: batch_y2s,
                        ret2['mask']: batch_mask2,
                        ret2['matches']: batch_feature_matches2,
                        flow: batch_flows,
                        ret1['use_theta_loss']: use_theta,
                        ret2['use_theta_loss']: use_theta,
                        use_temp_loss: use_temp,
                        ret1['use_black_loss']: use_black,
                        ret2['use_black_loss']: use_black,
                        ret1['use_theta_only']: theta_only,
                        ret2['use_theta_only']: theta_only

                    })
        t_e = time.time()
        tot_train_time += t_e - t_s
        '''
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline.json', 'w') as f:
            f.write(ctf)
        if (i == 200):
            break
        '''
        time_start = time.time()


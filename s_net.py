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
from spatial_transformer import transformer
import numpy as np
from tf_utils import weight_variable, bias_variable, dense_to_one_hot
import cv2
from resnet import *
import get_data
from config import *
import time
import utils
logger = utils.get_logger()
def get_theta_black_loss(theta):
    theta = tf.reshape(theta, (-1, 3, 3))
    theta = tf.cast(theta, 'float32')

    d = crop_rate
    target = tf.constant([-d, d, -d, d, -d, -d, d, d], shape=[8], dtype=tf.float32)
    target = tf.tile(target, [batch_size])
    target = tf.reshape(target, [batch_size, 2, -1])

    grid = tf.constant([-1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1], shape=[12], dtype=tf.float32)
    grid = tf.tile(grid, [batch_size])
    grid = tf.reshape(grid, [batch_size, 3, -1])

    T_g = tf.matmul(theta, grid)

    x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
    y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
    z_s = tf.slice(T_g, [0, 2, 0], [-1, 1, -1])

    t_1 = tf.ones(shape = tf.shape(x_s), dtype=tf.float32)
    t_0 = tf.zeros(shape = tf.shape(x_s), dtype=tf.float32)      

    sign_z = tf.where(tf.greater(z_s, t_0), t_1, t_0) * 2.0 - 1.0
    z_s = z_s + sign_z * 1e-5

    #op = tf.Print(theta, [z_s], summarize=24)
    #with tf.control_dependencies([op]):
    x_s = tf.div(x_s, z_s)
    y_s = tf.div(y_s, z_s)

    output = tf.concat([x_s, y_s], 1)
    
    #op = tf.Print(theta, [x_s], summarize=24)
    #op2 = tf.Print(theta, [y_s], summarize=24)
    #with tf.control_dependencies([op, op2]):
    #output = tf.slice(T_g, [0, 0, 0], [-1, 2, -1])
    #output = T_g[:, :2, :] / T_g[:, 2, None, :]
    one_ = tf.ones([batch_size, 2, 4])
    zero_ = tf.zeros([batch_size, 2, 4])
    black_err = tf.where(tf.greater(output, one_), output - one_, zero_) + tf.where(tf.greater(one_ * -1, output), one_ * -1 - output, zero_)
    return tf.reduce_mean(tf.abs(output - target)), tf.reshape(black_err, [batch_size, -1])

def reduce_layer(input):
    with tf.variable_scope('reduce_layer'):
        with tf.variable_scope('conv0'):
            conv0_ = conv_bn_relu_layer(input, [1, 1, 2048, 512], 1)

        with tf.variable_scope('conv1_0'):
            conv1_0_ = conv_bn_relu_layer2(conv0_, [1, 16, 512, 512], [1, 1])
        with tf.variable_scope('conv2_0'):
            conv2_0_ = conv_bn_relu_layer2(conv1_0_, [9, 1, 512, 512], [1, 1])

        with tf.variable_scope('conv1_1'):
            conv1_1_ = conv_bn_relu_layer2(conv0_, [9, 1, 512, 512], [1, 1])
        with tf.variable_scope('conv2_1'):
            conv2_1_ = conv_bn_relu_layer2(conv1_1_, [1, 16, 512, 512], [1, 1])
        with tf.variable_scope('conv2'):
            conv2_ = conv2_0_ + conv2_1_
        with tf.variable_scope('conv3'):
            conv3_ = conv_bn_relu_layer(conv2_, [1, 1, 512, 128], 1)
        with tf.variable_scope('conv4'):
            conv4_ = conv_bn_relu_layer(conv3_, [1, 1, 128, 32], 1)
        with tf.variable_scope('fc'):
            out = output_layer(tf.reshape(conv4_, [batch_size, 32]), 8)
    return out

def to_mat(x):
    return tf.reshape(x, [-1, 3, 3])

def warp_pts(x, theta_mat):
    logger.info('warp_pts: x.shape={}, theta_mat.shape={}'.format(x.shape, theta_mat.shape))
    x = tf.concat([x, tf.ones([tf.shape(x)[0], tf.shape(x)[1], 1])], axis=2)
    warpped = tf.matmul(x, tf.transpose(theta_mat, [0, 2, 1]))
    return warpped[:, :, :2] / warpped[:, :, 2, None]

def inference_stable_net(reuse):
    with tf.variable_scope('stable_net'):
        with tf.name_scope('input'):
            # %% Since x is currently [batch, height*width], we need to reshape to a
            # 4-D tensor to use it in a convolutional graph.  If one component of
            # `shape` is the special value -1, the size of that dimension is
            # computed so that the total size remains constant.  Since we haven't
            # defined the batch dimension's shape yet, we use -1 to denote this
            # dimension should not change size.
            x_tensor = tf.placeholder(tf.float32, [None, height, width, tot_ch], name = 'x_tensor')
            x_batch_size = tf.shape(x_tensor)[0]
            x = tf.slice(x_tensor, [0, 0, 0, before_ch], [-1, -1, -1, 1])
            mask = tf.placeholder(tf.float32, [None, max_matches])
            matches = tf.placeholder(tf.float32, [None, max_matches, 4])
            
            for i in range(tot_ch):
                temp = tf.slice(x_tensor, [0, 0, 0, i], [-1, -1, -1, 1])
                tf.summary.image('x' + str(i), temp)

        with tf.name_scope('label'):
            y = tf.placeholder(tf.float32, [None, height, width, 1])
            x4 = tf.slice(y, [0, 0, 0, 0], [-1, -1, -1, 1])
            tf.summary.image('label', x4)

        with tf.variable_scope('resnet', reuse=reuse): 
            config = {'stage_sizes' : [3, 4, 6, 3], 'channel_params' : [{'kernel_sizes':[1, 3, 1], 'channel_sizes':[64, 64, 256]}, 
                                                                        {'kernel_sizes':[1, 3, 1], 'channel_sizes':[128, 128, 512]}, 
                                                                        {'kernel_sizes':[1, 3, 1], 'channel_sizes':[256, 256, 1024]},
                                                                        {'kernel_sizes':[1, 3, 1], 'channel_sizes':[512, 512, 2048]}]}
            resnet = inference(x_tensor, tot_ch, config)

        with tf.variable_scope('fc', reuse=reuse):
            in_channel = resnet.get_shape().as_list()[-1]
            bn_layer = batch_normalization_layer(resnet, in_channel)
            relu_layer = tf.nn.relu(bn_layer)

            global_pool = tf.reduce_mean(relu_layer, [1, 2])
            theta = output_layer(global_pool, 8)

            #theta = reduce_layer(resnet)
            theta = tf.concat([theta, tf.ones([x_batch_size, 1], tf.float32)], 1)

        with tf.name_scope('theta_loss'):
            use_theta_loss = tf.placeholder(tf.float32)
            use_black_loss = tf.placeholder(tf.float32)
            theta_loss, black_pos = get_theta_black_loss(theta)
            black_pos = black_pos * black_pos
            theta_loss = theta_loss * use_theta_loss
            black_pos = black_pos * use_black_loss

        with tf.name_scope('feature_loss'):
            use_feature_loss = tf.placeholder(tf.float32)
            stable_pts = matches[:, :, :2]
            unstable_pts = matches[:, :, 2:]
            theta_mat = to_mat(theta)
            stable_warpped = warp_pts(stable_pts, theta_mat)
            before_mask = tf.reduce_sum(tf.abs(stable_warpped - unstable_pts), 2)
            #before_mask = tf.Print(before_mask, [tf.reduce_mean(before_mask), mask])
            logger.info('before_mask.shape={}'.format(before_mask.shape))
            assert(before_mask.shape[1] == max_matches)
            after_mask = tf.reduce_sum(before_mask * mask, axis=1) / (tf.maximum(tf.reduce_sum(mask, axis=1), 1))
            logger.info('after_mask.shape={}'.format(after_mask.shape))
            feature_loss = tf.reduce_mean(after_mask)
        regu_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        regu_loss = tf.add_n(regu_loss)
        out_size = (height, width)
        h_trans, black_pix = transformer(x, theta, out_size)
        black_pos_loss = tf.reduce_mean(black_pos)
        tf.add_to_collection('output', h_trans)
        with tf.name_scope('img_loss'):
            black_pix = tf.reshape(black_pix, [batch_size, height, width, 1]) 
            black_pix = tf.stop_gradient(black_pix)
            img_err = (h_trans - y) * (1 - black_pix)
            tf.summary.image('err', img_err * img_err)
            img_loss = tf.reduce_sum(tf.reduce_sum(img_err * img_err, [1, 2, 3]) / (tf.reduce_sum((1 - black_pix), [1, 2, 3]) + 1e-8), [0]) / batch_size
            
            #img_loss = tf.nn.l2_loss(h_trans - y) / batch_size
        use_theta_only = tf.placeholder(tf.float32)
        total_loss = theta_loss * theta_mul + ((1 - use_theta_only) * 
        (img_loss * img_mul + regu_loss * regu_mul + black_pos_loss * black_mul + feature_loss * feature_mul))

        '''
        with tf.name_scope('loss'):
            tf.summary.scalar('tot_loss',total_loss)
            tf.summary.scalar('theta_loss',theta_loss * theta_mul)
            tf.summary.scalar('img_loss',img_loss * img_mul)
            tf.summary.scalar('regu_loss',regu_loss * regu_mul)
        '''
    ret = {}
    ret['error'] = tf.abs(h_trans - y)
    ret['black_pos'] = black_pos
    ret['black_pix'] = black_pix
    ret['theta_loss'] = theta_loss * theta_mul
    ret['black_loss'] = black_pos_loss * black_mul
    ret['img_loss'] = img_loss * img_mul
    ret['regu_loss'] = regu_loss * regu_mul
    ret['feature_loss'] = feature_loss * feature_mul
    ret['x_tensor'] = x_tensor
    ret['use_theta_only'] = use_theta_only
    ret['y'] = y
    ret['mask'] = mask
    ret['matches'] = matches
    ret['output'] = h_trans
    ret['total_loss'] = total_loss
    ret['use_theta_loss'] = use_theta_loss
    ret['use_black_loss'] = use_black_loss
    ret['stable_warpped'] = stable_warpped
    ret['theta_mat'] = theta_mat
    return ret

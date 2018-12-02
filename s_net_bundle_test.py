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
from spatial_transformer2 import transformer
import numpy as np
from tf_utils import weight_variable, bias_variable, dense_to_one_hot
import cv2
from resnet import output_layer
import get_data
from config import *
import time
from tensorflow.contrib.slim.nets import resnet_v2
slim = tf.contrib.slim
import utils
logger = utils.get_logger()

def get_4_pts(theta, batch_size):
    with tf.name_scope('get_4_pts'):
        #v2_16 ##
        #big_p0 = tf.slice(theta, [0, 0], [-1, 2]) + tf.constant([-1, -1], shape=[2], dtype=tf.float32)
        #big_p1 = tf.slice(theta, [0, 2], [-1, 2]) + tf.constant([1, -1], shape=[2], dtype=tf.float32)
        #big_p2 = tf.slice(theta, [0, 4], [-1, 2]) + tf.constant([-1, 1], shape=[2], dtype=tf.float32)
        #big_p3 = tf.slice(theta, [0, 6], [-1, 2]) + tf.constant([1, 1], shape=[2], dtype=tf.float32)
        big_p0 = tf.constant([-1, -1], shape=[2], dtype=tf.float32)
        big_p1 = tf.constant([1, -1], shape=[2], dtype=tf.float32)
        big_p2 = tf.constant([-1, 1], shape=[2], dtype=tf.float32)
        big_p3 = tf.constant([1, 1], shape=[2], dtype=tf.float32)
        pts1_ = []
        pts2_ = []
        pts = []
        h = 2.0 / grid_h
        w = 2.0 / grid_w
        tot = 0

        for i in range(grid_h + 1):
            pts.append([])
            for j in range(grid_w + 1):
                hh = i * h / 2
                ww = j * w / 2
                p = big_p0 * (1 - hh) * (1 - ww) + big_p1 * (1 - hh) * ww + big_p2 * hh * (1 - ww) + big_p3 * hh * ww
                temp = tf.slice(theta, [0, tot * 2], [-1, 2])
                #test[v]
                #temp = tf.zeros([batch_size, 2])
                #test[^]
                tot += 1
                p = tf.reshape(p + temp, [batch_size, 1, 2])
                pts[i].append(tf.reshape(p, [batch_size, 2, 1]))
                pts2_.append(p)

        for i in range(grid_h):
            for j in range(grid_w):
                g = tf.concat([pts[i][j], pts[i][j + 1], pts[i + 1][j], pts[i + 1][j + 1]], axis=2)
                pts1_.append(tf.reshape(g, [batch_size, 1, 8]))

        pts1 = tf.reshape(tf.concat(pts1_, 1), [batch_size, grid_h, grid_w, 8])
        pts2 = tf.reshape(tf.concat(pts2_, 1), [batch_size, grid_h + 1, grid_w + 1, 2])

    return pts1, pts2

def get_theta_black_loss(theta, do_crop_rate):
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

    t_1 = tf.constant(1.0, shape=[batch_size, 1, 4])
    t_0 = tf.constant(0.0, shape=[batch_size, 1, 4])      

    sign_z = tf.where(tf.greater(z_s, t_0), t_1, t_0) * 2.0 - 1.0
    z_s = z_s + sign_z * 1e-8

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
    one_ = tf.ones([batch_size, 2, 4]) / do_crop_rate
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

def get_black_pos(pts):
    print(batch_size)
    print('==================')
    with tf.name_scope('black_pos'):
        one_ = tf.ones([batch_size, grid_h, grid_w, 8]) / do_crop_rate
        zero_ = tf.zeros([batch_size, grid_h, grid_w, 8])
        black_err = tf.where(tf.greater(pts, one_), pts - one_, zero_) + tf.where(tf.greater(one_ * -1, pts), one_ * -1 - pts, zero_)
    return tf.reshape(black_err, [batch_size, -1])

def calc_distortion_loss(p0, p1, p2, clock, hw):
    h = 2.0 / grid_h
    w = 2.0 / grid_w
    if (hw == 0):
        k = h / w
    else:
        k = w / h

    if (not clock):
        R_ = [0, -k, k, 0]
    else:
        R_ = [0, k, -k, 0]
    R = tf.constant(R_, shape=[4], dtype=tf.float32)
    R = tf.tile(R, [batch_size * grid_h * grid_w])
    R = tf.reshape(R, [-1, 2, 2])
    loss = tf.abs(tf.matmul(R, p1 - p0) - (p2 - p1))    #batch_size*grid_h*grid_w, 2, 1
    return loss * loss

def get_distortion_loss(pts):
    with tf.name_scope('distortion_loss'):
        pts = tf.reshape(pts, [-1, 2, 4])
        p0 = tf.slice(pts, [0, 0, 0], [-1, -1, 1])
        p1 = tf.slice(pts, [0, 0, 1], [-1, -1, 1])
        p2 = tf.slice(pts, [0, 0, 2], [-1, -1, 1])
        p3 = tf.slice(pts, [0, 0, 3], [-1, -1, 1])
        loss =          calc_distortion_loss(p0, p1, p3, 0, 0)
        loss = loss +   calc_distortion_loss(p1, p3, p2, 0, 1)
        loss = loss +   calc_distortion_loss(p3, p2, p0, 0, 0)
        loss = loss +   calc_distortion_loss(p2, p0, p1, 0, 1)
        loss = loss +   calc_distortion_loss(p1, p0, p2, 1, 0)
        loss = loss +   calc_distortion_loss(p0, p2, p3, 1, 1)
        loss = loss +   calc_distortion_loss(p2, p3, p1, 1, 0)
        loss = loss +   calc_distortion_loss(p3, p1, p0, 1, 1)
    return tf.reduce_mean(loss) / 8

def get_consistency_loss(pts):
    with tf.name_scope('consistency_loss'):
        p = []
        for i in range(grid_h + 1):
            p.append([])
            for j in range(grid_w + 1):
                pn = tf.slice(pts, [0, i, j, 0], [-1, 1, 1, -1])
                pn = tf.reshape(pn, [batch_size, 2, 1])
                p[i].append(pn)

        errs = []
        for i in range(grid_h + 1):
            for j in range(grid_w + 1):
                if (i > 1):
                    errs.append(tf.abs(2 * p[i - 1][j] - p[i][j] - p[i - 2][j]))
                if (j > 1):
                    errs.append(tf.abs(2 * p[i][j - 1] - p[i][j] - p[i][j - 2]))
                if (i < grid_h - 1):
                    errs.append(tf.abs(2 * p[i + 1][j] - p[i][j] - p[i + 2][j]))
                if (j < grid_w - 1):
                    errs.append(tf.abs(2 * p[i][j + 1] - p[i][j] - p[i][j + 2]))
        if (len(errs) == 0):
            loss = tf.reduce_mean(pts) * 0
        else:
            loss = tf.concat(errs, 2)
            loss = tf.reduce_mean(loss * loss)
    return loss

def to_mat(x):
    return tf.reshape(x, [-1, 3, 3])

def warp_pts(pts, flow):
    x = pts[:, :, 0]
    x = tf.clip_by_value((x + 1) / 2 * width, 0, width - 1)
    x = tf.cast(tf.round(x), tf.int32)
    y = pts[:, :, 1]
    y = tf.clip_by_value((y + 1) / 2 * height, 0, height - 1)
    y = tf.cast(tf.round(y), tf.int32)

    out = []
    for i in range(batch_size):
        flow_ = tf.reshape(flow[i, :, :, :], [-1, 2])
        xy = x[i, :] + y[i, :] * width
        temp = tf.gather(flow_, xy)
        print(temp.shape)
        out.append(tf.reshape(temp, [1, max_matches, 2]))
    return tf.concat(out, 0)

def get_resnet__(x_tensor, reuse, is_training, x_batch_size):
    with tf.variable_scope('resnet', reuse=reuse):
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            resnet, end_points = resnet_v2.resnet_v2_50(x_tensor, global_pool=False, is_training=is_training, reuse=reuse, output_stride=32)
        global_pool = tf.reduce_mean(resnet, [1, 2])
        with tf.variable_scope('fc'):
            #global_pool = slim.fully_connected(global_pool, 2048, scope='fc/fc_1')
            #global_pool = slim.fully_connected(global_pool, 1024, scope='fc/fc_2')
            #global_pool = slim.fully_connected(global_pool, 512, scope='fc/fc_3')
            theta = output_layer(global_pool, (grid_h + 1) * (grid_w + 1) * 2)
        with tf.name_scope('gen_theta'):
            id2_loss = tf.reduce_mean(tf.abs(tf.slice(theta, [0, 0], [-1, -1]))) * id_mul
    return theta, id2_loss, id2_loss

def get_resnet(x_tensor, reuse, is_training, x_batch_size):
    with tf.variable_scope('resnet', reuse=reuse):
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            resnet, end_points = resnet_v2.resnet_v2_50(x_tensor, global_pool=False, is_training=is_training, reuse=reuse, output_stride=32)
        global_pool = tf.reduce_mean(resnet, [1, 2])
        with tf.variable_scope('fc'):
            #test[v]
            #theta = output_layer(global_pool, 8)# + (grid_h + 1) * (grid_w + 1) * 2)
            theta = output_layer(global_pool, (grid_h + 1) * (grid_w + 1) * 2)
            #test[^]
        with tf.name_scope('gen_theta'):
            id_loss  = tf.reduce_mean(tf.abs(tf.slice(theta, [0, 0], [-1,  -1]))) * id_mul
            id2_loss = tf.reduce_mean(tf.abs(tf.slice(theta, [0, 8], [-1, -1]))) * id_mul
    #test[v]
    return theta, id_loss, id_loss#, id2_loss
    #test[^]

def inference_stable_net(reuse):
    with tf.variable_scope('stable_net'):
        with tf.name_scope('input'):
            # %% Since x is currently [batch, height*width], we need to reshape to a
            # 4-D tensor to use it in a convolutional graph.  If one component of
            # `shape` is the special value -1, the size of that dimension is
            # computed so that the total size remains constant.  Since we haven't
            # defined the batch dimension's shape yet, we use -1 to denote this
            # dimension should not change size.
            if input_mask:
                x_tensor = tf.placeholder(tf.float32, [None, height, width, tot_ch + before_ch + 1], name = 'x_tensor')
            else:
                x_tensor = tf.placeholder(tf.float32, [None, height, width, tot_ch + 1], name = 'x_tensor')
            x_batch_size = tf.shape(x_tensor)[0]
            if (input_mask):
                x = tf.slice(x_tensor, [0, 0, 0, before_ch + before_ch], [-1, -1, -1, 1])
                black_mask = tf.slice(x_tensor, [0, 0, 0, before_ch + before_ch + 1], [-1, -1, -1, 1])
            else:
                x = tf.slice(x_tensor, [0, 0, 0, before_ch], [-1, -1, -1, 1])
                black_mask = tf.slice(x_tensor, [0, 0, 0, before_ch + 1], [-1, -1, -1, 1])

            mask = tf.placeholder(tf.float32, [None, max_matches])
            matches = tf.placeholder(tf.float32, [None, max_matches, 4])
           
            if (input_mask):
                out_ch = tot_ch + before_ch + 1
            else:
                out_ch = tot_ch + 1
            for i in range(out_ch):
                temp = tf.slice(x_tensor, [0, 0, 0, i], [-1, -1, -1, 1])
                tf.summary.image('x' + str(i), temp)

        with tf.name_scope('label'):
            y = tf.placeholder(tf.float32, [None, height, width, 1])
            x4 = tf.slice(y, [0, 0, 0, 0], [-1, -1, -1, 1])
            tf.summary.image('label', x4)

        theta, id_loss, id2_loss = get_resnet(x_tensor, reuse = reuse, is_training=True, x_batch_size = x_batch_size)
        theta_infer, id_loss_infer, id2_loss_infer = get_resnet(x_tensor, reuse = True, is_training=False, x_batch_size = x_batch_size)
        #new


        pts1, pts2 = get_4_pts(theta, x_batch_size)
        _, pts2_infer = get_4_pts(theta_infer, x_batch_size)
        with tf.name_scope('inference'):
            h_trans_infer, black_pix_infer, _ = transformer(x, pts2_infer)
        with tf.name_scope('theta_loss'):
            use_theta_loss = tf.placeholder(tf.float32)
            theta_loss = id_loss #theta_loss * use_theta_loss + id_loss
            grid_theta_loss = id2_loss
        with tf.name_scope('black_loss'):
            use_black_loss = tf.placeholder(tf.float32)
            black_pos = get_black_pos(pts1)
            black_pos = black_pos * black_pos
            black_pos = black_pos * use_black_loss
            black_pos_loss = tf.reduce_mean(black_pos)

        with tf.name_scope('regu_loss'):
            #regu_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            #regu_loss = tf.add_n(regu_loss)


            regu_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            regu_loss = tf.add_n(regu_loss)
            #regu_loss = tf.add_n(slim.losses.get_regularization_losses())

        distortion_loss = get_distortion_loss(pts1)
        consistency_loss = get_consistency_loss(pts2)
        #neighbor_loss = get_neighbor_loss(pts)

        h_trans, black_pix, flow = transformer(x, pts2)


        with tf.name_scope('feature_loss'):
            use_feature_loss = tf.placeholder(tf.float32)
            stable_pts = matches[:, :, :2]
            unstable_pts = matches[:, :, 2:]
            stable_warpped = warp_pts(stable_pts, flow)
            before_mask = tf.reduce_sum(tf.abs(stable_warpped - unstable_pts), 2)
            assert(before_mask.shape[1] == max_matches)
            after_mask = tf.reduce_sum(before_mask * mask, axis=1) / (tf.maximum(tf.reduce_sum(mask, axis=1), 1))
            feature_loss = tf.reduce_mean(after_mask)

        tf.summary.image('output', h_trans)
        tf.add_to_collection('output', h_trans)
        with tf.name_scope('img_loss'):
            black_pix = tf.reshape(black_pix, [batch_size, height, width, 1]) 
            #black_pix = tf.stop_gradient(black_pix)
            img_err = (h_trans - y) * (1 - black_pix)
            tf.summary.image('err', img_err * img_err)
            img_loss = tf.reduce_sum(tf.reduce_sum(img_err * img_err, [1, 2, 3]) / (tf.reduce_sum((1 - black_pix), [1, 2, 3]) + 1e-8), [0]) / batch_size

        use_theta_only = tf.placeholder(tf.float32)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies([tf.group(*update_ops)]):
            total_loss = theta_loss * theta_mul + grid_theta_loss * grid_theta_mul + ((1 - use_theta_only) * 
            (img_loss * img_mul + regu_loss * regu_mul + black_pos_loss * black_mul + distortion_loss * distortion_mul + 
             consistency_loss * consistency_mul + feature_loss * feature_mul))
        
    ret = {}
    ret['error'] = tf.abs(h_trans - y)
    ret['black_pos'] = black_pos
    ret['black_pix'] = black_pix
    ret['theta_loss'] = theta_loss * theta_mul
    ret['grid_theta_loss'] = grid_theta_loss * grid_theta_mul
    ret['black_loss'] = black_pos_loss * black_mul
    ret['distortion_loss'] = distortion_loss * distortion_mul
    ret['consistency_loss'] = consistency_loss * consistency_mul
    ret['feature_loss'] = feature_loss * feature_mul
    ret['mask'] = mask
    ret['matches'] = matches
    ret['feature_loss'] = feature_loss * feature_mul
    ret['img_loss'] = img_loss * img_mul
    ret['regu_loss'] = regu_loss * regu_mul
    ret['x_tensor'] = x_tensor
    ret['use_theta_only'] = use_theta_only
    ret['y'] = y
    ret['output'] = h_trans
    ret['total_loss'] = total_loss
    ret['use_theta_loss'] = use_theta_loss
    ret['use_black_loss'] = use_black_loss
    ret['stable_warpped'] = stable_warpped
    #ret['theta_mat'] = theta_mat
    return ret


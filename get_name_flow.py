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
import get_data_flow
from config import *
import time
import s_net

ret1 = s_net.inference_stable_net(False)
ret2 = s_net.inference_stable_net(True)

with tf.name_scope('data_flow'):
    flow = tf.placeholder(tf.float32, [None, height, width, 2])
    x_flow = tf.slice(flow, [0, 0, 0, 0], [-1, -1, -1, 1])
    y_flow = tf.slice(flow, [0, 0, 0, 1], [-1, -1, -1, 1])

with tf.name_scope('temp_loss'):
    use_temp_loss = tf.placeholder(tf.float32)
    output2_aft_flow = interpolate(ret2['output'], x_flow, y_flow, (height, width))
    temp_loss = tf.nn.l2_loss(ret1['output'] - output2_aft_flow) / batch_size * use_temp_loss
with tf.name_scope('errors'):
    tf.summary.image('error_temp', tf.abs(ret1['output'] - output2_aft_flow))
    tf.summary.image('error_1', ret1['error'])
    tf.summary.image('error_2', ret2['error'])

with tf.name_scope('test_flow'):
    warped_y2 = interpolate(ret2['y'], x_flow, y_flow, (height, width))
    tf.summary.image('error_black_wy2', tf.abs(ret1['y'] - warped_y2))
    tf.summary.image('error_black_nowarp', tf.abs(ret2['y'] - ret1['y']))

loss_displayer = tf.placeholder(tf.float32)
with tf.name_scope('test_loss'):
    tf.summary.scalar('test_loss', loss_displayer, collections=['test'])

total_loss = ret1['total_loss'] + ret2['total_loss'] + temp_loss * temp_mul
with tf.name_scope('train_loss'):
    tf.summary.scalar('theta_loss', ret1['theta_loss'] + ret2['theta_loss'])
    tf.summary.scalar('img_loss', ret1['img_loss'] + ret2['img_loss'])
    tf.summary.scalar('regu_loss', ret1['regu_loss'] + ret2['regu_loss'])
    tf.summary.scalar('temp_loss', temp_loss * temp_mul)
    tf.summary.scalar('total_loss', total_loss)

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                           global_step=global_step,
                                           decay_steps=step_size,decay_rate=0.1, staircase=True)
opt = tf.train.AdamOptimizer(learning_rate)
optimizer = opt.minimize(total_loss, global_step=global_step)

print(ret1['x_tensor'].name)
print(ret1['output'].name)

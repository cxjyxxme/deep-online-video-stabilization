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
'''
# %% Load data
mnist_cluttered = np.load('./data/mnist_sequence1_sample_5distortions5x5.npz')

X_train = mnist_cluttered['X_train']
y_train = mnist_cluttered['y_train']
X_valid = mnist_cluttered['X_valid']
y_valid = mnist_cluttered['y_valid']
X_test = mnist_cluttered['X_test']
y_test = mnist_cluttered['y_test']

# % turn from dense to one hot representation
Y_train = dense_to_one_hot(y_train, n_classes=10)
Y_valid = dense_to_one_hot(y_valid, n_classes=10)
Y_test = dense_to_one_hot(y_test, n_classes=10)
'''

def get_theta_loss(theta):
    theta = tf.reshape(theta, (-1, 3, 3))
    theta = tf.cast(theta, 'float32')

    d = crop_rate
    '''
    target = tf.convert_to_tensor(np.array(  [[-d, d, -d, d], [-d, -d, d, d]]))
    target = tf.cast(target, 'float32')
    target = tf.expand_dims(target, 0)
    target = tf.reshape(target, [-1])
    '''
    target = tf.constant([-d, d, -d, d, -d, -d, d, d], shape=[8], dtype=tf.float32)
    target = tf.tile(target, tf.stack([batch_size]))
    target = tf.reshape(target, tf.stack([batch_size, 2, -1]))

    '''
    grid = tf.convert_to_tensor(np.array([[-1, 1, -1, 1], [-1, -1, 1, 1], [1, 1, 1, 1]]))
    grid = tf.cast(grid, 'float32')
    grid = tf.expand_dims(grid, 0)
    grid = tf.reshape(grid, [-1])
    '''

    grid = tf.constant([-1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1], shape=[12], dtype=tf.float32)
    grid = tf.tile(grid, tf.stack([batch_size]))
    grid = tf.reshape(grid, tf.stack([batch_size, 3, -1]))

    T_g = tf.matmul(theta, grid)
    output = tf.slice(T_g, [0, 0, 0], [-1, 2, -1])
    return tf.reduce_mean(tf.abs(output - target))
    
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
        
        for i in range(tot_ch):
            temp = tf.slice(x_tensor, [0, 0, 0, i], [-1, -1, -1, 1])
            tf.summary.image('x' + str(i), temp)

with tf.name_scope('label'):
	y = tf.placeholder(tf.float32, [None, height, width, 1])
        x4 = tf.slice(y, [0, 0, 0, 0], [-1, -1, -1, 1])
        tf.summary.image('label', x4)

with tf.variable_scope('resnet'): 
    config = {'stage_sizes' : [3, 4, 23], 'channel_params' : [  {'kernel_sizes':[1, 3, 1], 'channel_sizes':[64, 64, 256]}, 
                                                                {'kernel_sizes':[1, 3, 1], 'channel_sizes':[128, 128, 512]}, 
                                                                {'kernel_sizes':[1, 3, 1], 'channel_sizes':[256, 256, 1024]}]}
    resnet = inference(x_tensor, tot_ch, config)

with tf.variable_scope('fc'):
    in_channel = resnet.get_shape().as_list()[-1]
    bn_layer = batch_normalization_layer(resnet, in_channel)
    relu_layer = tf.nn.relu(bn_layer)
    global_pool = tf.reduce_mean(relu_layer, [1, 2])

    theta = output_layer(global_pool, 8)
    theta = tf.concat([theta, tf.ones([x_batch_size, 1], tf.float32)], 1)

with tf.name_scope('theta_loss'):
    use_theta_loss = tf.placeholder(tf.float32)
    theta_loss = get_theta_loss(theta) * use_theta_loss
regu_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
regu_loss = tf.add_n(regu_loss)
out_size = (height, width)
h_trans = transformer(x, theta, out_size)
tf.add_to_collection('output', h_trans)
tf.summary.image('result', h_trans)
img_loss = tf.nn.l2_loss(h_trans - y) / batch_size
#img_loss = tf.reduce_mean(tf.abs(h_trans - y))
tf.summary.image('error', tf.abs(h_trans - y))
total_loss = theta_loss * theta_mul + img_loss * img_mul + regu_loss * regu_mul
#total_loss = theta_loss
loss_displayer = tf.placeholder(tf.float32)
with tf.name_scope('loss'):
    tf.summary.scalar('tot_loss',total_loss)
    tf.summary.scalar('theta_loss',theta_loss * theta_mul)
    tf.summary.scalar('img_loss',img_loss * img_mul)
    tf.summary.scalar('regu_loss',regu_loss * regu_mul)

with tf.name_scope('test_loss'):
    tf.summary.scalar('test_loss', loss_displayer, collections=['test'])

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                           global_step=global_step,
                                           decay_steps=step_size,decay_rate=0.1, staircase=True)
opt = tf.train.AdamOptimizer(learning_rate)
optimizer = opt.minimize(total_loss, global_step=global_step)


with tf.name_scope('datas'):
    data_x, data_y = get_data.read_and_decode("data/train.tfrecords", int(training_iter * batch_size / train_data_size) + 2)
    test_x, test_y = get_data.read_and_decode("data/test.tfrecords", int(training_iter * batch_size * test_batches / test_data_size / test_freq) + 2)

    x_batch, y_batch = tf.train.shuffle_batch([data_x, data_y],
                                                batch_size=batch_size, capacity=1500,
                                                min_after_dequeue=1200, num_threads=3)
    test_x_batch, test_y_batch = tf.train.shuffle_batch([test_x, test_y],
                                                batch_size=batch_size, capacity=1500,
                                                min_after_dequeue=1200)
'''
with tf.variable_scope('SpatialTransformer', reuse=True):
    with tf.variable_scope('_transform', reuse=True):
        t_bf = tf.get_variable('before:0')
        t_af = tf.get_variable('after:0', [batch_size, 3, 3])
'''
merged = tf.summary.merge_all()
test_merged = tf.summary.merge_all("test")
saver = tf.train.Saver()
sv = tf.train.Supervisor(logdir='log', save_summaries_secs=0, saver=None)
with sv.managed_session(config=tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95))) as sess:
    #sess = tf.Session()
    #sess.run(tf.initialize_all_variables())
    #sess.run(tf.initialize_local_variables())
    threads = tf.train.start_queue_runners(sess=sess)
    #writer = tf.summary.FileWriter("log", sess.graph)

    time_start = time.time()
    tot_time = 0

    for i in range(training_iter):
        batch_xs, batch_ys = sess.run([x_batch, y_batch])
        if (i > no_theta_iter):
            use_theta = 0
        else:
            use_theta = 1
        if i % disp_freq == 0:
            print('time:' + str(tot_time) + 's')
            tot_time = 0
            time_start = time.time()
            t_loss, loss, summary = sess.run([theta_loss, total_loss, merged],
                            feed_dict={
                                x_tensor: batch_xs,
                                y: batch_ys,
                                use_theta_loss: use_theta
                            })
            #writer.add_summary(summary, i)
            sv.summary_writer.add_summary(summary, i)
            print('Iteration: ' + str(i) + ' Loss: ' + str(loss) + ' ThetaLoss: ' + str(t_loss*theta_mul))
            lr = sess.run(learning_rate)
            print(lr)
            time_end = time.time()
            print('disp time:' + str(time_end - time_start) + 's')
        if i % test_freq == 0:
            sum_test_loss = 0.0
            for j in range(test_batches):
                test_batch_xs, test_batch_ys = sess.run([test_x_batch, test_y_batch])
                loss = sess.run(total_loss,
                            feed_dict={
                                x_tensor: test_batch_xs,
                                y: test_batch_ys,
                                use_theta_loss: use_theta
                            })
                sum_test_loss += loss
            sum_test_loss /= test_batches
            print("Test Loss: " + str(sum_test_loss))
            summary = sess.run(test_merged,
                    feed_dict={
                        loss_displayer: sum_test_loss
                    })
            #writer.add_summary(summary, i)
            sv.summary_writer.add_summary(summary, i)
        if i % save_freq == 0:
            saver.save(sess, 'models/model', global_step=i)
        time_end = time.time()
        tot_time += time_end - time_start
        sess.run(optimizer, feed_dict={
            x_tensor: batch_xs, y: batch_ys, use_theta_loss: use_theta})   
        time_start = time.time()


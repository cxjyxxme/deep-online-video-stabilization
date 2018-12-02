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

# %% Graph representation of our network

# %% Placeholders for 40x40 resolution
with tf.name_scope('input'):
	x = tf.placeholder(tf.float32, [None, 1600])
	# %% Since x is currently [batch, height*width], we need to reshape to a
	# 4-D tensor to use it in a convolutional graph.  If one component of
	# `shape` is the special value -1, the size of that dimension is
	# computed so that the total size remains constant.  Since we haven't
	# defined the batch dimension's shape yet, we use -1 to denote this
	# dimension should not change size.
	x_tensor = tf.reshape(x, [-1, 40, 40, 1])
	x_tensor2 = tf.slice(x_tensor, [0, 10, 10, 0], [-1, 20, 20, -1])

with tf.name_scope('output'):
	y = tf.placeholder(tf.float32, [None, 10])

with tf.name_scope('loc1'):
	# %% We'll setup the two-layer localisation network to figure out the
	# %% parameters for an affine transformation of the input
	# %% Create variables for fully connected layer
	W_fc_loc1 = weight_variable([1600, 20])
	b_fc_loc1 = bias_variable([20])
	# %% Define the two layer localisation network
	h_fc_loc1 = tf.nn.tanh(tf.matmul(x, W_fc_loc1) + b_fc_loc1)
	# %% We can add dropout for regularizing and to reduce overfitting like so:
	keep_prob = tf.placeholder(tf.float32)
	h_fc_loc1_drop = tf.nn.dropout(h_fc_loc1, keep_prob)

with tf.name_scope('loc2'):
	W_fc_loc2 = weight_variable([20, 9])
	# Use identity transformation as starting point
	initial = np.array([[1., 0, 0], [0, 1., 0], [0, 0, 1.]])
	initial = initial.astype('float32')
	initial = initial.flatten()
	b_fc_loc2 = tf.Variable(initial_value=initial, name='b_fc_loc2')

	# %% Second layer
	h_fc_loc2 = tf.nn.tanh(tf.matmul(h_fc_loc1_drop, W_fc_loc2) + b_fc_loc2)

# %% We'll create a spatial transformer module to identify discriminative
# %% patches
out_size = (20, 20)
h_trans = transformer(x_tensor, h_fc_loc2, out_size)
cross_entropy = tf.nn.l2_loss(h_trans - x_tensor2)

# %% Define loss/eval/training functions
#cross_entropy = tf.reduce_mean(
#    tf.nn.softmax_cross_entropy_with_logits(logits = y_logits, labels = y))
opt = tf.train.AdamOptimizer()
optimizer = opt.minimize(cross_entropy)
grads = opt.compute_gradients(cross_entropy, [b_fc_loc2])

# %% Monitor accuracy
#correct_prediction = tf.equal(tf.argmax(y_logits, 1), tf.argmax(y, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

# %% We now create a new session to actually perform the initialization the
# variables:
sess = tf.Session()
sess.run(tf.initialize_all_variables())


# %% We'll now train in minibatches and report accuracy, loss:
iter_per_epoch = 100
n_epochs = 500
train_size = 10000

indices = np.linspace(0, 10000 - 1, iter_per_epoch)
indices = indices.astype('int')

writer = tf.summary.FileWriter("log", sess.graph)

for epoch_i in range(n_epochs):
    for iter_i in range(iter_per_epoch - 1):
        batch_xs = X_train[indices[iter_i]:indices[iter_i+1]]
        batch_ys = Y_train[indices[iter_i]:indices[iter_i+1]]

        if iter_i % 10 == 0:
            loss = sess.run(cross_entropy,
                            feed_dict={
                                x: batch_xs,
                                y: batch_ys,
                                keep_prob: 1.0
                            })
            print('Iteration: ' + str(iter_i) + ' Loss: ' + str(loss))

        sess.run(optimizer, feed_dict={
            x: batch_xs, y: batch_ys, keep_prob: 0.8})

        if iter_i % 100 == 1:
		ans = sess.run(h_trans, feed_dict={
		    x: batch_xs, y: batch_ys, keep_prob: 0.8})
		print(batch_xs.shape)
		temp = batch_xs[1, :]
		temp = np.reshape(temp, [40, 40, 1])
		cv2.imwrite('data/out.jpg', ans[1, :, :, :] * 255)
		cv2.imwrite('data/in.jpg', temp * 255)
    # theta = sess.run(h_fc_loc2, feed_dict={
    #        x: batch_xs, keep_prob: 1.0})
    # print(theta[0])

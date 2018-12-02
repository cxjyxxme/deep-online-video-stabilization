# Coder: Wenxin Xu
# Github: https://github.com/wenxinxu/resnet_in_tensorflow
# ==============================================================================
'''
This is the resnet structure
'''
import numpy as np
from hyper_parameters import *


BN_EPSILON = 0.001

def activation_summary(x):
    '''
    :param x: A Tensor
    :return: Add histogram summary and scalar summary of the sparsity of the tensor
    '''
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
    '''
    :param name: A string. The name of the new variable
    :param shape: A list of dimensions
    :param initializer: User Xavier as default.
    :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
    layers.
    :return: The created variable
    '''
    
    ## TODO: to allow different weight decay to fully connected layer and conv layer
    if is_fc_layer is True:
        regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay)
    else:
        regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay)

    new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                    regularizer=regularizer)
    return new_variables


def output_layer(input_layer, num_labels):
    '''
    :param input_layer: 2D tensor
    :param num_labels: int. How many output labels in total? (10 for cifar10 and 100 for cifar100)
    :return: output layer Y = WX + B
    '''
    input_dim = input_layer.get_shape().as_list()[-1]
    fc_w = create_variables(name='fc_weights', shape=[input_dim, num_labels], is_fc_layer=True,
                            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    fc_b = create_variables(name='fc_bias', shape=[num_labels], initializer=tf.zeros_initializer())

    fc_h = tf.matmul(input_layer, fc_w) + fc_b
    return fc_h


def batch_normalization_layer(input_layer, dimension):
    #return input_layer
    '''
    Helper function to do batch normalziation
    :param input_layer: 4D tensor
    :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
    :return: the 4D tensor after being normalized
    '''
    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    beta = tf.get_variable('beta', dimension, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable('gamma', dimension, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
    bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)

    return bn_layer


def conv_bn_relu_layer(input_layer, filter_shape, stride):
    '''
    A helper function to conv, batch normalize and relu the input tensor sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
    '''

    out_channel = filter_shape[-1]
    filter = create_variables(name='conv', shape=filter_shape)

    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    bn_layer = batch_normalization_layer(conv_layer, out_channel)

    output = tf.nn.relu(bn_layer)
    return output

def conv_bn_relu_layer2(input_layer, filter_shape, stride):
    '''
    A helper function to conv, batch normalize and relu the input tensor sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
    '''

    out_channel = filter_shape[-1]
    filter = create_variables(name='conv', shape=filter_shape)

    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride[0], stride[1], 1], padding='VALID')
    bn_layer = batch_normalization_layer(conv_layer, out_channel)

    output = tf.nn.relu(bn_layer)
    return output

def bn_relu_conv_layer(input_layer, filter_shape, stride):
    '''
    A helper function to batch normalize, relu and conv the input layer sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
    '''

    in_channel = input_layer.get_shape().as_list()[-1]

    bn_layer = batch_normalization_layer(input_layer, in_channel)
    relu_layer = tf.nn.relu(bn_layer)

    filter = create_variables(name='conv', shape=filter_shape)
    conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    return conv_layer


def residual_block(input_layer, params, first_block=False, inc_block = False):
    '''
    Defines a residual block in ResNet
    :param input_layer: 4D tensor
    :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
    :param first_block: if this is the first residual block of the whole network
    :return: 4D tensor.
    '''
    input_channel = input_layer.get_shape().as_list()[-1]

    convs = []
    for i in range(len(params['kernel_sizes'])):
        with tf.variable_scope('conv%d_in_block' % i):
            if (i == 0):
                stride = 1
                if ((inc_block) and (not first_block)):
                    stride = 2
                conv = conv_bn_relu_layer(input_layer, [params['kernel_sizes'][i], params['kernel_sizes'][i], input_channel, params['channel_sizes'][i]], stride)
            else:
                conv = conv_bn_relu_layer(convs[-1], [params['kernel_sizes'][i], params['kernel_sizes'][i], params['channel_sizes'][i - 1], params['channel_sizes'][i]], 1)
            convs.append(conv)
    if inc_block is True:
        stride = 1
        if (not first_block):
            stride = 2
        pooled_input = conv_bn_relu_layer(input_layer, [1, 1, input_channel, params['channel_sizes'][len(params['kernel_sizes']) - 1]], stride)
    else:
        pooled_input = input_layer

    output = convs[-1] + pooled_input
    return output


def inference(input_tensor_batch, input_channel, params):
    '''
    The main function that defines the ResNet. total layers = 1 + 2n + 2n + 2n +1 = 6n + 2
    :param input_tensor_batch: 4D tensor
    :param n: num_residual_blocks
    :param reuse: To build train graph, reuse=False. To build validation graph and share weights
    with train graph, resue=True
    :return: last layer in the network. Not softmax-ed
    '''

    layers = []
    with tf.variable_scope('conv0'):
        conv0 = conv_bn_relu_layer(input_tensor_batch, [7, 7, input_channel, 64], 2)
        activation_summary(conv0)
        layers.append(conv0)

    with tf.variable_scope('down_pooling'):
        down_pooling = tf.nn.max_pool(layers[-1], [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')
        layers.append(down_pooling)	

    for k in range(len(params['stage_sizes'])):
        with tf.variable_scope('stage_%d' %k):
            for i in range(params['stage_sizes'][k]):
                with tf.variable_scope('block_%d' %i):
                    first_block = (k == 0) and (i == 0)
                    inc_block = (i == 0)
                    conv = residual_block(layers[-1], params['channel_params'][k], first_block = first_block, inc_block = inc_block)
                    activation_summary(conv)
                    layers.append(conv)
    return layers[-1]


def test_graph(train_dir='logs'):
    '''
    Run this function to look at the graph structure on tensorboard. A fast way!
    :param train_dir:
    '''
    input_tensor = tf.constant(np.ones([128, 224, 224, 3]), dtype=tf.float32)
    config = {'stage_sizes' : [8, 8, 24], 'channel_params' : [  {'kernel_sizes':[1, 3, 1], 'channel_sizes':[64, 64, 256]}, 
                                                                {'kernel_sizes':[1, 3, 1], 'channel_sizes':[128, 128, 512]}, 
                                                                {'kernel_sizes':[1, 3, 1], 'channel_sizes':[256, 256, 1024]}]}
    result = inference(input_tensor, 3, config)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

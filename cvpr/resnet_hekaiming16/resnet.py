# Minhui huang
# Reproducing ResNet --cvpr16


import numpy as np
from hyper_parameters import *


BN_EPSILON = 0.001

def activation_summary(x):
    '''
    :param x: a tensor
    :return: add histogram summary and scalar summary of the sparsity of  the tensor
    '''

    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
    '''
    :param name: name of the new variable
    :param shape: dims of the new variable
    :param initializer: xavier as default
    :param is_fc_layer: if the variable is a fc layer's params
    :return: the created variable
    '''

    regularizer = tf.contrib.layers.l2_regularizer(scale=FLAG.weight_decay)
    new_variable = tf.get_variable(name, shape=shape, initializer=initializer, regularizer=regularizer)

    return new_variable


def output_layer(input_layer, num_labels):
    '''
    :param input_layer: 2D tensor
    :param num_labels: int number of total labels
    :return: output Y = WX + B
    '''
    input_dim = input_layer.get_shape().as_list[-1]
    fc_w = create_variables(name='fc_weights', shape=[input_dim, num_labels], is_fc_layer=True,
                            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    fc_b = create_variables(name='fc_bias', shape=[num_labels], initializer=tf.zeros_initializer())

    fc_h = tf.matmul(input_layer, fc_w) + fc_b

    return fc_h


def batch_nomalization_layer(input_layer, dimensions):
    '''

    :param input_layer: 4D tensor
    :param dimensions: The depth of the 4D tensor
    :return: the nomalized 4D tensor
    '''
    mean, variance = tf.nn.moments(input_layer, axes=[0,1,2])
    beta = tf.get_variable(name='bata', dimensions, tf.float32, initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable(name='gamma', dimensions, tf.float32, initializer=tf.constant_initializer(1.0, tf.float32))

    bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)

    return bn_layer


def conv_bn_relu_layer(input_layer, filter_shape, stride):
    '''
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
    '''

    out_channel = filter_shape[-1]
    filter = create_variables(name='conv', shape=filter_shape)

    conv = tf.nn.conv_2d(input_layer, filter, stride=[1,stride, stride, 1], padding='SAME')
    bn_layer = batch_nomalization_layer(conv, out_channel)

    output = tf.nn.relu(bn_layer)
    return output


def bn_relu_conv_layer(input_layer, filter_shape, stride):
    '''
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
    '''

    in_channel = input_layer.get_shape().as_list()[-1]

    bn_layer = batch_nomalization_layer(input_layer, in_channel)
    relu_layer = tf.nn.relu(bn_layer)

    filter = create_variables(name='conv', shape=filter_shape)
    conv = tf.nn.conv_2d(relu_layer, filter, stride=[1,stride, stride, 1], padding='SAME')

    return conv

def residual_block(input_layer, output_channel, first_block=False):
    '''

    :param input_layer: 4D tensor
    :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
    :param first_block: if it is the first block of resnet
    :return: 4D tensor
    '''

    input_channel = input_layer.get_shape().as_list()[-1]

    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')


    with tf.variable_scope('conv1_in_block'):
        if first_block:
            filter = create_variables(name='conv', shape=[3,3,input_channel,output_channel])
            conv1 = tf.nn.conv_2d(input_layer, filter=filter, stride=[1,1,1,1], padding='SAME')
        else:
            conv1 = bn_relu_conv_layer(input_layer, [3,3,input_channel, output_channel], stride)

    with tf.variable_scope('conv2_in_block'):
        conv2 = bn_relu_conv_layer(conv1, [3,3,output_channel, output_channel], 1)

    if increase_dim:
        
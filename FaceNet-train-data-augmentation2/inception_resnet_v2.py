# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Contains the definition of the Inception Resnet V2 architecture.
As described in http://arxiv.org/abs/1602.07261.
  Inception-v4, Inception-ResNet and the Impact of Residual Connections
    on Learning
  Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow
if tensorflow.__version__.startswith('1.'):
    import tensorflow as tf
    import tensorflow.contrib.slim as slim
else:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    import tf_slim as slim

#Inception-Resnet-A
def tf_block35(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 35x35 resnet block."""
    with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            # tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
            tower_conv = conv2d(net,filters=32,k_size=1)
        with tf.variable_scope('Branch_1'):
            # tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv1_0 = conv2d(net,filters=32,k_size=1)
            # tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
            tower_conv1_1 = conv2d(tower_conv1_0,filters=32,k_size=3)
        with tf.variable_scope('Branch_2'):
            # tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv2_0 = conv2d(net,filters=32,k_size=1)
            # tower_conv2_1 = slim.conv2d(tower_conv2_0, 48, 3, scope='Conv2d_0b_3x3')
            tower_conv2_1 = conv2d(tower_conv2_0,filters=48,k_size=3)
            # tower_conv2_2 = slim.conv2d(tower_conv2_1, 64, 3, scope='Conv2d_0c_3x3')
            tower_conv2_2 = conv2d(tower_conv2_1,filters=64,k_size=3)
        mixed = tf.concat([tower_conv, tower_conv1_1, tower_conv2_2], 3)
        # up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,activation_fn=None, scope='Conv2d_1x1')
        up = conv2d(mixed,filters=net.get_shape()[3],k_size=1,activation=None)

        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net
def block35(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 35x35 resnet block."""
    with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
            tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv2_1 = slim.conv2d(tower_conv2_0, 48, 3, scope='Conv2d_0b_3x3')
            tower_conv2_2 = slim.conv2d(tower_conv2_1, 64, 3, scope='Conv2d_0c_3x3')
        mixed = tf.concat([tower_conv, tower_conv1_1, tower_conv2_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net

# Inception-Resnet-B
def tf_block17(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 17x17 resnet block."""
    with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            # tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
            tower_conv = conv2d(net,filters=192,k_size=1)
        with tf.variable_scope('Branch_1'):
            # tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
            tower_conv1_0 = conv2d(net,filters=128,k_size=1)
            # tower_conv1_1 = slim.conv2d(tower_conv1_0, 160, [1, 7], scope='Conv2d_0b_1x7')
            tower_conv1_1 = conv2d(tower_conv1_0,filters=160,k_size=[1,7])

            # tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [7, 1],scope='Conv2d_0c_7x1')
            tower_conv1_2 = conv2d(tower_conv1_1,filters=192,k_size=[7,1])

        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        # up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,activation_fn=None, scope='Conv2d_1x1')
        up = conv2d(mixed, filters=net.get_shape()[3], k_size=1, activation=None)


        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net

def block17(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 17x17 resnet block."""
    with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 160, [1, 7],
                                        scope='Conv2d_0b_1x7')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [7, 1],
                                        scope='Conv2d_0c_7x1')
        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net


# Inception-Resnet-C
def tf_block8(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 8x8 resnet block."""
    with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            # tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
            tower_conv = conv2d(net, filters=192, k_size=1)
        with tf.variable_scope('Branch_1'):
            # tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
            tower_conv1_0 = conv2d(net,filters=192,k_size=1)
            # tower_conv1_1 = slim.conv2d(tower_conv1_0, 224, [1, 3],scope='Conv2d_0b_1x3')
            tower_conv1_1 = conv2d(tower_conv1_0,filters=224,k_size=[1, 3])

            # tower_conv1_2 = slim.conv2d(tower_conv1_1, 256, [3, 1],scope='Conv2d_0c_3x1')
            tower_conv1_2 = conv2d(tower_conv1_1,filters=256,k_size=[3, 1])

        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        # up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,activation_fn=None, scope='Conv2d_1x1')
        up = conv2d(mixed, filters=net.get_shape()[3], k_size=1, activation=None)

        net += scale * up
        if activation_fn is not None:
            net = activation_fn(net)
    return net
def block8(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 8x8 resnet block."""
    with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 224, [1, 3],
                                        scope='Conv2d_0b_1x3')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 256, [3, 1],
                                        scope='Conv2d_0c_3x1')
        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn is not None:
            net = activation_fn(net)
    return net
def conv2d(tf_input, filters=32, k_size=3, stride=1, padding='same',activation=tf.nn.relu):
    net = tf.layers.conv2d(
        inputs=tf_input,
        filters=filters,
        kernel_size=k_size,
        strides=stride,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0),
        padding=padding,
        activation=activation)
    return net
def inception_resnet_v2_Johnny(inputs,bottleneck_layer_size=128):

    # 149 x 149 x 32
    net = conv2d(inputs, filters=32, k_size=3, stride=2, padding='valid')
    # 147 x 147 x 32
    net = conv2d(net, filters=32, k_size=3, stride=1, padding='valid')
    # 147 x 147 x 64
    net = conv2d(net, filters=64, k_size=3)
    # 73 x 73 x 64
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[3, 3], strides=2)
    # 73 x 73 x 80
    net = conv2d(net, filters=80, k_size=1, padding='valid')
    # 71 x 71 x 192
    net = conv2d(net, filters=192, k_size=3, padding='valid')
    # 35 x 35 x 192
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[3, 3], strides=2)
    # 35 x 35 x 320
    with tf.variable_scope('Mixed_5b'):
        with tf.variable_scope('Branch_0'):
            # tower_conv = slim.conv2d(net, 96, 1, scope='Conv2d_1x1')
            tower_conv = conv2d(net, filters=96, k_size=1)
        with tf.variable_scope('Branch_1'):
            # tower_conv1_0 = slim.conv2d(net, 48, 1, scope='Conv2d_0a_1x1')
            tower_conv1_0 = conv2d(net, filters=48, k_size=1)
            # tower_conv1_1 = slim.conv2d(tower_conv1_0, 64, 5,scope='Conv2d_0b_5x5')
            tower_conv1_1 = conv2d(tower_conv1_0, filters=64, k_size=5)

        with tf.variable_scope('Branch_2'):
            # tower_conv2_0 = slim.conv2d(net, 64, 1, scope='Conv2d_0a_1x1')
            tower_conv2_0 = conv2d(net, filters=64, k_size=1)
            # tower_conv2_1 = slim.conv2d(tower_conv2_0, 96, 3,scope='Conv2d_0b_3x3')
            tower_conv2_1 = conv2d(tower_conv2_0, filters=96, k_size=3)

            # tower_conv2_2 = slim.conv2d(tower_conv2_1, 96, 3,scope='Conv2d_0c_3x3')
            tower_conv2_2 = conv2d(tower_conv2_1, filters=96, k_size=3)

        with tf.variable_scope('Branch_3'):
            # tower_pool = slim.avg_pool2d(net, 3, stride=1, padding='SAME',scope='AvgPool_0a_3x3')
            tower_pool = tf.layers.average_pooling2d(inputs=net, pool_size=[3, 3], strides=1,padding='same')

            # tower_pool_1 = slim.conv2d(tower_pool, 64, 1,scope='Conv2d_0b_1x1')
            tower_pool_1 = conv2d(tower_pool, filters=64, k_size=1)

        net = tf.concat([tower_conv, tower_conv1_1,tower_conv2_2, tower_pool_1], 3)


    # net = slim.repeat(net, 10, block35, scale=0.17)
    for i in range(10):
        net = tf_block35(net,scale=0.17)

    # 17 x 17 x 1024
    with tf.variable_scope('Mixed_6a'):
        with tf.variable_scope('Branch_0'):
            # tower_conv = slim.conv2d(net, 384, 3, stride=2, padding='VALID',scope='Conv2d_1a_3x3')
            tower_conv = conv2d(net,filters=384,k_size=3,stride=2,padding='valid')

        with tf.variable_scope('Branch_1'):
            # tower_conv1_0 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
            tower_conv1_0 = conv2d(net,filters=256,k_size=1)
            # tower_conv1_1 = slim.conv2d(tower_conv1_0, 256, 3,scope='Conv2d_0b_3x3')
            tower_conv1_1 = conv2d(tower_conv1_0,filters=256,k_size=3)

            # tower_conv1_2 = slim.conv2d(tower_conv1_1, 384, 3,stride=2, padding='VALID',scope='Conv2d_1a_3x3')
            tower_conv1_2 = conv2d(tower_conv1_1,filters=384,k_size=3,stride=2,padding='valid')


        with tf.variable_scope('Branch_2'):
            # tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',scope='MaxPool_1a_3x3')
            tower_pool = tf.layers.max_pooling2d(inputs=net, pool_size=[3, 3], strides=2,padding='valid')

        net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)

    # net = slim.repeat(net, 20, block17, scale=0.10)
    for i in range(20):
        net = tf_block17(net,scale=0.1)

    with tf.variable_scope('Mixed_7a'):
        with tf.variable_scope('Branch_0'):
            # tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
            tower_conv = conv2d(net,filters=256,k_size=1)
            # tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2,padding='VALID', scope='Conv2d_1a_3x3')
            tower_conv_1 = conv2d(tower_conv,filters=384,k_size=3,stride=2,padding='valid')

        with tf.variable_scope('Branch_1'):
            # tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
            tower_conv1 = conv2d(net,filters=256,k_size=1)
            # tower_conv1_1 = slim.conv2d(tower_conv1, 288, 3, stride=2,padding='VALID', scope='Conv2d_1a_3x3')

            tower_conv1_1 = conv2d(tower_conv1,filters=288,k_size=3,stride=2,padding='valid')
        with tf.variable_scope('Branch_2'):
            # tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
            tower_conv2 = conv2d(net,filters=256,k_size=1)
            # tower_conv2_1 = slim.conv2d(tower_conv2, 288, 3,scope='Conv2d_0b_3x3')
            tower_conv2_1 = conv2d(tower_conv2,filters=288,k_size=3)

            # tower_conv2_2 = slim.conv2d(tower_conv2_1, 320, 3, stride=2,padding='VALID', scope='Conv2d_1a_3x3')
            tower_conv2_2 = conv2d(tower_conv2_1,filters=320,k_size=3,stride=2,padding='valid')

        with tf.variable_scope('Branch_3'):
            # tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',scope='MaxPool_1a_3x3')
            tower_pool = tf.layers.max_pooling2d(inputs=net, pool_size=[3, 3], strides=2,padding='valid')

        net = tf.concat([tower_conv_1, tower_conv1_1,tower_conv2_2, tower_pool], 3)

    # net = slim.repeat(net, 9, block8, scale=0.20)
    for i in range(9):
        net = tf_block8(net,scale=0.2)

    net = block8(net, activation_fn=None)

    net = conv2d(net, filters=1536, k_size=1)

    with tf.variable_scope('Logits'):
        # end_points['PrePool'] = net
        # pylint: disable=no-member
        # net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',scope='AvgPool_1a_8x8')
        net = tf.layers.max_pooling2d(inputs=net, pool_size=net.get_shape()[1:3], strides=2,padding='valid')

        # net = slim.flatten(net)
        net = tf.layers.flatten(net)

        #net = slim.dropout(net, dropout_keep_prob, is_training=False,scope='Dropout')
        #end_points['PreLogitsFlatten'] = net

        # net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None,
        #                            scope='Bottleneck', reuse=False)
        net = tf.layers.dense(inputs=net, units=bottleneck_layer_size, activation=None)

        return net

def inference_2(inputs, bottleneck_layer_size=128, weight_decay=0.0, reuse=None):

    #----tf.int32 to tf.bool
    #new_phase_train = tf.cast(phase_train,tf.bool)

    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
}
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=slim.initializers.xavier_initializer(),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        # return inception_resnet_v2(images, phase_train,
        #       dropout_keep_prob=keep_probability, bottleneck_layer_size=bottleneck_layer_size, reuse=reuse)
        end_points = {}

        # phase_train_tranform = tf.where(phase_train == 0,False,True)
        #is_training = tf.cond(tf.greater(phase_train, 0), lambda: True, lambda: False)
        scope = 'InceptionResnetV2'
        with tf.variable_scope(scope, 'InceptionResnetV2', [inputs], reuse=reuse):
            with slim.arg_scope([slim.batch_norm, slim.dropout]):

                with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                    stride=1, padding='SAME'):
                    # 149 x 149 x 32
                    net = slim.conv2d(inputs, 32, 3, stride=2, padding='VALID',
                                      scope='Conv2d_1a_3x3')
                    end_points['Conv2d_1a_3x3'] = net
                    # 147 x 147 x 32
                    net = slim.conv2d(net, 32, 3, padding='VALID',
                                      scope='Conv2d_2a_3x3')
                    end_points['Conv2d_2a_3x3'] = net
                    # 147 x 147 x 64
                    net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
                    end_points['Conv2d_2b_3x3'] = net
                    # 73 x 73 x 64
                    net = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                          scope='MaxPool_3a_3x3')
                    end_points['MaxPool_3a_3x3'] = net
                    # 73 x 73 x 80
                    net = slim.conv2d(net, 80, 1, padding='VALID',
                                      scope='Conv2d_3b_1x1')
                    end_points['Conv2d_3b_1x1'] = net
                    # 71 x 71 x 192
                    net = slim.conv2d(net, 192, 3, padding='VALID',
                                      scope='Conv2d_4a_3x3')
                    end_points['Conv2d_4a_3x3'] = net
                    # 35 x 35 x 192
                    net = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                          scope='MaxPool_5a_3x3')
                    end_points['MaxPool_5a_3x3'] = net

                    # 35 x 35 x 320
                    with tf.variable_scope('Mixed_5b'):
                        with tf.variable_scope('Branch_0'):
                            tower_conv = slim.conv2d(net, 96, 1, scope='Conv2d_1x1')
                        with tf.variable_scope('Branch_1'):
                            tower_conv1_0 = slim.conv2d(net, 48, 1, scope='Conv2d_0a_1x1')
                            tower_conv1_1 = slim.conv2d(tower_conv1_0, 64, 5,
                                                        scope='Conv2d_0b_5x5')
                        with tf.variable_scope('Branch_2'):
                            tower_conv2_0 = slim.conv2d(net, 64, 1, scope='Conv2d_0a_1x1')
                            tower_conv2_1 = slim.conv2d(tower_conv2_0, 96, 3,
                                                        scope='Conv2d_0b_3x3')
                            tower_conv2_2 = slim.conv2d(tower_conv2_1, 96, 3,
                                                        scope='Conv2d_0c_3x3')
                        with tf.variable_scope('Branch_3'):
                            tower_pool = slim.avg_pool2d(net, 3, stride=1, padding='SAME',
                                                         scope='AvgPool_0a_3x3')
                            tower_pool_1 = slim.conv2d(tower_pool, 64, 1,
                                                       scope='Conv2d_0b_1x1')
                        net = tf.concat([tower_conv, tower_conv1_1,
                                         tower_conv2_2, tower_pool_1], 3)

                    end_points['Mixed_5b'] = net
                    net = slim.repeat(net, 10, block35, scale=0.17)

                    # 17 x 17 x 1024
                    with tf.variable_scope('Mixed_6a'):
                        with tf.variable_scope('Branch_0'):
                            tower_conv = slim.conv2d(net, 384, 3, stride=2, padding='VALID',
                                                     scope='Conv2d_1a_3x3')
                        with tf.variable_scope('Branch_1'):
                            tower_conv1_0 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                            tower_conv1_1 = slim.conv2d(tower_conv1_0, 256, 3,
                                                        scope='Conv2d_0b_3x3')
                            tower_conv1_2 = slim.conv2d(tower_conv1_1, 384, 3,
                                                        stride=2, padding='VALID',
                                                        scope='Conv2d_1a_3x3')
                        with tf.variable_scope('Branch_2'):
                            tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                                         scope='MaxPool_1a_3x3')
                        net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)

                    end_points['Mixed_6a'] = net
                    net = slim.repeat(net, 20, block17, scale=0.10)

                    with tf.variable_scope('Mixed_7a'):
                        with tf.variable_scope('Branch_0'):
                            tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                            tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2,
                                                       padding='VALID', scope='Conv2d_1a_3x3')
                        with tf.variable_scope('Branch_1'):
                            tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                            tower_conv1_1 = slim.conv2d(tower_conv1, 288, 3, stride=2,
                                                        padding='VALID', scope='Conv2d_1a_3x3')
                        with tf.variable_scope('Branch_2'):
                            tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                            tower_conv2_1 = slim.conv2d(tower_conv2, 288, 3,
                                                        scope='Conv2d_0b_3x3')
                            tower_conv2_2 = slim.conv2d(tower_conv2_1, 320, 3, stride=2,
                                                        padding='VALID', scope='Conv2d_1a_3x3')
                        with tf.variable_scope('Branch_3'):
                            tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                                         scope='MaxPool_1a_3x3')
                        net = tf.concat([tower_conv_1, tower_conv1_1,
                                         tower_conv2_2, tower_pool], 3)

                    end_points['Mixed_7a'] = net

                    net = slim.repeat(net, 9, block8, scale=0.20)
                    net = block8(net, activation_fn=None)

                    net = slim.conv2d(net, 1536, 1, scope='Conv2d_7b_1x1')
                    end_points['Conv2d_7b_1x1'] = net

                    with tf.variable_scope('Logits'):
                        end_points['PrePool'] = net
                        # pylint: disable=no-member
                        net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                                              scope='AvgPool_1a_8x8')
                        net = slim.flatten(net)

                        # net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                        #                    scope='Dropout')

                        end_points['PreLogitsFlatten'] = net

                    net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None,
                                               scope='Bottleneck', reuse=False)

        return net

def inference(images, keep_probability, phase_train,
              bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
    #----tf.int32 to tf.bool
    #new_phase_train = tf.cast(phase_train,tf.bool)

    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
}
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=slim.initializers.xavier_initializer(), 
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        return inception_resnet_v2(images, phase_train,
              dropout_keep_prob=keep_probability, bottleneck_layer_size=bottleneck_layer_size, reuse=reuse)


def inception_resnet_v2(inputs, is_training,
                        dropout_keep_prob=0.8,
                        bottleneck_layer_size=128,
                        reuse=None,
                        scope='InceptionResnetV2'):
    """Creates the Inception Resnet V2 model.
    Args:
      inputs: a 4-D tensor of size [batch_size, height, width, 3].
      num_classes: number of predicted classes.
      is_training: whether is training or not.
      dropout_keep_prob: float, the fraction to keep before final layer.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.
    Returns:
      logits: the logits outputs of the model.
      end_points: the set of end_points from the inception model.
    """
    end_points = {}

    #phase_train_tranform = tf.where(phase_train == 0,False,True)
    # is_training = tf.cond(tf.greater(phase_train, 0), lambda: True, lambda: False)

  
    with tf.variable_scope(scope, 'InceptionResnetV2', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride=1, padding='SAME'):
      
                # 149 x 149 x 32
                net = slim.conv2d(inputs, 32, 3, stride=2, padding='VALID',
                                  scope='Conv2d_1a_3x3')
                end_points['Conv2d_1a_3x3'] = net
                # 147 x 147 x 32
                net = slim.conv2d(net, 32, 3, padding='VALID',
                                  scope='Conv2d_2a_3x3')
                end_points['Conv2d_2a_3x3'] = net
                # 147 x 147 x 64
                net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
                end_points['Conv2d_2b_3x3'] = net
                # 73 x 73 x 64
                net = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                      scope='MaxPool_3a_3x3')
                end_points['MaxPool_3a_3x3'] = net
                # 73 x 73 x 80
                net = slim.conv2d(net, 80, 1, padding='VALID',
                                  scope='Conv2d_3b_1x1')
                end_points['Conv2d_3b_1x1'] = net
                # 71 x 71 x 192
                net = slim.conv2d(net, 192, 3, padding='VALID',
                                  scope='Conv2d_4a_3x3')
                end_points['Conv2d_4a_3x3'] = net
                # 35 x 35 x 192
                net = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                      scope='MaxPool_5a_3x3')
                end_points['MaxPool_5a_3x3'] = net
        
                # 35 x 35 x 320
                with tf.variable_scope('Mixed_5b'):
                    with tf.variable_scope('Branch_0'):
                        tower_conv = slim.conv2d(net, 96, 1, scope='Conv2d_1x1')
                    with tf.variable_scope('Branch_1'):
                        tower_conv1_0 = slim.conv2d(net, 48, 1, scope='Conv2d_0a_1x1')
                        tower_conv1_1 = slim.conv2d(tower_conv1_0, 64, 5,
                                                    scope='Conv2d_0b_5x5')
                    with tf.variable_scope('Branch_2'):
                        tower_conv2_0 = slim.conv2d(net, 64, 1, scope='Conv2d_0a_1x1')
                        tower_conv2_1 = slim.conv2d(tower_conv2_0, 96, 3,
                                                    scope='Conv2d_0b_3x3')
                        tower_conv2_2 = slim.conv2d(tower_conv2_1, 96, 3,
                                                    scope='Conv2d_0c_3x3')
                    with tf.variable_scope('Branch_3'):
                        tower_pool = slim.avg_pool2d(net, 3, stride=1, padding='SAME',
                                                     scope='AvgPool_0a_3x3')
                        tower_pool_1 = slim.conv2d(tower_pool, 64, 1,
                                                   scope='Conv2d_0b_1x1')
                    net = tf.concat([tower_conv, tower_conv1_1,
                                        tower_conv2_2, tower_pool_1], 3)
        
                end_points['Mixed_5b'] = net
                net = slim.repeat(net, 10, block35, scale=0.17)
        
                # 17 x 17 x 1024
                with tf.variable_scope('Mixed_6a'):
                    with tf.variable_scope('Branch_0'):
                        tower_conv = slim.conv2d(net, 384, 3, stride=2, padding='VALID',
                                                 scope='Conv2d_1a_3x3')
                    with tf.variable_scope('Branch_1'):
                        tower_conv1_0 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                        tower_conv1_1 = slim.conv2d(tower_conv1_0, 256, 3,
                                                    scope='Conv2d_0b_3x3')
                        tower_conv1_2 = slim.conv2d(tower_conv1_1, 384, 3,
                                                    stride=2, padding='VALID',
                                                    scope='Conv2d_1a_3x3')
                    with tf.variable_scope('Branch_2'):
                        tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                                     scope='MaxPool_1a_3x3')
                    net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)
        
                end_points['Mixed_6a'] = net
                net = slim.repeat(net, 20, block17, scale=0.10)
        
                with tf.variable_scope('Mixed_7a'):
                    with tf.variable_scope('Branch_0'):
                        tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                        tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2,
                                                   padding='VALID', scope='Conv2d_1a_3x3')
                    with tf.variable_scope('Branch_1'):
                        tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                        tower_conv1_1 = slim.conv2d(tower_conv1, 288, 3, stride=2,
                                                    padding='VALID', scope='Conv2d_1a_3x3')
                    with tf.variable_scope('Branch_2'):
                        tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                        tower_conv2_1 = slim.conv2d(tower_conv2, 288, 3,
                                                    scope='Conv2d_0b_3x3')
                        tower_conv2_2 = slim.conv2d(tower_conv2_1, 320, 3, stride=2,
                                                    padding='VALID', scope='Conv2d_1a_3x3')
                    with tf.variable_scope('Branch_3'):
                        tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                                     scope='MaxPool_1a_3x3')
                    net = tf.concat([tower_conv_1, tower_conv1_1,
                                        tower_conv2_2, tower_pool], 3)
        
                end_points['Mixed_7a'] = net
        
                net = slim.repeat(net, 9, block8, scale=0.20)
                net = block8(net, activation_fn=None)
        
                net = slim.conv2d(net, 1536, 1, scope='Conv2d_7b_1x1')
                end_points['Conv2d_7b_1x1'] = net
        
                with tf.variable_scope('Logits'):
                    end_points['PrePool'] = net
                    #pylint: disable=no-member
                    net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                                          scope='AvgPool_1a_8x8')
                    net = slim.flatten(net)
          
                    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                       scope='Dropout')
          
                    end_points['PreLogitsFlatten'] = net
                
                net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None, 
                        scope='Bottleneck', reuse=False)
  
    return net

import tensorflow as tf
import numpy as np
import cv2
import time
import os
import sys

class mfnet_core(object):
    def __init__(self, nameScope='mfnet_core', trainable=True, bnPhase=True, reuse=False, activation = tf.nn.elu):
        self._reuse = reuse
        self._trainable = trainable
        self._bnPhase = bnPhase
        self._activation = activation
        self._nameScope = nameScope
        self.variables = None
        self.update_ops = None
        self.saver = None

        # print('init func')

    def _conv(self, inputs, filters, kernel_size, strides=1):
        hidden = tf.layers.conv2d(
            inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
            padding='same', activation=None, trainable=self._trainable, use_bias=False,
            reuse= False
        )
        hidden = tf.layers.batch_normalization(hidden)
        return hidden

    def _maxpool(self, inputs, pool_size=(2,2), strides=2, padding='same'):
        hidden = tf.layers.max_pooling2d(inputs=inputs, pool_size=pool_size, strides=strides, padding=padding)
        return hidden

    def __call__(self, InputImgs):
        # print self._nameScope

        with tf.variable_scope(self._nameScope, reuse=self._reuse):
            h1 = self._conv(inputs=InputImgs, filters=32, kernel_size=3)
            p1 = self._maxpool(inputs=h1)

            h2 = self._conv(inputs=p1, filters=64, kernel_size=3)
            p2 = self._maxpool(inputs=h2)

            h31 = self._conv(inputs=p2, filters=128, kernel_size=3)
            h32 = self._conv(inputs=h31, filters=64, kernel_size=1)
            h33 = self._conv(inputs=h32, filters=128, kernel_size=3)
            p3 = self._maxpool(inputs=h33)

            h41 = self._conv(inputs=p3, filters=256, kernel_size=3)
            h42 = self._conv(inputs=h41, filters=128, kernel_size=1)
            h43 = self._conv(inputs=h42, filters=256, kernel_size=3)
            p4 = self._maxpool(inputs=h43)

            h51 = self._conv(inputs=p4, filters=512, kernel_size=3)
            h52 = self._conv(inputs=h51, filters=256, kernel_size=1)
            h53 = self._conv(inputs=h52, filters=512, kernel_size=3)
            h54 = self._conv(inputs=h53, filters=256, kernel_size=1)
            h55 = self._conv(inputs=h54, filters=512, kernel_size=3)
            p5 = self._maxpool(inputs=h55)

            h61 = self._conv(inputs=p5, filters=1024, kernel_size=3)
            h62 = self._conv(inputs=h61, filters=512, kernel_size=1)
            h63 = self._conv(inputs=h62, filters=1024, kernel_size=3)
            h64 = self._conv(inputs=h63, filters=512, kernel_size=1)
            h65 = self._conv(inputs=h64, filters=1024, kernel_size=3)

        self._reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._nameScope)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self._nameScope)
        self.saver = tf.train.Saver(var_list=self.variables)
        outputs = h65

        return outputs

class MF_Classification(object):
    def __init__(self, outputDim, nameScope='mf_rpn', trainable=True,
                 bnPhase=True, reuse=False, coreActivation=tf.nn.leaky_relu,
                 lastLayerActivation=None,
                 lastLayerPooling=None):
        self._outputDim = outputDim
        self._nameScope = nameScope
        self._trainable = trainable
        self._bnPhase = bnPhase
        self._reuse = reuse
        self._coreActivation = coreActivation
        self._lastActivation = lastLayerActivation
        self._lastPool = lastLayerPooling
        self.variables = None
        self.update_ops = None
        self.saver = None
        self._mfnet_core = None

    def __call__(self, InputImgs):
        # print self._nameScope
        self._mfnet_core = mfnet_core(nameScope=self._nameScope+"_MFNetCore", trainable=self._trainable,
                                      bnPhase=self._bnPhase, reuse=self._reuse, activation=self._coreActivation)


class MF_Detection(object):
    def __init__(self, outputDim, nameScope='mf_rpn', trainable=True,
                 bnPhase=True, reuse=False, coreActivation=tf.nn.leaky_relu,
                 lastLayerActivation=None,
                 lastLayerPooling=None):
        self._outputDim = outputDim
        self._nameScope = nameScope
        self._trainable = trainable
        self._bnPhase = bnPhase
        self._reuse = reuse
        self._coreActivation = coreActivation
        self._lastActivation = lastLayerActivation
        self._lastPool = lastLayerPooling
        self.variables = None
        self.update_ops = None
        self.saver = None
        self._mfnet_core = None

        # print 'init'

    def __call__(self, InputImgs):
        # print self._nameScope
        self._mfnet_core = mfnet_core(nameScope=self._nameScope+"_MFCore", trainable=self._trainable,
                                      bnPhase=self._bnPhase, reuse=self._reuse, activation=self._coreActivation)

        hidden = self._mfnet_core(InputImgs)

        with tf.variable_scope(self._nameScope+'_Detection', reuse=self._reuse):
            h1 = tf.layers.conv2d(inputs=hidden, filters=1024, kernel_size=3, strides=1, padding='same', activation=None,
                                      trainable=self._trainable, use_bias=False)
            h2 = tf.layers.conv2d(inputs=h1, filters=1024, kernel_size=3, strides=1, padding='same', activation=None,
                                      trainable=self._trainable, use_bias=False)
            h3 = tf.layers.conv2d(inputs=h2, filters=1024, kernel_size=3, strides=1, padding='same', activation=None,
                                      trainable=self._trainable, use_bias=False)

            # print 'h3 shape is {}'.format(h3.shape)
            output = tf.layers.conv2d(inputs=h3, filters=self._outputDim, kernel_size=1, strides=1, padding='same',
                                      activation=None, trainable=self._trainable, use_bias=False)
            # print 'output shape is {}'.format(output.shape)

        self._reuse = True
        self.variables = [self._mfnet_core.variables, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._nameScope+"_Detection")]
        self.update_ops = [self._mfnet_core.update_ops, tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self._nameScope+"_Detection")]
        self.allVariables = self._mfnet_core.variables + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._nameScope+"_Detection")
        self.allUpdate_ops = self._mfnet_core.update_ops + tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self._nameScope+"_Detection")
        self.coreVariables = self._mfnet_core.variables
        self.detectorVariables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._nameScope+"_Detection")
        self.coreSaver = tf.train.Saver(var_list=self.coreVariables)
        self.detectorSaver = tf.train.Saver(var_list=self.detectorVariables)

        return output

# x = tf.placeholder(tf.float32, [None, 416, 416, 12], 'input')
#
# y = MF_Detection(30)
# y(x)


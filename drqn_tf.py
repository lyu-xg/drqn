import gym

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

class Qnetwork():
    def __init__(self, h_size, a_size, rnn_cell, scopeName):
        self.frameInput = tf.placeholder(shape=[None, 110, 84, 3], dtype=tf.float32)
        self.conv1 = slim.convolution2d(
            inputs=self.frameInput, num_outputs=32,
            kernel_size=(8, 8), stride=(4, 4), padding='VALID',
            biases_initializer=None, scope=scopeName+'_conv1'
        )
        self.conv2 = slim.convolution2d(
            inputs=self.conv1, num_outputs = 64,
            kernel_size=(4, 4), stride=(2, 2), padding='VALID',
            biases_initializer=None, scope=scopeName+'_conv2'
        )
        self.conv3 = slim.convolution2d(
            inputs=self.conv1, num_outputs = 64,
            kernel_size=(3, 3), stride=(1, 1), padding='VALID',
            biases_initializer=None, scope=scopeName+'_conv3'
        )
        
        # Length of the trace
        self.trainLength = tf.placeholder(dtype=tf.int32)
        
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])
        self.convFlat = tf.reshape(
            slim.flatten(self.conv3), [self.batch_size, self.trainLength, h_size])
        
        self.state_init = rnn_cell.zero_state(self.batch_size, tf.float32)
        self.rnn, self.rnn_state = tf.nn.dynamic_rnn(
            inputs=self.convFlat, cell=rnn_cell, dtype=tf.float32,
            initial_state=self.state_init, scope=scopeName+'_rnn'
        )
        self.rnn = tf.reshape(self.rnn, shape=[-1, h_size])
        
        self.streamA, self.streamV = tf.split(self.rnn, 2, axis=1)
        print(self.streamA.shape, self.streamV.shape)
        self.AW = tf.Variable(tf.random_normal([h_size//2, a_size]))
        self.VW = tf.Variable(tf.random_normal([h_size//2, 1]))
        self.A = tf.matmul(self.streamA, self.AW)
        self.V = tf.matmul(self.streamV, self.VW)
        
        self.salience = tf.gradients(self.A, self.frameInput)
        
        self.Qout = self.V + \
            (self.A - tf.reduce_mean(self.A, 
                                     reduction_indices=1, 
                                     keep_dims=True))
        self.predict = tf.argmax(self.Qout, 1)
        
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
        
        self.Q = tf.reduce_sum(self.Qout * self.actions_onehot,
                               reduction_indices=1)
        
        # only train on first half of trace per Lample & Chatlot 2016
        self.mask = tf.concat((tf.zeros((self.batch_size, self.trainLength//2)),
                               tf.ones((self.batch_size, self.trainLength//2))), 1)
        self.mask = tf.to_float(self.mask)
        
        self.loss = tf.losses.huber_loss(self.Q * self.mask,
                                         self.targetQ * self.mask, delta=1)
        
        
        
if __name__ == '__main__':
    h_size = 512
    a_size = 4
    q = Qnetwork(h_size,
                 a_size, 
                 tf.nn.rnn_cell.LSTMCell(num_units=h_size,state_is_tuple=True),
                 'main')
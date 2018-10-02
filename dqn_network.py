import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


class Qnetwork():
    def __init__(self, h_size, a_size, stack_size, scopeName):
        self.h_size, self.a_size, self.stack_size = h_size, a_size, stack_size
        self.scalarInput = tf.placeholder(shape=[None, 84*84], dtype=tf.float32)
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])
        self.trainLength = tf.placeholder(dtype=tf.int32, shape=[])

        self.frames = tf.reshape(self.scalarInput, [-1, 84, 84, stack_size])
        self.conv1 = slim.convolution2d(
            inputs=self.frames, num_outputs=32,
            kernel_size=(8, 8), stride=(4, 4), padding='VALID',
            biases_initializer=None, scope=scopeName+'_conv1'
        )
        self.conv2 = slim.convolution2d(
            inputs=self.conv1, num_outputs=64,
            kernel_size=(4, 4), stride=(2, 2), padding='VALID',
            biases_initializer=None, scope=scopeName+'_conv2'
        )
        self.conv3 = slim.convolution2d(
            inputs=self.conv2, num_outputs=64,
            kernel_size=(3, 3), stride=(1, 1), padding='VALID',
            biases_initializer=None, scope=scopeName+'_conv3'
        )
        self.conv4 = slim.convolution2d(
            inputs=self.conv3, num_outputs=h_size,
            kernel_size=(7, 7), stride=(1, 1), padding='VALID',
            biases_initializer=None, scope=scopeName+'_conv4'
        )
        
        self.convFlat = tf.reshape(
            slim.flatten(self.conv4), [self.batch_size, h_size])
        
        self.streamA, self.streamV = tf.split(self.convFlat, 2, axis=1)

        self.AW = tf.Variable(tf.random_normal([h_size//2, a_size]))
        self.VW = tf.Variable(tf.random_normal([h_size//2, 1]))
        self.A = tf.matmul(self.streamA, self.AW)
        self.V = tf.matmul(self.streamV, self.VW)

        self.salience = tf.gradients(self.A, self.scalarInput)

        self.Qout = self.V + \
            (self.A - tf.reduce_mean(self.A, axis=1, keepdims=True))
        self.predict = tf.argmax(self.Qout, 1)
        self.action = self.predict[-1]

        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(
            self.actions, a_size, dtype=tf.float32)

        self.Q = tf.reduce_sum(self.Qout * self.actions_onehot,
                               reduction_indices=1)

        self.loss = tf.losses.huber_loss(self.Q, self.targetQ)

        if scopeName == 'main':
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('Q', self.Qout)

        self.trainer = tf.train.RMSPropOptimizer(
            0.00025, momentum=0.95, epsilon=0.01)
        self.updateModel = self.trainer.minimize(self.loss)

    def get_action(self, sess, frames):
        return sess.run(self.action, feed_dict={
            self.scalarInput: np.vstack(np.array(frames)/255.0),
            self.batch_size: 1
        })

if __name__=='__main__':
    q = Qnetwork(512, 4, 7, 'main')
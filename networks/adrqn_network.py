import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


class Qnetwork():
    def __init__(self, h_size, a_size, action_h_size, rnn_cell, scopeName):
        self.h_size, self.a_size = h_size, a_size
        self.scalarInput = tf.placeholder(shape=[None, 7056], dtype=tf.float32, name='frameInput')
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name='batchSize')
        self.trainLength = tf.placeholder(dtype=tf.int32, shape=[], name='trainLength')

        self.actionsInput = tf.placeholder(shape=[None], dtype=tf.int32, name='actionsInput')
        self.actionsInputOnehot = tf.one_hot(self.actionsInput, a_size)
        self.actionsInputWeights = tf.Variable(tf.random_normal([a_size, action_h_size]))
        self.actionsInputProjected = tf.matmul(self.actionsInputOnehot, self.actionsInputWeights)
        

        self.frameShape = tf.constant((84, 84, 1), dtype=tf.int32)
#         self.frames = tf.reshape(self.scalarInput, tf.concat(([self.batch_size*self.trainLength], self.frameShape), 0))
        self.frames = tf.reshape(self.scalarInput, [-1, 84, 84, 1])
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

        # self.convFlat = tf.reshape(
        #     slim.flatten(self.conv4), [self.batch_size, self.trainLength, h_size])
        # print(self.convFlat.shape)
        
        
        self.rnnInput = tf.concat([
                tf.reshape(slim.flatten(self.conv4), [self.batch_size, self.trainLength, h_size]), 
                tf.reshape(self.actionsInputProjected, [self.batch_size, self.trainLength, action_h_size])],
            2)
        print(self.rnnInput.shape)

        self.state_init = rnn_cell.zero_state(self.batch_size, tf.float32)
        self.rnn, self.rnn_state = tf.nn.dynamic_rnn(
            inputs=self.rnnInput, cell=rnn_cell, dtype=tf.float32,
            initial_state=self.state_init, scope=scopeName+'_rnn'
        )
        self.rnn = tf.reshape(self.rnn, shape=[-1, h_size])

        self.streamA, self.streamV = tf.split(self.rnn, 2, axis=1)
        print(self.streamA.shape)

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

        # only train on first half of every trace per Lample & Chatlot 2016
        self.mask = tf.concat((tf.zeros((self.batch_size, self.trainLength//2)),
                               tf.ones((self.batch_size, self.trainLength//2))), 1)
        self.mask = tf.reshape(self.mask, [-1])

        self.loss = tf.losses.huber_loss(
            self.Q * self.mask, self.targetQ * self.mask)

        if scopeName == 'main':
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('Q', self.Qout)
            tf.summary.histogram('hidden', self.rnn_state)

        self.trainer = tf.train.RMSPropOptimizer(
            0.00025, momentum=0.95, epsilon=0.01)
        self.updateModel = self.trainer.minimize(self.loss)

    def get_action_and_next_state(self, sess, state, prev_actions, frames):
        state = state or (np.zeros((1, self.h_size)),) * 2
        return sess.run([self.action, self.rnn_state], feed_dict={
            self.scalarInput: np.vstack(np.array(frames)/255.0),
            self.actionsInput: prev_actions,
            self.trainLength: len(frames),
            self.state_init: state,
            self.batch_size: 1
        })

if __name__ == '__main__':
    q = Qnetwork(512, 4, 256, tf.nn.rnn_cell.LSTMCell(num_units=512), 'main')
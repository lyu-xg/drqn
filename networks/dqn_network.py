import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from common import getTargetUpdateOps

class Qnetwork():
    def __init__(self, h_size, a_size, stack_size, scopeName):
        self.h_size, self.a_size, self.stack_size = h_size, a_size, stack_size
        
        self.scalarInput = tf.placeholder(shape=[None, stack_size, 84*84], dtype=tf.uint8)
        self.frames = tf.reshape(self.scalarInput/255, [-1, stack_size, 84, 84])

        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])
        

        with tf.variable_scope(scopeName):
            self.construct_network()

        if scopeName == 'main':
            self.target_network = Qnetwork(h_size, a_size, stack_size, 'target')
            self.construct_target_and_loss()

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())

    def construct_network(self):
        # where all the variables resides
        self.conv1 = slim.convolution2d(
            inputs=self.frames, num_outputs=32,
            kernel_size=(8, 8), stride=(4, 4), padding='VALID',
            data_format='NCHW',
            biases_initializer=None
        )
        self.conv2 = slim.convolution2d(
            inputs=self.conv1, num_outputs=64,
            kernel_size=(4, 4), stride=(2, 2), padding='VALID',
            data_format='NCHW',
            biases_initializer=None
        )
        self.conv3 = slim.convolution2d(
            inputs=self.conv2, num_outputs=64,
            kernel_size=(3, 3), stride=(1, 1), padding='VALID',
            data_format='NCHW',
            biases_initializer=None
        )
        self.conv4 = slim.convolution2d(
            inputs=self.conv3, num_outputs=self.h_size,
            kernel_size=(7, 7), stride=(1, 1), padding='VALID',
            data_format='NCHW',
            biases_initializer=None
        )
        
        self.convFlat = tf.reshape(
            slim.flatten(self.conv4), [self.batch_size, self.h_size])
        
        self.streamA, self.streamV = tf.split(self.convFlat, 2, axis=1)

        xavier_init = tf.contrib.layers.xavier_initializer()
        self.AW = tf.Variable(xavier_init([self.h_size//2, self.a_size]))
        self.VW = tf.Variable(xavier_init([self.h_size//2, 1]))
        self.A = tf.matmul(self.streamA, self.AW)
        self.V = tf.matmul(self.streamV, self.VW)

        self.Qout = self.V + \
            (self.A - tf.reduce_mean(self.A, axis=1, keepdims=True))
        self.predict = tf.argmax(self.Qout, 1)
        self.action = self.predict[-1]

    def construct_target_and_loss(self):
        # where target and loss are computed
        # self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.transition_rewards = tf.placeholder(tf.float32, shape=[None])
        self.transition_terminals = tf.placeholder(tf.float32, shape=[None])
        self.transition_actions = tf.placeholder(tf.int32, shape=None)
        
        # in order to do a single pass, we set the first part of the batch
        # to be S, and the second half of the batch to be S_next
        self.Q_s, self.Q_s_next = tf.split(self.Qout, 2)
        
        # select action using main network, using Q values from target network
        self.targetQ = self.select_actions(self.target_network.Qout, tf.argmax(self.Q_s_next, 1))
        self.Q       = self.select_actions(self.Q_s,                 self.transition_actions)

        # here goes BELLMAN
        self.target = (self.transition_rewards + 
                       0.99 * self.targetQ * (- self.transition_terminals + 1))
    
        self.loss = tf.losses.huber_loss(self.Q, tf.stop_gradient(self.target))

        tf.summary.scalar('loss', self.loss)
        # tf.summary.histogram('Q', self.Q)
        # tf.summary.histogram('targetQ', self.targetQ)

        self.trainer = tf.train.RMSPropOptimizer(
            0.00025, momentum=0.95, epsilon=0.01)
        self.updateModel = self.trainer.minimize(self.loss)

    def get_action(self, frames):
        # frames.shape: [batch_size, stack_size, frame_shape]
        return self.sess.run(self.action, feed_dict={
            self.scalarInput: frames,
            self.batch_size: len(frames)
        })
    
    def one_hot(self, actions):
        return tf.one_hot(actions, self.a_size, on_value=1.0, off_value=0.0, dtype=tf.float32)

    def select_actions(self, Q_values, actions):
        # Q_values: (batch_size, a_size)
        # actions: (batch_size,)
        return tf.reduce_sum(Q_values * self.one_hot(actions), axis=1)

    def update_model(self, S, A, R, S_next, T, additional_ops=[]):
        return self.sess.run([self.updateModel] + additional_ops, feed_dict={
            self.target_network.scalarInput: S_next,
            self.target_network.batch_size: len(S),
            # SINGLE FORWARD PASS WITH DOUBLE THE BATCH SIZE
            self.scalarInput: np.concatenate([S, S_next]),
            self.batch_size: len(S) * 2,
            self.transition_actions: A,
            self.transition_rewards: R,
            self.transition_terminals: T
        })

    def update_target_network(self):
        updateOps = getTargetUpdateOps(tf.trainable_variables())

    # def __del__(self):
    #     self.sess.close()

if __name__=='__main__':
    q = Qnetwork(512, 4, 7, 'main')
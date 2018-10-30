import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim import convolution2d as conv2d
from tensorflow.contrib.slim import flatten

class Qnetwork():
    def __init__(self, h_size, a_size, stack_size, scopeName, model='dqn'):
        self.h_size, self.a_size, self.stack_size = \
            h_size, a_size, stack_size

        # expected inputs        
        self.frames = tf.placeholder(shape=[None, stack_size, 84, 84], dtype=tf.uint8)
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])

        self.ZERO_STATE = (np.zeros((1, self.h_size)),) * 2

        with tf.variable_scope(scopeName):
            try:
                eval('self.construct_{}()'.format(model))
            except AttributeError:
                print('model {} not implemented.'.format(model))
                raise SystemExit

        if scopeName == 'main':
            self.target_network = Qnetwork(h_size, a_size, stack_size, 'target',
                                           model=model)
            self.construct_target_and_loss()
            self.construct_target_update_ops()

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())

    def __del__(self):
        if hasattr(self, 'sess'):
            self.sess.close()

    ###########################################################################
    # Top Level TensorFlow Graph Constructors
    ###########################################################################
    def construct_dqn(self):
        dense = self.dense_from_conv(self.construct_convs())
        self.dueling_q(dense)

    def construct_drqn(self):
        lstm = self.lstm_from_conv(self.construct_convs())
        self.dueling_q(lstm)

    ###########################################################################
    # TensorFlow Graph Layer Constructors (side effects of creating TF ops)
    ###########################################################################
    def dueling_q(self, dense):
        streamA, streamV = tf.split(tf.layers.flatten(dense), 2, axis=1)

        xavier_init = tf.contrib.layers.xavier_initializer()
        Advantages = tf.matmul(streamA, tf.Variable(xavier_init([self.h_size//2, self.a_size])))
        Value      = tf.matmul(streamV, tf.Variable(xavier_init([self.h_size//2, 1])))

        self.Qout = Value + \
            (Advantages - tf.reduce_mean(Advantages, axis=1, keepdims=True))
        self.predict = tf.argmax(self.Qout, 1)
        self.action = self.predict[-1]

    def lstm_from_conv(self, conv_res):
        self.trace_length = tf.placeholder(tf.int32, shape=[])
        traces = tf.reshape(flatten(self.dense_from_conv(conv_res)),
                            (self.batch_size, self.trace_length, self.h_size))
        traces = tf.Print(traces, [tf.shape(traces)])
        rnn_cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units=self.h_size)
        self.lstm_state = rnn_cell.zero_state(self.batch_size, tf.float32)
        self.rnn, self.new_state = tf.nn.dynamic_rnn(
            inputs=traces,
            cell=rnn_cell, dtype=tf.float32,
            initial_state=self.lstm_state
        )
        self.reset_hidden_state()
        return tf.reshape(self.rnn, (self.batch_size * self.trace_length, self.h_size))


    def dense_from_conv(self, conv_res):
        # using conv kernel as dense layer weight matrix rows, for faster Cudnn performance
        return conv2d(
            inputs=conv_res, num_outputs=self.h_size,
            kernel_size=(7, 7), stride=(1, 1), padding='VALID',
            data_format='NCHW',
            biases_initializer=None
        )

    def construct_convs(self):
        # input shape: (N, Channal, Height, Weight) hence 'NCHW'
        # return the end-result tensor of convolution layers
        # using 'NCHW' format so that it maximizes runtime for Nvidia drivers
        # output shape: (N, h_size)
        with tf.variable_scope('convs'):
            conv1 = conv2d(
                inputs=self.frames/255, num_outputs=32,
                kernel_size=(8, 8), stride=(4, 4), padding='VALID',
                data_format='NCHW',
                biases_initializer=None
            )
            conv2 = conv2d(
                inputs=conv1, num_outputs=64,
                kernel_size=(4, 4), stride=(2, 2), padding='VALID',
                data_format='NCHW',
                biases_initializer=None
            )
            return conv2d(
                inputs=conv2, num_outputs=64,
                kernel_size=(3, 3), stride=(1, 1), padding='VALID',
                data_format='NCHW',
                biases_initializer=None
            )

    def construct_training_inputs(self):
        self.transition_rewards = tf.placeholder(tf.float32, shape=[None])
        self.transition_terminals = tf.placeholder(tf.float32, shape=[None])
        self.transition_actions = tf.placeholder(tf.int32, shape=[None])

    def construct_Q_and_doubleTargetQ(self):
        # in order to do a single pass, we set the first part of the batch
        # to be S, and the second half of the batch to be S_next
        Q_s, Q_s_next = tf.split(self.Qout, 2)
        
        # select action using main network, using Q values from target network
        Q       = self.select_actions(Q_s,                      self.transition_actions)
        targetQ = self.select_actions(self.target_network.Qout, tf.argmax(Q_s_next, 1))

        # here goes BELLMAN
        target = (self.transition_rewards + 
                  0.99 * targetQ * (- self.transition_terminals + 1))
        return Q, tf.stop_gradient(target)

    def construct_target_and_loss(self, lstm=False):
        # where target and loss are computed
        self.construct_training_inputs()
        Q, targetQ = self.construct_Q_and_doubleTargetQ()

        # For DRQN,
        # only train on first half of every trace per Lample & Chatlot 2016
        if lstm:
            targetQ = discard_first_half_trace(targetQ)
            Q       = discard_first_half_trace(Q)
        
        self.loss = tf.losses.huber_loss(Q, targetQ)
        self.RMSprop_trainer(self.loss)
        

    def RMSprop_trainer(self, loss):
        tf.summary.scalar('loss', loss)
        # tf.summary.histogram('Q', self.Q)
        # tf.summary.histogram('targetQ', self.targetQ)

        self.trainer = tf.train.RMSPropOptimizer(
            0.00025, momentum=0.95, epsilon=0.01)
        self.updateModel = self.trainer.minimize(self.loss)

    def construct_target_update_ops(self):
        self.target_update_ops = []
        for m, t in zip(tf.trainable_variables(scope='main'),
                        tf.trainable_variables(scope='target')):
            self.target_update_ops.append(t.assign(m.value()))
        
    ###########################################################################
    # Helper Methods (functional)
    ###########################################################################
    def one_hot(self, actions):
        return tf.one_hot(actions, self.a_size, on_value=1.0, off_value=0.0, dtype=tf.float32)

    def select_actions(self, Q_values, actions):
        # Q_values: (batch_size, a_size)
        # actions: (batch_size,)
        return tf.reduce_sum(Q_values * self.one_hot(actions), axis=1)

    def discard_first_half_trace(self, batch):
        # batch shape: (batch_size * trace_length,)
        traces = tf.reshape(batch, (self.batch_size, self.trace_length))
        _, traces = tf.split(traces, 2, axis=1)
        return traces
    ###########################################################################
    # Exposed Methods (used when training agents)
    ###########################################################################
    def get_action(self, frames):
        # frames.shape: [batch_size, stack_size, frame_shape]
        return self.sess.run(self.action, feed_dict={
            self.frames: [np.stack(frames)],
            self.batch_size: len(frames)
        })

    def get_action_stateful(self, frames):
        action, self.hidden_state = self.sess.run([self.action, self.new_state], feed_dict={
            self.frames: np.reshape(frames, (len(frames), 1, 84, 84)),
            self.batch_size: 1,
            self.trace_length: len(frames),
            self.lstm_state: self.hidden_state
        })
        return action

    def reset_hidden_state(self):
        self.hidden_state = self.ZERO_STATE

    def update_model(self, S, A, R, S_next, T, additional_ops=[], additional_feeds={}):
        # print(S[0])
        return self.sess.run([self.updateModel] + additional_ops, feed_dict={
            self.target_network.frames: S_next,
            self.target_network.batch_size: len(S),
            # SINGLE FORWARD PASS WITH DOUBLE THE BATCH SIZE
            self.frames: np.concatenate([S, S_next]),
            self.batch_size: len(S) * 2,
            self.transition_actions: A,
            self.transition_rewards: R,
            self.transition_terminals: T,
            **additional_feeds
        })

    def update_model_stateful(self, S, A, R, S_next, T, addtional_ops=[], addtional_feeds={}):
        # S.shape: (batch_size, 10, 84, 84)
        batch_size, trace_length = len(S)//10, 10
        lstm_feeds = {
            self.lstm_state: self.ZERO_STATE,
            self.target_network.lstm_state: self.ZERO_STATE,
            self.trace_length: trace_length,
            self.target_network.trace_length: trace_length,
            self.batch_size: batch_size * 2,
            self.target_network.batch_size: batch_size
        }

        return self.update_model(np.stack(S), A, R, np.stack(S_next), T,
            additional_ops=addtional_ops,
            additional_feeds={**lstm_feeds, **addtional_feeds})

    def update_target_network(self):
        self.sess.run(self.target_update_ops)

if __name__=='__main__':
    q = Qnetwork(512, 4, 7, 'main')
import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim import convolution2d as conv2d
from tensorflow.contrib.slim import flatten

def tfprint(op, msg):
    return tf.Print(op, [tf.shape(op)], message=msg)

class Qnetwork():
    def __init__(self, h_size, a_size, stack_size, scopeName, train_batch_size=32, \
                 train_trace_length=10, model='dqn', model_kwargs={}, **kwargs):
        self.h_size, self.a_size, self.stack_size, self.model, self.train_batch_size, self.train_trace_length= \
            h_size, a_size, stack_size, model, train_batch_size, train_trace_length

        # self.adrqn = self.model.endswith('adrqn')
        self.num_quant = kwargs.get('num_quant', 1)
        self.autoencode = kwargs.get('autoencode', False)

        # expected inputs        
        self.frames = tf.placeholder(shape=[None, stack_size, 84, 84], dtype=tf.uint8)
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])

        with tf.variable_scope(scopeName):
            try:
                eval('self.construct_{}(**model_kwargs)'.format(model.split('-')[-1]))
            except AttributeError:
                print('model {} not implemented.'.format(model))
                raise SystemExit

        if scopeName == 'main':
            self.target_network = Qnetwork(h_size, a_size, stack_size, 'target',
                                           model=model, model_kwargs=model_kwargs, num_quant=self.num_quant)
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

    def construct_adrqn(self, action_hidden_size=0):
        self.action_h_size = max(action_hidden_size, self.a_size)
        assert not action_hidden_size % 2 # has to be even in order to split for dueling

        lstm = self.lstm_from_conv(self.construct_convs())
        self.dueling_q(lstm)

    ###########################################################################
    # TensorFlow Graph Layer Constructors (side effects of creating TF ops)
    ###########################################################################
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

    def dueling_q(self, dense):
        streamA, streamV = tf.split(dense, 2, axis=1)

        xavier_init = tf.contrib.layers.xavier_initializer()
        A = tf.matmul(streamA, tf.Variable(xavier_init([self.h_size//2, self.a_size * self.num_quant])))
        V = tf.matmul(streamV, tf.Variable(xavier_init([self.h_size//2, self.num_quant])))

        if not self.distributional:
            self.Qout = V + (A - tf.reduce_mean(A, axis=1, keepdims=True))
            self.predict = tf.argmax(self.Qout, 1)
        else:
            A = tf.reshape(A, (-1, self.a_size, self.num_quant))
            A -= tf.reduce_mean(A, axis=1, keepdims=True)
            self.Qout = A + tf.reshape(V, (-1, 1, self.num_quant))
            self.Qout_mean = tf.reduce_mean(self.Qout, axis=2)
            self.predict = tf.argmax(self.Qout_mean, 1)

        self.action = self.predict[-1]

    def lstm_from_conv(self, conv_res):
        self.trace_length = tf.placeholder(tf.int32, shape=[])
        conv_res = flatten(conv_res)
        conv_res_len = 64 * 7 * 7
        if self.adrqn:
            prev_actions = self.construct_action_projection()
            conv_res = tf.concat([conv_res, prev_actions], 1)
            conv_res_len += self.action_h_size
        traces = tf.reshape(conv_res,
                            (self.batch_size, self.trace_length, conv_res_len))
        self.rnn_cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units=self.h_size)
        self.lstm_state = self.rnn_cell.zero_state(self.batch_size, tf.float32)
        self.rnn, self.new_state = tf.nn.dynamic_rnn(
            self.rnn_cell,
            traces,
            dtype=tf.float32,
            initial_state=self.lstm_state
        )
        self.reset_hidden_state()
        return tf.reshape(
            tf.nn.dropout(self.rnn, 0.75),
            (self.batch_size * self.trace_length, self.h_size))


    def dense_from_conv(self, conv_res):
        # using conv kernel as dense layer weight matrix rows, for faster Cudnn performance
        return flatten(conv2d(
            inputs=conv_res, num_outputs=self.h_size,
            kernel_size=(7, 7), stride=(1, 1), padding='VALID',
            data_format='NCHW',
            biases_initializer=None
        ))

    def construct_training_inputs(self):
        self.transition_rewards = tf.placeholder(tf.float32, shape=[None])
        self.transition_terminals = tf.placeholder(tf.float32, shape=[None])
        self.transition_actions = tf.placeholder(tf.int32, shape=[None])

    def construct_action_projection(self):
        self.prev_actions_input = tf.placeholder(tf.int32, shape=[None])
        actions = tf.one_hot(self.prev_actions_input, self.a_size)
        actions = tf.layers.dense(actions, self.action_h_size, activation='relu')
        return actions

    def construct_Q_and_doubleTargetQ(self):
        # in order to do a single pass, we set the first part of the batch
        # to be S, and the second half of the batch to be S_next
        Q_s, Q_s_next = tf.split(self.Qout, 2)

        double_actions = tf.argmax(Q_s_next, 1, output_type=tf.int32) if not self.distributional \
                    else tf.argmax(tf.split(self.Qout_mean, 2)[1], 1, output_type=tf.int32)
        # select action using main network, using Q values from target network
        Q       = self.select_actions(Q_s, self.transition_actions)
        targetQ = self.select_actions(self.target_network.Qout, double_actions)
        
        # Q = tfprint(Q, 'Q')
        # targetQ = tfprint(targetQ, 'targetQ')

        # here goes BELLMAN
        target = (tf.reshape(self.transition_rewards, (-1,1)) + 
                  0.99 * targetQ * (- tf.reshape(self.transition_terminals, (-1,1)) + 1))
        return Q, tf.stop_gradient(target)

    def construct_distQ_and_doubleTargetQ(self):
        Q_s, Q_s_next = tf.split(self.Qout, 2)
        


    def construct_target_and_loss(self):
        # where target and loss are computed
        self.construct_training_inputs()
        Q, targetQ = self.construct_Q_and_doubleTargetQ()

        # For DRQN,
        # only train on first half of every trace per Lample & Chatlot 2016
        
        self.loss = self.construct_loss(Q, targetQ)
        tf.summary.scalar('loss', self.loss)
        if self.distributional:
            self.Adam_trainer()
        else:
            self.RMSprop_trainer()


    def construct_loss(self, Q, targetQ):
        if self.distributional:
            loss = self.quantile_dist_loss(Q, targetQ)
        else:
            if self.lstm:
                targetQ = self.discard_first_half_trace(targetQ)
                Q       = self.discard_first_half_trace(Q)
            loss = tf.losses.huber_loss(Q, targetQ)
        
        if self.autoencode:
            pass # TODO
        return loss
        

    def RMSprop_trainer(self):
        self.trainer = tf.train.RMSPropOptimizer(
            0.00025, momentum=0.95, epsilon=0.01)
        self.updateModel = self.trainer.minimize(self.loss)

    def Adam_trainer(self):
        self.trainer = tf.train.AdamOptimizer(0.00005, epsilon=0.01/32)
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
        # Q_values: (batch_size, a_size, num_quant)
        # actions: (batch_size,)
        return tf.gather_nd(Q_values,
            tf.transpose(tf.stack([tf.constant(list(range(self.train_batch_size*self.train_trace_length))), actions])))

    def discard_first_half_trace(self, batch):
        # batch shape: (batch_size * trace_length,)
        traces = tf.reshape(batch, (-1, self.trace_length, self.num_quant))
        _, traces = tf.split(traces, 2, axis=1)
        return traces

    @staticmethod
    def huber_loss(residual, delta=1.0):
        residual = tf.abs(residual)
        # condition = tf.less(residual, delta)
        small_res = 0.5 * tf.square(residual) * tf.cast(residual <= delta, tf.float32)
        large_res = delta * residual - 0.5 * tf.square(delta) * tf.cast(residual > delta, tf.float32)
        return small_res + large_res

    # @staticmethod
    def rep_row(self, row, times=0):
        times = times or self.num_quant
        return tf.reshape(tf.tile(row, [times]), [times,-1])

    def quantile_dist_loss(self, dist, target_dist):
        remaing_dist_len = self.train_batch_size * self.train_trace_length // 2
        dist = tf.reshape(self.discard_first_half_trace(dist), (remaing_dist_len, self.num_quant))
        target_dist = tf.reshape(self.discard_first_half_trace(target_dist), (remaing_dist_len, self.num_quant))

        losses = []
        tau_row = list((2 * np.array(list(range(self.num_quant))) + 1) / (2 * self.num_quant))
        tau = tf.constant([tau_row for _ in range(self.num_quant)], dtype=tf.float32)

        for b in range(remaing_dist_len):
            residual = self.rep_row(target_dist[b]) - tf.transpose(self.rep_row(dist[b]))
            residual_countweights = tau - tf.cast(residual < 0, tf.float32)
            losses.append(tf.reduce_mean(self.huber_loss(residual) * residual_countweights))


        # dist.shape and target_dist.shape: (batch_size, num_quantile)
        # J = tf.map_fn(lambda b: self.rep_row(b, self.num_quant), target_dist)
        # I = tf.map_fn(lambda b: tf.transpose(self.rep_row(b, self.num_quant)), dist)
        # residual = J - I

        # tau_row = list((2 * np.array(list(range(self.num_quant))) + 1) / (2 * self.num_quant)) # evenly spaced from 0 to 1
        # tau = tf.constant([[tau_row for _ in range(self.num_quant)] 
        #                   for _ in range(self.train_batch_size * self.train_trace_length)])

        # residual = self.huber_loss(residual)
        # residual_counterweights = tf.cast(tau, tf.float32) - tf.cast(residual < 0, tf.float32)

        return tf.add_n(losses)

    @property
    def ZERO_STATE(self):
        return (np.zeros((1, self.h_size)),) * 2

    @property
    def lstm(self):
        return self.model.endswith('drqn')

    @property
    def adrqn(self):
        return self.model.endswith('adrqn')
    
    @property
    def distributional(self):
        return self.model.startswith('dist')
    ###########################################################################
    # Exposed Methods (used when training agents)
    ###########################################################################
    def get_action(self, frames):
        # frames.shape: [batch_size, stack_size, frame_shape]
        return self.sess.run(self.action, feed_dict={
            self.frames: [np.stack(frames)],
            self.batch_size: len(frames)
        })

    def get_action_stateful(self, frames, prev_a=0, state=None):
        feed = {
            self.frames: np.reshape(frames, (len(frames), 1, 84, 84)),
            self.batch_size: 1,
            self.trace_length: len(frames),
            self.lstm_state: state or self.hidden_state
        }
        if self.adrqn:
            feed[self.prev_actions_input] = [prev_a]
        action, new_state = self.sess.run([self.action, self.new_state],
                                                  feed_dict=feed)
        if not state:
            self.hidden_state = new_state
        else:
            state = new_state
        return action, state

    def reset_hidden_state(self):
        self.hidden_state = self.ZERO_STATE

    def update_model(self, S, A, R, S_next, T, additional_ops=[], additional_feeds={}):
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

    def update_model_stateful(self, S, A, R, S_next, T, *args, addtional_ops=[], addtional_feeds={}):
        # S.shape: (batch_size, 10, 84, 84)
        batch_size, trace_length = len(S)//10, 10
        lstm_feeds = {
            self.trace_length: trace_length,
            self.target_network.trace_length: trace_length,
            self.batch_size: batch_size * 2,
            self.target_network.batch_size: batch_size
        }
        if args and self.adrqn:
            A_prev, = args
            lstm_feeds[self.prev_actions_input] = A_prev
            lstm_feeds.update({
                self.prev_actions_input: np.concatenate([A_prev, A]),
                self.target_network.prev_actions_input: A
            })
        
        return self.update_model(np.stack(S), A, R, np.stack(S_next), T,
            additional_ops=addtional_ops,
            additional_feeds={**lstm_feeds, **addtional_feeds})

    def update_target_network(self):
        self.sess.run(self.target_update_ops)

if __name__=='__main__':
    q = Qnetwork(512, 4, 7, 'main')
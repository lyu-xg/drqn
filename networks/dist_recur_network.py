import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


class Qnetwork():
    def __init__(self, h_size, a_size, rnn_cell, scopeName, num_quant=50, discount=0.99, **kwargs):
        self.scopeName = scopeName
        self.h_size, self.a_size, self.num_quant, self.discount = h_size, a_size, num_quant, discount
        self.scalarInput = tf.placeholder(shape=[None, 7056], dtype=tf.uint8)
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])
        self.trainLength = tf.placeholder(dtype=tf.int32, shape=[])

        self.frameShape = tf.constant((84, 84, 1), dtype=tf.int32)
#         self.frames = tf.reshape(self.scalarInput, tf.concat(([self.batch_size*self.trainLength], self.frameShape), 0))
        self.frames = tf.reshape(self.scalarInput/255, [-1, 84, 84, 1])
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
            slim.flatten(self.conv4), [self.batch_size, self.trainLength, h_size])

        self.state_init = rnn_cell.zero_state(self.batch_size, tf.float32)
        self.rnn, self.rnn_state = tf.nn.dynamic_rnn(
            inputs=self.convFlat, cell=rnn_cell, dtype=tf.float32,
            initial_state=self.state_init, scope=scopeName+'_rnn'
        )
        self.rnn = tf.reshape(self.rnn, shape=[-1, h_size])

        # self.streamA, self.streamV = tf.split(self.rnn, 2, axis=1)

        xavier_init = tf.contrib.layers.xavier_initializer()
        self.QW = tf.Variable(xavier_init([h_size, a_size*num_quant]))
        self.Qout = tf.reshape(tf.matmul(self.rnn, self.QW), (-1, a_size, num_quant)) # Qout exploded

        # self.QW = tf.Variable(tf.random_normal([h_size//2, a_size*num_quant]))
        # self.VW = tf.Variable(tf.random_normal([h_size//2, 1*num_quant]))
        # self.A = tf.matmul(self.streamA, self.AW)
        # self.V = tf.matmul(self.streamV, self.VW)

        # self.V = tf.layers.dense(self.streamV, 1      * num_quant, activation='relu')
        # self.A = tf.layers.dense(self.streamA, a_size * num_quant, activation='relu')

        # self.Qout = self.combine_V_A(tf.reshape(self.V, (1,      num_quant)),
        #                              tf.reshape(self.A, (a_size, num_quant)))

        # self.salience = tf.gradients(self.A, self.scalarInput)

        # self.Qout = self.V + \
        #     (self.A - tf.reduce_mean(self.A, axis=1, keepdims=True))

        # self.Qout = tf.layers.dense(self.rnn, a_size * num_quant, activation='relu')

        self.Qout_mean = self.dist_mean(self.Qout)

        self.predict = tf.argmax(self.Qout_mean, axis=1)
        self.action = self.predict[0] # only make sense when stepping to get optimal a

        ##############################################################################
        # Training stuff and loss

        # None stands for batch_size
        self.sample_terminals = tf.placeholder(tf.int32, shape=[None], name='sample_terminals')
        self.sample_rewards = tf.placeholder(tf.float32, shape=[None], name='sample_rewards')
        self.doubleQ = tf.placeholder(tf.float32, shape=(None, self.num_quant), name='doubleQ')
        # self.targetQ = tf.placeholder(shape=[None, num_quant], dtype=tf.float32)
        end_multiplier = tf.cast(- (self.sample_terminals - 1), tf.float32)
        end_mul_tiles = tf.transpose(self.rep_row(end_multiplier, self.num_quant))
        # reward_tiles = tf.tile(tf.reshape(self.sample_rewards, [-1,1]), [1, self.num_quant])
        reward_tiles = tf.transpose(self.rep_row(self.sample_rewards, self.num_quant))
        terminal_tiles = tf.transpose(self.rep_row(end_multiplier, self.num_quant))

        print(terminal_tiles.shape, terminal_tiles.dtype)
        self.targetQ = reward_tiles + (self.discount * self.doubleQ * end_mul_tiles)

        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)

        select = tf.concat([tf.reshape(tf.range(self.batch_size*self.trainLength), [-1, 1]),
                            tf.reshape(self.actions, (-1, 1))], axis=1)
        self.Q = tf.gather_nd(self.Qout, select)
        print('Q shape', self.Q.shape)

        self.loss = self.quantile_dist_loss(self.Q, self.targetQ) #* self.mask
        print('loss shape', self.loss.shape)
        if scopeName == 'main':
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('Q', self.Qout)
            tf.summary.histogram('hidden', self.rnn_state)
        # self.debug = self.rnn

        self.trainer = tf.train.AdamOptimizer(
            learning_rate=0.00005, epsilon=0.01/32)
        self.updateModel = self.trainer.minimize(self.loss)

    # def combine_V_A(self, V, A):
    #     # A.shape: (action x num_quatile)
    #     # V.shape: (1 x num_quatile)
    #     A_mean = tf.reduce_mean(A, axis=0)
    # need to figure out how to get and subtract mean for distributions
    
    def dist_mean(self, dists, keepdims=False):
        # dists.shape: (batch_size, a_size, num_quantile)
        return tf.reduce_mean(dists, axis=2, keepdims=keepdims) / self.num_quant


    def get_action_and_next_state(self, sess, state, frames):
        state = state or (np.zeros((1, self.h_size)),) * 2
        return sess.run([self.action, self.rnn_state], feed_dict={
            self.scalarInput: np.vstack(np.array(frames)),
            self.trainLength: len(frames),
            self.state_init: state,
            self.batch_size: 1
        })

    @staticmethod
    def rep_row(row, times):
        return tf.reshape(tf.tile(row, [times]), [times,-1])

    @staticmethod
    def huber_loss(residual, delta=1.0):
        residual = tf.abs(residual)
        # condition = tf.less(residual, delta)
        small_res = 0.5 * tf.square(residual) * tf.cast(residual <= delta, tf.float32)
        large_res = delta * residual - 0.5 * tf.square(delta) * tf.cast(residual > delta, tf.float32)
        return small_res + large_res

    def quantile_dist_loss(self, dist, target_dist):
        # dist.shape and target_dist.shape: (batch_size, num_quantile)
        J = tf.map_fn(lambda b: self.rep_row(b, self.num_quant), target_dist)
        I = tf.map_fn(lambda b: tf.transpose(self.rep_row(b, self.num_quant)), dist)
        residual = J - I
        I_indexes = tf.map_fn(
            lambda x: self.rep_row(tf.range(self.num_quant), self.num_quant),
            tf.range(self.batch_size * self.trainLength))

        tau = (2 * I_indexes + 1) / (2 * self.num_quant) # evenly spaced from 0 to 1
        
        residual = self.huber_loss(residual)
        residual_counterweights = tf.cast(tau, tf.float32) - tf.cast(residual < 0, tf.float32)

        if self.scopeName == 'main':
            tf.summary.histogram('targetQ', J)
            tf.summary.histogram('predictQ', I)
            tf.summary.histogram('residual', residual)
            tf.summary.histogram('residual_counterweights', residual_counterweights)
            

        self.debug = (residual[-1], residual_counterweights[-1])

        # only train on first half of every trace per Lample & Chatlot 2016
        mask = tf.concat((tf.zeros((self.batch_size, self.trainLength//2, self.num_quant, self.num_quant)),
                          tf.ones((self.batch_size, self.trainLength//2, self.num_quant, self.num_quant))), 1)
        mask = tf.reshape(mask, [-1, self.num_quant, self.num_quant])

        loss = tf.reduce_sum(residual * residual_counterweights / self.num_quant * mask)
        
        return loss
if __name__ == '__main__':
    q = Qnetwork(512, 256, tf.nn.rnn_cell.LSTMCell(num_units=512), 'main', num_quant=128)
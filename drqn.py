import gym
import os
import csv
import signal
import argparse
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from buffer import TraceBuf
from common import epsilon_at, downsample, to_grayscale, checkpoint_dir, checkpoint_exists
from common_tf import checkpoint, load_checkpoint

Exiting = 0


def signal_handler(sig, frame):
    global Exiting
    print('signal captured, trying to save states.', flush=1)
    Exiting += 1
    if Exiting > 3:
        raise SystemExit


class Qnetwork():
    def __init__(self, h_size, a_size, rnn_cell, scopeName):
        self.h_size, self.a_size = h_size, a_size
        self.scalarInput = tf.placeholder(shape=[None, 7056], dtype=tf.float32)
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])
        self.trainLength = tf.placeholder(dtype=tf.int32, shape=[])

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

        self.convFlat = tf.reshape(
            slim.flatten(self.conv4), [self.batch_size, self.trainLength, h_size])

        self.state_init = rnn_cell.zero_state(self.batch_size, tf.float32)
        self.rnn, self.rnn_state = tf.nn.dynamic_rnn(
            inputs=self.convFlat, cell=rnn_cell, dtype=tf.float32,
            initial_state=self.state_init, scope=scopeName+'_rnn'
        )
        self.rnn = tf.reshape(self.rnn, shape=[-1, h_size])

        self.streamA, self.streamV = tf.split(self.rnn, 2, axis=1)

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


# In[3]:


def preprocess(s):
    return downsample(to_grayscale(s[8:-12]), (84, 84)).reshape((7056,))


def reset(env, buf, noop_max=30):
    if buf.trans_cache:
        buf._flush()
    S = [preprocess(env.reset())]
    life = None
    for _ in range(np.random.randint(noop_max)):
        s, r, t, l = env.step(0)
        s, life = preprocess(s), l['ale.lives']
        if t:
            return reset(env, buf, noop_max)
        buf.append_trans((S[-1], 0, r, s, t))
        S.append(s)
    return S, life


def getTargetUpdateOps(tfVars):
    # tfVars consists of all trainble TF Variables
    # where the first half is from the main network
    #  and the second half is from the target network
    # RETURNS: list of operations which when run,
    #          updates the target network with main network's values
    return [vt.assign(vm.value())
            for i, (vm, vt) in enumerate(zip(tfVars[:len(tfVars)//2],
                                             tfVars[len(tfVars)//2:]))]


# def updateTarget(op_holder, sess):
#     # for op in op_holder:
#     sess.run(op)
#     total_vars = len(tf.trainable_variables())
#     a = tf.trainable_variables()[0].eval(session=sess)
#     b = tf.trainable_variables()[total_vars//2].eval(session=sess)
#     print("Target Set", "Success." if a.all() == b.all() else "Failed.")


def train(trace_length, render_eval=False, h_size=512, target_update_freq=10000,
          ckpt_freq=100000, summary_freq=1000, eval_freq=10000,
          batch_size=32, env_name='SpaceInvaders', total_iteration=5e7,
          pretrain_steps=50000):
    global Exiting
    # env_name += 'NoFrameskip-v4'
    identity = 'stack={},env={},mod={}'.format(trace_length, env_name, 'drqn')

    env = gym.make(env_name)
    a_size = env.action_space.n

    tf.reset_default_graph()
    cell = tf.nn.rnn_cell.LSTMCell(num_units=h_size)
    cellT = tf.nn.rnn_cell.LSTMCell(num_units=h_size)
    mainQN = Qnetwork(h_size, a_size, cell, 'main')
    targetQN = Qnetwork(h_size, a_size, cellT, 'target')
    init = tf.global_variables_initializer()
    updateOps = getTargetUpdateOps(tf.trainable_variables())
    saver = tf.train.Saver(max_to_keep=5)

    sess = tf.Session()
    summary_writer = tf.summary.FileWriter('./log/' + identity, sess.graph)

    if checkpoint_exists(identity):
        (_, env, last_iteration, is_done,
         prev_life_count, action, state, S) = load_checkpoint(sess, saver, identity)
        start_time = time.time()
    # else:
        exp_buf = TraceBuf(trace_length, size=400000)
        last_iteration = 1 - pretrain_steps
        is_done = True
        prev_life_count = None
        state = None

    sess.run(init)

    updateTarget(updateOps, sess)
    summaryOps = tf.summary.merge_all()

    eval_summary_ph = tf.placeholder(tf.float32, shape=(2,), name='evaluation')
    evalOps = (tf.summary.scalar('performance', eval_summary_ph[0]),
               tf.summary.scalar('perform_std', eval_summary_ph[1]))

    for i in range(last_iteration, int(total_iteration)):
        if is_done:
            exp_buf.flush_episode()
            S, prev_life_count = reset(env, exp_buf)
            state, action = sess.run((mainQN.rnn_state, mainQN.action), feed_dict={
                mainQN.scalarInput: np.vstack(np.array(S)/255.0),
                mainQN.trainLength: len(S),
                mainQN.state_init: (np.zeros((1, h_size)),) * 2,
                mainQN.batch_size: 1
            })

        S = [S[-1]]
        for _ in range(4):
            s, r, is_done, info = env.step(action)
            s, life_count = preprocess(s), info['ale.lives']
            exp_buf.append_trans((
                S[-1], action, np.sign(r), s,
                (prev_life_count and life_count < prev_life_count or is_done)
            ))
            S.append(s)
            prev_life_count = life_count

        feed = {
            mainQN.scalarInput: np.vstack(np.array(S[1:])/255.0),
            mainQN.trainLength: 4,
            mainQN.state_init: state,
            mainQN.batch_size: 1
        }
        if np.random.random() < epsilon_at(i):
            action = env.action_space.sample()
            state = sess.run(mainQN.rnn_state, feed_dict=feed)
        else:
            action, state = sess.run(
                (mainQN.action, mainQN.rnn_state), feed_dict=feed)

        if not i:
            start_time = time.time()

        if i <= 0:
            continue

        if Exiting or not i % ckpt_freq:
            checkpoint(sess, saver, identity,
                       exp_buf, env, i, is_done,
                       prev_life_count, action, state, S)
            if Exiting:
                raise SystemExit

        if not i % target_update_freq:
            sess.run(updateOps)
            cur_time = time.time()
            print('[{}] took {} seconds to {} steps'.format(
                i, cur_time-start_time, target_update_freq))
            start_time = cur_time

        #　TRAINING STARTS
        state_train = (np.zeros((batch_size, h_size)),) * 2

        trainBatch = exp_buf.sample_traces(batch_size)

        Q1 = sess.run(mainQN.predict, feed_dict={
            mainQN.scalarInput: np.vstack(trainBatch[:, 3]/255.0),
            mainQN.trainLength: trace_length,
            mainQN.state_init: state_train,
            mainQN.batch_size: batch_size
        })
        Q2 = sess.run(targetQN.Qout, feed_dict={
            targetQN.scalarInput: np.vstack(trainBatch[:, 3]/255.0),
            targetQN.trainLength: trace_length,
            targetQN.state_init: state_train,
            targetQN.batch_size: batch_size
        })
        end_multiplier = - (trainBatch[:, 4] - 1)
        doubleQ = Q2[range(batch_size * trace_length), Q1]
        targetQ = trainBatch[:, 2] + (0.99 * doubleQ * end_multiplier)

        _, summary = sess.run((mainQN.updateModel, summaryOps), feed_dict={
            mainQN.scalarInput: np.vstack(trainBatch[:, 0]/255.0),
            mainQN.targetQ: targetQ,
            mainQN.actions: trainBatch[:, 1],
            mainQN.trainLength: trace_length,
            mainQN.state_init: state_train,
            mainQN.batch_size: batch_size
        })

        if not i % summary_freq:
            summary_writer.add_summary(summary, i)
        if not i % eval_freq:
            eval_res = np.array(
                evaluate(sess, mainQN, env_name, is_render=render_eval))
            perf, perf_std = sess.run(
                evalOps, feed_dict={eval_summary_ph: eval_res})
            summary_writer.add_summary(perf, i)
            summary_writer.add_summary(perf_std, i)
    # In the end
    sess.close()
    env.close()
    checkpoint(sess, saver, identity)


import time


def evaluate(sess, mainQN, env_name, action_repeat=6, scenario_count=3, is_render=False):
    start_time = time.time()
    env = gym.make(env_name)
    # step 6 frame with same action, and use trace size = 6

    def get_action(S):
        return sess.run(mainQN.action, feed_dict={
            mainQN.scalarInput: np.vstack(np.array(S)/255.0),
            mainQN.trainLength: len(S),
            mainQN.state_init: (np.zeros((1, mainQN.h_size)),) * 2,
            mainQN.batch_size: 1
        })

    def run_scenario():
        S, R, t = [preprocess(env.reset())], 0, 0
        noop = np.random.randint(30)
        for _ in range(noop):
            frame, r, t, _ = env.step(0)
            S += (preprocess(frame),)
            R += r
        action = get_action(S)
        while not t:
            S.clear()
            for _ in range(action_repeat):
                #                 print(action)
                frame, r, t, _ = env.step(action)
                R += r
                S += (preprocess(frame),)
                if is_render:
                    env.render()
            action = get_action(S)
        return R

    res = np.array([run_scenario() for _ in range(scenario_count)])
    print(time.time() - start_time, 'seconds to evaluate')
    env.close()
    return np.mean(res), np.std(res)


def main():
    signal.signal(signal.SIGINT, signal_handler)
    parser = argparse.ArgumentParser()
    parser.add_argument('trace_length', action='store', type=int, default=10)
    parser.add_argument('-e', '--env_name', action='store',
                        default='SpaceInvadersNoFrameskip-v4')
    train(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()

import signal
import time
import argparse
import numpy as np
import tensorflow as tf

import common as util
from buffer import ActionTraceBuf
from myenv import Env
from networks.adrqn_network import Qnetwork


Exiting = 0
def train(trace_length, render_eval=False, h_size=512, action_h_size=512,
          target_update_freq=10000, ckpt_freq=500000, summary_freq=1000, eval_freq=10000,
          batch_size=32, env_name='SpaceInvaders', total_iteration=5e7,
          pretrain_steps=50000):
    global Exiting
    # env_name += 'NoFrameskip-v4'
    identity = 'stack={},env={},mod={}'.format(trace_length, env_name, 'adrqn')

    env = Env(env_name=env_name, skip=4)
    a_size = env.n_actions

    tf.reset_default_graph()
    cell = tf.nn.rnn_cell.LSTMCell(num_units=h_size)
    cellT = tf.nn.rnn_cell.LSTMCell(num_units=h_size)
    mainQN = Qnetwork(h_size, a_size, action_h_size, cell, 'main')
    targetQN = Qnetwork(h_size, a_size, action_h_size, cellT, 'target')
    init = tf.global_variables_initializer()
    updateOps = util.getTargetUpdateOps(tf.trainable_variables())
    saver = tf.train.Saver(max_to_keep=5)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(init)
    summary_writer = tf.summary.FileWriter('./log/' + identity, sess.graph)

    if util.checkpoint_exists(identity):
        (exp_buf, env, last_iteration, is_done,
         prev_life_count, action, state, S) = util.load_checkpoint(sess, saver, identity)
        start_time = time.time()
    else:
        exp_buf = ActionTraceBuf(trace_length, scenario_size=2500)
        last_iteration = 1 - pretrain_steps
        is_done = True
        action = 0
        prev_life_count = None
        state = None
        sess.run(updateOps)

    summaryOps = tf.summary.merge_all()

    eval_summary_ph = tf.placeholder(tf.float32, shape=(2,), name='evaluation')
    evalOps = (tf.summary.scalar('performance', eval_summary_ph[0]),
               tf.summary.scalar('perform_std', eval_summary_ph[1]))
    online_summary_ph = tf.placeholder(tf.float32, shape=(2,), name='online')
    onlineOps = (tf.summary.scalar('online_performance', online_summary_ph[0]),
                 tf.summary.scalar('online_scenario_length', online_summary_ph[1]))

    for i in range(last_iteration, int(total_iteration)):
        if is_done:
            total_scenario_reward = exp_buf.get_cache_total_reward()
            if i > 0:
                online_perf_and_length = np.array(
                    [total_scenario_reward, len(exp_buf.trans_cache)])
                online_perf, online_episode_count = sess.run(onlineOps, feed_dict={
                    online_summary_ph: online_perf_and_length})
                summary_writer.add_summary(online_perf, i)
                summary_writer.add_summary(online_episode_count, i)
            exp_buf.flush_scenario()
            s, r, prev_life_count = env.reset()
            S = [s]
            action, state = mainQN.get_action_and_next_state(sess, None, [action], S)

        S = [S[-1]]
        for _ in range(4):
            s, r, is_done, life_count = env.step(action)
            exp_buf.append_trans((
                S[-1], action, r, s,  # not cliping reward (huber loss)
                (prev_life_count and life_count < prev_life_count or is_done)
            ))
            S.append(s)
            prev_life_count = life_count

        action, state = mainQN.get_action_and_next_state(sess, state, [action]*len(S), S)
        if np.random.random() < util.epsilon_at(i):
            action = env.rand_action()

        if not i:
            start_time = time.time()

        if i <= 0:
            continue

        if Exiting or not i % ckpt_freq:
            util.checkpoint(sess, saver, identity,
                       exp_buf, env, i, is_done,
                       prev_life_count, action, state, S)
            if i % ckpt_freq:
                raise SystemExit

        if not i % target_update_freq:
            sess.run(updateOps)
            cur_time = time.time()
            print('[{}{}:{}] took {} seconds to {} steps'.format(
                'ADRQN', trace_length, i, cur_time-start_time, target_update_freq), flush=1)
            start_time = cur_time

        #　TRAINING STARTS
        state_train = (np.zeros((batch_size, h_size)),) * 2

        trainBatch = exp_buf.sample_traces(batch_size)

        Q1 = sess.run(mainQN.predict, feed_dict={
            mainQN.scalarInput: np.vstack(trainBatch[:, 3]/255.0),
            targetQN.actionsInput: np.vstack(trainBatch[:, 1]),
            mainQN.trainLength: trace_length,
            mainQN.state_init: state_train,
            mainQN.batch_size: batch_size
        })
        Q2 = sess.run(targetQN.Qout, feed_dict={
            targetQN.scalarInput: np.vstack(trainBatch[:, 3]/255.0),
            targetQN.actionsInput: np.vstack(trainBatch[:, 1]),
            targetQN.trainLength: trace_length,
            targetQN.state_init: state_train,
            targetQN.batch_size: batch_size
        })
        end_multiplier = - (trainBatch[:, 4] - 1)
        doubleQ = Q2[range(batch_size * trace_length), Q1]
        targetQ = trainBatch[:, 2] + (0.99 * doubleQ * end_multiplier)

        # print(targetQ.shape)
        _, summary = sess.run((mainQN.updateModel, summaryOps), feed_dict={
            mainQN.scalarInput: np.vstack(trainBatch[:, 0]/255.0),
            mainQN.actionsInput: np.vstack(trainBatch[:, 5]),
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
    util.checkpoint(sess, saver, identity)


def evaluate(sess, mainQN, env_name, skip=6, scenario_count=3, is_render=False):
    start_time = time.time()
    env = Env(env_name=env_name, skip=skip)

    def total_scenario_reward():
        (s, R, _), t, state = env.reset(), False, None
        while not t:
            action, state = mainQN.get_action_and_next_state(sess, state, [s])
            s, r, t, _ = env.step(action)
            R += r
            if is_render:
                env.render()
        return R

    res = np.array([total_scenario_reward() for _ in range(scenario_count)])
    print(time.time() - start_time, 'seconds to evaluate', flush=1)
    return np.mean(res), np.std(res)


def main():
    signal.signal(signal.SIGINT, util.signal_handler)
    parser = argparse.ArgumentParser()
    parser.add_argument('trace_length', action='store', type=int, default=10)
    parser.add_argument('-e', '--env_name', action='store',
                        default='SpaceInvadersNoFrameskip-v4')
    train(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()

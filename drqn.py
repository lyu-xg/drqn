import signal
import time
import os
import argparse
import numpy as np
import tensorflow as tf

import common as util
from buffer import FixedTraceBuf
from myenv import Env
from networks.dqn_network import Qnetwork
# from networks.dist_recur_network import Qnetwork as dist_Qnetwork

KICKSTART_EXP_BUF_FILE = 'trace_buf_random_policy_50000.p'

def train(trace_length, render_eval=False, h_size=512, target_update_freq=10000,
          ckpt_freq=500000, summary_freq=1000, eval_freq=10000,
          batch_size=8, env_name='Pong', total_iteration=5e7,
          pretrain_steps=50000, num_quant=0):
    # network = dist_Qnetwork if num_quant else Qnetwork
    # env_name += 'NoFrameskip-v4'
    identity = 'stack={},env={},mod={},h_size={}'.format(
        trace_length, env_name, 'drqn', h_size)
    if num_quant:
        identity += ',quantile={}'.format(num_quant)
    print(identity)

    train_len = trace_length * batch_size

    env = Env(env_name=env_name, skip=4)
    
    mainQN = Qnetwork(h_size, env.n_actions, 1, 'main', model='drqn')
    saver = tf.train.Saver(max_to_keep=5)

    summary_writer = tf.summary.FileWriter('./log/' + identity, mainQN.sess.graph)

    if util.checkpoint_exists(identity):
        (exp_buf, env, last_iteration, is_done,
         prev_life_count, action, mainQN.hidden_state, S) = util.load_checkpoint(mainQN.sess, saver, identity)
        start_time = time.time()
    else:
        exp_buf = FixedTraceBuf(trace_length, buf_length=500000)
        last_iteration = 1 - pretrain_steps
        if os.path.isfile(KICKSTART_EXP_BUF_FILE):
            exp_buf, last_iteration = util.load(KICKSTART_EXP_BUF_FILE), 1
        is_done = True
        prev_life_count = None
        mainQN.update_target_network()

    summaryOps = tf.summary.merge_all()

    eval_summary_ph = tf.placeholder(tf.float32, shape=(2,), name='evaluation')
    evalOps = (tf.summary.scalar('performance', eval_summary_ph[0]),
               tf.summary.scalar('perform_std', eval_summary_ph[1]))
    online_summary_ph = tf.placeholder(tf.float32, shape=(2,), name='online')
    onlineOps = (tf.summary.scalar('online_performance', online_summary_ph[0]),
                 tf.summary.scalar('online_scenario_length', online_summary_ph[1]))

    for i in range(last_iteration, int(total_iteration)):
        if is_done:
            scen_R, scen_L = exp_buf.flush_scenario()
            if i > 0:
                online_perf_and_length = np.array([scen_R, scen_L])
                online_perf, online_episode_count = mainQN.sess.run(onlineOps, feed_dict={
                    online_summary_ph: online_perf_and_length})
                summary_writer.add_summary(online_perf, i)
                summary_writer.add_summary(online_episode_count, i)
            
            s, r, prev_life_count = env.reset()
            S = [s]
            action = mainQN.get_action_stateful(S)

        S = [S[-1]]
        for _ in range(4):
            s, r, is_done, life_count = env.step(action, epsilon=util.epsilon_at(i))
            exp_buf.append_trans((
                [S[-1]], action, r, [s],  # not cliping reward (huber loss)
                (prev_life_count and life_count < prev_life_count or is_done)
            ))
            S.append(s)
            prev_life_count = life_count

        action = mainQN.get_action_stateful(S)

        if not i:
            start_time = time.time()
            util.pickle.dump(exp_buf, open(KICKSTART_EXP_BUF_FILE, 'wb'))


        if util.Exiting or not i % ckpt_freq:
            util.checkpoint(mainQN.sess, saver, identity,
                       exp_buf, env, i, is_done,
                       prev_life_count, action, mainQN.hidden_state, S)
            if util.Exiting:
                raise SystemExit

        if i < 0:
            continue

        if not i % target_update_freq:
            mainQN.update_target_network()
            cur_time = time.time()
            print('[{}{}:{}] took {} seconds to {} steps'.format(
                'dRqn', trace_length, i, cur_time-start_time, target_update_freq * 4), flush=1)
            start_time = cur_time

        for _ in range(4):
            batch = np.transpose(exp_buf.sample_traces(batch_size))
            for i in range(5):
                print(batch[i].dtype)
            _, summary = mainQN.update_model_stateful(
                *batch,
                addtional_ops=[summaryOps])
                                     
        
        if not i % summary_freq:
            summary_writer.add_summary(summary, i)
        if not i % eval_freq:
            eval_res = np.array(
                evaluate(sess, mainQN, env_name, is_render=render_eval))
            perf, perf_std = sess.run(
                evalOps, feed_dict={eval_summary_ph: eval_res})
            summary_writer.add_summary(perf, i)
            summary_writer.add_summary(perf_std, i)
            print(identity)
    # In the end
    util.checkpoint(sess, saver, identity)


def evaluate(sess, mainQN, env_name, skip=4, scenario_count=3, is_render=False):
    start_time = time.time()
    env = Env(env_name=env_name, skip=skip)

    def total_scenario_reward():
        (s, R, _), t, state = env.reset(), False, None
        # frame_count = 0
        while not t:
            # frame_count += 4
            action, state = mainQN.get_action_stateful([s])
            s, r, t, _ = env.step(action, epsilon=0.1)
            R += r
            if is_render:
                env.render()
            # if frame_count and not frame_count % 10000:
            #     print(frame_count, action)
        return R

    res = np.array([total_scenario_reward() for _ in range(scenario_count)])
    print(time.time() - start_time, 'seconds to evaluate', flush=1)
    print('Eval mean', np.mean(res))
    return np.mean(res), np.std(res)


def main():
    signal.signal(signal.SIGINT, util.signal_handler)
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--trace_length', action='store', type=int, default=10)
    parser.add_argument('-d', '--h_size', action='store', type=int, default=512)
    parser.add_argument('-e', '--env_name', action='store', default='Pong')
    parser.add_argument('-q', '--num_quant', action='store', type=int, default=0)
    train(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()

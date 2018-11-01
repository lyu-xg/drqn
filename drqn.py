import signal
import time
import os
import argparse
import numpy as np
import tensorflow as tf

import common as util
from buffer import FixedTraceBuf, FixedActionTraceBuf
from myenv import Env
from networks.dqn_network import Qnetwork


def train(trace_length, render_eval=False, h_size=512, target_update_freq=10000,
          ckpt_freq=500000, summary_freq=1000, eval_freq=10000,
          batch_size=32, env_name='Pong', total_iteration=5e7, use_actions=0,
          pretrain_steps=50000, num_quant=0):
    # network = dist_Qnetwork if num_quant else Qnetwork
    # env_name += 'NoFrameskip-v4'
    model = 'drqn' if not use_actions else 'adrqn'
    if num_quant:
        model = 'dist-' + model
    KICKSTART_EXP_BUF_FILE = 'trace_buf_random_policy_50000.p'

    model_args = {}
    identity = 'stack={},env={},mod={},h_size={}'.format(
        trace_length, env_name, model, h_size)
    if num_quant:
        identity += ',quantile={}'.format(num_quant)
    if use_actions:
        identity += ',action_dim={}'.format(use_actions)
        FixedTraceBuf = FixedActionTraceBuf
        model_args['action_hidden_size'] = use_actions
        KICKSTART_EXP_BUF_FILE = 'action_' + KICKSTART_EXP_BUF_FILE
    print(identity)

    train_len = trace_length * batch_size

    env = Env(env_name=env_name, skip=4)
    
    mainQN = Qnetwork(h_size, env.n_actions, 1, 'main', model=model, model_kwargs=model_args)
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
            print('Filling buffer with random episodes on disk.')
            exp_buf, last_iteration = util.load(KICKSTART_EXP_BUF_FILE), 1
        is_done = True
        prev_life_count = None
        mainQN.update_target_network()

    summaryOps = tf.summary.merge_all()

    eval_summary_ph = tf.placeholder(tf.float32, shape=(4,), name='evaluation')
    evalOps = (tf.summary.scalar('performance', eval_summary_ph[0]),
               tf.summary.scalar('perform_std', eval_summary_ph[1]),
               tf.summary.scalar('flicker_performance', eval_summary_ph[2]),
               tf.summary.scalar('flicker_perform_std', eval_summary_ph[3]))
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
            
            S, r, prev_life_count = env.reset()
            S = np.reshape(S, (1, 84, 84))
            mainQN.reset_hidden_state()
            
        action, _ = mainQN.get_action_stateful(S, prev_a=0)
        S_new, r, is_done, life_count = env.step(action, epsilon=util.epsilon_at(i))
        S_new = np.reshape(S_new, (1, 84, 84))
        exp_buf.append_trans((
            S, action, r, S_new,  # not cliping reward (huber loss)
            (prev_life_count and life_count < prev_life_count or is_done)
        ))
        S = S_new
        prev_life_count = life_count

        if not i:
            start_time = time.time()
            util.pickle.dump(exp_buf, open(KICKSTART_EXP_BUF_FILE, 'wb'))


        if util.Exiting or not i % ckpt_freq:
            util.checkpoint(mainQN.sess, saver, identity,
                       exp_buf, env, i, is_done,
                       prev_life_count, action, mainQN.hidden_state, S)
            if util.Exiting:
                raise SystemExit

        if i < 0: continue

        # TRAIN
        _, summary = mainQN.update_model_stateful(
            *exp_buf.sample_traces(batch_size),
            addtional_ops=[summaryOps])
                                     
        # Summary
        if not i % summary_freq:
            summary_writer.add_summary(summary, i)

        # Target Update
        if not i % target_update_freq:
            mainQN.update_target_network()
            cur_time = time.time()
            print(i, identity)
            print('[{}{}:{}] took {} seconds to {} steps'.format(
                model, trace_length, i, cur_time-start_time, target_update_freq), flush=1)
            start_time = cur_time

        # Evaluate
        if not i % eval_freq:
            eval_res = np.array(evaluate(mainQN, env_name, is_render=render_eval))
            eval_vals = mainQN.sess.run(
                evalOps, feed_dict={eval_summary_ph: eval_res})
            for v in eval_vals:
                summary_writer.add_summary(v, i)

    util.checkpoint(mainQN.sess, saver, identity)


def evaluate(mainQN, env_name, skip=4, scenario_count=5, is_render=False):
    start_time = time.time()
    def total_scenario_reward(flicker=0):
        env = Env(env_name=env_name, skip=skip, flicker=flicker, force=True)
        (s, R, _), t, action, state = env.reset(), False, 0, mainQN.ZERO_STATE
        # frame_count = 0
        while not t:
            # frame_count += 4
            action, state = mainQN.get_action_stateful([s], prev_a=action, state=state)
            s, r, t, _ = env.step(action, epsilon=0.1)
            R += r
            if is_render:
                env.render()
            # if frame_count and not frame_count % 10000:
            #     print(frame_count, action)
        return R

    res = np.array([total_scenario_reward() for _ in range(scenario_count)])
    res_flicker = np.array([total_scenario_reward(flicker=.5) for _ in range(scenario_count)])

    print(time.time() - start_time, 'seconds to evaluate', flush=1)
    print(res, res_flicker)
    print('Eval mean', np.mean(res), np.mean(res_flicker))
    return np.mean(res), np.std(res), np.mean(res_flicker), np.std(res_flicker)


def main():
    signal.signal(signal.SIGINT, util.signal_handler)
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--trace_length', action='store', type=int, default=10)
    parser.add_argument('-d', '--h_size', action='store', type=int, default=512)
    parser.add_argument('-e', '--env_name', action='store', default='Pong')
    parser.add_argument('-q', '--num_quant', action='store', type=int, default=0)
    parser.add_argument('-a', '--use_actions', action='store', type=int, default=0)
    train(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()

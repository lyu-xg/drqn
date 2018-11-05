import signal
import os
import argparse
import numpy as np
import tensorflow as tf

import common as util
from buffer import StackBuf, FrameBuf
from myenv import Env
from dqn_network import Qnetwork


EVAL_EPSILON = 0.05
MAX_EVAL_STEP = 60 * 60 * 5 # 60HZ for 5 minutes

def reset(stack_length, env, frame_buf):
    frame, R, lives = env.reset()
    frame_buf.append(frame)
    for _ in range(1, stack_length):
        frame, r, terminal, lives = env.step(0)
        frame_buf.append(frame)
        R += r
        if terminal:
            return reset(stack_length, env, frame_buf)
    return R, lives
    

def train(stack_length, render_eval=False, h_size=512, target_update_freq=10000,
          ckpt_freq=500000, summary_freq=1000, eval_freq=10000,
          batch_size=32, env_name='Pong', total_iteration=5e7,
          pretrain_steps=50000):
    # KICKSTART_EXP_BUF_FILE = 'cache/stack_buf_random_policy_{}.p'.format(pretrain_steps)
    identity = 'stack={},env={},mod={}'.format(stack_length, env_name, 'dqn')

    env = Env(env_name=env_name, skip=4)

    tf.reset_default_graph()

    # loads of side effect! e.g. initialize session, creating graph, etc.
    mainQN = Qnetwork(h_size, env.n_actions, stack_length, 'main', train_batch_size=batch_size)
    
    saver = tf.train.Saver(max_to_keep=5)
    summary_writer = tf.summary.FileWriter('./log/' + identity, mainQN.sess.graph)

    if util.checkpoint_exists(identity):
        (exp_buf, env, last_iteration, is_done,
         prev_life_count, action, frame_buf) = util.load_checkpoint(mainQN.sess, saver, identity)
        start_time = util.time()
    else:
        frame_buf = FrameBuf(size=stack_length)
        exp_buf, last_iteration = (
            (StackBuf(size=util.MILLION), 1 - pretrain_steps))
            # if not os.path.isfile(KICKSTART_EXP_BUF_FILE)
            # else (util.load(KICKSTART_EXP_BUF_FILE), 1))
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

    # Main Loop
    for i in range(last_iteration, int(total_iteration)):
        if is_done:
            scen_reward, scen_length = exp_buf.get_and_reset_reward_and_length()
            if i > 0:
                online_perf, online_episode_count = mainQN.sess.run(onlineOps, feed_dict={
                    online_summary_ph: np.array([scen_reward, scen_length])})
                summary_writer.add_summary(online_perf, i)
                summary_writer.add_summary(online_episode_count, i)

            _, prev_life_count = reset(stack_length, env, frame_buf)
            action = mainQN.get_action(list(frame_buf))

        s, r, is_done, life_count = env.step(action, epsilon=util.epsilon_at(i))
        exp_buf.append_trans((
            list(frame_buf), action, r, list(frame_buf.append(s)),  # not cliping reward (huber loss)
            (prev_life_count and life_count < prev_life_count or is_done)
        ))
        prev_life_count = life_count
        action = mainQN.get_action(list(frame_buf))

        if not i:
            start_time = util.time()
            util.pickle.dump(exp_buf, open(KICKSTART_EXP_BUF_FILE, 'wb'))
            
        if i <= 0: continue

        if util.Exiting or not i % ckpt_freq:
            util.checkpoint(mainQN.sess, saver, identity,
                       exp_buf, env, i, is_done,
                       prev_life_count, action, frame_buf)
            if util.Exiting:
                raise SystemExit

        if not i % target_update_freq:
            mainQN.update_target_network()
            cur_time = util.time()
            print('[{}{}:{}] took {} seconds to {} steps'.format(
                'dqn', stack_length, util.unit_convert(i), (cur_time-start_time)//1, target_update_freq), flush=1)
            start_time = cur_time

        #　TRAIN
        trainBatch = exp_buf.sample_batch(batch_size)

        _, summary = mainQN.update_model(*trainBatch, additional_ops=[summaryOps])

        if not i % summary_freq:
            summary_writer.add_summary(summary, i)
        if not i % eval_freq:
            eval_res = np.array(
                evaluate(mainQN, env_name, is_render=render_eval))
            perf, perf_std = mainQN.sess.run(
                evalOps, feed_dict={eval_summary_ph: eval_res})
            summary_writer.add_summary(perf, i)
            summary_writer.add_summary(perf_std, i)
    # In the end
    util.checkpoint(mainQN.sess, saver, identity)


def evaluate(mainQN, env_name, skip=4, scenario_count=5, is_render=False):
    start_time = util.time()
    env = Env(env_name=env_name, skip=skip)
    frame_buf = FrameBuf(size=mainQN.stack_size)
    def total_scenario_reward():
        t = 0
        R, _ = reset(mainQN.stack_size, env, frame_buf)
        for _ in range(MAX_EVAL_STEP):
            action = mainQN.get_action(list(frame_buf))
            s, r, t, _ = env.step(action, epsilon=EVAL_EPSILON)
            frame_buf.append(s)
            R += r
            if is_render:
                env.render()
            if t: break
        return R

    res = np.array([total_scenario_reward() for _ in range(scenario_count)])
    print((util.time() - start_time)//2, 'seconds to evaluate', flush=1)
    print('Eval:', res, 'mean =', np.mean(res))
    return np.mean(res), np.std(res)


def main():
    signal.signal(signal.SIGINT, util.signal_handler)
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--stack_length', action='store', type=int, default=4)
    parser.add_argument('-e', '--env_name', action='store', default='Pong')
    args = vars(parser.parse_args())
    train(**args)

if __name__ == '__main__':
    main()

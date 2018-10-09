import signal
import argparse
import numpy as np
import tensorflow as tf

import common as util
from buffer import StackBuf, FrameBuf
from myenv import Env
from networks.dqn_network import Qnetwork


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
          batch_size=32, env_name='SpaceInvaders', total_iteration=5e7,
          pretrain_steps=5000):
    identity = 'stack={},env={},mod={}'.format(stack_length, env_name, 'dqn')

    env = Env(env_name=env_name, skip=4)
    a_size = env.n_actions

    tf.reset_default_graph()
    mainQN = Qnetwork(h_size, a_size, stack_length, 'main')
    targetQN = Qnetwork(h_size, a_size, stack_length, 'target')
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
         prev_life_count, action, frame_buf) = util.load_checkpoint(sess, saver, identity)
        start_time = util.time()
    else:
        exp_buf = StackBuf(size=util.MILLION)
        frame_buf = FrameBuf(size=stack_length)
        last_iteration = 1 - pretrain_steps
        is_done = True
        prev_life_count = None
        sess.run(updateOps)

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
                online_perf, online_episode_count = sess.run(onlineOps, feed_dict={
                    online_summary_ph: np.array([scen_reward, scen_length])})
                summary_writer.add_summary(online_perf, i)
                summary_writer.add_summary(online_episode_count, i)

            _, prev_life_count = reset(stack_length, env, frame_buf)
            action = mainQN.get_action(sess, list(frame_buf))

        s, r, is_done, life_count = env.step(action)
        exp_buf.append_trans((
            list(frame_buf), action, r, list(frame_buf.append(s)),  # not cliping reward (huber loss)
            (prev_life_count and life_count < prev_life_count or is_done)
        ))
        prev_life_count = life_count

        action = (env.rand_action() 
                  if np.random.random() < util.epsilon_at(i)
                  else mainQN.get_action(sess, list(frame_buf)))

        if not i:
            start_time = util.time()
            
        if i <= 0:
            continue

        if util.Exiting or not i % ckpt_freq:
            util.checkpoint(sess, saver, identity,
                       exp_buf, env, i, is_done,
                       prev_life_count, action, frame_buf)
            if util.Exiting:
                raise SystemExit

        if not i % target_update_freq:
            sess.run(updateOps)
            cur_time = util.time()
            print('[{}{}:{}] took {} seconds to {} steps'.format(
                'dqn', stack_length, i, cur_time-start_time, target_update_freq), flush=1)
            start_time = cur_time

        #　TRAIN
        trainBatch = exp_buf.sample_batch(batch_size)

        Q1 = sess.run(mainQN.predict, feed_dict={
            mainQN.scalarInput: np.vstack(trainBatch[3]/255.0),
            mainQN.batch_size: batch_size
        })
        Q2 = sess.run(targetQN.Qout, feed_dict={
            targetQN.scalarInput: np.vstack(trainBatch[3]/255.0),
            targetQN.batch_size: batch_size
        })
        end_multiplier = - (trainBatch[4] - 1)
        doubleQ = Q2[range(batch_size), Q1]
        targetQ = trainBatch[2] + (0.99 * doubleQ * end_multiplier)

        # print(targetQ.shape)
        _, summary = sess.run((mainQN.updateModel, summaryOps), feed_dict={
            mainQN.scalarInput: np.vstack(trainBatch[0]/255.0),
            mainQN.targetQ: targetQ,
            mainQN.actions: trainBatch[1],
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
    start_time = util.time()
    env = Env(env_name=env_name, skip=skip)
    frame_buf = FrameBuf(size=mainQN.stack_size)
    def total_scenario_reward():
        t = 0
        R, _ = reset(mainQN.stack_size, env, frame_buf)
        while not t:
            action = mainQN.get_action(sess, list(frame_buf))
            s, r, t, _ = env.step(action)
            frame_buf.append(s)
            R += r
            if is_render:
                env.render()
        return R

    res = np.array([total_scenario_reward() for _ in range(scenario_count)])
    print(util.time() - start_time, 'seconds to evaluate', flush=1)
    return np.mean(res), np.std(res)


def main():
    signal.signal(signal.SIGINT, util.signal_handler)
    parser = argparse.ArgumentParser()
    parser.add_argument('stack_length', action='store', type=int, default=4)
    parser.add_argument('-e', '--env_name', action='store',
                        default='SpaceInvadersNoFrameskip-v4')
    train(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()

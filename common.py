import os
import pickle
import tensorflow as tf
import numpy as np
from time import time

MILLION = int(1e6)

def epsilon_at(i, anneal_bottom=1e6):
    return (-0.9/anneal_bottom) * i + 1 if i < anneal_bottom else .1

ckpt_prefix = './ckpts/'
def checkpoint_dir(identity):
    return ckpt_prefix + str(identity) + '/'

def checkpoint_exists(identity):
    return os.path.isdir(checkpoint_dir(identity))

Exiting = 0
def signal_handler(sig, frame):
    global Exiting
    # if 'Exiting' not in globals():
    #     Exiting = 0
    print('signal captured, trying to save states.', flush=1)
    Exiting += 1
    if Exiting > 2:
        print('okay got it, exiting without saving.', flush=1)
        raise SystemExit

def save(obj, filename):
    pickle.dump(obj, open(filename, 'wb'))

def load(filename):
    return pickle.load(open(filename, 'rb'))

def unit_convert(i):
    if i // MILLION:
        return '{}M'.format(i/MILLION)
    else:
        return '{}K'.format(i/1000)

def checkpoint(sess, saver, identity, *args):
    print('checkpointing.')
    ckpt_dir = checkpoint_dir(identity)
    if not checkpoint_exists(identity):
        os.makedirs(ckpt_dir)
    saver.save(sess, ckpt_dir+'tf_vars.ckpt')
    # for i, arg in enumerate(args):
    save(args, ckpt_dir + 'run_state.p')
    print('checkpoiting finished.')


def load_checkpoint(sess, saver, identity):
    print('loading from checkpoint')
    ckpt_dir = checkpoint_dir(identity)
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)
    res = load(ckpt_dir + 'run_state.p')
    print('loading finished.')
    return res
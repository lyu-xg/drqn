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

def signal_handler(sig, frame):
    global Exiting
    if 'Exiting' not in globals():
        Exiting = 0
    print('signal captured, trying to save states.', flush=1)
    Exiting += 1
    if Exiting > 3:
        print('okay got it, exiting without saving.', flush=1)
        raise SystemExit

def getTargetUpdateOps(tfVars):
    # tfVars consists of all trainble TF Variables
    # where the first half is from the main network
    #  and the second half is from the target network
    # RETURNS: list of operations which when run,
    #          updates the target network with main network's values
    return [vt.assign(vm.value())
            for vm, vt in zip(tfVars[:len(tfVars)//2],
                              tfVars[len(tfVars)//2:])]


def checkpoint(sess, saver, identity, *args):
    print('checkpointing.')
    ckpt_dir = checkpoint_dir(identity)
    if not checkpoint_exists(identity):
        os.makedirs(ckpt_dir)
    saver.save(sess, ckpt_dir+'tf_vars.ckpt')
    for i, arg in enumerate(args):
        pickle.dump(arg, open(ckpt_dir + '{}.p'.format(i), 'wb'))
    print('checkpoiting finished.')


def load_checkpoint(sess, saver, identity):
    print('loading from checkpoint')
    ckpt_dir = checkpoint_dir(identity)
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)
    res, i = [], 0
    while True:
        print(i, end='')
        f_path = ckpt_dir + '{}.p'.format(i)
        if not os.path.isfile(f_path):
            break
        res.append(pickle.load(open(f_path, 'rb')))
        i += 1
    print('...checkpoint loaded.')
    return res
import pickle
import os
import gym
import tensorflow as tf
from common import checkpoint_dir, checkpoint_exists

def checkpoint(sess, saver, identity, *args):
    ckpt_dir = checkpoint_dir(identity)
    if not checkpoint_exists(identity):
        os.makedirs(ckpt_dir)
    saver.save(sess, ckpt_dir+'tf_vars.ckpt')
    for i, arg in enumerate(args):
        pickle.dump(arg, open(ckpt_dir+'{}.p'.format(i), 'wb'))
        
def load_checkpoint(sess, saver, identity):
    ckpt_dir = checkpoint_dir(identity)
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)
    res, i = [], 0
    while True:
        f_path = ckpt_dir + '{}.p'.format(i)
        if not os.path.isfile(f_path): break
        res.append(pickle.load(open(f_path, 'rb')))
        i += 1
    return res
    

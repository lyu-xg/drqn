import pickle
import os
from keras.models import load_model
from keras import backend as K
from common import checkpoint_dir, checkpoint_exists

# Note: pass in_keras=False to use this function with raw numbers of numpy arrays for testing
def huber_loss(a, b, in_keras=True):
    error = a - b
    quadratic_term = error*error / 2
    linear_term = abs(error) - 1/2
    use_linear_term = (abs(error) > 1.0)
    if in_keras:
        # Keras won't let us multiply floats by booleans, so we explicitly cast the booleans to floats
        use_linear_term = K.cast(use_linear_term, 'float32')
    return use_linear_term * linear_term + (1-use_linear_term) * quadratic_term

def load_model_from(model_to_load):
    return load_model(model_to_load, custom_objects={'huber_loss': huber_loss})

def copy_weights(model1, model2):
    model2.set_weights(model1.get_weights())

def checkpoint(exp_buf, frame_buf, model, env, i, identity):
    print('check-pointing...', end=' ', flush=1)
    ckpt_dir = checkpoint_dir(identity)
    if not checkpoint_exists(identity):
        os.makedirs(ckpt_dir)
    def dump(obj, obj_name):
        pickle.dump(obj, open(ckpt_dir+obj_name, 'wb'))

    dump(exp_buf, 'exp_buf.p')
    dump(frame_buf, 'frame_buf.p')
    model.save(ckpt_dir+'model.h5')
    dump(env, 'env.p')
    dump(i, 'i.p')
    print('done.', flush=1)


def load_checkpoint(identity):
    ckpt_dir = checkpoint_dir(identity)
    def load(obj_name):
        return pickle.load(open(ckpt_dir+obj_name, 'rb'))
    return (load('exp_buf.p'),\
        load('frame_buf.p'),\
        load_model_from(ckpt_dir+'model.h5'),\
        load('env.p'),\
        load('i.p'))
import numpy as np
import argparse
import signal
import gym
from time import time as current_time
from keras.layers import Input, Lambda, Reshape, Conv2D, Flatten, Dense, multiply
from keras.models import Model
from keras.optimizers import RMSprop

from buffer import ExpBuf, Logger, FrameBuf
from evaluate import evaluate
from common import *
from common_keras import *


# global vars
MILLION = int(1e6)
DOWNSAMPLE_SIZE = (110, 84)
BATCH_SIZE = 32
Exiting = 0

def signal_handler(sig, frame):
    global Exiting
    print('signal captured, trying to save states.', flush=1)
    Exiting += 1
    if Exiting > 3:
        raise SystemExit

def init_ddqn(frame_shape, n_actions):
    # With the functional API we need to define the inputs.
    frames_input = Input(frame_shape, name='frames')
    actions_input = Input((n_actions,), name='mask')

    # convert RGB color value from range [0,255] to [0,1]
    normalized = Lambda(lambda x: x / 255.0)(frames_input)
        
    conv_1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu', data_format='channels_first')(normalized)
    
    conv_2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu', data_format='channels_first')(conv_1)
    
    conv_3 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', data_format='channels_first')(conv_2)
    
    conv_3_V = Lambda(lambda x:x[:,:32,:,:])(conv_3)
    conv_3_A = Lambda(lambda x:x[:,32:,:,:])(conv_3)
    
    # Flattening the second convolutional layer.
    conv_flattened_V = Flatten()(conv_3_V)
    conv_flattened_A = Flatten()(conv_3_A)
    
    
    hidden_V = Dense(512, activation='relu')(conv_flattened_V)
    hidden_A = Dense(512, activation='relu')(conv_flattened_A)
    
    output_V = Dense(1)(hidden_V)
    output_A = Dense(n_actions)(hidden_A)
    
    # combine the two streams: value and advantages
    def combine_V_A(V_A):
        V, A = V_A
        A -= K.mean(A)
        A += V
        return A
    
    output = Lambda(combine_V_A, output_shape=(n_actions,))([output_V, output_A])
    # Finally, we multiply the output by the mask!
    filtered_output = multiply([output, actions_input])
    
    print(filtered_output.shape)

    model = Model(inputs=[frames_input, actions_input], outputs=filtered_output)
    optimizer = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    model.compile(optimizer, loss=huber_loss)
    return model


def init_dqn(frame_shape, n_actions):
    # With the functional API we need to define the inputs.
    frames_input = Input(frame_shape, name='frames')
    actions_input = Input((n_actions,), name='mask')

    # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0, 1].
    normalized = Lambda(lambda x: x / 255.0)(frames_input)
        
    conv_1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu', data_format='channels_first')(normalized)
    
    conv_2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu', data_format='channels_first')(conv_1)
    
    conv_3 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', data_format='channels_first')(conv_2)
    
    # Flattening the second convolutional layer.
    conv_flattened = Flatten()(conv_3)
    
    # "The final hidden layer is fully-connected and consists of 256 rectifier units."
    hidden = Dense(512, activation='relu')(conv_flattened)
    
    # "The output layer is a fully-connected linear layer with a single output for each valid action."
    output = Dense(n_actions)(hidden)
    # Finally, we multiply the output by the mask!
    filtered_output = multiply([output, actions_input])

    model = Model(inputs=[frames_input, actions_input], outputs=filtered_output)
    optimizer = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    model.compile(optimizer, loss=huber_loss)
    return model

def get_models(frame_shape, n_actions, init_model_fn):
    model = init_model_fn(frame_shape, n_actions)
    targetModel = init_model_fn(frame_shape, n_actions)
    copy_weights(model, targetModel)
    return model, targetModel

def get_dqn_models(frame_shape, n_actions):
    return get_models(frame_shape, n_actions, init_dqn)

def get_ddqn_models(frame_shape, n_actions):
    return get_models(frame_shape, n_actions, init_ddqn)

def fit_batch(model, target_model, batch, n_actions, discount=0.99):
    start_states, a, next_states, rewards, is_terminal = zip(*batch)

    actions = one_hot(a, n_actions)

    next_states = np.array(next_states)
    start_states = np.array(start_states)

    next_actions = np.argmax(
        model.predict([next_states, np.ones((len(batch), n_actions))]),
        axis=1)
    # First, predict the Q values of the next states. Note how we are passing ones as the mask.
    next_Q_values = target_model.predict([next_states, one_hot(next_actions, n_actions)])

    # TRY IT OUT AND TRAIN

    # The Q values of the terminal states is 0 by definition, so override them
    next_Q_values[is_terminal] = 0
    # The Q values of each start state is the reward + gamma * the max next state Q value
    Q_values = np.array(rewards) + discount * np.max(next_Q_values, axis=1)
    # Fit the keras model. Note how we are passing the actions as the mask and multiplying
    # the targets by the actions.
    return model.fit(
        [start_states, actions], actions * Q_values[:, None],
        epochs=1, batch_size=BATCH_SIZE, verbose=0, shuffle=False
    )



MODELS = {
    'ddqn': get_ddqn_models,
    'dqn':  get_dqn_models,
    'DDQN': get_ddqn_models,
    'DQN':  get_dqn_models,
    # 'drqn': init_drqn,
    # 'DRQN': init_drqn,
}
def train(stack_size, env_name, load, save, runto_finish, model, total_iteration=5e7):
    global Exiting
    identity = 'stack={},env={},mod={}'.format(stack_size, env_name, model)
    frame_shape = (stack_size,) + DOWNSAMPLE_SIZE
    logger = Logger('log/{}.log'.format(identity))

    if checkpoint_exists(identity) and load:
        exp_buf, frame_buf, model, env, i = load_checkpoint(identity)
        print(i)
        (last_iteration, \
        is_done, prev_life_count, prev_action, prev_action_taken) = i
        A_n = env.action_space.n
        _, target_model = MODELS[model](frame_shape, A_n)
        copy_weights(model, target_model)
        start_time = current_time()
        print(last_iteration, is_done, prev_action_taken, prev_life_count)
    else:
        env = gym.make(env_name)
        A_n = env.action_space.n
        model, target_model = MODELS[model](frame_shape, A_n)
        #, model_to_load='model-SI.h5')
        exp_buf = ExpBuf(size=MILLION)
        frame_buf = FrameBuf(size=stack_size)
        last_iteration = 1-50000

        is_done = True
        prev_life_count = None
        prev_action = 0
        prev_action_taken = 0

    total_reward = 0

    for i in range(last_iteration, int(total_iteration)):
        if is_done:
            # env.render()
            prev_life_count = reset(env, frame_buf)
        
        epsilon = epsilon_at(i)
        if prev_action_taken < 3: # take same action 3 times.
            action = prev_action
        elif np.random.random() < epsilon:
            # note that during pre-run, i < 0, therefore epsilon > 1
            action = env.action_space.sample()
        else:
            action = np.argmax(model.predict([[frame_buf.toarray()],np.ones((1,A_n))]))

        prev_action = action
        prev_action_taken = (prev_action_taken + 1) % 4
        
        new_frame, reward, is_done, life_count = step(env, action, clip=False)
        
        exp_buf.append((list(frame_buf),
                        action,
                        list(frame_buf.append(new_frame)),
                        np.sign(reward),
                        (prev_life_count and life_count<prev_life_count) or is_done))
        
        prev_action = action
        prev_life_count = life_count
        
        total_reward += reward

        if i == 0:
            start_time = current_time()
            last_iteration = 0

        if i >= 0:
            # TODO this effects the timer
            history = fit_batch(model, target_model, exp_buf.sample_batch(BATCH_SIZE), A_n)
            loss = history.history['loss'][0]
        else:
            loss = 0
        
        if not i%1000:
            # env.render()
            copy_weights(model, target_model)
            m, std = evaluate(model, env_name, scenario_count=10)
            if i>0:
                print_time_estimate(start_time, iteration=(i-last_iteration), total=(total_iteration-last_iteration))
            summary = (i, total_reward, epsilon, loss, m, std)
            print('frame={}, total_reward={}, ε={:.2f}, loss={}, performance={}, std={:.2f}'.format(*summary), flush=True)
            logger.log(','.join(map(str, summary)))
            total_reward = 0
            if Exiting or not runto_finish and i>0 and current_time() - start_time >= 84600:
                logger._flush()
                if save:
                    checkpoint(exp_buf, frame_buf, model, env, (i, \
                        is_done, prev_life_count, prev_action, prev_action_taken), identity)
                raise SystemExit
        if save and not i%100000:
            checkpoint(exp_buf, frame_buf, model, env, (i, \
                       is_done, prev_life_count, prev_action, prev_action_taken), identity)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    parser = argparse.ArgumentParser()
    parser.add_argument('stack_size', action='store', type=int, default=4)
    parser.add_argument('-e', '--env_name', action='store', default='SpaceInvadersNoFrameskip-v4')
    parser.add_argument('-l', '--load', type=int, default=1)
    parser.add_argument('-s', '--save', type=int, default=1)
    parser.add_argument('-f', '--runto_finish', type=int, default=0)
    parser.add_argument('-m', '--model', action='store', default='ddqn')
    # parser.add_argument('total_iteration', action='store', type=int, default=50000000)
    train(**vars(parser.parse_args()))

if __name__ == '__main__':
    main()
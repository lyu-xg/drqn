import cv2
import gym
import numpy as np

ZERO_FRAME = np.zeros((7056,), dtype=np.int8)

def to_grayscale(frame):
    return np.mean(frame, axis=2).astype(np.uint8)

def downsample(frame, shape):
    # cv2 reverses the input shape for some reason..
    # so let's reverse it back..
    return cv2.resize(frame, tuple(reversed(shape)))

class Env:
    def __init__(self, env_name='SpaceInvaders', skip=4, noop=30):
        if '-' not in env_name:
            env_name += 'NoFrameskip-v4'
        # prefix parameters format: [<param>@<value>:]*
        # example env_name: flicker@.7:block@0:SpaceInvaders
        lst = env_name.split(':')
        options, gymEnvName = dict(prefix.split('@') for prefix in lst[:-1]), lst[-1]

        # setup flickering class var
        self.flicker_prob = float(options.get('flicker', 0))
        if self.flicker_prob:
            print('enabling flickering at prob=', self.flicker_prob, flush=1, sep='')

        self.env = gym.make(gymEnvName)
        assert (skip >= 2)
        self.skip = skip
        self.n_actions = self.env.action_space.n
        self.noop = noop

    def preprocess(self, s):
        if self.flicker_prob and np.random.random() < self.flicker_prob:
            return ZERO_FRAME
        # return downsample(to_grayscale(s[8:-12]), (84, 84)).reshape((7056,))
        return downsample(to_grayscale(s), (84, 84)).reshape((7056,))

    def rand_action(self):
        return self.env.action_space.sample()

    def reset(self):
        self.frame_count = 0
        frames, rewards = [self.preprocess(self.env.reset())], 0
        for _ in range(np.random.randint(self.skip - 1, self.noop)):
            frame, reward, terminal, summary = self.env.step(self.rand_action())
            frames.append(self.preprocess(frame))
            if terminal:
                return self.reset()
            rewards += reward

        return np.maximum(frames[-2], frames[-1]), rewards, summary['ale.lives']

    def step(self, action, epsilon=0.1):
        self.frame_count += self.skip
        frames, rewards = [], 0
        if epsilon and np.random.random() < epsilon:
            action = self.rand_action()
        for _ in range(self.skip):
            frame, reward, terminal, summary = self.env.step(action)
            frames.append(self.preprocess(frame))
            rewards += reward
            if terminal:
                break
        max_frame = (np.maximum(frames[-2], frames[-1])
                     if len(frames) > 1
                     else frames[0])
        return max_frame, rewards, terminal or self.frame_count > 3e5, summary['ale.lives']

    def render(self):
        self.env.render()

    def __del__(self):
        self.env.close()
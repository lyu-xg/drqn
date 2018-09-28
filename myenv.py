import cv2
import gym
import numpy as np


def to_grayscale(frame):
    return np.mean(frame, axis=2).astype(np.uint8)


def downsample(frame, shape):
    # cv2 reverses the input shape for some reason..
    # so let's reverse it back..
    return cv2.resize(frame, tuple(reversed(shape)))


def preprocess(s, flicker_prob=0):
    if flicker_prob and np.random.random() < flicker_prob:
        return np.zeros((84, 84))
    return downsample(to_grayscale(s[8:-12]), (84, 84)).reshape((7056,))


class Env:
    def __init__(self, env_name='SpaceInvaders', skip=4, noop=30):
        if '-' not in env_name:
            env_name += 'NoFrameskip-v4'
        self.env = gym.make(env_name)
        assert (skip >= 2)
        self.skip = skip
        self.n_actions = self.env.action_space.n
        self.noop = noop

    def rand_action(self):
        return self.env.action_space.sample()

    def reset(self):
        frames, rewards = [preprocess(self.env.reset())], 0
        for _ in range(np.random.randint(self.skip - 1, self.noop)):
            frame, reward, terminal, summary = self.env.step(self.rand_action())
            frames.append(preprocess(frame))
            if terminal:
                return self.reset()
            rewards += reward

        return np.maximum(frames[-2], frames[-1]), rewards, summary['ale.lives']

    def step(self, action):
        frames, rewards = [], 0
        for _ in range(self.skip):
            frame, reward, terminal, summary = self.env.step(action)
            frames.append(preprocess(frame))
            rewards += reward
            if terminal:
                break
        max_frame = (np.maximum(frames[-2], frames[-1])
                     if len(frames) > 1
                     else frames[0])
        return max_frame, rewards, terminal, summary['ale.lives']

    def render(self):
        self.env.render()

    def __del__(self):
        self.env.close()
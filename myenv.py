import cv2
import gym
import numpy as np


def to_grayscale(frame):
    return np.mean(frame, axis=2).astype(np.uint8)


def downsample(frame, shape):
    # cv2 reverses the input shape for some reason..
    # so let's reverse it back..
    return cv2.resize(frame, tuple(reversed(shape)))


def preprocess(frame, flicker=False, flicker_prob=0, shape=(110, 84)):
    if flicker and np.random.random() < flicker_prob:
        return np.zeros(shape)
    return to_grayscale(downsample(frame, shape))


class Env:
    def __init__(self, env_name='SpaceInvaders', skip=4):
        if '-' not in env_name:
            env_name += 'NoFrameskip-v4'
        self.env = gym.make(env_name)
        assert (skip >= 2)
        self.skip = skip
        self.n_actions = self.env.action_space.n

    def reset(self):
        frames, rewards = [preprocess(self.env.reset())], 0
        for _ in range(self.skip - 1):
            frame, reward, terminal, summary = self.env.step(
                self.env.action_space.sample())
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
        return max_frame, rewards, summary['ale.lives']

    def render(self):
        self.env.render()

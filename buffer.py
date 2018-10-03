import numpy as np
from common import MILLION


class ExpBuf:
    def __init__(self, size=1000000):
        # Pro-tip: when implementing a ring buffer, always allocate one extra element,
        # this way, self.start == self.end always means the buffer is EMPTY, whereas
        # if you allocate exactly the right number of elements, it could also mean
        # the buffer is full. This greatly simplifies the rest of the code.
        self.data = [None] * (size + 1)
        self.start = 0
        self.end = 0

    def append(self, element):
        self.data[self.end] = element
        self.end = (self.end + 1) % len(self.data)
        # end == start and yet we just added one element. This means the buffer has one
        # too many element. Remove the first element by incrementing start.
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)

    def __getitem__(self, idx):
        return self.data[(self.start + idx) % len(self.data)]

    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.data) - self.start
        else:
            return self.end - self.start

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def sample_batch(self, size):
        return np.array([self[i] for i in np.random.choice(len(self), size)])


class StackBuf(ExpBuf):
    def __init__(self, size=MILLION):
        super().__init__(size=size)
        self.scenario_reward = 0
        self.scenario_length = 0
    
    def append_trans(self, transition):
        '''
        transition: (s', a, r, s, t)
        '''
        super().append(transition)
        self.scenario_reward += transition[2]
        self.scenario_length += 1

    def get_and_reset_reward_and_length(self):
        r, self.scenario_reward = self.scenario_reward, 0
        l, self.scenario_length = self.scenario_reward, 0
        return r, l

    def sample_batch(self, size):
        batch = super().sample_batch(size)
        S_prime, A, R, S, T = zip(*batch)
        S_prime = np.array([np.array(s) for s in S_prime])
        S = np.array([np.array(s) for s in S])
        return S_prime, np.array(A), np.array(R), S, np.array(T)

class TraceBuf:
    def __init__(self, trace_length, scenario_size=10000):
        # uses the ring buf above to save episodes
        self.buf = ExpBuf(size=scenario_size)
        self.trans_cache = []
        self.trace_length = trace_length

    def flush_scenario(self):
        if len(self.trans_cache) >= self.trace_length:
            self.buf.append(np.array(self.trans_cache))
        self.trans_cache.clear()

    def get_cache_total_reward(self):
        return sum(ep[2] for ep in self.trans_cache)

    def append_trans(self, trans):
        # s_prime, a, r, s, t
        self.trans_cache.append(list(trans))

    def append_episode(self, ep):
        assert(len(ep) >= self.trace_length)
        self.buf.append(ep)

    def slice_ep(self, ep):
        anchor = np.random.randint(0, len(ep) + 1 - self.trace_length)
        return ep[anchor:anchor+self.trace_length]

    def sample_traces(self, batch_size):
        traces = np.array([self.slice_ep(e)
                           for e in self.buf.sample_batch(batch_size)])
        return np.reshape(traces, (batch_size * self.trace_length, 5))


class Logger:
    def __init__(self, filename, cache_size=50):
        self.cache = []
        self.filename = filename
        self.cache_size = cache_size

    def __del__(self):
        self._flush()

    def log(self, item):
        self.cache.append(item)
        if len(self.cache) >= self.cache_size:
            self._flush()

    def _flush(self):
        with open(self.filename, 'a+') as fp:
            fp.write('\n'.join(map(str, self.cache))+'\n')
        self.cache.clear()


from collections import deque
from imgutil import display_frames_as_gif


class FrameBuf:
    def __init__(self, size=4):
        self.frames = deque()
        self.size = size

    def append(self, frame):
        self.frames.append(frame)
        if len(self) > self.size:
            self.frames.popleft()
        return self

    def toarray(self):
        return np.array(list(self))

    def __iter__(self):
        return iter(self.frames)

    def __len__(self):
        return len(self.frames)

    def __str__(self):
        return '\n'.join('{} with shape {}'.format(type(f), f.shape) for f in self)

    def show(self):
        display_frames_as_gif(self.toarray())


if __name__ == '__main__':
    sb = StackBuf()
    print(len(sb.data))
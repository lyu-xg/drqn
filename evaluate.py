import argparse
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import gym
import time
import numpy as np
from common import preprocess, step_multiple, reset
from common_keras import load_model_from
from buffer import FrameBuf

def main(model_filename, env_name, scenario_count, is_render):
	mean, std = evaluate(
		load_model_from(model_filename),
		env_name,
		scenario_count,
		is_render)
	print('Mean:', mean, '±', std, flush=True)


def evaluate(model, env_name, scenario_count, is_render=False):
	env = gym.make(env_name)
	frame_buf = FrameBuf(size=model.input_shape[0][1])

	def run_scenario(): # returns total reward
		reset(env, frame_buf)
		terminal, total_r = 0, 0

		while not terminal:
			Q = model.predict([[frame_buf.toarray()],np.ones((1,env.action_space.n))])
			optimal_action = np.argmax(Q)

			r, terminal = step_multiple(env, optimal_action, frame_buf, 6, clip=False)
			total_r += r
			
			if is_render: env.render()
		return total_r

	res = np.array([run_scenario() for _ in range(scenario_count)])
	# print('\n'.join(str(r) for r in res), flush=True)
	return np.mean(res), np.std(res)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('model_filename', action='store')
	parser.add_argument('-e', '--env_name', action='store', default='SpaceInvadersNoFrameskip-v4')
	parser.add_argument('-c', '--scenario_count', help='number of scenarios to run',
						action='store', type=int, default=10)
	parser.add_argument('-r', '--is_render', action='store_true', default=False)
	parser.parse_args()
	main(**vars(parser.parse_args()))
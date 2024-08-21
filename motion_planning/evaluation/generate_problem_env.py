# generate and filter test cases using BIT*

# f"{str.rsplit(os.path.abspath(__file__), '/', 2)[0]}/models/env_file/{''.join(robot_name_list)}_{problem_num}_{obstacle_num}.npz"
import os
import pickle
from argparse import ArgumentParser
from tqdm import tqdm

import numpy as np
import pytorch_lightning as pl

from environment import ArmEnv
from neural_cbf.systems import ArmDynamics

from motion_planning.baseline.tsa import RRT_plan
from motion_planning.baseline.bit_star import BITStar


def generate_refined_test_cases(**kwargs):
	file_name = f"../../models/motion_planning/problems/pkl/{kwargs['robot_name']}_{kwargs['problem_num']}_{kwargs['obstacle_num']}_v{kwargs['version']}_{kwargs['seed']}_refined.pkl"

	environment = ArmEnv([kwargs['robot_name']], GUI=0, config_file='')
	robot = environment.robot_list[0]

	nominal_params = {'version': kwargs['version']}
	dynamics_model = ArmDynamics(
		nominal_params,
		dt=1/120,
		dis_threshold=0.05,
		controller_dt=1/30,
		env=environment,
		robot=robot
	)

	seed = kwargs['seed']
	pl.seed_everything(seed)

	pbar = tqdm(range(kwargs['problem_num']), desc='Generating test cases')
	for _ in pbar:
		while True:
			positions, sizes, start_config, end_config = propose_test_cases(dynamics_model,
																			obstacle_num=kwargs['obstacle_num'])

			environment.reset_env(enable_object=False, obs_configs=(positions, sizes))
			dynamics_model.set_goal(end_config)

			# BIT*
			solution = BITStar(env=environment, init_state=start_config, goal_state=end_config,
								dynamics_model=dynamics_model, steer_type='line', batch_size=50)
			if solution['success'] and solution['time_dict']['total_time']:
				with open(file_name, 'ab') as handle:
					solution.update({'obstacle_positions': positions,
									 'obstacle_sizes': sizes,
									 'init_config': start_config,
									 'goal_config': end_config})
					pickle.dump(solution, handle, protocol=pickle.HIGHEST_PROTOCOL)
				break
			else:
				print('no solution')


def propose_test_cases(dynamics_model, obstacle_num):
	while True:
		try:
			sizes = np.zeros((0, 3))
			for ii in range(obstacle_num):
				idx = np.random.randint(0, 3)
				size = np.random.uniform(low=(0.05, 0.05, 0.05), high=(0.25, 0.25, 0.25), size=(1, 3))
				if idx == 0:  # pad
					size[0, ii % 3] = 0.05
				elif idx == 1:  # stick
					size[0, ii % 3] = 0.05
					size[0, (ii + 1) % 3] = 0.05
				else:  # box
					pass
				sizes = np.concatenate((sizes, size), axis=0)
			while True:
				positions = np.random.uniform(low=(-0.5, -0.5, 0), high=(0.5, 0.5, 1), size=(obstacle_num, 3))  # magician and panda
				# positions = np.random.uniform(low=(-1, -1, 0), high=(1, 1, 1), size=(obstacle_num, 3))  # kuka-ext
				far_from_base = [not (positions[ii][2]-sizes[ii][2] < 0.3 and (abs(abs(positions[ii][0]) - sizes[ii][0]) < 0.1 or abs(abs(positions[ii][1]) - sizes[ii][1]) < 0.1)) for ii in range(obstacle_num)]
				if all(far_from_base):
					break
			dynamics_model.env.reset_env(obs_configs=(positions, sizes), enable_object=False)
			for _ in range(10):
				safe_configs = dynamics_model.sample_safe(2)
				start_config = safe_configs[0, :dynamics_model.n_dims].numpy()
				end_config = safe_configs[1, :dynamics_model.n_dims].numpy()
				if np.linalg.norm(start_config - end_config) > 0.7:
					return positions, sizes, start_config, end_config
		except RuntimeWarning:
			continue


if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument('--robot_name', type=str)

	# test cases params
	parser.add_argument('--problem_num', type=int, default=200)
	parser.add_argument('--obstacle_num', type=int, default=8)
	parser.add_argument('--version', type=str, default='supersafe')
	parser.add_argument('--seed', type=int, default=1234)

	args = parser.parse_args()

	generate_refined_test_cases(**vars(args))

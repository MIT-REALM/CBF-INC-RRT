import os
import time
import argparse
import pickle
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import matplotlib.pyplot as plt
import seaborn

import pybullet as p

from environment import ArmEnv

from neural_cbf.controllers import NeuralLidarCBFController, ImitationController, ReinforcementController
from neural_cbf.systems import ArmLidar
from motion_planning.evaluation.eval_rrt_all import get_test_cases

def init_dynamics(**kwargs):
	robot_name = kwargs['robot_name']
	config_file = f"../../models/motion_planning/problems/refined_pkl/{robot_name}_easy.pkl"
	environment = ArmEnv([robot_name], GUI=0, config_file=config_file)
	robot = environment.robot_list[0]

	nominal_params = {'name': 1}
	dynamics_model = ArmLidar(
		nominal_params,
		dt=kwargs['simulation_dt'],
		dis_threshold=0.02,
		controller_dt=kwargs['controller_period'],
		n_obs=1024,
		point_in_dataset_pc=1024,
		list_sensor=robot.body_joints,
		env=environment,
		robot=robot,
		observation_type='uniform_lidar' if kwargs['obs_only'] else 'uniform_surface',
		add_normal=True,
		point_dim=3,
	)
	return dynamics_model


def init_controller(dynamics_model, **kwargs):
	robot_name = kwargs['robot_name']
	method_model_dir = {
		'cbf': f"../../models/neural_cbf/collection/{robot_name}_lidar/",
		'RL': f"../../models/neural_cbf/RL_baseline/reinforcement_Lidar_{robot_name}_best",
		'IL': f"../../models/neural_cbf/IL_baseline/imitation_Lidar_{robot_name}",
	}
	controller = {'cbf': NeuralLidarCBFController, 'RL': ReinforcementController, 'IL': ImitationController}

	args.cbf_relaxation_penalty = 500.
	device = 'cpu' if not torch.cuda.is_available() else f"cuda:{int(float(kwargs['devices']))}"

	if kwargs['method_name'] == 'cbf':
		model_file_list = os.listdir(method_model_dir[args.method_name])
		for file in model_file_list:
			if file.endswith('.ckpt'):
				model_file = os.path.join(method_model_dir[args.method_name], file)
				break
		neural_controller = NeuralLidarCBFController.load_from_checkpoint(
			model_file,
			dynamics_model=dynamics_model, scenarios=[{'name': 1}],
			loss_config={},  # not used
			use_neural_actor=0,
			cbf_relaxation_penalty=args.cbf_relaxation_penalty,
			datamodule=None, experiment_suite=None,
			map_location=device,
			**kwargs,
		)
	elif kwargs['method_name'] in ['RL', 'IL']:
		model_file = method_model_dir[args.method_name] + '.pt'
		if robot_name == 'magician':
			neural_controller = controller[args.method_name](dynamics_model=dynamics_model,
															 feature_dim=32, per_feature_dim=64, use_bn=False,
															 hidden_size=48, hidden_layers=2, DEVICE=device, **kwargs)
		elif robot_name == 'panda':
			neural_controller = controller[args.method_name](dynamics_model=dynamics_model,
															 feature_dim=128, per_feature_dim=128, use_bn=False,
															 hidden_size=128, hidden_layers=1, DEVICE=device, **kwargs)
		else:
			raise NotImplementedError(f"robot {robot_name} not implemented")
		neural_controller.load(model_file, device=device)
	else:
		raise NotImplementedError(f"method {kwargs['method_name']} not implemented")

	neural_controller.to(device)
	return neural_controller


def eval_in_obs_env(controller, **kwargs):
	robot_name = kwargs['robot_name']
	config_file = f"../../models/motion_planning/problems/refined_pkl/{robot_name}_{kwargs['level']}.pkl"
	positions, sizes, start_configs, end_configs = get_test_cases(config_file, **kwargs)
	pbar = tqdm(range(kwargs['end_idx'] - kwargs['start_idx']))

	env = controller.dynamics_model.env
	robot = controller.dynamics_model.robot
	obs_num = 8

	goal_reached = []
	safe_states = []
	t_step = []

	for idx in pbar:
		safe = 0
		unsafe = 0
		goal_state = end_configs[idx]
		init_state = start_configs[idx]
		obs_position = positions[idx][:obs_num]
		obs_size = sizes[idx][:obs_num]

		env.reset_env(enable_object=False, obs_configs=(obs_position, obs_size))
		dynamics_model.set_goal(goal_state)
		robot.set_joint_position(robot.body_joints, init_state)

		controller_update_freq = dynamics_model.controller_dt // dynamics_model.dt
		x_current = dynamics_model.complete_sample_with_observations(
			torch.tensor(init_state, device=controller.device).unsqueeze(0), 1)

		for tstep in range(kwargs['max_steps']):
			if tstep % controller_update_freq == 0:
				u_current = controller.u(x_current)
				if isinstance(u_current, tuple):
					u_current, u_time_dict = u_current
				else:
					u_time_dict = {}
			if (tstep + 1) % controller_update_freq == 0:
				x_current, tt_list = dynamics_model.closed_loop_dynamics(x_current, u_current, update_observation=True,
																		 return_time=True)
			else:
				x_current, tt_list = dynamics_model.closed_loop_dynamics(x_current, u_current, update_observation=False,
																		 return_time=True)
			if dynamics_model.unsafe_mask(x_current):
				unsafe += 1
				# print(f"unsafe at step {tstep}")
				break
			else:
				safe += 1
			if dynamics_model.goal_mask(x_current):
				goal_reached.append(1)
				t_step.append(tstep)
				# print(f"goal reached at step {tstep}")
				break
			if kwargs['obs_only']:
				for iidx, obs in zip(range(len(env.obstacle_ids) - 1), env.obstacle_ids[1:]):
					obs_velocity = np.zeros(3)
					obs_velocity[iidx % 3] = 0.03
					env.obs_positions[iidx] += obs_velocity * dynamics_model.dt
					p.resetBasePositionAndOrientation(obs, env.obs_positions[iidx], [0, 0, 0, 1])
		safe_states.append([safe, unsafe])

	# with open(f"../../models/motion_planning/table3_supp/{robot_name}_{kwargs['method_name']}_{kwargs['seed']}_{kwargs['level']}.pkl", 'wb') as f:
	# 	pickle.dump({'goal_reached': np.sum(goal_reached),
	# 				 't_step': t_step, 'arg': kwargs,
	# 				 'safe_state': safe_states}, f, protocol=pickle.HIGHEST_PROTOCOL)
	print("=====================================")
	print(f"method: {kwargs['method_name']}, robot: {kwargs['robot_name']}, obs_only: {kwargs['obs_only']}")
	print(f"goal reached: {np.sum(goal_reached) / (kwargs['end_idx'] - kwargs['start_idx'])}")
	print(f"avg time to goal: {np.mean(t_step)}")
	# print(
	# 	f"state un-safety rate: {np.sum([x[1] for x in safe_states]) / np.sum([kwargs['max_steps'] for x in safe_states])}")
	print(f"traj un-safety rate: unsafe: {np.sum([bool(x[1]) for x in safe_states]) / len(safe_states)}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--robot_name', type=str, default='panda')
	parser.add_argument('--method_name', type=str, default='cbf', help='select from [cbf, IL, RL]')
	parser.add_argument('--obs_only', type=bool, default=False)

	# simulation params
	parser.add_argument('--controller_period', type=float, default=1 / 30)
	parser.add_argument('--simulation_dt', type=float, default=1 / 120)
	parser.add_argument('--level', type=str, default='easy')

	# planning params
	# parser.add_argument('--accelerator', type=str, default='gpu', help='cpu or gpu')
	parser.add_argument('--seed', type=int, default=20)
	parser.add_argument('--devices', type=str, default=0, help='gpu id')
	parser.add_argument('--start_idx', type=int, default=0)
	parser.add_argument('--end_idx', type=int, default=20)

	# tuning params
	parser.add_argument('--cbf_alpha', type=float, default=1)
	parser.add_argument('--max_steps', type=int, default=300, help="maximum steps")

	args = parser.parse_args()

	dynamics_model = init_dynamics(**vars(args))
	neural_controller = init_controller(dynamics_model, **vars(args))
	eval_in_obs_env(neural_controller, **vars(args))

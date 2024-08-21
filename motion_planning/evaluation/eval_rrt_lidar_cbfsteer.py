import os
import sys
from time import time, sleep
from tqdm import tqdm
import yaml
import argparse

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import pybullet as p

sys.path.append(os.getcwd())

from environment import ArmEnv
from neural_cbf.systems import ArmLidar
from neural_cbf.controllers import NeuralLidarCBFController
from motion_planning.baseline.tsa import RRT_plan
from neural_cbf.training.utils import current_git_hash


def eval_rrt_lidar(seed, environment, robot, method_model_dir, obs_positions, obs_sizes, start_configs, end_configs, **kwargs):
	pl.seed_everything(seed)

	with open(method_model_dir + 'hparams.yaml', 'r') as f:
		args = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))

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
		observation_type='uniform_surface',  #args.observation_type,
		add_normal=True,
		point_dim=args.point_dim,
	)

	args.cbf_relaxation_penalty = 500.
	device = 'cpu' if not torch.cuda.is_available() else f"cuda:{int(float(kwargs['devices']))}"

	model_file_list = os.listdir(method_model_dir)
	for file in model_file_list:
		if file.endswith('.ckpt'):
			model_file = os.path.join(method_model_dir, file)
			break
	print(model_file)
	cbf_controller = NeuralLidarCBFController.load_from_checkpoint(
		model_file,
		dynamics_model=dynamics_model, scenarios=[nominal_params],
		loss_config={},  # not used
		use_neural_actor=0,
		cbf_relaxation_penalty=args.cbf_relaxation_penalty,
		datamodule=None, experiment_suite=None,
		map_location=device,
		**kwargs,
	)

	cbf_controller.to(device)

	cbf_controller.h_nn.eval()
	cbf_controller.encoder.eval()
	cbf_controller.pc_head.eval()

	solutions = []
	pbar = tqdm(range(kwargs['end_idx'] - kwargs['start_idx']))
	for idx in pbar:
		goal_state = end_configs[idx]
		init_state = start_configs[idx]
		environment.reset_env(enable_object=False, obs_configs=(obs_positions[idx], obs_sizes[idx]))
		dynamics_model.set_goal(goal_state)

		solutions.append(RRT_plan(env=environment, init_state=init_state, goal_state=goal_state,
								  dynamics_model=dynamics_model,
								  controller=cbf_controller, RRT_PARAM=kwargs['RRT_STEP'],
								  steer_type='cbf', model_eps=kwargs['goal_biasing'],
								  T=kwargs['max_node'], stop_when_success=True))
		# print(solutions[-1])

	return solutions


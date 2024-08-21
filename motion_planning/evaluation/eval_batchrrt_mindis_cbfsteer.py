import os
import sys
from time import time, sleep
from tqdm import tqdm
from argparse import ArgumentParser

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import pybullet as p

sys.path.append(os.getcwd())

from neural_cbf.systems import ArmMindis
from neural_cbf.controllers import NeuralMindisCBFController
from motion_planning.baseline.batch_tsa import batch_RRT_plan


def eval_batchrrt_mindis(seed, environment, robot, method_model_dir, obs_positions, obs_sizes, start_configs,
						 end_configs, **kwargs):
	pl.seed_everything(seed)

	nominal_params = {'name': 1}
	dynamics_model = ArmMindis(
		nominal_params,
		dt=kwargs['simulation_dt'],
		dis_threshold=0.02,
		controller_dt=kwargs['controller_period'],
		env=environment,
		robot=robot
	)

	model_file_list = os.listdir(method_model_dir)
	for file in model_file_list:
		if file.endswith('.ckpt'):
			model_file = os.path.join(method_model_dir, file)
			break
	cbf_controller = NeuralMindisCBFController.load_from_checkpoint(
		model_file,
		dynamics_model=dynamics_model, scenarios=[nominal_params],
		datamodule=None, experiment_suite=None,
		# new args
		cbf_alpha=kwargs['cbf_alpha'],
		safe_level=0.01,
		unsafe_level=0.01,
		cbf_hidden_layers=2,
		cbf_hidden_size=48,
		use_neural_actor=False,
		cbf_relaxation_penalty=500
	)
	if not torch.cuda.is_available():
		cbf_controller.to('cpu')
	else:
		cbf_controller.to(f"cuda:{int(float(kwargs['devices']))}")

	solutions = []
	pbar = tqdm(range(kwargs['end_idx'] - kwargs['start_idx']))
	for idx in pbar:
		goal_state = end_configs[idx]
		init_state = start_configs[idx]
		environment.reset_env(enable_object=False, obs_configs=(obs_positions[idx], obs_sizes[idx]))
		dynamics_model.set_goal(goal_state)

		solutions.append(batch_RRT_plan(env=environment, init_state=init_state, goal_state=goal_state,
										dynamics_model=dynamics_model, batch=1,
										controller=cbf_controller, RRT_PARAM=kwargs['RRT_STEP'],
										steer_type='cbf', model_eps=kwargs['goal_biasing'],
										T=kwargs['max_node'], stop_when_success=True))

	return solutions

import os
import sys
from time import time, sleep
from tqdm import tqdm
from argparse import ArgumentParser

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

sys.path.append(os.getcwd())

from neural_cbf.systems import ArmDynamics
from neural_cbf.controllers import EightBlocksController
from motion_planning.baseline.tsa import RRT_plan


def eval_rrt_docbf(seed, environment, robot, method_model_dir, obs_positions, obs_sizes, start_configs, end_configs,
					**kwargs):
	pl.seed_everything(seed)

	nominal_params = {'name': 1}
	dynamics_model = ArmDynamics(
		nominal_params,
		dt=kwargs['simulation_dt'],
		dis_threshold=0.02,
		controller_dt=kwargs['controller_period'],
		env=environment,
		robot=robot
	)

	solutions = []
	pbar = tqdm(range(kwargs['end_idx'] - kwargs['start_idx']))
	for idx in pbar:
		goal_state = end_configs[idx]
		init_state = start_configs[idx]
		environment.reset_env(enable_object=False, obs_configs=(obs_positions[idx], obs_sizes[idx]))
		dynamics_model.set_goal(goal_state)

		# reload controller for each problem
		cbf_controller = EightBlocksController(
			dynamics_model=dynamics_model,
			scenarios=[nominal_params],
			experiment_suite=None,
			cbf_lambda=kwargs['cbf_alpha'],
			cbf_relaxation_penalty=5000,
			controller_period=dynamics_model.controller_dt,
			obs_sizes=obs_sizes[idx],
			obs_positions=obs_positions[idx]
		)

		solutions.append(RRT_plan(env=environment, init_state=init_state, goal_state=goal_state,
								  dynamics_model=dynamics_model,
								  controller=cbf_controller, RRT_PARAM=kwargs['RRT_STEP'],
								  steer_type='cbf', model_eps=kwargs['goal_biasing'],
								  T=kwargs['max_node'], stop_when_success=True))

	return solutions

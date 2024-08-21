import os
import sys

from argparse import ArgumentParser
from importlib_metadata import requires

import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import numpy as np
import pybullet as p

from environment import ArmEnv
from neural_cbf.systems import ArmMindis
from neural_cbf.controllers import NeuralMindisCBFController

from neural_cbf.datamodules.episodic_datamodule import (
	EpisodicDataModule,
)
from neural_cbf.experiments import (
	ExperimentSuite,
	BFContourExperiment,
	RolloutStateSpaceExperiment,
)

# from neural_cbf.training.utils import current_git_hash

torch.multiprocessing.set_sharing_strategy("file_system")


def main(args):
	# Define the scenarios
	nominal_params = {"m1": 5.76}
	scenarios = [
		nominal_params,
	]

	# Define environment and agent
	environment = ArmEnv([args.robot_name], GUI=False, config_file="")
	robot = environment.robot_list[0]

	# Define the dynamics model
	dynamics_model = ArmMindis(
		nominal_params,
		dt=args.simulation_dt,
		controller_dt=args.controller_period,
		dis_threshold=args.dis_threshold,
		env=environment,
		robot=robot
	)

	# Define goal_state
	goal_state = torch.tensor(robot.q0).float()
	dynamics_model.set_goal(goal_state)

	# Initialize the DataModule
	initial_conditions = [tuple(robot.body_range[i]) for i in range(robot.body_dim)]
	data_module = EpisodicDataModule(
		dynamics_model,
		initial_conditions,
		max_episode=args.max_episode,
		trajectories_per_episode=args.trajectories_per_episode,
		trajectory_length=args.trajectory_length,
		fixed_samples=args.fixed_samples,
		val_split=args.val_split,
		batch_size=args.batch_size,
		quotas={"safe": args.safe_portion, "goal": args.goal_portion, "unsafe": args.unsafe_portion},
	)

	# Define the experiment suite
	exp_suite_list = []

	if args.exp_cbf_contour:
		default_state = dynamics_model.complete_sample_with_observations(dynamics_model.goal_state.reshape(1, -1), num_samples=1).squeeze()
		cbf_contour_experiment = BFContourExperiment(
			"cbf_Contour",
			domain=[tuple(robot.body_range[args.contour_x_idx]), tuple(robot.body_range[args.contour_y_idx])],
			n_grid=30,
			x_axis_index=args.contour_x_idx,
			y_axis_index=args.contour_y_idx,
			x_axis_label=f"$\\theta_{args.contour_x_idx}$",
			y_axis_label=f"$\\theta_{args.contour_y_idx}$",
			default_state=default_state,
			plot_unsafe_region=True,
		)
		exp_suite_list.append(cbf_contour_experiment)

	if args.exp_rollout:
		ul, ll = dynamics_model.state_limits
		start_x = torch.cat([
			torch.lerp(ll, ul, 0.2 * torch.ones(ll.shape[-1]).double()).reshape(1, -1),
			torch.lerp(ll, ul, 0.8 * torch.ones(ll.shape[-1]).double()).reshape(1, -1),
		], dim=0).float()
		start_x = dynamics_model.complete_sample_with_observations(start_x, num_samples=start_x.shape[0])

		rollout_experiment = RolloutStateSpaceExperiment(
			"Rollout",
			start_x,
			args.rollout_x_idx,
			f"$\\theta_{args.rollout_x_idx}$",
			args.rollout_y_idx,
			f"$\\theta_{args.rollout_y_idx}$",
			scenarios=scenarios,
			n_sims_per_start=args.rollout_n_sim_per_start,
			t_sim=args.rollout_t_sim,
		)
		exp_suite_list.append(rollout_experiment)

	experiment_suite = ExperimentSuite(exp_suite_list)

	# Initialize the controller
	loss_config = {
        "u_coef_in_training": args.u_coef_in_training,
		"safe_classification_weight": args.safe_classification_weight,
		"unsafe_classification_weight": args.unsafe_classification_weight,
		"descent_violation_weight": args.descent_violation_weight,
		"actor_weight": args.actor_weight,
	}
	cbf_controller = NeuralMindisCBFController(dynamics_model, scenarios, data_module, experiment_suite,
											   safe_level=args.safe_level,
											   unsafe_level=args.unsafe_level,
											   cbf_hidden_layers=args.cbf_hidden_layers,
											   cbf_hidden_size=args.cbf_hidden_size,
											   cbf_alpha=args.cbf_alpha,
											   learn_shape_epochs=args.learn_shape_epochs,
											   loss_config=loss_config,
											   all_hparams=args,
											   use_neural_actor=False,
											   cbf_relaxation_penalty=5000,
											   )

	# Initialize the logger and trainer
	tb_logger = pl_loggers.TensorBoardLogger(
		save_dir=os.path.abspath(__file__).rsplit('/', 3)[0] + f"/models/neural_cbf/{dynamics_model}",
		name=f"{args.version}",
	)
	trainer = pl.Trainer(
		logger=tb_logger,
		reload_dataloaders_every_epoch=True,
		max_epochs=args.max_epochs,
		gpus=args.devices,  # only supporting single-GPU at present
	)

	# Train
	pl.seed_everything(args.seed)
	torch.autograd.set_detect_anomaly(True)
	trainer.fit(cbf_controller)


if __name__ == "__main__":
	parser = ArgumentParser()

	# experiment params
	parser.add_argument('--robot_name', type=str, default='magician')
	parser.add_argument('--version', type=str, default="multiple-seeds")

	# simulation params
	parser.add_argument('--dis_threshold', type=float, default=0.05)
	parser.add_argument('--controller_period', type=float, default=1 / 30)
	parser.add_argument('--simulation_dt', type=float, default=1 / 120)

	# CBF definition params
	parser.add_argument('--safe_level', type=float, default=0.1, help='h_safe < -safe_level')
	parser.add_argument('--unsafe_level', type=float, default=0.1, help='h_unsafe > unsafe_level')

	# training params
	parser.add_argument('--seed', type=int, default=1)
	parser.add_argument('--accelerator', type=str, default='gpu', help='cpu or gpu')
	parser.add_argument('--devices', type=str, default="0", help='gpu id')
	parser.add_argument('--max_epochs', type=int, default=101)
	parser.add_argument('--learn_shape_epochs', type=int, default=-1,
						help='different from max_epochs when training a neural policy')

	# neural network params
	parser.add_argument('--cbf_hidden_layers', type=int, default=2)
	parser.add_argument('--cbf_hidden_size', type=int, default=48)
	parser.add_argument('--cbf_alpha', type=float, default=1, help='lambda in (L_f V + L_g V u + lambda V <= 0)')

	# loss config params
	parser.add_argument('--u_coef_in_training', type=float, default=5e-1,
						help='control signal amplification coefficient in training')
	parser.add_argument('--safe_classification_weight', type=float, default=20,
						help='weight of safe region classification loss')
	parser.add_argument('--unsafe_classification_weight', type=float, default=20,
						help='weight of unsafe region classification loss')
	parser.add_argument('--descent_violation_weight', type=float, default=2, help='weight of descent violation loss')
	parser.add_argument('--actor_weight', type=float, default=1e-2)

	# datamodule params
	parser.add_argument('--batch_size', type=int, default=256)
	parser.add_argument('--safe_portion', type=float, default=0.49, help='portion of safe dps in dataset')
	parser.add_argument('--unsafe_portion', type=float, default=0.49, help='portion of unsafe dps in dataset')
	parser.add_argument('--goal_portion', type=float, default=0.02, help='portion of goal dps in dataset')
	parser.add_argument('--val_split', type=float, default=0.1, help='portion of validation dps in dataset')
	parser.add_argument('--noise_level', type=float, default=0.3)
	parser.add_argument('--max_episode', type=int, default=100)
	parser.add_argument('--trajectories_per_episode', type=int, default=10)
	parser.add_argument('--trajectory_length', type=int, default=150)
	parser.add_argument('--fixed_samples', type=int, default=750)
	# ## for debugging
	# parser.add_argument('--max_episode', type=int, default=2)
	# parser.add_argument('--trajectories_per_episode', type=int, default=5)
	# parser.add_argument('--trajectory_length', type=int, default=50)
	# parser.add_argument('--fixed_samples', type=int, default=30)

	# experiment-suite params
	parser.add_argument('--exp_cbf_contour', action='store_false')
	parser.add_argument('--contour_x_idx', type=int, default=1)
	parser.add_argument('--contour_y_idx', type=int, default=3)
	parser.add_argument('--exp_rollout', action='store_false')
	parser.add_argument('--rollout_x_idx', type=int, default=1)
	parser.add_argument('--rollout_y_idx', type=int, default=3)
	parser.add_argument('--rollout_t_sim', type=float, default=2.)
	parser.add_argument('--rollout_n_sim_per_start', type=int, default=2)

	args = parser.parse_args()

	main(args)

import os
import time
import argparse
import yaml

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import matplotlib.pyplot as plt
import seaborn

import pybullet as p

from environment import ArmEnv

from neural_cbf.controllers import NeuralLidarCBFController
from neural_cbf.datamodules.episodic_datamodule import (
	EpisodicDataModule,
)
from neural_cbf.systems import ArmLidar
from neural_cbf.experiments import (
	ExperimentSuite,
	BFContourExperiment,
	LidarRolloutExperiment,
)
from neural_cbf.training.utils import current_git_hash
from neural_cbf.systems.utils import grav, Scenario, cartesian_to_spherical, spherical_to_cartesian

from PIL import Image
import cv2

# batch_size = 1


def init_val(path, args):
	# initialize models and parameters for loaded controllers
	nominal_params = {}
	scenarios = [
		nominal_params,
	]
	# Define environment and agent
	config_file = ''
	# config_file = '../../models/env_file/panda_100_8_v1_refined.npz'
	environment = ArmEnv([args.robot_name], GUI=0, config_file=config_file)
	robot = environment.robot_list[0]

	# Define the dynamics model
	dynamics_model = ArmLidar(
		nominal_params,
		dis_threshold=args.dis_threshold,
		dt=args.simulation_dt,
		controller_dt=args.controller_period,
		n_obs=args.n_observation,
		point_dim=args.point_dim,
		add_normal=bool('norm' in args.dataset_name),
		point_in_dataset_pc=args.n_observation_dataset,
		list_sensor=robot.body_joints,
		env=environment,
		robot=robot,
		observation_type=args.observation_type,
	)
	dynamics_model.compute_linearized_controller(None)

	# start_x = torch.tensor(np.load(config_file)['init_configs'][0]).unsqueeze(0)
	# goal_state = torch.tensor(np.load(config_file)['goal_configs'][0])

	# Define goal_state
	# goal_state = dynamics_model.sample_boundary(1, data_collection=True)[0, :7]
	# goal_state = torch.Tensor(p.calculateInverseKinematics(robot.robotId, 6, [0.3, -0.4, 0.5])[:7])
	goal_state = torch.tensor(robot.q0).float()
	dynamics_model.set_goal(goal_state)

	# Initialize the DataModule
	initial_conditions = [tuple(robot.body_range[i]) for i in range(robot.body_dim)]
	data_module = None #EpisodicDataModule(
	# 	dynamics_model,
	# 	initial_conditions,
	# 	total_point=args.n_observation_dataset,
	# 	max_episode=args.max_episode,
	# 	trajectories_per_episode=args.trajectories_per_episode,
	# 	trajectory_length=args.trajectory_length,
	# 	fixed_samples=args.fixed_samples,
	# 	val_split=args.val_split,
	# 	batch_size=args.batch_size,
	# 	noise_level=args.noise_level,
	# 	quotas={"safe": args.safe_portion, "goal": args.goal_portion, "unsafe": args.unsafe_portion},
	# 	name=args.dataset_name,
	# 	shuffle=False,
	# )

	# start_x = torch.tensor([
	# 	[0.00887519, 0.50546576, -0.69052917, -2.2909179, 2.95208592, 2.29793418, 2.93001438] # + [0 for _ in range(8)],
	# # 	[0.00887519, -0.50546576, -0.69052917, -2.2909179, 2.95208592, 2.29793418, 2.93001438] + [0 for _ in range(
	# # 		# 8)],
	# # 		[-2.60887519, -1.30546576, -1.69052917, -2.2909179, 2.95208592, 3.59793418, 2.93001438]
	# 			# dynamics_model.o_dims + dynamics_model.state_aux_dims)],
	# # 		[0.00887519, -0.50546576, -0.69052917, -2.2909179, 2.95208592, 3.59793418, 2.93001438] + [0 for _ in range(
	# # 		dynamics_model.o_dims + dynamics_model.state_aux_dims)],
	# ])
	# # start_x = dynamics_model.sample_safe(1)
	# # start_x = dynamics_model.sample_boundary(1, data_collection=True)
	ul, ll = dynamics_model.state_limits
	start_x = torch.cat([
		torch.lerp(ll, ul, 0.4 * torch.ones(ll.shape[-1]).double()).reshape(1, -1),
		# torch.lerp(ll, ul, 0.8 * torch.ones(ll.shape[-1]).double()).reshape(1, -1),
	], dim=0).float()
	start_x = dynamics_model.complete_sample_with_observations(start_x, num_samples=start_x.shape[0])

	x_idx = 0
	y_idx = 2
	rollout_experiment = LidarRolloutExperiment(
		"Rollout",
		start_x,
		x_idx,
		f"$\\theta_{x_idx}$",
		y_idx,
		f"$\\theta_{y_idx}$",
		scenarios=scenarios,
		n_sims_per_start=1,
		t_sim=20,
		compare_nominal=False,
	)

	default_state = start_x
	# default_state = dynamics_model.sample_boundary(1).squeeze()
	# # default_state = dynamics_model.complete_sample_with_observations(dynamics_model.goal_state.reshape(1, -1),
	# # 																 num_samples=1).squeeze()

	# Define the experiment suite
	h_contour_experiment = BFContourExperiment(
		"h_Contour",
		domain=[tuple(robot.body_range[x_idx]), tuple(robot.body_range[y_idx])],
		n_grid=40,
		x_axis_index=x_idx,
		y_axis_index=y_idx,
		x_axis_label=f"$\\theta_{x_idx}$",
		y_axis_label=f"$\\theta_{y_idx}$",
		default_state=default_state,
		plot_unsafe_region=True,
	)

	experiment_suite = ExperimentSuite([rollout_experiment, h_contour_experiment])

	loss_config = {
		"u_coef_in_training": args.u_coef_in_training,
		"safe_classification_weight": args.safe_classification_weight,
		"unsafe_classification_weight": args.unsafe_classification_weight,
		"descent_violation_weight": args.descent_violation_weight,
		"hdot_divergence_weight": args.hdot_divergence_weight,
	}
	return NeuralLidarCBFController.load_from_checkpoint(path, dynamics_model=dynamics_model, scenarios=scenarios,
														 datamodule=data_module, experiment_suite=experiment_suite,
														 use_bn=args.use_bn,
														 cbf_hidden_layers=args.cbf_hidden_layers,
														 cbf_hidden_size=args.cbf_hidden_size,
														 cbf_alpha=args.cbf_alpha,
														 cbf_relaxation_penalty=args.cbf_relaxation_penalty,
														 feature_dim=args.feature_dim,
														 per_feature_dim=args.per_feature_dim,
														 loss_config=loss_config,
														 controller_period=args.controller_period,
														 all_hparams=args,
														 use_neural_actor=0,
														 map_location='cpu')


def vis_traj_rollout(controller: NeuralLidarCBFController):
	"""
	Visualize trajectories from two-link-arm RolloutStateSpaceExperiments.
	"""
	# Tweak experiment params
	controller.experiment_suite.experiments[0].t_sim = 9.

	# Run the experiments and save the results
	controller.experiment_suite.experiments[0].run_and_plot(
		controller, display_plots=True
	)
	print('finished')


def vis_CBF_contour(controller: NeuralLidarCBFController):
	# Run the experiments and save the results
	controller.experiment_suite.experiments[1].run_and_plot(
		controller_under_test=controller, display_plots=True
	)
	print('finished CBF contour')


@torch.no_grad()
def check_evaluation(controller: NeuralLidarCBFController):
	controller.datamodule.prepare_data()
	# # just check below z=0
	# below_z = 0
	# training_unsafe_mask = torch.nonzero(controller.datamodule.x_training_mask['unsafe']).squeeze()
	# for i in range(training_unsafe_mask.shape[0]):
	# 	q = controller.datamodule.x_training[training_unsafe_mask[i], :7]
	# 	if controller.dynamics_model.robot.forward_kinematics([-2], q)[0][0][2] < 0.05:
	# 		below_z += 1
	# 	if i %500 == 0:
	# 		print(f"below z: {below_z} / {i}")

	batch_size = 50
	for i in range(30):
		init_idx = i * batch_size + 1000
		end_idx = init_idx + batch_size
		data_x, goal_mask, safe_mask, unsafe_mask, boundary_mask, JP, JR = controller.datamodule.training_data[torch.arange(init_idx, end_idx)]
		data_x = data_x[:, :-1]

		eps = controller.safe_level
		h_value = controller.h(data_x)

		#   1.) h < 0 in the safe region
		safe_violation = F.relu(eps + h_value[safe_mask]).squeeze()
		safe_h_term = 20 * safe_violation.mean()
		safe_h_acc = (safe_violation <= eps).sum() / safe_violation.nelement()

		#   2.) h > 0 in the unsafe region
		unsafe_violation = F.relu(eps - h_value[unsafe_mask]).squeeze()
		unsafe_h_term = 20 * unsafe_violation.mean()
		unsafe_h_acc = (unsafe_violation <= eps).sum() / unsafe_violation.nelement()
		# print(f"safe_h_acc: {safe_h_acc}, unsafe_h_acc: {unsafe_h_acc}, safe_h_term: {safe_h_term}, unsafe_h_term: {unsafe_h_term}")

		#   3.) hdot + alpha * h < 0 in all regions
		_, Lf_V, Lg_V, _ = controller.V_with_lie_derivatives(data_x, (JP, JR))

		Lg_V_no_grad = Lg_V.detach().clone().squeeze(1)  # bs * n_control

		qp_sol = controller.u(data_x)[0]
		x_next = controller.dynamics_model.batch_lookahead(data_x, qp_sol * controller.dynamics_model.dt, data_jacobian=(JP, JR))
		hdot_simulated = (controller.h(x_next) - h_value) / controller.dynamics_model.dt

		hdot = hdot_simulated
		alpha = controller.clf_lambda # torch.where(h < 0, 2 * self.clf_lambda, self.clf_lambda).type_as(x)
		qp_relaxation = F.relu(hdot + torch.multiply(alpha, h_value))
		print(f"qp_relaxation: {qp_relaxation.mean():.4f}, qp_relaxation: {qp_relaxation.max():.4f}, "
			  f"safe: {(qp_relaxation[safe_mask] <= 0).sum() /  qp_relaxation[safe_mask].nelement():.4f}, "
			  f"unsafe: {(qp_relaxation[unsafe_mask] <= 0).sum() /  qp_relaxation[unsafe_mask].nelement():.4f}, "
			  f"boundary: {(qp_relaxation[boundary_mask] <= 0).sum() /  qp_relaxation[boundary_mask].nelement():.4f}")
		# print(f"relaxation_safe: {qp_relaxation[safe_mask].mean()}, relaxation_unsafe: {qp_relaxation[unsafe_mask].mean()}, "
		# 	  f"relaxation_boundary: {qp_relaxation[boundary_mask].mean()}")

@torch.no_grad()
def vis_misclassification(controller: NeuralLidarCBFController, log_path: str):
	controller.datamodule.prepare_data()
	init_idx = 0
	end_idx = 20
	data_x, goal_mask, safe_mask, unsafe_mask, boundary_mask, JP, JR, io_label = controller.datamodule.training_data[torch.arange(init_idx, end_idx)]
	x = controller.dynamics_model.datax_to_x(data_x, io_label)
	# x = controller.dynamics_model.datax_lookahead_prepare(data_x, data_lookahead)[0, :, :]

	eps = controller.safe_level
	h_value = controller.h(x)

	#   1.) h < 0 in the safe region
	safe_violation = F.relu(eps + h_value).squeeze()
	# safe_h_term = (1 / eps) * safe_violation[safe_mask].mean()
	# safe_h_acc = (safe_violation[safe_mask] <= eps).sum() / safe_violation[safe_mask].nelement()

	#   2.) h > 0 in the unsafe region
	unsafe_violation = F.relu(eps - h_value).squeeze()
	# unsafe_h_term = (1 / eps) * unsafe_violation[unsafe_mask].mean()
	# unsafe_h_acc = (unsafe_violation[unsafe_mask] <= eps).sum() / unsafe_violation[unsafe_mask].nelement()

	log_fig_path = log_path + '/data_classification/'
	if not os.path.exists(log_fig_path):
		os.makedirs(log_fig_path)
		os.makedirs(log_fig_path + 'gt_safe/')
		os.makedirs(log_fig_path + 'gt_unsafe/')
		os.makedirs(log_fig_path + 'safe/')
		os.makedirs(log_fig_path + 'unsafe/')

	for idx in range(10):
	# 	if safe_violation[idx] < eps and safe_mask[idx]:
	# 		draw_environment(controller, x[idx], idx + init_idx, log_fig_path + 'safe/')
		if unsafe_violation[idx] < eps and unsafe_mask[idx]:
			draw_environment(controller, x[idx], idx + init_idx, log_fig_path + 'unsafe/')
	# exit()

	# safe misclassification
	for idx in range(x.shape[0]):
		# if safe_violation[idx] > eps and safe_mask[idx]:
		# 	draw_environment(controller, x[idx], idx + init_idx, log_fig_path + 'gt_safe/')
		if unsafe_violation[idx] > eps and unsafe_mask[idx]:
			draw_environment(controller, x[idx], idx + init_idx, log_fig_path + 'gt_unsafe/')
			# break


	# print(safe_mask)
	# print(unsafe_mask)
	# print(safe_violation.squeeze())
	# print(unsafe_violation.squeeze())
	pass

@torch.no_grad()
def statistics_safe_level(controller: NeuralLidarCBFController):
	controller.datamodule.prepare_data()
	init_idx = 0
	end_idx = 20
	data_x, goal_mask, safe_mask, unsafe_mask, data_lookahead = controller.datamodule.training_data[
		torch.arange(init_idx, end_idx)]
	x = controller.dynamics_model.datax_to_x(data_x)
	# safe_h_acc = (safe_violation[safe_mask] <= eps).sum() / safe_violation[safe_mask].nelement()

def draw_environment(controller: NeuralLidarCBFController, x: torch.Tensor, idx: int, fig_path):
	controller.dynamics_model.env.reset_env(np.array([]), tidy_env=True)

	robot = controller.dynamics_model.robot
	q = x[:controller.dynamics_model.n_dims]
	robot.set_joint_position(robot.body_joints, q)

	p_p = [torch.Tensor(p.getLinkState(robot.robotId, sensor_idx)[4]) for sensor_idx in controller.dynamics_model.list_sensor]
	p_r = [torch.Tensor(p.getMatrixFromQuaternion(p.getLinkState(robot.robotId, sensor_idx)[5])).reshape(3, 3) for
		   sensor_idx in controller.dynamics_model.list_sensor]
	O = x[controller.dynamics_model.n_dims:].reshape(-1, controller.dynamics_model.ray_per_sensor, controller.dynamics_model.point_dims)
	if controller.dynamics_model.point_dims == 4:
		G = [p_p[i] + spherical_to_cartesian(O[i, :, :3]) @ p_r[i].T for i in range(len(controller.dynamics_model.list_sensor))]
	else:
		G = [p_p[i] + O[i, :, :3] @ p_r[i].T for i in range(len(controller.dynamics_model.list_sensor))]
	G = torch.vstack(G).tolist()

	for pt in G:
		vid = p.createVisualShape(p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 1, 1, 1])
		p.createMultiBody(baseVisualShapeIndex=vid, basePosition=pt)

	width = 1280
	height = 720
	total_frame = 30
	video = []
	for i_frame in range(total_frame):
		projectionMatrix = p.computeProjectionMatrixFOV(
			fov=20,
			aspect=width / height,
			nearVal=0.1,
			farVal=50
		)
		viewMatrix = p.computeViewMatrix(
			cameraEyePosition=[3 * np.cos(i_frame/total_frame * 2 * np.pi), 3 * np.sin((i_frame/total_frame * 2 * np.pi)), 1.5],
			cameraTargetPosition=[0, 0, 0.5],
			cameraUpVector=[0, 0, 1]
		)
		width, height, rgbImg, depthImg, segImg = p.getCameraImage(
			width=width,
			height=height,
			viewMatrix=viewMatrix,
			projectionMatrix=projectionMatrix,
			renderer=p.ER_BULLET_HARDWARE_OPENGL
		)
		video.append(rgbImg)
		im = Image.fromarray(rgbImg)

		if not os.path.exists(f"{fig_path}/{idx}/"):
			os.makedirs(f"{fig_path}/{idx}/")
		im.save(f"{fig_path}/{idx}/{i_frame}.png")

	name = idx
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	out = cv2.VideoWriter(f'{fig_path}/{name}.mp4', fourcc, 24, (width, height))
	for i_img, img in enumerate(video):
		img_new = cv2.imread(f"{fig_path}/{idx}/{i_img}.png")
		out.write(img_new)
	out.release()


def statistics_robustness_observation(controller: NeuralLidarCBFController):
# 	controller.datamodule.prepare_data()
# 	init_idx = 1000
# 	x, goal_mask, safe_mask, unsafe_mask, lookahead = controller.datamodule.training_data[
# 		torch.arange(init_idx, init_idx + args.batch_size)]
	batch_size = 256
	N_test = 20
	q = torch.Tensor(np.random.uniform(low=controller.dynamics_model.state_limits[1],
									   high=controller.dynamics_model.state_limits[0],
									   size=(batch_size, controller.dynamics_model.n_dims)))
	dq = torch.Tensor(N_test, controller.dynamics_model.n_dims).uniform_(1e-3, 2e-3)

	results = []
	for i in range(N_test):
		x = controller.dynamics_model.complete_sample_with_observations(q + dq[i, :], batch_size)
		results.append(controller.h(x))

	results = torch.cat(results, dim=1).detach().numpy()
	# print(np.mean(results, axis=1))
	# print(np.std(results, axis=1))

	plt.figure(figsize=(9, 3))
	plt.subplot(121)
	plt.hist(np.std(results, axis=1), 10)
	plt.yscale("log")
	# plt.xlim(0., 0.025)
	plt.title("std distribution")
	plt.grid(True)

	plt.subplot(122)
	plt.hist(results.max(axis=1) - results.min(axis=1), 10)
	plt.yscale("log")
	plt.xlim(0., 0.07)
	plt.title("(max-min) distribution")
	plt.grid(True)

	plt.show()


if __name__ == "__main__":
	# Load the checkpoint file. This should include the experiment suite used during training.
	robot_name = "panda"
	log_dir = "../../models/neural_cbf/"
	git_version = f"collection/{robot_name}_lidar/"

	log_file = ""  # specify the checkpoint file

	# load arguments from yaml
	with open(log_dir + git_version + 'hparams.yaml', 'r') as f:
		args = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))
	args.accelerator = 'cpu'
	args.n_observation = 1024
	# args.simulation_dt = 0.01
	# args.controller_period = 0.01
	args.cbf_relaxation_penalty = 50000.
	args.cbf_alpha = 20
	# args.dis_threshold = 0.02
	# args.observation_type = 'uniform_lidar'

	# args.use_bn = 0

	# args.dataset_name = 'prob08_motor'
	# args.max_episode=100
	# args.trajectories_per_episode=40
	# args.trajectory_length=35

	neural_controller = init_val(log_dir + git_version + log_file, args)

	neural_controller.h_nn.eval()
	neural_controller.encoder.eval()
	neural_controller.pc_head.eval()


	# neural_controller.h_alpha=0.3
	# vis_misclassification(neural_controller, log_path = log_dir+ git_version)
	# vis_traj_rollout(neural_controller)
	vis_CBF_contour(neural_controller)
	# statistics_robustness_observation(neural_controller)
	# check_evaluation(neural_controller)

	# neural_controller.datamodule.prepare_data()
	# idx = 117
	# data_x, goal_mask, safe_mask, unsafe_mask, boundary_mask, JP, JR, io_label = neural_controller.datamodule.training_data[idx]
	# x = neural_controller.dynamics_model.datax_to_x(data_x.unsqueeze(0), io_label.unsqueeze(0))
	# draw_environment(neural_controller, fig_path = log_dir+ git_version, idx=idx, x=x.squeeze())

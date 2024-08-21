"""DataModule for aggregating data points over a series of episodes, with additional
sampling from fixed sets.

Code based on the Pytorch Lightning example at
pl_examples/domain_templates/reinforce_learn_Qnet.py
"""
import os
from typing import List, Callable, Tuple, Dict, Optional

import random
import torch
import pytorch_lightning as pl
import tqdm
from torch.utils.data import TensorDataset, DataLoader

from neural_cbf.systems import ControlAffineSystem

import matplotlib.pyplot as plt


class EpisodicDataModule(pl.LightningDataModule):
	"""
	DataModule for sampling from a replay buffer
	"""

	def __init__(
			self,
			model: ControlAffineSystem,
			initial_domain: List[Tuple[float, float]],
			max_episode: int = 1,
			trajectories_per_episode: int = 100,
			trajectory_length: int = 5000,
			fixed_samples: int = 100000,
			val_split: float = 0.1,
			batch_size: int = 64,
			noise_level: float = 0,
			total_point: int = 0,
			quotas: Optional[Dict[str, float]] = None,
			shuffle: bool = True,
			name: str = "",
	):
		"""Initialize the DataModule

		args:
			model: the dynamics model to use in simulation
			initial_domain: the initial_domain to sample from, expressed as a list of
							 tuples denoting the min/max range for each dimension
			trajectories_per_episode: the number of rollouts to conduct at each episode
			trajectory_length: the number of samples to collect in each trajectory
			fixed_samples: the number of uniform samples to collect
			val_split: the fraction of sampled data to reserve for validation
			total_point: number of available points in global point cloud (in a single data point)
			batch_size: the batch size
			quotas: a dictionary specifying the minimum percentage of the
					fixed samples that should be taken from the safe,
					unsafe, boundary, and goal sets. Expects keys to be either "safe",
					"unsafe", "boundary", or "goal".
		"""
		super().__init__()
		self.name = name

		self.model = model
		self.n_dims = model.n_dims  # copied for convenience
		self.shuffle = shuffle

		if "lidar" in str(self.model):
			self.total_points = total_point

		# Save the parameters
		self.max_episode = max_episode
		self.trajectories_per_episode = trajectories_per_episode
		self.trajectory_length = trajectory_length
		self.fixed_samples = fixed_samples
		self.val_split = val_split
		self.batch_size = batch_size
		self.noise_level = noise_level
		if quotas is not None:
			self.quotas = quotas
		else:
			self.quotas = {}

		# Define the sampling intervals for initial conditions as a hyper-rectangle
		assert len(initial_domain) == self.n_dims
		self.initial_domain = initial_domain

		# Save the min, max, central point, and range tensors
		self.x_max, self.x_min = model.state_limits
		self.x_center = (self.x_max + self.x_min) / 2.0
		self.x_range = self.x_max - self.x_min

	def sample_trajectories(
			self, simulator: Callable[[torch.Tensor, int], torch.Tensor],
			episode_num=0, trajectory_num=0,
	) -> (torch.Tensor, dict):
		"""
		Generate new data points by simulating a bunch of trajectories

		args:
			simulator: a function that simulates the given initial conditions out for
					   the specified number of timesteps
		"""
		x_sim = []
		x_mask = {'safe': [], 'unsafe': [], 'goal': [], 'boundary': []}

		for episode in range(episode_num if episode_num else self.max_episode):
			complete_episode = False
			while not complete_episode:
				try:
					self.model.env.reset_env(obs_configs=self.model.env.get_env_config(int(random.uniform(0, 600))),
											 enable_object=False)
					self.model.set_goal(self.model.sample_state_space(1)[0, :self.model.q_dims])

					# Start by sampling from initial conditions from the given region
					traj_number = trajectory_num if trajectory_num else self.trajectories_per_episode
					x_init = self.model.sample_boundary(traj_number, max_tries=2)[:, :self.n_dims]
					complete_episode = True
				except RuntimeWarning:
					pass

			# Simulate each initial condition out for the specified number of steps
			batch_x_sim = simulator(x_init, self.trajectory_length, noise_level=self.noise_level,
									collect_dataset=True, use_motor_control=random.random() < 0.8)

			# Reshape the data into a single replay buffer
			if hasattr(self.model, "o_dims_in_dataset"):
				batch_x_sim = batch_x_sim.view(-1,
											   self.n_dims + self.model.o_dims_in_dataset + self.model.state_aux_dims_in_dataset)
			elif hasattr(self.model, "o_dims"):
				batch_x_sim = batch_x_sim.view(-1, self.n_dims + self.model.o_dims + self.model.state_aux_dims)
			else:
				batch_x_sim = batch_x_sim.view(-1, self.n_dims)
			x_sim.append(batch_x_sim)
			x_mask['safe'].append(self.model.safe_mask(batch_x_sim))
			x_mask['unsafe'].append(self.model.unsafe_mask(batch_x_sim))
			x_mask['goal'].append(self.model.goal_mask(batch_x_sim))
			x_mask['boundary'].append(self.model.boundary_mask(batch_x_sim))

		# Return the sampled data
		x_sim = torch.cat(x_sim, dim=0)
		x_mask['safe'] = torch.cat(x_mask['safe'], dim=0)
		x_mask['unsafe'] = torch.cat(x_mask['unsafe'], dim=0)
		x_mask['goal'] = torch.cat(x_mask['goal'], dim=0)
		x_mask['boundary'] = torch.cat(x_mask['boundary'], dim=0)
		return x_sim, x_mask

	def sample_fixed(self) -> (torch.Tensor, dict):
		"""
		Generate new data points by sampling uniformly from the state space
		"""
		samples = []
		x_mask = {'safe': [], 'unsafe': [], 'goal': [], 'boundary': []}
		add_env_idx = 0
		for episode in tqdm.tqdm(range(self.max_episode)):
			complete_episode = False
			while not complete_episode:
				try:
					sample = []
					self.model.env.reset_env(obs_configs=self.model.env.get_env_config(episode + add_env_idx),
											 enable_object=False)
					# self.model.set_goal(self.model.sample_safe(1)[0, :self.model.q_dims])
					# Figure out how many points are to be sampled at random, how many from the
					# goal, safe, or unsafe regions specifically
					allocated_samples = 0
					for region_name, quota in self.quotas.items():
						num_samples = int(self.fixed_samples * quota)
						allocated_samples += num_samples

						if region_name == "goal":
							sample.append(self.model.sample_goal(num_samples))
						elif region_name == "safe":
							sample.append(self.model.sample_safe(num_samples))
						elif region_name == "unsafe":
							sample.append(self.model.sample_unsafe(num_samples))
						elif region_name == "boundary":
							sample.append(self.model.sample_boundary(num_samples))
					complete_episode = True
				except RuntimeWarning:
					add_env_idx += 1
					continue

			sample = torch.vstack(sample)
			samples.append(sample)
			x_mask['safe'].append(self.model.safe_mask(sample))
			x_mask['unsafe'].append(self.model.unsafe_mask(sample))
			x_mask['goal'].append(self.model.goal_mask(sample))
			x_mask['boundary'].append(self.model.boundary_mask(sample))

		samples = torch.vstack(samples)
		x_mask['safe'] = torch.cat(x_mask['safe'], dim=0)
		x_mask['unsafe'] = torch.cat(x_mask['unsafe'], dim=0)
		x_mask['goal'] = torch.cat(x_mask['goal'], dim=0)
		x_mask['boundary'] = torch.cat(x_mask['boundary'], dim=0)

		return samples, x_mask

	def prepare_data(self):
		"""Create the dataset"""
		# load dataset if saved, prefer not saving in checkpoint
		dataset_path = os.path.abspath(__file__).rsplit('/', 3)[0] + f'/models/neural_cbf/{str(self.model)}/data/'
		if not os.path.exists(dataset_path):
			os.makedirs(dataset_path)
		if "lidar" in str(self.model):
			list_sensor = self.model.list_sensor
			ray_per_sensor_ds = self.model.point_in_dataset_pc
			observation_type = self.model.observation_type
			dataset_path = dataset_path + f'{int(100 * self.model.dis_threshold)}_{self.model.env.obstacle_num}_' \
										  f'{ray_per_sensor_ds}_{observation_type}_' \
										  f'{int(self.noise_level * 1e2)}_{self.max_episode}_{self.trajectories_per_episode}_{self.trajectory_length}_' \
										  f'{self.fixed_samples}.pt'
		elif "mindis" in str(self.model):
			dataset_path = dataset_path + f'{int(100 * self.model.dis_threshold)}_{self.model.env.obstacle_num}_' \
										  f'{int(self.noise_level * 1e2)}_{self.max_episode}_{self.trajectories_per_episode}_{self.trajectory_length}_' \
										  f'{self.fixed_samples}.pt'
		else:
			raise NotImplementedError(f"Unknown model {self.model}")

		if os.path.exists(dataset_path):
			print(f'Using saved dataset from {dataset_path}')
			data = torch.load(dataset_path)
			self.x_training = data['x_training']
			self.x_validation = data['x_validation']
			self.x_training_mask = data['x_training_mask']
			self.x_validation_mask = data['x_validation_mask']
			if "lidar" in str(self.model):
				self.x_training_lookahead = data['x_training_lookahead']
				self.x_validation_lookahead = data['x_validation_lookahead']
		# create new dataset and save for future use
		else:
			print(f'Didn\'t find datset at {dataset_path}. Collecting new dataset......')

			# Augment those points with samples from the fixed range
			x_sample, x_sample_mask = self.sample_fixed()

			# Get some data points from simulations
			x_sim, x_sim_mask = self.sample_trajectories(self.model.noisy_simulator)

			x = torch.cat((x_sim, x_sample), dim=0)
			x_mask = {key: torch.cat((x_sim_mask[key], x_sample_mask[key]), dim=0) for key in x_sim_mask.keys()}

			# Randomly split data into training and test sets
			random_indices = torch.randperm(x.shape[0])
			val_pts = int(x.shape[0] * self.val_split)
			validation_indices = random_indices[:val_pts]
			training_indices = random_indices[val_pts:]
			self.x_training = x[training_indices]
			self.x_validation = x[validation_indices]
			self.x_training_mask = {key: x_mask[key][training_indices] for key in x_mask.keys()}
			self.x_validation_mask = {key: x_mask[key][validation_indices] for key in x_mask.keys()}
			to_save_dict = {'x_training': self.x_training,
							'x_validation': self.x_validation,
							'x_training_mask': self.x_training_mask,
							'x_validation_mask': self.x_validation_mask}
			if "lidar" in str(self.model):
				JP_training, JR_training = self.model.get_batch_jacobian(self.x_training)
				JP_validation, JR_validation = self.model.get_batch_jacobian(self.x_validation)
				self.x_training_lookahead = {'J_P': JP_training, 'J_R': JR_training}
				self.x_validation_lookahead = {'J_P': JP_validation, 'J_R': JR_validation}
				to_save_dict['x_training_lookahead'] = self.x_training_lookahead
				to_save_dict['x_validation_lookahead'] = self.x_validation_lookahead
			torch.save(to_save_dict, dataset_path)

		# Print dataset statistics
		print("Full dataset:")
		print(f"\t{self.x_training.shape[0]} training")
		print(f"\t{self.x_validation.shape[0]} validation")
		print("\t----------------------")
		print(f"\t{self.x_training_mask['goal'].sum()} goal points")
		print(f"\t({self.x_validation_mask['goal'].sum()} val)")
		print(f"\t{self.x_training_mask['safe'].sum()} safe points")
		print(f"\t({self.x_validation_mask['safe'].sum()} val)")
		print(f"\t{self.x_training_mask['unsafe'].sum()} unsafe points")
		print(f"\t({self.x_validation_mask['unsafe'].sum()} val)")
		print(f"\t{self.x_training_mask['boundary'].sum()} boundary points")
		print(f"\t({self.x_validation_mask['boundary'].sum()} val)")

		# Turn these into tensor datasets
		if "lidar" in str(self.model):
			if True: #self.x_training_mask['unsafe'].sum() > 3 * self.x_training_mask['safe'].sum():
				self.downsample_unsafe()
			else:
				self.training_data = TensorDataset(
					self.x_training,
					self.x_training_mask['goal'],
					self.x_training_mask['safe'],
					self.x_training_mask['unsafe'],
					self.x_training_mask['boundary'],
					self.x_training_lookahead['J_P'],
					self.x_training_lookahead['J_R'],
				)
				self.validation_data = TensorDataset(
					self.x_validation,
					self.x_validation_mask['goal'],
					self.x_validation_mask['safe'],
					self.x_validation_mask['unsafe'],
					self.x_validation_mask['boundary'],
					self.x_validation_lookahead['J_P'],
					self.x_validation_lookahead['J_R'],
				)
		else:
			self.training_data = TensorDataset(
				self.x_training,
				self.x_training_mask['goal'],
				self.x_training_mask['safe'],
				self.x_training_mask['unsafe'],
			)
			self.validation_data = TensorDataset(
				self.x_validation,
				self.x_validation_mask['goal'],
				self.x_validation_mask['safe'],
				self.x_validation_mask['unsafe'],
			)

	# downsample unsafe and boundary data
	def downsample_unsafe(self) -> None:
		# safe_training = torch.nonzero(self.x_training_mask['safe'].squeeze()).squeeze()
		# unsafe_training = torch.nonzero(self.x_training_mask['unsafe'].squeeze()).squeeze()
		# boundary_training = torch.nonzero(self.x_training_mask['boundary'].squeeze()).squeeze()
		# training_index = torch.cat((safe_training,
		# 							unsafe_training[torch.randint(0, unsafe_training.shape[0], (int(safe_training.shape[0] * 2.5),))],
		# 							boundary_training[torch.randint(0, boundary_training.shape[0], (int(safe_training.shape[0] * 1.5),))]), dim=0).reshape(-1)
		#
		# safe_validation = torch.nonzero(self.x_validation_mask['safe'].squeeze()).squeeze()
		# unsafe_validation = torch.nonzero(self.x_validation_mask['unsafe'].squeeze()).squeeze()
		# boundary_validation = torch.nonzero(self.x_validation_mask['boundary'].squeeze()).squeeze()
		# validation_index = torch.cat((safe_validation,
		# 							unsafe_validation[torch.randint(0, unsafe_validation.shape[0], (int(safe_validation.shape[0] * 2.5),))],
		# 							boundary_validation[torch.randint(0, boundary_validation.shape[0], (int(safe_validation.shape[0] * 1.5),))]), dim=0).reshape(-1)

		# self.training_data = TensorDataset(
		# 	torch.index_select(self.x_training, 0, training_index),
		# 	torch.index_select(self.x_training_mask['goal'], 0, training_index),
		# 	torch.index_select(self.x_training_mask['safe'], 0, training_index),
		# 	torch.index_select(self.x_training_mask['unsafe'], 0, training_index),
		# 	torch.index_select(self.x_training_mask['boundary'], 0, training_index),
		# 	torch.index_select(self.x_training_lookahead['J_P'], 0, training_index),
		# 	torch.index_select(self.x_training_lookahead['J_R'], 0, training_index),
		# )
		# # self.validation_data = TensorDataset(
		# # 	torch.index_select(self.x_validation, 0, validation_index),
		# # 	torch.index_select(self.x_validation_mask['goal'], 0, validation_index),
		# # 	torch.index_select(self.x_validation_mask['safe'], 0, validation_index),
		# # 	torch.index_select(self.x_validation_mask['unsafe'], 0, validation_index),
		# # 	torch.index_select(self.x_validation_mask['boundary'], 0, validation_index),
		# # 	torch.index_select(self.x_validation_lookahead['J_P'], 0, validation_index),
		# # 	torch.index_select(self.x_validation_lookahead['J_R'], 0, validation_index),
		# # )
		self.training_data = TensorDataset(
			self.x_training,
			self.x_training_mask['goal'],
			self.x_training_mask['safe'],
			self.x_training_mask['unsafe'],
			self.x_training_mask['boundary'],
			self.x_training_lookahead['J_P'],
			self.x_training_lookahead['J_R'],
		)
		self.validation_data = TensorDataset(
			self.x_validation,
			self.x_validation_mask['goal'],
			self.x_validation_mask['safe'],
			self.x_validation_mask['unsafe'],
			self.x_validation_mask['boundary'],
			self.x_validation_lookahead['J_P'],
			self.x_validation_lookahead['J_R'],
		)

	# def add_data(self, simulator: Callable[[torch.Tensor, int], torch.Tensor]) -> None:
	# 	"""
	# 	Augment the training and validation datasets by simulating and sampling
	#
	# 	args:
	# 		simulator: a function that simulates the given initial conditions out for
	# 					the specified number of timesteps
	# 	"""
	# 	# print("\nAdding data!\n")
	# 	# Get some data points from simulations
	# 	x_sim, x_sim_mask = self.sample_trajectories(simulator, episode_num=1, trajectory_num=10)
	# 	x = x_sim.type_as(self.x_training)
	# 	x_mask = {key: x_sim_mask[key] for key in x_sim_mask.keys()}
	# 	# print(f"Sampled {x.shape[0]} new points")
	#
	# 	# Get the lookahead Jacobians for the new points if necessary
	# 	if "lidar" in str(self.model):
	# 		JP, JR = self.get_batch_jacobian(x)
	#
	# 	# Randomly split data into training and test sets
	# 	random_indices = torch.randperm(x.shape[0])
	# 	val_pts = int(x.shape[0] * self.val_split)
	# 	validation_indices = random_indices[:val_pts]
	# 	training_indices = random_indices[val_pts:]
	#
	# 	n_train = training_indices.shape[0]
	# 	n_val = validation_indices.shape[0]
	# 	# print(f"\t{training_indices.shape[0]} train, {validation_indices.shape[0]} val")
	#
	# 	# If we've exceeded the maximum number of points, forget the oldest
	# 	# print("Forgetting...")
	# 	# And then keep only the most recent points
	# 	self.x_training = torch.cat((self.x_training[n_train:], x[training_indices]), dim=0)
	# 	self.x_validation = torch.cat((self.x_validation[n_val:], x[validation_indices]), dim=0)
	# 	for key in self.x_training_mask.keys():
	# 		self.x_training_mask[key] = torch.cat((self.x_training_mask[key][n_train:], x_mask[key][training_indices]),
	# 											  dim=0)
	# 		self.x_validation_mask[key] = torch.cat(
	# 			(self.x_validation_mask[key][n_val:], x_mask[key][validation_indices]), dim=0)
	# 	if "lidar" in str(self.model):
	# 		self.x_training_lookahead['J_P'] = torch.cat(
	# 			(self.x_training_lookahead['J_P'][n_train:], JP[training_indices]), dim=0)
	# 		self.x_training_lookahead['J_R'] = torch.cat(
	# 			(self.x_training_lookahead['J_R'][n_train:], JR[training_indices]), dim=0)
	# 		self.x_validation_lookahead['J_P'] = torch.cat(
	# 			(self.x_validation_lookahead['J_P'][n_val:], JP[validation_indices]), dim=0)
	# 		self.x_validation_lookahead['J_R'] = torch.cat(
	# 			(self.x_validation_lookahead['J_R'][n_val:], JR[validation_indices]), dim=0)
	# 		if self.model.use_inout_label:
	# 			self.x_training_inoutlabel = torch.cat(
	# 				(self.x_training_inoutlabel[n_train:], x_inoutlabel[training_indices]), dim=0)
	# 			self.x_validation_inoutlabel = torch.cat(
	# 				(self.x_validation_inoutlabel[n_val:], x_inoutlabel[validation_indices]), dim=0)
	#
	# 	# Save the new datasets
	# 	if "lidar" in str(self.model):
	# 		if self.model.use_inout_label:
	# 			training_index = torch.randint(0, self.x_training.shape[0], (self.x_training.shape[0]//2,))
	# 			self.training_data = TensorDataset(
	# 				torch.index_select(self.x_training, 0, training_index),
	# 				torch.index_select(self.x_training_mask['goal'], 0, training_index),
	# 				torch.index_select(self.x_training_mask['safe'], 0, training_index),
	# 				torch.index_select(self.x_training_mask['unsafe'], 0, training_index),
	# 				torch.index_select(self.x_training_mask['boundary'], 0, training_index),
	# 				torch.index_select(self.x_training_lookahead['J_P'], 0, training_index),
	# 				torch.index_select(self.x_training_lookahead['J_R'], 0, training_index),
	# 				torch.index_select(self.x_training_inoutlabel, 0, training_index),
	# 			)
	# 			self.validation_data = TensorDataset(
	# 				self.x_validation,
	# 				self.x_validation_mask['goal'],
	# 				self.x_validation_mask['safe'],
	# 				self.x_validation_mask['unsafe'],
	# 				self.x_validation_mask['boundary'],
	# 				self.x_validation_lookahead['J_P'],
	# 				self.x_validation_lookahead['J_R'],
	# 				self.x_validation_inoutlabel,
	# 			)
	# 		else:
	# 			raise NotImplementedError
	# 	else:
	# 		raise NotImplementedError

	# # Print dataset statistics
	# print("Full dataset:")
	# print(f"\t{self.x_training.shape[0]} training")
	# print(f"\t{self.x_validation.shape[0]} validation")
	# print("\t----------------------")
	# print(f"\t{self.x_training_mask['goal'].sum()} goal points")
	# print(f"\t({self.x_validation_mask['goal'].sum()} val)")
	# print(f"\t{self.x_training_mask['safe'].sum()} safe points")
	# print(f"\t({self.x_validation_mask['safe'].sum()} val)")
	# print(f"\t{self.x_training_mask['unsafe'].sum()} unsafe points")
	# print(f"\t({self.x_validation_mask['unsafe'].sum()} val)")
	# print(f"\t{self.x_training_mask['boundary'].sum()} boundary points")
	# print(f"\t({self.x_validation_mask['boundary'].sum()} val)")

	def setup(self, stage=None):
		"""Setup -- nothing to do here"""
		pass

	def train_dataloader(self):
		"""Make the DataLoader for training data"""
		return DataLoader(
			self.training_data,
			batch_size=self.batch_size,
			num_workers=4,
			shuffle=True,
		)

	def val_dataloader(self):
		"""Make the DataLoader for validation data"""
		return DataLoader(
			self.validation_data,
			batch_size=self.batch_size,
			num_workers=4,
			shuffle=True,
		)

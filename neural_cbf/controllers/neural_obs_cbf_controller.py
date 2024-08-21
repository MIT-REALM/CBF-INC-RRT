import itertools
import time
from functools import partial
from typing import Tuple, List, Optional
from collections import OrderedDict
import random
import tqdm

import pybullet as p

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import pytorch_lightning as pl

import matplotlib.pyplot as plt

from neural_cbf.systems import ArmDynamics
from neural_cbf.systems.utils import ScenarioList, cartesian_to_spherical, spherical_to_cartesian
from neural_cbf.controllers.cbf_controller import CBFController
from neural_cbf.controllers.utils import PointNetfeat, PointNetVanillaEncoder
from neural_cbf.datamodules.episodic_datamodule import EpisodicDataModule
from neural_cbf.experiments import ExperimentSuite


class NeuralObsCBFController(pl.LightningModule, CBFController):
	"""
	A neural CBF controller.
	1) Differs from the CBFController in that it uses a neural network to learn the CBF.
	2) Differs from neural_bf_controller that the control input is solved with QP here.

	More specifically, the CBF controller looks for a h such that

	h(safe) < 0
	h(unsafe) > 0
	dh/dt <= -lambda h

	This proves forward invariance of the 0-sublevel set of h, and since the safe set is
	a subset of this sublevel set, we prove that the unsafe region is not reachable from
	the safe region.

	Note that V in the code is just intended to reuse CLF code :(

	h:
		(observations -> encoder) + state -> fully-connected layers -> h

	u is determined by solving QP

	Note that V in the code is just intended to reuse CLF code :(
	"""

	def __init__(
			self,
			dynamics_model: ArmDynamics,
			scenarios: ScenarioList,
			datamodule: EpisodicDataModule,
			experiment_suite: ExperimentSuite,
			**kwargs,
	):
		"""Initialize the controller.

		args:
			dynamics_model: the control-affine dynamics of the underlying system
			scenarios: a list of parameter scenarios to train on
			datamodule: the datamodule to use for training & validation
			experiment_suite: defines the experiments to run during training
		"""
		super(NeuralObsCBFController, self).__init__(
			dynamics_model=dynamics_model,
			scenarios=scenarios,
			experiment_suite=experiment_suite,
			cbf_lambda=kwargs["cbf_alpha"],
			cbf_relaxation_penalty=kwargs["cbf_relaxation_penalty"],
			controller_period=dynamics_model.controller_dt,
			safe_level=kwargs["safe_level"],
			unsafe_level=kwargs["unsafe_level"],
		)

		if "all_hparams" in kwargs:
			self.save_hyperparameters(kwargs["all_hparams"])

		self.dynamics_model = dynamics_model
		self.datamodule = datamodule

		self.cbf_alpha = kwargs["cbf_alpha"]
		assert self.cbf_alpha > 0
		assert self.cbf_alpha * self.dynamics_model.controller_dt <= 1

		# neural actor setting
		self.use_neural_actor = kwargs["use_neural_actor"]

		# Several common parameters for the neural network
		self.h_hidden_layers = kwargs["cbf_hidden_layers"]
		self.h_hidden_size = kwargs["cbf_hidden_size"]

		# Save the other parameters
		if "loss_config" in kwargs:
			self.loss_config = kwargs["loss_config"]
			self.learn_shape_epochs = kwargs["learn_shape_epochs"]

	def prepare_data(self):
		return self.datamodule.prepare_data()

	def setup(self, stage: Optional[str] = None):
		return self.datamodule.setup(stage)

	def train_dataloader(self):
		return self.datamodule.train_dataloader()

	def val_dataloader(self):
		return self.datamodule.val_dataloader()

	def test_dataloader(self):
		return self.datamodule.test_dataloader()

	def h(self, datax: torch.Tensor) -> torch.Tensor:
		"""Return the CBF value for the observations o

		args:
			x: bs x self.n_dims_extended tensor of state and observation
		returns:
			h: bs x 1 tensor of BF values
		"""
		pass

	def h_with_jacobian(self, datax: torch.Tensor, data_jacobian: tuple) -> Tuple[
		torch.Tensor, torch.Tensor, dict]:
		"""Computes the CLBF value and its Jacobian

		args:
			x: bs x (self.dynamics_model.n_dims + o_dims) the points at which to evaluate the CLBF
		returns:
			V: bs tensor of CLBF values
			JV: bs x 1 x self.dynamics_model.n_dims Jacobian of each row of V wrt x
		"""
		pass

	def V(self, x: torch.Tensor):
		"""Return the CBF value for the observations o"""
		return self.h(x)

	def V_with_jacobian(self, x: torch.Tensor, data_jacobian: tuple) -> Tuple[
		torch.Tensor, torch.Tensor, dict]:
		return self.h_with_jacobian(x, data_jacobian)

	def forward(self, x):
		"""Determine the control input for a given state using a QP

		args:
			x: bs x self.n_dims_extended, tensor of state and observation
		returns:
			u: bs x self.dynamics_model.n_controls, tensor of control inputs
		"""
		assert x.shape[1] == self.n_dims_extended
		return self.u(x)

	def boundary_loss(
			self,
			h: torch.Tensor,
			goal_mask: torch.Tensor,
			safe_mask: torch.Tensor,
			unsafe_mask: torch.Tensor,
			accuracy: bool = False,
	) -> List[Tuple[str, torch.Tensor]]:
		"""
		Evaluate the loss on the CLBF due to boundary conditions

		args:
			x: the points at which to evaluate the loss,
			goal_mask: the points in x marked as part of the goal
			safe_mask: the points in x marked safe
			unsafe_mask: the points in x marked unsafe
			accuracy: if True, return the accuracy (from 0 to 1) as well as the losses
		returns:
			loss: a list of tuples containing ("category_name", loss_value).
		"""
		loss = []

		#   2.) h < 0 in the safe region
		h_safe = h[safe_mask]
		safe_violation = F.relu(self.safe_level + h_safe)
		safe_h_term = self.loss_config["safe_classification_weight"] * safe_violation.mean()
		loss.append(("BF safe region term", safe_h_term))
		if accuracy:
			safe_h_acc = (safe_violation <= self.safe_level).sum() / safe_violation.nelement()
			loss.append(("BF safe region accuracy", safe_h_acc))

		#   3.) h > 0 in the unsafe region
		h_unsafe = h[unsafe_mask]
		unsafe_violation = F.relu(self.unsafe_level - h_unsafe)
		unsafe_h_term = self.loss_config["unsafe_classification_weight"] * unsafe_violation.mean()
		loss.append(("BF unsafe region term", unsafe_h_term))
		if accuracy:
			# print((unsafe_violation <= self.unsafe_level).sum(), unsafe_violation.nelement())
			unsafe_h_acc = (unsafe_violation <= self.unsafe_level).sum() / unsafe_violation.nelement()
			loss.append(("BF unsafe region accuracy", unsafe_h_acc))

		return loss

	def training_step(self, batch, batch_idx):
		"""Conduct the training step for the given batch"""
		# Extract the input and masks from the batch
		if len(batch) == 4:
			data_x, goal_mask, safe_mask, unsafe_mask = batch
		elif len(batch) == 7:
			data_x, goal_mask, safe_mask, unsafe_mask, boundary_mask, JP, JR = batch
			data_jacobian = (JP, JR)
		else:
			raise ValueError("Please check batch data, component number must be 4 or 7")

		h = self.h(data_x)

		# Compute the losses
		component_losses = {}
		component_losses.update(
			self.boundary_loss(h, goal_mask, safe_mask, unsafe_mask)
		)
		if self.current_epoch > self.learn_shape_epochs:
			if len(batch) == 4:
				component_losses.update(
					self.descent_loss(
						data_x, goal_mask, safe_mask, unsafe_mask, requires_grad=True
					)
				)
			elif len(batch) == 7:
				component_losses.update(
					self.descent_loss(
						data_x, goal_mask, safe_mask, unsafe_mask, boundary_mask, data_jacobian, requires_grad=True
					)
				)

		# Compute the overall loss by summing up the individual losses
		total_loss = torch.tensor(0.0).type_as(data_x)
		# For the objectives, we can just sum them
		for key, loss_value in component_losses.items():
			if not torch.isnan(loss_value):
				total_loss += loss_value
			component_losses[key] = loss_value.detach().data

		batch_dict = {"loss": total_loss, **component_losses}

		return batch_dict

	def training_epoch_end(self, outputs):
		"""This function is called after every epoch is completed."""
		# Outputs contains a list for each optimizer, and we need to collect the losses
		# from all of them if there is a nested list
		if isinstance(outputs[0], list):
			outputs = itertools.chain(*outputs)

		# Gather up all of the losses for each component from all batches
		losses = {}
		for batch_output in outputs:
			for key in batch_output.keys():
				# if we've seen this key before, add this component loss to the list
				if key in losses:
					losses[key].append(batch_output[key])
				else:
					# otherwise, make a new list
					losses[key] = [batch_output[key]]

		# Average all the losses
		avg_losses = {}
		for key in losses.keys():
			key_losses = torch.stack(losses[key])
			avg_losses[key] = torch.nansum(key_losses) / key_losses.shape[0]

		# Log the overall loss...
		self.log("Total loss / train", avg_losses["loss"], sync_dist=True)
		# And all component losses
		for loss_key in avg_losses.keys():
			# We already logged overall loss, so skip that here
			if loss_key == "loss":
				continue
			# Log the other losses
			self.log(loss_key + " / train", avg_losses[loss_key], sync_dist=True)

	def validation_step(self, batch, batch_idx):
		"""Conduct the validation step for the given batch"""
		# Extract the input and masks from the batch
		if len(batch) == 4:
			data_x, goal_mask, safe_mask, unsafe_mask = batch
		elif len(batch) == 7:
			data_x, goal_mask, safe_mask, unsafe_mask, boundary_mask, JP, JR = batch
			data_jacobian = (JP, JR)
		else:
			raise ValueError("Please check batch data, component number must be 4 or 7")

		h = self.h(data_x)

		# Get the various losses
		component_losses = {}
		component_losses.update(
			self.boundary_loss(h, goal_mask, safe_mask, unsafe_mask)
		)
		if self.current_epoch > self.learn_shape_epochs:
			if len(batch) == 4:
				component_losses.update(
					self.descent_loss(data_x, goal_mask, safe_mask, unsafe_mask)
				)
			elif len(batch) == 7:
				component_losses.update(
					self.descent_loss(data_x, goal_mask, safe_mask, unsafe_mask, boundary_mask, data_jacobian)
				)

		# Compute the overall loss by summing up the individual losses
		total_loss = torch.tensor(0.0).type_as(data_x)
		# For the objectives, we can just sum them
		for key, loss_value in component_losses.items():
			if not torch.isnan(loss_value):
				total_loss += loss_value
			component_losses[key] = loss_value.detach().data

		# Also compute the accuracy associated with each loss
		component_losses.update(
			self.boundary_loss(h, goal_mask, safe_mask, unsafe_mask, accuracy=True)
		)
		if len(batch) == 4:
			component_losses.update(
				self.descent_loss(data_x, goal_mask, safe_mask, unsafe_mask, accuracy=True)
			)
		elif len(batch) == 7:
			component_losses.update(
				self.descent_loss(data_x, goal_mask, safe_mask, unsafe_mask, boundary_mask, data_jacobian, accuracy=True)
			)

		batch_dict = {"val_loss": total_loss, **component_losses}

		return batch_dict

	def validation_epoch_end(self, outputs):
		"""This function is called after every epoch is completed."""
		# Gather up all of the losses for each component from all batches
		losses = {}
		for batch_output in outputs:
			for key in batch_output.keys():
				# if we've seen this key before, add this component loss to the list
				if key in losses:
					losses[key].append(batch_output[key])
				else:
					# otherwise, make a new list
					losses[key] = [batch_output[key]]

		# Average all the losses
		avg_losses = {}
		for key in losses.keys():
			key_losses = torch.stack(losses[key])
			avg_losses[key] = torch.nansum(key_losses) / key_losses.shape[0]

		# Log the overall loss...
		self.log("Total loss / val", avg_losses["val_loss"], sync_dist=True)
		# And all component losses
		for loss_key in avg_losses.keys():
			# We already logged overall loss, so skip that here
			if loss_key == "val_loss":
				continue
			# Log the other losses
			self.log(loss_key + " / val", avg_losses[loss_key], sync_dist=True)

		# **Now entering spicetacular automation zone**
		# We automatically run experiments every few epochs
		# # finetuning
		# for _ in tqdm.tqdm(range(10), desc="collecting data"):
		# 	if np.random.rand() > max(1 - self.current_epoch / 100, 0.1): # 0.02
		# 		self.datamodule.add_data(partial(self.dynamics_model.simulate, controller=partial(self.u, use_datax=True),
		# 										 data_collection=True))
		# 	else:
		# 		self.datamodule.add_data(partial(self.dynamics_model.simulate,
		# 										 controller=partial(self.u_reference),
		# 										 data_collection=True))

		# down sampling unsafe data if yumi
		if 'lidar' in str(self.dynamics_model) and self.current_epoch % 3==0:
			print("="*20, "down sampling unsafe data", "="*20)
			self.datamodule.downsample_unsafe()
		else:
			print("="*20, "no down sampling", "="*20)

		for key in self.datamodule.x_training_mask.keys():
			self.log(f"training_data / {key}", self.datamodule.x_training_mask[key].sum(), sync_dist=True)
			self.log(f"validation_data / {key}", self.datamodule.x_validation_mask[key].sum(), sync_dist=True)

		# Only plot every 5 epochs
		if self.current_epoch % 5 != 0:
			return

		self.experiment_suite.run_all_and_log_plots(
			self, self.logger, self.current_epoch
		)

	@pl.core.decorators.auto_move_data
	def simulator_fn(
			self,
			x_init: torch.Tensor,
			num_steps: int,
			relaxation_penalty: Optional[float] = None,
	):
		# Choose parameters randomly
		random_scenario = {}
		for param_name in self.scenarios[0].keys():
			param_max = max([s[param_name] for s in self.scenarios])
			param_min = min([s[param_name] for s in self.scenarios])
			random_scenario[param_name] = random.uniform(param_min, param_max)

		return self.dynamics_model.simulate(
			x_init,
			num_steps,
			self.u,
			guard=self.dynamics_model.out_of_bounds_mask,
			controller_period=self.controller_period,
			params=random_scenario,
		)

	def configure_optimizers(self):
		cbf_params = list(self.parameters())

		cbf_opt = torch.optim.Adam(
			cbf_params,
			lr=5e-4
		)

		self.opt_idx_dict = {0: "cbf"}

		return [cbf_opt]

	def u(self, datax: torch.Tensor, u_nominal=None):
		"""Get the control input for a given state"""
		if False:  #self.use_neural_actor:
			t_dict = {}
			a = time.time()
			if torch.cuda.is_available():
				torch.cuda.synchronize()
			default_nominal = (u_nominal is None)
			if hasattr(self.dynamics_model, "datax_to_x"):
				x = self.dynamics_model.datax_to_x(datax).type_as(self.h_layers["input_linear"].weight)
				bs = x.shape[0]

				assert x.shape[1] == self.n_dims_extended

				state = x[:, :self.dynamics_model.n_dims]
				observation = x[:, self.dynamics_model.n_dims:]

				# encoding
				with torch.no_grad():
					feature = self.pc_head(observation)
					encoded_obs = self.encoder(feature)

				if u_nominal is None:
					u_nominal = self.u_reference(x)

				u_residual = self.actor_nn(torch.cat([state, encoded_obs, u_nominal], dim=-1))
			else:
				if u_nominal is None:
					u_nominal = self.u_reference(datax)
				u_residual = self.actor_nn(torch.cat([datax, u_nominal], dim=-1))
			b=time.time()
			t_dict["actor_forward"] = b-a
			upper_u_lim, lower_u_lim = self.dynamics_model.control_limits
			u_residual = torch.lerp(lower_u_lim.type_as(u_residual), upper_u_lim.type_as(u_residual), u_residual)
			u_before = u_residual + u_nominal

			u_clamped = []
			for dim_idx in range(self.dynamics_model.n_controls):
				u_clamped.append(torch.clamp(
					u_before[:, dim_idx],
					min=lower_u_lim[dim_idx].item(),
					max=upper_u_lim[dim_idx].item(),
				))
			u = torch.stack(u_clamped, dim=-1)
			c=time.time()
			t_dict["clamp_action"] = c-b

			if default_nominal:
				return u, t_dict
			else:
				return u, u_residual
		else:
			return CBFController.u(self, datax)

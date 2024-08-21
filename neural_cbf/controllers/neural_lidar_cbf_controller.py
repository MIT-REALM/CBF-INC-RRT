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
import pytorch_lightning as pl

import matplotlib.pyplot as plt

from neural_cbf.systems import ArmLidar
from neural_cbf.systems.utils import ScenarioList
from neural_cbf.controllers import NeuralObsCBFController
from neural_cbf.controllers.utils import PointNetfeat, PointNetVanillaEncoder
from neural_cbf.datamodules.episodic_datamodule import EpisodicDataModule
from neural_cbf.experiments import ExperimentSuite


class NeuralLidarCBFController(NeuralObsCBFController):
	"""
	h:  (observations -> encoder) + state -> fully-connected layers -> h
	"""

	def __init__(
			self,
			dynamics_model: ArmLidar,
			scenarios: ScenarioList,
			datamodule: EpisodicDataModule,
			experiment_suite: ExperimentSuite,
			**kwargs,
	):
		"""Initialize the controller.

		args:
			dynamics_model: the control-affine dynamics of the underlying system
			scenarios: a list of parameter scenarios to train on
			experiment_suite: defines the experiments to run during training
			cbf_hidden_layers: number of hidden layers to use for the CBF network
			cbf_hidden_size: number of neurons per hidden layer in the CBF network
			cbf_lambda: convergence rate for the CBF
			cbf_relaxation_penalty: the penalty for relaxing CBF conditions.
			controller_period: the timestep to use in simulating forward Vdot
			learn_shape_epochs: number of epochs to spend just learning the shape
			state_only: if True, define the barrier function in terms of robot state
		"""
		super(NeuralLidarCBFController, self).__init__(
			dynamics_model=dynamics_model,
			scenarios=scenarios,
			datamodule=datamodule,
			experiment_suite=experiment_suite,
			**kwargs,
		)

		self.all_encoded_obs_dim = kwargs["feature_dim"]
		self.n_dims_extended = self.dynamics_model.n_dims + self.dynamics_model.o_dims

		# ----------------------------------------------------------------------------
		# Define the encoder network
		# ----------------------------------------------------------------------------
		self.pc_head = PointNetfeat(num_sensor=len(self.dynamics_model.list_sensor),
									ray_per_sensor=self.dynamics_model.ray_per_sensor,
									input_channel=self.dynamics_model.point_dims,
									output_channel=kwargs["per_feature_dim"],
									use_bn=kwargs["use_bn"], )
		self.encoder = PointNetVanillaEncoder(num_sensor=len(self.dynamics_model.list_sensor),
											  ray_per_sensor=self.dynamics_model.ray_per_sensor,
											  input_channel=kwargs["per_feature_dim"],
											  output_dim=self.all_encoded_obs_dim)

		# ----------------------------------------------------------------------------
		# Define the BF network, which we denote h
		# ----------------------------------------------------------------------------
		num_h_inputs = self.dynamics_model.n_dims + self.all_encoded_obs_dim

		# CBF head
		self.h_layers: OrderedDict[str, nn.Module] = OrderedDict()
		self.h_layers["input_linear"] = nn.Linear(num_h_inputs, self.h_hidden_size)
		self.h_layers["input_activation"] = nn.LeakyReLU(0.1)
		for i in range(self.h_hidden_layers):
			self.h_layers[f"layer_{i}_linear"] = nn.Linear(
				self.h_hidden_size, self.h_hidden_size
			)
			self.h_layers[f"layer_{i}_activation"] = nn.LeakyReLU(0.1)
		self.h_layers["output_linear"] = nn.Linear(self.h_hidden_size, 1)
		self.h_nn = nn.Sequential(self.h_layers)

		# ----------------------------------------------------------------------------
		# Define the actor network, which we denote u
		# ----------------------------------------------------------------------------
		self.use_neural_actor = kwargs["use_neural_actor"]
		if self.use_neural_actor:
			# actor head
			self.actor_layers: OrderedDict[str, nn.Module] = OrderedDict()
			self.actor_layers["input_linear"] = nn.Linear(num_h_inputs + self.dynamics_model.n_controls,
														  self.h_hidden_size)
			self.actor_layers["input_activation"] = nn.ReLU()
			for i in range(self.h_hidden_layers):
				self.actor_layers[f"layer_{i}_linear"] = nn.Linear(
					self.h_hidden_size, self.h_hidden_size
				)
				self.actor_layers[f"layer_{i}_activation"] = nn.ReLU()
			self.actor_layers["output_linear"] = nn.Linear(self.h_hidden_size, self.dynamics_model.n_dims)
			self.actor_layers["output_clamp"] = nn.Sigmoid()
			self.actor_nn = nn.Sequential(self.actor_layers)

	# @torch.autocast('cuda' if torch.cuda.is_available() else 'cpu')
	def h(self, datax: torch.Tensor):
		"""Return the CBF value for the observations o

		args:
			x: bs x self.n_dims_extended tensor of state and observation
		returns:
			h: bs x 1 tensor of BF values
		"""
		x = self.dynamics_model.datax_to_x(datax)
		bs = x.shape[0]
		assert x.shape[1] == self.n_dims_extended

		state = x[:, :self.dynamics_model.n_dims]
		observation = x[:, self.dynamics_model.n_dims:]

		# encoding
		feature = self.pc_head(observation)
		encoded_obs = self.encoder(feature)
		# Then get the barrier function value.
		h = self.h_nn(torch.cat([state, encoded_obs], dim=-1))

		return h

	def h_with_jacobian(self, datax: torch.Tensor, data_jacobian: tuple) -> Tuple[
		torch.Tensor, torch.Tensor, dict]:
		"""Computes the CLBF value and its Jacobian

		args:
			x: bs x (self.dynamics_model.n_dims + o_dims) the points at which to evaluate the CLBF
		returns:
			V: bs tensor of CLBF values
			JV: bs x 1 x self.dynamics_model.n_dims Jacobian of each row of V wrt x
		"""
		bs = datax.shape[0]
		feature_level = False
		dq_scale = self.dynamics_model.controller_dt/2

		t_dict = {}

		self.pc_head.eval()
		if torch.cuda.is_available():
			torch.cuda.synchronize()
		t0 = time.time()
		# prepare x_prime=x_{t+1}, shape: (bs * q_dims) * x_dim
		with torch.no_grad():
			dq1 = dq_scale * torch.eye(self.dynamics_model.q_dims, device=datax.device).unsqueeze(0).expand(bs, -1, -1)
			dq2 = -dq_scale * torch.eye(self.dynamics_model.q_dims, device=datax.device).unsqueeze(0).expand(bs, -1, -1)
			dqs = torch.cat([dq1, dq2], dim=1)
			assert datax.shape[
					   1] == self.dynamics_model.n_dims + self.dynamics_model.o_dims_in_dataset + self.dynamics_model.state_aux_dims_in_dataset

			if torch.cuda.is_available():
				torch.cuda.synchronize()
			t00 = time.time()
			datax_prime = [self.dynamics_model.batch_lookahead(datax, dqs[:, i, :], data_jacobian) for i in
						   range(dqs.shape[1])]
			datax_prime = torch.cat(datax_prime, dim=1).reshape(-1, datax.shape[1])
		if torch.cuda.is_available():
			torch.cuda.synchronize()
		t_dict['prepare_x_prime_1'] = time.time() - t00
		t_dict['prepare_x_prime'] = time.time() - t0

		if feature_level:  # numerical on feature level and symbolic to cbf value
			raise NotImplementedError('Did not implement feature level for h(data_x)')
			feature = self.pc_head(observation)
			with torch.enable_grad():
				state_for_grad = torch.autograd.Variable(state.data, requires_grad=True)
				feature_for_grad = torch.autograd.Variable(feature.data, requires_grad=True)

				encoded_obs = self.encoder(feature_for_grad)
				h = self.h_nn(torch.cat([state_for_grad, encoded_obs], dim=-1))
				Jh_q = torch.autograd.grad(h.sum(), state_for_grad, create_graph=True, retain_graph=True)[0].unsqueeze(
					1)
				ph_pf = torch.autograd.grad(h.sum(), feature_for_grad, create_graph=True, retain_graph=True)[
					0].unsqueeze(1)
			with torch.no_grad():
				dfdq = torch.zeros((bs, feature.shape[1], self.dynamics_model.q_dims)).type_as(x)
				# x_prime[:, self.dynamics_model.n_dims:] += torch.Tensor(x_prime.shape[0], x_prime.shape[1]-self.dynamics_model.n_dims).uniform_(-1e-5, 1e-5).type_as(x_prime)
				feature_prime = self.pc_head(x_prime[:, self.dynamics_model.n_dims:]).reshape(bs, -1, feature.shape[-1])
				for dim in range(self.dynamics_model.q_dims):
					dfdq[:, :, dim] = (feature_prime[:, dim, :] - feature[:, :]) / dqs[dim][dim]
			J = Jh_q + torch.bmm(ph_pf, dfdq.type_as(x))
		else:  # pure numerical estimation
			with torch.no_grad():
				if torch.cuda.is_available():
					torch.cuda.synchronize()
				t1 = time.time()
				all_h = self.h(torch.cat((datax, datax_prime), dim=0))
				if torch.cuda.is_available():
					torch.cuda.synchronize()
				t2 = time.time()
				h = all_h[:bs]
				h_prime = all_h[bs:].reshape(bs, 1, -1)
				J = (h_prime[:, :, :self.dynamics_model.q_dims] - h.unsqueeze(1)) / (dq_scale * 2)

			if torch.cuda.is_available():
				torch.cuda.synchronize()
			t3 = time.time()
			t_dict['single_prime_forward'] = t2 - t1
			t_dict['jacobian'] = t3 - t2

		if self.h_nn.training:
			self.pc_head.train()

		return h, J, t_dict

	def descent_loss(
			self,
			data_x: torch.Tensor,
			goal_mask: torch.Tensor,
			safe_mask: torch.Tensor,
			unsafe_mask: torch.Tensor,
			boundary_mask: torch.Tensor,
			data_jacobian: Tuple[torch.Tensor, torch.Tensor],
			accuracy: bool = False,
			requires_grad: bool = False,
	) -> List[Tuple[str, torch.Tensor]]:
		"""
		Evaluate the loss on the CBF due to the descent condition

		args:
			x: the points at which to evaluate the loss,
			goal_mask: the points in x marked as part of the goal
			safe_mask: the points in x marked safe
			unsafe_mask: the points in x marked unsafe
			accuracy: if True, return the accuracy (from 0 to 1) as well as the losses
		returns:
			loss: a list of tuples containing ("category_name", loss_value).
		"""
		# Compute loss to encourage satisfaction of the following conditions...
		loss = []

		bs = safe_mask.shape[0]
		ul, ll = self.dynamics_model.control_limits
		upper_limit = ul.unsqueeze(0).expand(bs, -1).type_as(data_x)
		lower_limit = ll.unsqueeze(0).expand(bs, -1).type_as(data_x)

		qp_coef = self.loss_config["descent_violation_weight"]
		# qp_coef = min(max(0, (self.current_epoch-self.learn_shape_epochs)/50), 1) * self.loss_config["descent_violation_weight"]

		if self.use_neural_actor:
			u_goal_reaching = torch.lerp(lower_limit, upper_limit,
										 torch.Tensor(*upper_limit.shape).uniform_(0, 1).type_as(data_x))

			h = self.h(data_x)
			u, u_residual = self.u(data_x, u_goal_reaching)

			datax_next = self.dynamics_model.batch_lookahead(data_x, u * self.dynamics_model.dt,
															 data_jacobian=data_jacobian)
			hdot_simulated = (self.h(datax_next) - h) / self.dynamics_model.dt

			hdot = hdot_simulated
			alpha = torch.where(h < 0, 10, self.clf_lambda).type_as(u)
			qp_relaxation = F.relu(hdot + torch.multiply(alpha, h))

			# Minimize the qp relaxation to encourage satisfying the decrease condition
			qp_relaxation_loss = qp_relaxation.mean() * qp_coef
			loss.append(("QP relaxation", qp_relaxation_loss))
			loss.append(("residual", torch.norm(u_residual, p=2, dim=1).mean() * self.loss_config["actor_weight"]))
		else:
			_, Lf_V, Lg_V, _ = self.V_with_lie_derivatives(data_x, data_jacobian)

			Lg_V_no_grad = Lg_V.detach().clone().squeeze(1)  # bs * n_control

			h = self.h(data_x)
			u_coef = self.loss_config["u_coef_in_training"]
			u = torch.where(Lg_V_no_grad < 0, upper_limit * u_coef, lower_limit * u_coef)

			hdot_expected = (Lf_V.squeeze(1).squeeze(1) + torch.bmm(Lg_V, u.unsqueeze(2)).squeeze(1).squeeze(
				1)).unsqueeze(1)
			datax_next = self.dynamics_model.batch_lookahead(data_x, u * self.dynamics_model.controller_dt,
															 data_jacobian=data_jacobian)
			hdot_simulated = (self.h(datax_next) - h) / self.dynamics_model.controller_dt

			hdot = hdot_simulated
			alpha = self.clf_lambda  # torch.where(h < 0, 2 * self.clf_lambda, self.clf_lambda).type_as(x)
			# qp_relaxation = F.relu(hdot + torch.multiply(alpha, h + self.safe_level))
			qp_relaxation = F.relu(hdot + torch.multiply(alpha, torch.minimum(h, 2 * self.unsafe_level * torch.ones(*h.shape).type_as(h))).detach())

			# Minimize the qp relaxation to encourage satisfying the decrease condition
			qp_relaxation_loss = qp_relaxation.mean() * qp_coef / alpha
			loss.append(("QP relaxation", qp_relaxation_loss))

			loss.append(("hdot divergence",
						 self.loss_config["hdot_divergence_weight"] * torch.abs(hdot_simulated - hdot_expected).mean()))

		if accuracy:
			qp_acc_safe = (qp_relaxation[safe_mask] <= alpha * self.safe_level).sum() / qp_relaxation[safe_mask].nelement()
			qp_acc_unsafe = (qp_relaxation[unsafe_mask] <= alpha * self.safe_level).sum() / qp_relaxation[unsafe_mask].nelement()
			qp_acc_boundary = (qp_relaxation[boundary_mask] <= alpha * self.safe_level).sum() / qp_relaxation[boundary_mask].nelement()
			loss.append(("boundary condition accuracy/safe", qp_acc_safe))
			loss.append(("boundary condition accuracy/unsafe", qp_acc_unsafe))
			loss.append(("boundary condition accuracy/boundary", qp_acc_boundary))

		return loss

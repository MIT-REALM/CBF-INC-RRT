import itertools
from typing import Tuple, List, Optional
from collections import OrderedDict
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import pytorch_lightning as pl

import matplotlib.pyplot as plt

from neural_cbf.systems import ArmMindis
from neural_cbf.systems.utils import ScenarioList
from neural_cbf.controllers import NeuralObsCBFController
from neural_cbf.datamodules.episodic_datamodule import EpisodicDataModule
from neural_cbf.experiments import ExperimentSuite


class NeuralMindisCBFController(NeuralObsCBFController):
    def __init__(
        self,
        dynamics_model: ArmMindis,
        scenarios: ScenarioList,
        datamodule: EpisodicDataModule,
        experiment_suite: ExperimentSuite,
        **kwargs,
    ):
        super(NeuralMindisCBFController, self).__init__(
            dynamics_model=dynamics_model,
            scenarios=scenarios,
            datamodule=datamodule,
            experiment_suite=experiment_suite,
            **kwargs,
        )

        # ----------------------------------------------------------------------------
        # Define the CBF network, which we denote h
        # ----------------------------------------------------------------------------
        self.n_dims_extended = self.dynamics_model.n_dims + self.dynamics_model.o_dims
        num_h_inputs = self.n_dims_extended

        self.h_layers: OrderedDict[str, nn.Module] = OrderedDict()
        self.h_layers["input_linear"] = nn.Linear(num_h_inputs, self.h_hidden_size)
        self.h_layers["input_activation"] = nn.ReLU()
        for i in range(self.h_hidden_layers):
            self.h_layers[f"layer_{i}_linear"] = nn.Linear(
                self.h_hidden_size, self.h_hidden_size
            )
            self.h_layers[f"layer_{i}_activation"] = nn.ReLU()
        self.h_layers["output_linear"] = nn.Linear(self.h_hidden_size, 1)
        self.h_nn = nn.Sequential(self.h_layers)

        # ----------------------------------------------------------------------------
        # Define the neural actor network, which we denote u
        # ----------------------------------------------------------------------------
        if self.use_neural_actor:
            self.actor_layers: OrderedDict[str, nn.Module] = OrderedDict()
            # input: state, o, do/dq, nominal control
            self.actor_layers["input_linear"] = nn.Linear(num_h_inputs + 2 * dynamics_model.n_dims, self.h_hidden_size)
            self.actor_layers["input_activation"] = nn.ReLU()
            for i in range(self.h_hidden_layers):
                self.actor_layers[f"layer_{i}_linear"] = nn.Linear(
                    self.h_hidden_size, self.h_hidden_size
                )
                self.actor_layers[f"layer_{i}_activation"] = nn.ReLU()
            self.actor_layers["output_linear"] = nn.Linear(self.h_hidden_size, dynamics_model.n_dims)
            self.actor_nn = nn.Sequential(self.actor_layers)

    def h(self, datax: torch.Tensor):
        """
        only using state and mindis observation
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        x = datax[:, :self.n_dims_extended]
        h = self.h_nn(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return h

    def h_with_jacobian(self, x: torch.Tensor, data_jacobian: tuple = None):
        """Computes the CLBF value and its Jacobian

        args:
            x: bs x self.dynamics_model.n_dims the points at which to evaluate the CLBF
        returns:
            V: bs tensor of CLBF values
            JV: bs x 1 x self.dynamics_model.n_dims Jacobian of each row of V wrt x
        """
        x_norm = x[:, :self.n_dims_extended]
        with torch.enable_grad():
            x_for_grad = torch.autograd.Variable(x_norm.data, requires_grad=True)
            h = self.h(x_for_grad)
            Jh = torch.autograd.grad(h.sum(), x_for_grad, create_graph=True, retain_graph=True)[0].unsqueeze(1)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dodx = x[:, self.n_dims_extended:].type_as(x)
            dodx = dodx.reshape(-1, 1, self.dynamics_model.n_dims)
            J = Jh[:, :, :self.dynamics_model.n_dims] + torch.bmm(Jh[:, :, self.dynamics_model.n_dims:], dodx)
        return h, J, {}

    def forward(self, x):
        """Determine the control input for a given state using a QP

        args:
            x: bs x self.dynamics_model.n_dims tensor of state
        returns:
            u: bs x self.dynamics_model.n_controls tensor of control inputs
        """
        return self.u(x[:, :self.n_dims_extended])

    def descent_loss(
        self,
        x: torch.Tensor,
        goal_mask: torch.Tensor,
        safe_mask: torch.Tensor,
        unsafe_mask: torch.Tensor,
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
        upper_limit = ul.unsqueeze(0).expand(bs, -1).type_as(x)
        lower_limit = ll.unsqueeze(0).expand(bs, -1).type_as(x)

        qp_coef = self.loss_config["descent_violation_weight"]
        _, Lf_V, Lg_V, _ = self.V_with_lie_derivatives(x)

        Lg_V_no_grad = Lg_V.detach().clone().squeeze(1)  # bs * n_control

        h = self.h(x)
        u_coef = self.loss_config["u_coef_in_training"]
        if self.use_neural_actor:
            u_goal_reaching = torch.lerp(lower_limit, upper_limit,
                                         torch.Tensor(*upper_limit.shape).uniform_(0, 1).type_as(x))
            u, u_residual = self.u(x, u_goal_reaching)
            loss.append(("residual", torch.norm(u_residual, p=2, dim=1).mean() * self.loss_config["actor_weight"]))
        else:
            u = torch.where(Lg_V_no_grad < 0, upper_limit * u_coef, lower_limit * u_coef)

        hdot_expected = (Lf_V.squeeze(1).squeeze(1) + torch.bmm(Lg_V, u.unsqueeze(2)).squeeze(1).squeeze(
            1)).unsqueeze(1)

        hdot = hdot_expected
        alpha = self.clf_lambda  # torch.where(h < 0, 2 * self.clf_lambda, self.clf_lambda).type_as(x)
        qp_relaxation = F.relu(hdot + torch.multiply(alpha, h + self.safe_level))

        # Minimize the qp relaxation to encourage satisfying the decrease condition
        qp_relaxation_loss = qp_relaxation.mean() * qp_coef
        loss.append(("QP relaxation", qp_relaxation_loss))

        if accuracy:
            qp_acc_safe = (qp_relaxation[safe_mask] <= alpha * self.safe_level).sum() / qp_relaxation[safe_mask].nelement()
            qp_acc_unsafe = (qp_relaxation[unsafe_mask] <= alpha * self.safe_level).sum() / qp_relaxation[unsafe_mask].nelement()
            boundary_mask = torch.logical_not(torch.logical_or(safe_mask, unsafe_mask))
            qp_acc_boundary = (qp_relaxation[boundary_mask] <= alpha * self.safe_level).sum() / qp_relaxation[
                boundary_mask].nelement()
            loss.append(("boundary condition accuracy/safe", qp_acc_safe))
            loss.append(("boundary condition accuracy/unsafe", qp_acc_unsafe))
            loss.append(("boundary condition accuracy/boundary", qp_acc_boundary))

        return loss

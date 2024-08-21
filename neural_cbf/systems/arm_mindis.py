import time
from typing import Tuple, List, Optional, Callable

import os
import torch
import numpy as np
import tqdm
from torch.autograd.functional import jacobian

from environment import ArmEnv, BasicRobot

from neural_cbf.systems import ArmDynamics
from .utils import grav, Scenario


class ArmMindis(ArmDynamics):
	"""
	The observation is the distance to the closest obstacle, and the aux is the derivative of the distance
	"""

	# Number of states and controls

	def __init__(
			self,
			nominal_params: Scenario,
			dt: float,
			controller_dt: Optional[float],
			dis_threshold: float,
			env: ArmEnv = None,
			robot: BasicRobot = None
	):
		"""
		Initialization.

		args:
			nominal_params: a dictionary giving the parameter values for the system. Not used.
			dt: the timestep to use for the simulation
			controller_dt: the timestep for the LQR discretization
		"""
		super().__init__(nominal_params, dt, controller_dt, dis_threshold=dis_threshold,
						 use_linearized_controller=False, env=env, robot=robot)

		self.compute_linearized_controller(None)

	def __str__(self):
		return f"{str(self.robot)}_mindis_Dynamics"

	@property
	def o_dims(self) -> int:
		return 1

	@property
	def state_aux_dims(self) -> int:
		return self.robot.body_dim

	def calc_do_dq(self, sd):
		nhat_AB_W = np.array(sd[5]) - np.array(sd[6])
		nhat_AB_W /= sd[8]
		q = self.robot.get_joint_position(self.robot.body_joints + self.robot.ee_joints)

		frame_A = self.env.p.getLinkState(sd[1], sd[3])[4:6]
		frame_A_inv = self.env.p.invertTransform(frame_A[0], frame_A[1])
		p_ACa = self.env.p.multiplyTransforms(frame_A_inv[0], frame_A_inv[1], sd[5], [0, 0, 0, 1])[0]

		Jacobian = self.env.p.calculateJacobian(sd[1], sd[3], p_ACa, objPositions=q,
									   objVelocities=[0. for _ in range(self.robot.body_dim + self.robot.ee_dim)],
									   objAccelerations=[0. for _ in range(self.robot.body_dim + self.robot.ee_dim)])[0]
		do_dx = nhat_AB_W.T @ (np.array(Jacobian)[:, self.robot.body_joints])
		return do_dx

	def _get_observation_with_state(self, state):
		self.robot.set_joint_position(self.robot.body_joints, state[:self.q_dims])

		dis = []
		do_dqs = []
		for obstacle in self.env.obstacle_ids:
			sd_all = list(self.env.p.getClosestPoints(self.robot.robotId, obstacle, 5, linkIndexB=-1))
			for sd in sd_all[1:]:
				dis.append(sd[8])
				do_dqs.append(sd)

		dis = np.array(dis)
		shortest_dis = np.min(dis)
		idx = np.where(dis == shortest_dis)[0]
		if idx.shape[0] == 1:
			do_dq = self.calc_do_dq(do_dqs[idx[0]])  # Robot is closest to only 1 object
		else:
			do_dq = self.calc_do_dq(do_dqs[idx[0]])

		return np.concatenate(([shortest_dis], do_dq.squeeze()))


if __name__ == '__main__':
	problem_num = 1000
	obstacle_num = 8

	robot_name = 'yumi'
	nominal_params = {"m1": 5.76}
	controller_period = 1 / 30
	simulation_dt = 1 / 120
	environment = ArmEnv([robot_name], GUI=True, config_file='')
	robot = environment.robot_list[0]
	dynamics_model = ArmMindis(
		nominal_params,
		dt=simulation_dt,
		dis_threshold=0.02,
		controller_dt=controller_period,
		env=environment,
		robot=robot
	)

	while True:
		environment.p.stepSimulation()

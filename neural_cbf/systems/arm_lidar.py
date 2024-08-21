"""Define observation-related dynamics for a 7-DoF Franka Panda robot"""
import time
from typing import Tuple, List, Optional, Callable, Dict

import os
import torch
import numpy as np
import tqdm

from environment import ArmEnv, BasicRobot

from neural_cbf.systems import ArmDynamics
from .utils import grav, cartesian_to_spherical, spherical_to_cartesian
from .utils import Scenario


class ArmLidar(ArmDynamics):
	"""
	The observation is a point cloud in world frame.

	State: q + o + aux
	"""

	# Number of states and controls

	def __init__(
			self,
			nominal_params: Scenario,
			dt: float,
			controller_dt: Optional[float],
			dis_threshold: float,
			env: ArmEnv = None,
			robot: BasicRobot = None,
			## observation-related, more than base dynamics
			n_obs: int = 128,
			point_in_dataset_pc: int = 0,
			list_sensor=None,
			observation_type: str = '',
			point_dim=4,
			add_normal=False,
	):
		"""
		Initialization.

		args:
			nominal_params: a dictionary giving the parameter values for the system.
							Requires keys ["m", "I", "r"]
			dt: the timestep to use for the simulation
			controller_dt: the timestep for the LQR discretization. Defaults to dt
		raises:
			ValueError if nominal_params are not valid for this system
		"""
		super().__init__(nominal_params, dt, controller_dt, dis_threshold=dis_threshold,
						 use_linearized_controller=False, env=env, robot=robot)

		self.compute_linearized_controller(None)
		self.observation_type = observation_type

		self.add_normal = add_normal
		self.point_dims = point_dim + 3 * int(self.add_normal)

		# initialize sensor
		self.ray_per_sensor = n_obs
		self.point_in_dataset_pc = point_in_dataset_pc

		self.list_sensor = list_sensor

		if self.observation_type == 'uniform_lidar':
			self.rayFromLocal = np.zeros((self.point_in_dataset_pc, 3))
			theta = np.random.normal((0, 0, 0), (1, 1, 1), (self.point_in_dataset_pc, 3))
			self.rayToLocal = theta / np.linalg.norm(theta, axis=1, keepdims=True)

	def __str__(self):
		return f"{str(self.robot)}_lidar_Dynamics"

	@property
	def o_dims_in_dataset(self) -> int:
		return self.point_in_dataset_pc * (3 + 3 * int(self.add_normal))

	@property
	def o_dims(self) -> int:
		return len(self.list_sensor) * self.ray_per_sensor * self.point_dims

	@property
	def state_aux_dims_in_dataset(self) -> int:
		return len(self.list_sensor) * (3 + 9)

	@property
	def state_aux_dims(self) -> int:
		return 0

	def _get_observation_with_state(self, state):
		if self.observation_type == 'uniform_surface':
			obs = self.env.sample_obstacle_surface(self.point_in_dataset_pc, add_normal=self.add_normal)
			return obs.reshape(-1)
		elif self.observation_type == 'uniform_lidar':
			"""
			uniformly ejecting light rays from the sensor on a uniform sphere surface, and check the collision
			"""
			obs = np.zeros((self.point_in_dataset_pc, self.point_dims))
			sensors = [self.robot.body_joints[-1], self.robot.body_joints[self.robot.body_dim // 2]]
			each_obstacle = np.random.randint(low=0, high=len(sensors), size=self.point_in_dataset_pc)

			fk = self.robot.forward_kinematics(self.list_sensor, state[:self.q_dims])
			for sensor in sensors:
				origin = fk[sensor][0]
				orientation = fk[sensor][1]

				which_point = np.where(each_obstacle == sensors.index(sensor))[0]

				rayFrom = self.rayFromLocal[which_point] + origin
				rayTo = self.rayToLocal[which_point] + origin
				raw_results = self.env.p.rayTestBatch(rayFrom.reshape((-1, 3)), rayTo.reshape((-1, 3)), numThreads=0)
				raw_results_position = np.array(
					[result[3] for result in raw_results if result[0] in self.env.obstacle_ids])
				raw_results_normal = np.array([result[4] for result in raw_results if result[0] in self.env.obstacle_ids])

				from scipy.spatial.transform import Rotation as R
				for pt in raw_results_position:
					start_point = rayFrom[0]
					end_point = pt
					diff_vector = end_point - start_point
					length = np.linalg.norm(diff_vector)
					diff_unit = diff_vector / length

					# Calculate the midpoint of the line (this is where the cylinder will be positioned)
					mid_point = (start_point + end_point) / 2.0

					# # Calculate the orientation of the line
					# orient = R.from_euler('xyz', [np.arctan(np.sqrt(diff_vector[0]**2 + diff_vector[1]**2)/diff_vector[2]), 0, -np.arctan(diff_vector[1] / diff_vector[0])])
					# orient = R.from_euler('zyx', [0, np.arcsin(diff_vector[2] / np.linalg.norm(diff_vector)), -np.arctan(diff_vector[1] / diff_vector[0])])
					axis = np.cross([0, 0, 1], diff_unit)
					axis = axis / np.linalg.norm(axis)
					angle = np.arccos(np.dot([0, 0, 1], diff_unit))
					orient = R.from_rotvec(axis * angle)
					orient = orient.as_quat()

					# Create the line (a thin cylinder) as a dynamic body
					line = self.env.p.createMultiBody(
						baseMass=0,
						baseInertialFramePosition=[0, 0, 0],
						baseVisualShapeIndex=self.env.p.createVisualShape(
							self.env.p.GEOM_CYLINDER,
							radius=0.002,  # you can adjust this as needed
							length=length,
							rgbaColor=[253/255, 16/255, 16/255, 0.5],
						),
						basePosition=list(mid_point),
						baseOrientation=orient,
					)

					# vid = p.createVisualShape(p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 0, 1, 1])
					# p.createMultiBody(baseVisualShapeIndex=vid, basePosition=list(mid_point))

				refined_results = raw_results_position
				if self.add_normal:
					refined_results = np.concatenate((refined_results, raw_results_normal), axis=1)

				if refined_results.shape[0] < which_point.shape[0]:
					refined_results = np.concatenate((refined_results, np.expand_dims(refined_results[0, :], 0).repeat(
						which_point.shape[0] - refined_results.shape[0], axis=0)), axis=0)

				obs[which_point, :] = refined_results
			return obs.reshape(-1)
		else:
			raise NotImplementedError(f"Unknown observation type: {self.observation_type}")

	def complete_sample_with_observations(self, x, num_samples: int) -> torch.Tensor:
		"""
		input: bs * (n_dim + o_dim + aux_dim)
		"""
		samples = torch.zeros(num_samples, self.n_dims + self.o_dims_in_dataset + self.state_aux_dims_in_dataset,
							  device=x.device)
		samples[:, :self.n_dims] = x
		for i in range(num_samples):
			o = self._get_observation_with_state(x[i, :self.q_dims])
			samples[i, self.n_dims:-self.state_aux_dims_in_dataset] = torch.tensor(o, device=x.device)
			samples[i, -self.state_aux_dims_in_dataset:] = self.get_aux(x[i, :self.q_dims])
		return samples

	def get_aux(self, state):
		state_aux = []
		x_fk = self.robot.forward_kinematics(self.list_sensor, state[:self.q_dims])
		for i in range(len(x_fk)):
			p_p = torch.tensor(x_fk[i][0], device=state.device)
			p_r = torch.tensor(x_fk[i][1], device=state.device)
			state_aux.append(torch.cat((p_p.reshape(1, -1), p_r.reshape(1, -1)), dim=1))
		return torch.cat(state_aux, dim=0).reshape(-1)

	def get_jacobian(self, state):
		J_p = torch.zeros((len(self.list_sensor), 3, self.q_dims))
		J_R = torch.zeros((len(self.list_sensor), 3, 3, self.q_dims))
		for a_idx, sensor_link in zip(range(len(self.list_sensor)), self.list_sensor):
			J_p[a_idx, :, :] = torch.from_numpy(self.robot.get_jacobian(state.tolist(), sensor_link, [0, 0, 0])[0])
			J_W = self.robot.get_jacobian(state.tolist(), sensor_link, [0, 0, 0])[1]
			# dW_dq to dR_dq
			transformation = self.get_aux(state).reshape(-1, 12).float()
			R_matrix = transformation[a_idx, 3:].reshape(3, 3)
			J_R[a_idx, 0, 1, :] = torch.from_numpy(-J_W[2, :])
			J_R[a_idx, 0, 2, :] = torch.from_numpy(J_W[1, :])
			J_R[a_idx, 1, 0, :] = torch.from_numpy(J_W[2, :])
			J_R[a_idx, 1, 2, :] = torch.from_numpy(-J_W[0, :])
			J_R[a_idx, 2, 0, :] = torch.from_numpy(-J_W[1, :])
			J_R[a_idx, 2, 1, :] = torch.from_numpy(J_W[0, :])
			J_R[a_idx, :, :, :] = torch.einsum('ibk,bj->ijk', J_R[a_idx, :, :, :], R_matrix)
		return J_p.unsqueeze(0), J_R.unsqueeze(0)

	def get_batch_jacobian(self, x):
		Js_P = []
		Js_R = []
		for idx in range(x.shape[0]):
			J_P, J_R = self.get_jacobian(x[idx, :self.n_dims])
			Js_P.append(J_P)
			Js_R.append(J_R)
		Js_P = torch.cat(Js_P, dim=0)
		Js_R = torch.cat(Js_R, dim=0)
		return Js_P, Js_R

	def datax_to_x(self, x: torch.Tensor):
		# x: bs * (n_dim + o_dim_in_dataset + aux_dim_in_dataset)
		bs = x.shape[0]
		q = x[:, :self.n_dims]
		global_obs = x[:, self.n_dims:-self.state_aux_dims_in_dataset].reshape(bs, -1, 3 + 3 * int(self.add_normal))
		transformation = x[:, -self.state_aux_dims_in_dataset:].reshape(bs, -1, 12)

		origins = transformation[:, :, :3]
		rotation_matrixs = transformation[:, :, 3:].reshape(bs, -1, 3, 3)

		obs = torch.zeros((bs, len(self.list_sensor), self.ray_per_sensor, self.point_dims), device=x.device)
		for idx in range(len(self.list_sensor)):
			origin = origins[:, idx, :]
			rotation_matrix = rotation_matrixs[:, idx, :, :]

			sampled_index = torch.randint(low=0, high=global_obs.shape[1],
										  size=(self.ray_per_sensor, 1), device=x.device).squeeze().int()
			raw_results = torch.index_select(global_obs, dim=1, index=sampled_index)
			refined_results = torch.bmm(torch.transpose(rotation_matrix, 1, 2),
										torch.transpose(raw_results[:, :, :3] - origin.unsqueeze(1), 1, 2))
			refined_results = torch.transpose(refined_results, 1, 2)
			if self.point_dims == 4:
				refined_results = cartesian_to_spherical(refined_results.reshape(-1, 3)).reshape(bs, -1, 4)
			if self.add_normal:
				refined_results = torch.cat(
					(refined_results, torch.transpose(torch.bmm(torch.transpose(rotation_matrix, 1, 2),
																torch.transpose(raw_results[:, :, 3:], 1, 2)), 1, 2)),
					dim=2)
			obs[:, idx, :, :] = refined_results
		return torch.cat((q, obs.reshape(bs, -1)), dim=1)

	def batch_lookahead(self, datax, dqs, data_jacobian):
		"""
		estimate next-step observation, x: state + observation, dqs: a list of possible dq
		(now estimation/learning for future work especially in dynamic env)

		one dqs for one x,
		"""
		if self.point_dims in [3, 4, 6]:
			datax_next = datax.detach().clone()
			datax_next[:, :self.n_dims] += dqs
			if len(data_jacobian) > 0:
				datax_next[:, -self.state_aux_dims_in_dataset:] = \
					self._jacobian_to_auxlookahead(datax[:, -self.state_aux_dims_in_dataset:], data_jacobian, dqs)
			else:
				for idx in range(datax.shape[0]):
					datax_next[idx, -self.state_aux_dims_in_dataset:] = self.get_aux(datax_next[idx, :self.n_dims])
			return datax_next
		else:
			raise NotImplementedError(f"Unknown point dimension {self.point_dims}.")

	def _jacobian_to_auxlookahead(self, aux, jacobian, dq):
		"""
		batch-wise, jacobian: (J_p, J_R), dq: bs * q_dim
		return auxillary dimensions in state, defined by list_sensor * (p + R)
		"""
		bs = aux.shape[0]
		q_dim = dq.shape[1]
		transformation = aux.reshape(bs, -1, 12)
		ps = transformation[:, :, :3]
		Rs = transformation[:, :, 3:].reshape(bs, -1, 3, 3)

		dq = dq.unsqueeze(1).expand(-1, transformation.shape[1], -1).reshape(-1, q_dim, 1)

		p_next = ps + torch.bmm(jacobian[0].reshape(dq.shape[0], 3, q_dim), dq).reshape(*ps.shape)
		R_next = Rs + torch.bmm(jacobian[1].reshape(dq.shape[0], 9, q_dim), dq).reshape(*Rs.shape)
		R_next = R_next.reshape(-1, 3, 3)
		# R_next /= (torch.linalg.det(R_next)).unsqueeze(1).unsqueeze(1).expand(-1, 3, 3)
		aux_next = torch.cat((p_next, R_next.reshape(bs, -1, 9)), dim=2).reshape(bs, -1)
		return aux_next

	def closed_loop_dynamics(
			self, x: torch.Tensor, u: torch.Tensor, collect_dataset: bool = False,
			use_motor_control: bool = False, update_observation: bool = True, return_time: bool = False
	) -> torch.Tensor:
		"""
		Uh-oh! closed_loop_dynamics is different from any other system!!
		due to existence of observation

		This one returns x_next,
		others: xdot
		"""
		if return_time:
			ttt_list = [0, 0]  # [offline, online]
			t0=time.time()
		batch_size = x.shape[0]

		x_next = torch.zeros((batch_size, self.n_dims + self.o_dims_in_dataset + self.state_aux_dims_in_dataset),
							 device=x.device)

		step = 10 if collect_dataset else 1
		if return_time:
			ttt_list[0] += time.time() - t0
			t0=time.time()

		for i in range(batch_size):
			q_dot = u[i]
			xdot = q_dot.type_as(x)
			if use_motor_control:
				# assert batch_size == 1
				self.robot.set_joint_position(self.robot.body_joints, x[i, :self.n_dims])
				self.env.p.setJointMotorControlArray(self.robot.robotId, self.robot.body_joints, self.env.p.VELOCITY_CONTROL,
											targetVelocities=xdot)
				for _ in range(step):
					self.env.p.stepSimulation()
				x_next[i, :self.n_dims] = torch.tensor(self.robot.get_joint_position(self.robot.body_joints),
													   device=x.device)
			else:
				x_next[i, :self.n_dims] = x[i, :self.n_dims] + xdot * self.dt * step
				self.robot.set_joint_position(self.robot.body_joints, x_next[i, :self.n_dims])

			if return_time:
				ttt_list[0] += time.time() - t0
				t0=time.time()

			# observation
			if update_observation:
				o = torch.tensor(self._get_observation_with_state(x_next[i, :self.n_dims]), device=x.device)
			else:
				o = x[i, self.n_dims:-self.state_aux_dims_in_dataset].clone()

			x_next[i, self.n_dims:-self.state_aux_dims_in_dataset] = o
			x_next[i, -self.state_aux_dims_in_dataset:] = self.get_aux(x_next[i, :self.q_dims])

			if return_time:
				ttt_list[1] += time.time() - t0
				t0=time.time()

		if return_time:
			return x_next, ttt_list
		else:
			return x_next


if __name__ == '__main__':
	problem_num = 1000
	obstacle_num = 8

	robot_name = 'yumi'
	nominal_params = {"m1": 5.76}
	controller_period = 1 / 30
	simulation_dt = 1 / 120
	environment = ArmEnv([robot_name], GUI=True, config_file='')
	robot = environment.robot_list[0]
	dynamics_model = ArmLidar(
		nominal_params,
		dt=simulation_dt,
		dis_threshold=0.02,
		controller_dt=controller_period,
		env=environment,
		robot=robot
	)

	while True:
		environment.p.stepSimulation()

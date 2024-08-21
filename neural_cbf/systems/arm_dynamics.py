import time
from typing import Tuple, List, Optional, Callable, Dict
import warnings

import os
import torch
import numpy as np
import tqdm

from environment import ArmEnv, BasicRobot

from neural_cbf.systems import ControlAffineSystem
# from .utils import Scenario
Scenario = Dict[str, float]
ScenarioList = List[Scenario]

class ArmDynamics(ControlAffineSystem):
	"""
	Represents a velocity controller for generic robot arm

	The system has state
		x = [theta_1, ..., theta_n, o, aux]
	representing the angles and observations of the robot arm,

	and it has control inputs
		u = [u_1, ..., u_n]
	representing the velocity at each joint.
	"""

	def __init__(
			self,
			nominal_params,
			dt: float,
			controller_dt: Optional[float],
			dis_threshold: float,
			use_linearized_controller: bool = False,
			env: ArmEnv = None,
			robot: BasicRobot = None
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
		super().__init__(nominal_params, dt, controller_dt, use_linearized_controller=use_linearized_controller)

		# Minimum distance threshold for avoiding collision
		self.dis_threshold = dis_threshold

		self.env = env
		self.robot = robot

		self.goal_state = np.array(self.robot.q0)
		self.intermediate_goals = np.array(self.robot.q0).reshape(1, -1)

		self.compute_linearized_controller(None)

	def set_goal(self, q_goal):
		assert q_goal.shape == self.goal_state.shape
		assert q_goal.shape[-1] == self.intermediate_goals.shape[-1]
		self.goal_state = q_goal
		self.intermediate_goals = q_goal.reshape(1, -1)
		self.compute_linearized_controller(None)

	def set_intermediate_goals(self, q_goal):
		assert q_goal.shape[-1] == self.intermediate_goals.shape[-1]
		self.intermediate_goals = q_goal.reshape(-1, q_goal.shape[-1])
		self.compute_linearized_controller(None)

	def __str__(self):
		return f"{str(self.robot)}_Dynamics"

	@property
	def q_dims(self) -> int:  # joint
		pass
		return self.robot.body_dim

	@property
	def n_dims(self) -> int:  # state may be different from joint when represented with sin/cos
		return self.robot.body_dim

	@property
	def n_controls(self) -> int:
		return self.robot.body_dim

	@property
	def o_dims(self) -> int:
		pass

	@property
	def state_aux_dims(self) -> int:
		pass

	@property
	def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Return a tuple (upper, lower) describing the expected range of states for this
		system
		"""
		# define upper and lower limits based around the nominal equilibrium input
		lower_limit = torch.tensor(self.robot.body_range[:, 0])
		upper_limit = torch.tensor(self.robot.body_range[:, 1])
		return (upper_limit, lower_limit)

	@property
	def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Return a tuple (upper, lower) describing the range of allowable control
		limits for this system
		"""
		upper_limit = torch.Tensor([self.env.p.getJointInfo(self.robot.robotId, j)[11] for j in self.robot.body_joints])
		lower_limit = -upper_limit
		return (upper_limit, lower_limit)

	def validate_params(self, params: Scenario) -> bool:
		return True

	def u_nominal(
			self, x: torch.Tensor, params: Optional[Scenario] = None,
	) -> torch.Tensor:
		"""
        Compute the nominal control for the nominal parameters, using LQR unless
        overridden

		args:
			x: bs x self.n_dims tensor of state
			params: the model parameters used
		returns:
			u_nominal: bs x self.n_controls tensor of controls
		"""
		x_touse = x[:, :self.n_dims]
		# Compute nominal control from feedback + equilibrium control
		K = self.K.type_as(x_touse)

		if isinstance(self.intermediate_goals, np.ndarray):
			goal = torch.from_numpy(self.intermediate_goals).to(x_touse.device).float()
		else:
			goal = self.intermediate_goals.clone().to(x_touse.device)
		assert goal.shape[-1] == x_touse.shape[-1]
		assert len(goal.shape) == len(x_touse.shape)
		u_nominal = -(K @ (x_touse - goal).T).T

		# Adjust for the equilibrium setpoint
		u = u_nominal + self.u_eq.type_as(x_touse)

		# Clamp given the control limits
		upper_u_lim, lower_u_lim = self.control_limits
		for dim_idx in range(self.n_controls):
			u[:, dim_idx] = torch.clamp(
				u[:, dim_idx],
				min=lower_u_lim[dim_idx].item(),
				max=upper_u_lim[dim_idx].item(),
			)

		return u

	@torch.enable_grad()
	def compute_A_matrix(self, scenario: Optional[Scenario]) -> np.ndarray:
		"""Compute the linearized continuous-time state-state derivative transfer matrix
		about the goal point"""
		return np.zeros((self.n_dims, self.n_dims))

	def check_collision_free_soft(self) -> bool:
		# body, no need to check gripper-gripper collision
		self.env.p.performCollisionDetection()
		if not self.robot.check_self_collision_free():
			return False

		# obstacle, including plane
		for all_link_idx in range(-1, self.robot.n_joints):
			for obstacle in self.env.obstacle_ids:
				checking_result = self.env.p.getClosestPoints(self.robot.robotId, obstacle, self.dis_threshold,
													 linkIndexA=all_link_idx)
				if bool(checking_result) and not (all_link_idx in [-1, 0] and obstacle == self.env.obstacle_ids[0]):
					return False
		return self.check_collision_free_hard()

	def check_collision_free_hard(self) -> bool:
		# used in unsafe mask, w/o distance threshold
		self.env.p.performCollisionDetection()
		return len(self.env.p.getContactPoints(self.robot.robotId)) == 0

	def safe_mask(self, x):
		"""
		Return the mask of x indicating safe regions for the obstacle task
		Only using q

		args:
			x: a tensor of points in the state space
		"""
		safe_mask = torch.ones_like(x[:, 0]).type_as(x).to(dtype=torch.bool)
		# Check if robot links are in collision with any obstacles
		for i in range(x.shape[0]):
			self.robot.set_joint_position(self.robot.body_joints, x[i, :self.q_dims])
			safe_mask[i].logical_and_(torch.tensor(self.check_collision_free_soft()))

		# Also constrain to be within the state limit
		# x_max, x_min = self.state_limits
		# safe_limit_coef = 0.95  # 0.8
		# up_mask = torch.all(x[:, :self.n_dims] <= safe_limit_coef * x_max.type_as(x), dim=1)
		# lo_mask = torch.all(x[:, :self.n_dims] >= safe_limit_coef * x_min.type_as(x), dim=1)
		# safe_mask.logical_and_(up_mask)
		# safe_mask.logical_and_(lo_mask)
		return safe_mask

	def unsafe_mask(self, x):
		"""
		Return the mask of x indicating unsafe regions for the obstacle task

		args:
			x: a tensor of points in the state space
		"""
		unsafe_mask = torch.zeros_like(x[:, 0]).type_as(x).to(dtype=torch.bool)

		# Check if robot links are in collision with any obstacles
		for i in range(x.shape[0]):
			self.robot.set_joint_position(self.robot.body_joints, x[i, :self.q_dims])
			unsafe_mask[i].logical_or_(torch.logical_not(torch.tensor(self.check_collision_free_hard())))

		# # Also constrain to be within the state limit
		# x_max, x_min = self.state_limits
		# limit_mask = (torch.all(x[:, :self.n_dims] >= x_max.type_as(x), dim=1)).logical_or_(
		#     torch.all(x[:, :self.n_dims] <= x_min.type_as(x), dim=1))
		# unsafe_mask.logical_or_(limit_mask)

		return unsafe_mask

	def goal_mask(self, x, velocity_limit: bool = False):
		"""Return the mask of x indicating points in the goal set (within 0.2 m of the
		goal).

		args:
			x: a tensor of points in the state space
		"""
		goal_mask = torch.ones_like(x[:, 0]).type_as(x).to(dtype=torch.bool)

		# Define the goal region as being near the goal
		# near_goal_xy = torch.ones_like(x[:, 0], dtype=torch.bool)
		# Forward kinematics
		# for i in range(x.shape[0]):
		#     self.plant.SetPositions(self.plant_context, x[i, :2])
		#     pos = self.plant.EvalBodyPoseInWorld(self.plant_context, self.ee_body).translation()[:2]  # Drake returns pos in 3D, project into 2D for our system
		#     near_goal_xy[i] = torch.tensor(np.abs(pos - self.ee_goal_pos) <= 0.1, dtype=torch.bool)

		near_goal_xy = (x[:, :self.q_dims] - torch.tensor(self.goal_state[:self.q_dims]).type_as(x)).norm(dim=1) <= 0.2
		goal_mask.logical_and_(near_goal_xy)
		if velocity_limit:
			near_goal_theta_velocity_1 = x[:, 2].abs() <= 0.1
			near_goal_theta_velocity_2 = x[:, 3].abs() <= 0.1
			goal_mask.logical_and_(near_goal_theta_velocity_1)
			goal_mask.logical_and_(near_goal_theta_velocity_2)

		# The goal set has to be a subset of the safe set
		goal_mask.logical_and_(self.safe_mask(x))

		return goal_mask

	def _get_observation_with_state(self, state):
		pass

	def complete_sample_with_observations(self, x, num_samples: int) -> torch.Tensor:
		"""
		input: bs * (n_dim + o_dim + aux_dim)
		"""
		if self.o_dims:
			samples = torch.zeros(num_samples, self.n_dims + self.o_dims + self.state_aux_dims).type_as(x)
			samples[:, :self.n_dims] = x
			for i in range(num_samples):
				o = self._get_observation_with_state(x[i, :self.q_dims])
				samples[i, self.n_dims:] = torch.tensor(o).type_as(samples)
			return samples
		else:
			return x

	def _get_sdf(self):
		dis = []
		for obstacle in self.env.obstacle_ids:
			sd_all = list(self.env.p.getClosestPoints(self.robot.robotId, obstacle, 5, linkIndexB=-1))
			for sd in sd_all:
				if sd[3] >= 0:
					dis.append(sd[8])

		dis = np.array(dis)
		shortest_dis = np.min(dis)
		return shortest_dis

	def sample_goal(self, num_samples: int, eps=0.02) -> torch.Tensor:
		"""

		Sample uniformly from the goal. May return some points that are not in the
		goal, so watch out (only a best-effort sampling).

		Only used when preparing data.
		"""
		x = torch.Tensor(num_samples, self.n_dims).uniform_(-1.0, 1.0)
		for i in range(num_samples):
			x[i, :self.n_dims] = x[i, :self.n_dims] * eps + self.goal_state

		return self.complete_sample_with_observations(x, num_samples)

	def sample_safe(self, num_samples: int, max_tries: int = 100) -> torch.Tensor:
		"""Sample uniformly from the goal. May return some points that are not in the
		goal, so watch out (only a best-effort sampling)."""
		x = ControlAffineSystem.sample_safe(self, num_samples, max_tries)

		return self.complete_sample_with_observations(x, num_samples)

	def sample_unsafe(self, num_samples: int, max_tries: int = 100) -> torch.Tensor:
		"""Sample uniformly from the goal. May return some points that are not in the
		goal, so watch out (only a best-effort sampling)."""
		x = ControlAffineSystem.sample_unsafe(self, num_samples, max_tries)

		return self.complete_sample_with_observations(x, num_samples)

	def sample_boundary(self, num_samples: int, max_tries: int = 100) -> torch.Tensor:
		"""Sample uniformly from the boundary"""
		x = ControlAffineSystem.sample_boundary(self, num_samples, max_tries)

		return self.complete_sample_with_observations(x, num_samples)

	def _f(self, x: torch.Tensor, params: Scenario):
		"""
		Return the control-independent part of the control-affine dynamics.

		args:
			x: bs x self.n_dims tensor of state
			params: a dictionary giving the parameter values for the system. If None,
					default to the nominal parameters used at initialization
		returns:
			f: bs x self.n_dims x 1 tensor
		"""
		# Extract batch size and set up a tensor for holding the result
		batch_size = x.shape[0]
		f = torch.zeros((batch_size, self.n_dims, 1))
		f = f.type_as(x)

		return f

	def _g(self, x: torch.Tensor, params: Scenario):
		"""
		Return the control-dependent part of the control-affine dynamics.

		args:
			x: bs x self.n_dims tensor of state
			params: a dictionary giving the parameter values for the system. If None,
					default to the nominal parameters used at initialization
		returns:
			g: bs x self.n_dims x self.n_controls tensor
		"""
		# Extract batch size and set up a tensor for holding the result
		if len(x.shape) > 1:
			batch_size = x.shape[0]
			g = torch.eye(n=self.n_dims, m=self.n_controls).unsqueeze(0).expand(batch_size, -1, -1)
		else:
			g = torch.eye(n=self.n_dims, m=self.n_controls)

		return g.to(x.device)

	@property
	def u_eq(self):
		u_eq = torch.zeros(
			(
				1,
				self.n_controls,
			)
		)

		return u_eq

	@property
	def goal_point(self):
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			return torch.tensor(self.goal_state)

	def noisy_simulator(self, x_init: torch.Tensor, num_steps: int, collect_dataset: bool = False,
						noise_level=1e-2, use_motor_control: bool = False) -> torch.Tensor:
		"""
		Simulate the system forward using the noisy nominal controller

		args:
			x_init - bs x n_dims tensor of initial conditions
			num_steps - a positive integer
		returns
			a bs x num_steps x self.n_dims tensor of simulated trajectories
		"""
		# Call the simulate method using the nominal controller
		assert collect_dataset
		return self.simulate(
			x_init, num_steps, controller=self.u_nominal, guard=self.out_of_bounds_mask, noise_level=noise_level,
			use_motor_control=use_motor_control, collect_dataset=collect_dataset,
		)

	def simulate(
			self,
			x_init: torch.Tensor,
			num_steps: int,
			collect_dataset: bool,
			controller: Callable[[torch.Tensor], torch.Tensor],
			controller_period: Optional[float] = None,
			guard: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
			noise_level: float = 0,
			use_motor_control: bool = False,
			params: Optional[Scenario] = None,
	) -> torch.Tensor:
		"""
		Simulate the system for the specified number of steps using the given controller
		(the batch come from different trajectories within same episode)

		args:
			x_init - bs x n_dims tensor of initial conditions
			num_steps - a positive integer
			controller - a mapping from state to control action
			controller_period - the period determining how often the controller is run
								(in seconds). If none, defaults to self.dt
			guard - a function that takes a bs x n_dims tensor and returns a length bs
					mask that's True for any trajectories that should be reset to x_init
			params - a dictionary giving the parameter values for the system. If None,
					 default to the nominal parameters used at initialization
		returns
			a bs x num_steps x self.n_dims tensor of simulated trajectories. If an error
			occurs on any trajectory, the simulation of all trajectories will stop and
			the second dimension will be less than num_steps
		"""

		assert controller.__name__ == "u_nominal"
		assert collect_dataset == True

		# Create a tensor to hold the simulation results
		init_states = self.complete_sample_with_observations(x_init, x_init.shape[0])
		x_sim = torch.zeros(x_init.shape[0], num_steps, init_states.shape[-1]).type_as(x_init)
		u = torch.zeros(x_init.shape[0], self.n_controls).type_as(x_init)

		# Compute controller update frequency
		if controller_period is None:
			controller_period = self.dt
		controller_update_freq = int(controller_period / self.dt)

		# Run the simulation until it's over or an error occurs
		x_sim[:, 0, :] = init_states

		for tstep in range(1, num_steps):

			# Get the current state
			x_current = x_sim[:, tstep - 1, :]
			if tstep == 1 or tstep % controller_update_freq == 0:
				u = controller(x_current)
				if isinstance(u, tuple):
					u = u[0]
				upper_u_lim, lower_u_lim = self.control_limits
				if noise_level > 0:
					u += torch.mul(
						torch.Tensor(x_init.shape[0], self.n_controls).uniform_(-noise_level, noise_level).type_as(
							x_init), self.control_limits[0])
					for dim_idx in range(self.n_controls):
						u[:, dim_idx] = torch.clamp(
							u[:, dim_idx],
							min=lower_u_lim[dim_idx].item(),
							max=upper_u_lim[dim_idx].item(),
						)

			# Simulate forward using the dynamics
			x_sim[:, tstep, :] = self.closed_loop_dynamics(x_current, u, use_motor_control=use_motor_control, collect_dataset=collect_dataset)

		return x_sim


	def closed_loop_dynamics(
			self, x: torch.Tensor, u: torch.Tensor, collect_dataset: bool = False,
			use_motor_control: bool = False, params: Optional[Scenario] = None, update_observation: bool = True,
			return_time: bool = False,
	) -> torch.Tensor:
		"""
		Damn! closed_loop_dynamics is different from any other system!!
		due to existence of observation

		This one returns x_next,
		others: xdot
		"""
		if return_time:
			t0=time.time()
		batch_size = x.shape[0]

		if self.o_dims:
			x_next = torch.zeros((batch_size, self.n_dims + self.o_dims + self.state_aux_dims)).type_as(x)
		else:
			x_next = torch.zeros((batch_size, self.n_dims)).type_as(x)

		step = 10 if collect_dataset else 1

		for i in range(batch_size):
			q_dot = u[i]
			xdot = q_dot.type_as(x)
			if use_motor_control:
				self.robot.set_joint_position(self.robot.body_joints, x[i, :self.n_dims])
				self.env.p.setJointMotorControlArray(self.robot.robotId, self.robot.body_joints, self.env.p.VELOCITY_CONTROL,
											targetVelocities=xdot)
				for _ in range(step):
					self.env.p.stepSimulation()
				x_next[i, :self.n_dims] = torch.tensor(self.robot.get_joint_position(self.robot.body_joints),
													   device=x.device)
			else:
				x_next[i, :self.n_dims] = xdot * self.dt * step + x[i, :self.n_dims]
				self.robot.set_joint_position(self.robot.body_joints, x_next[i, :self.n_dims])
		if return_time:
			t1 = time.time()

		# observation
		if update_observation:
			x_next = self.complete_sample_with_observations(x_next[:, :self.n_dims], batch_size)
		if return_time:
			t2 = time.time()

		if return_time:
			return x_next, [t1-t0, t2-t1]
		else:
			return x_next


if __name__ == '__main__':
	problem_num = 1000
	obstacle_num = 8

	robot_name = 'iiwa'
	nominal_params = {"m1": 5.76}
	controller_period = 1/30
	simulation_dt = 1/120
	environment = ArmEnv([robot_name], GUI=True, config_file='')
	robot = environment.robot_list[0]
	dynamics_model = ArmDynamics(
		nominal_params,
		dt=simulation_dt,
		dis_threshold=0.02,
		controller_dt=controller_period,
		env=environment,
		robot=robot
	)

	while True:
		environment.p.stepSimulation()

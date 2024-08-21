import os
import numpy as np
import pybullet_utils.bullet_client as bc
import pybullet
import warnings
import pybullet_data

from environment.franka_panda import FrankaPanda
from environment.magician import Magician


def sample_surface(center, half_extent, idx, num, add_normal=False):
	a = np.zeros((num, 3))
	b = np.random.uniform(low=(-1, -1, -1), high=(1, 1, 1), size=(num, 3))
	for ii in range(3):
		a[:, ii] = center[ii] + half_extent[ii] * b[:, ii]
	a[:, idx // 2] = center[idx // 2] + half_extent[idx // 2] * np.sign(idx % 2 - 0.5)
	if add_normal:
		a = np.concatenate((a, np.sign(idx % 2 - 0.5) * np.eye(3)[idx // 2].reshape(1, 3).repeat(num, axis=0)), axis=1)
	return a


class ArmEnv:
	'''
	Separate arm interface with maze environment ones
	'''

	def __init__(self, robot_name_list, config_file: str, GUI=False, include_floor=True):
		print("Initializing environment...")

		self.robot_name_list = robot_name_list
		self.robot_list = []
		self.obstacle_ids = []

		self.include_floor = include_floor
		print(f"included floor: {self.include_floor}")

		if GUI:
			self.p = bc.BulletClient(connection_mode=pybullet.GUI,
									 options='--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0')
		else:
			self.p = bc.BulletClient(connection_mode=pybullet.DIRECT)

		self.use_training_env = (len(config_file) == 0)
		if self.use_training_env:
			# training environment file
			self.env_config_file = f'{str.rsplit(os.path.abspath(__file__), "/", 2)[0]}/models/env_file/env_600_4.npz'
			self.obstacle_num = 4
		else:
			self.env_config_file = config_file

		if not os.path.exists(self.env_config_file) and self.use_training_env:
			self._generate_env_config(self.env_config_file, 4)

		self.env_config = np.load(self.env_config_file, allow_pickle=True)

		self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
		self.reset_env(self.get_env_config(-1), enable_object=False)

	def __str__(self):
		return self.robot_name_list

	def reset_env(self, obs_configs, enable_object=False):
		self.p.resetSimulation()
		self.robot_list = []
		assert len(self.robot_name_list) <= 1, "Only support one robot now."

		if self.include_floor:
			plane = self.p.createCollisionShape(self.p.GEOM_PLANE)
			self.plane = self.p.createMultiBody(0, plane)

		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			for robot_name in self.robot_name_list:
				if robot_name == "panda":
					self.robot_list.append(FrankaPanda(self.p))
				elif robot_name == "magician":
					self.robot_list.append(Magician(self.p))
				else:
					raise NotImplementedError(f"Robot {robot_name} not supported yet.")

				if self.include_floor:
					self.p.setCollisionFilterPair(self.robot_list[-1].robotId, self.plane, -1, -1, 0)
					self.p.setCollisionFilterPair(self.robot_list[-1].robotId, self.plane, 1, -1, 0)

		if enable_object:
			if len(self.robot_list) > 1:
				raise NotImplementedError
			else:
				object = self.p.createCollisionShape(self.p.GEOM_CYLINDER, radius=0.01,
													 height=np.random.uniform(0.15, 0.25))
				objectPos, objectOrn = self.robot_list[0].get_link_PosOrn(self.robot_list[0].body_joints[-1])
				objectOrn = \
					self.p.multiplyTransforms([0, 0, 0], [1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0], [0, 0, 0], objectOrn)[
						1]
				self.object = self.p.createMultiBody(0.01, object, basePosition=objectPos, baseOrientation=objectOrn)
				self.robot_list[0].end_effector.activate(self.object)

		self._generate_obstacle(obs_configs=obs_configs)

	# p.setGravity(0, 0, -10)

	def get_env_config(self, idx):
		# return obstacle positions and sizes
		if idx >= 0:
			obs_positions = self.env_config['obstacle_positions'][idx]
			if 'obstacle_sizes' in self.env_config.keys():
				obs_sizes = self.env_config['obstacle_sizes'][idx]
			else:
				obs_sizes = np.array([[0.05, 0.05, 0.1] for _ in range(obs_positions.shape[0])])
			return obs_positions, obs_sizes
		else:
			return np.array([[0.3, 0.17, 0.3], [-0.4, 0.23, 0.4], [0.0, -0.23, 0.6]]), \
				np.array([[0.05, 0.05, 0.1], [0.05, 0.05, 0.1], [0.05, 0.05, 0.1]])

	def sample_obstacle_surface(self, total_num, add_normal=False):
		# note: no self-collision information
		for_floor = 0
		each_obstacle = np.random.randint(low=0, high=len(self.obstacle_ids) - int(self.include_floor),
										  size=total_num - for_floor)
		# points on floor
		points_global = np.random.uniform(low=(-1, -1), high=(1, 1), size=(for_floor, 2))
		points_global = np.concatenate((points_global, np.zeros((for_floor, 1 + 3 * int(add_normal)))), axis=1)

		# points on obstacles
		for obs_idx in range(len(self.obstacle_ids) - int(self.include_floor)):
			obs_position = self.obs_positions[obs_idx]
			obs_size = self.obs_sizes[obs_idx]
			rand_idx = np.random.randint(0, 6, np.where(each_obstacle == obs_idx)[0].shape[0])
			for jj in range(6):
				new_array = sample_surface(obs_position, obs_size, jj, np.where(rand_idx == jj)[0].shape[0],
										   add_normal=add_normal)
				points_global = np.concatenate((points_global, new_array), axis=0)
		return points_global

	# ===================== internal module ===========================

	def _generate_env_config(self, file_name, obstacle_num, problem_num=1000):
		# generate or load test cases
		if len(self.robot_name_list) > 1:
			raise NotImplementedError('Environment config with multiple arms not implemented.')
		else:
			positions = np.zeros([0, obstacle_num, 3])

			while positions.shape[0] < problem_num:
				try:
					position = np.random.uniform(low=(-0.5, -0.5, 0), high=(0.5, 0.5, 1), size=(obstacle_num, 3))
					positions = np.concatenate([positions, np.expand_dims(position, axis=0)], axis=0)
				except AssertionError:
					continue
			assert file_name.endswith('.npz')
			np.savez(file_name, obstacle_positions=positions)

	def _generate_obstacle(self, obs_configs):
		self.obs_positions, self.obs_sizes = obs_configs
		if self.include_floor:
			self.obstacle_ids = [self.plane]
		else:
			self.obstacle_ids = []
		for obs_position, obs_size in zip(self.obs_positions, self.obs_sizes):
			halfExtents = obs_size
			basePosition = obs_position
			baseOrientation = [0, 0, 0, 1]
			self.obstacle_ids.append(self._create_voxel(halfExtents, basePosition, baseOrientation, color='random'))

	def _create_voxel(self, halfExtents, basePosition, baseOrientation, color='random'):
		voxColId = self.p.createCollisionShape(self.p.GEOM_BOX, halfExtents=halfExtents)
		if color == 'random':
			voxVisID = self.p.createVisualShape(shapeType=self.p.GEOM_BOX,
												rgbaColor=[58 / 256, 107 / 256, 53 / 256, 1],
												# np.random.uniform(0, 1, size=3).tolist() + [0.8],
												halfExtents=halfExtents)
		else:
			voxVisID = self.p.createVisualShape(shapeType=self.p.GEOM_BOX,
												rgbaColor=color,
												halfExtents=halfExtents)
		voxId = self.p.createMultiBody(baseMass=0,
									   baseCollisionShapeIndex=voxColId,
									   baseVisualShapeIndex=voxVisID,
									   basePosition=basePosition,
									   baseOrientation=baseOrientation)
		return voxId


if __name__ == '__main__':
	nominal_params = {"m1": 5.76}
	controller_period = 1 / 30
	simulation_dt = 1 / 120
	environment = ArmEnv(['panda'], GUI=True, config_file='')

	while True:
		environment.p.stepSimulation()
		print(environment.robot_list[0].check_self_collision_free())

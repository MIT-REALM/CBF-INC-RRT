import time
import os
import sys

import numpy as np
import torch

import pinocchio as pin


class BasicRobot:
	def __init__(self, bullet_client_id, urdf_file, global_scaling=1.):
		# pybullet model
		self.p = bullet_client_id
		self.robotId = self.p.loadURDF(urdf_file, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True,
									   # flags=p.URDF_IGNORE_COLLISION_SHAPES, #p.URDF_USE_SELF_COLLISION,
									   globalScaling=global_scaling)

		self.pin_model = pin.buildModelFromUrdf(urdf_file)
		self.pin_data = self.pin_model.createData()

		self._link_name_to_index()
		self.q0 = None

		# bound
		self.n_joints = self.p.getNumJoints(self.robotId)
		joints = [self.p.getJointInfo(self.robotId, i) for i in range(self.n_joints)]

		self.body_joints = [j[0] for j in joints if j[2] == self.p.JOINT_REVOLUTE]
		self.body_range = np.array(
			[(self.p.getJointInfo(self.robotId, jointId)[8], self.p.getJointInfo(self.robotId, jointId)[9]) for jointId
			 in
			 self.body_joints])
		self.body_dim = len(self.body_joints)

		self.ee_joints = [j[0] for j in joints if j[2] == self.p.JOINT_PRISMATIC]
		self.ee_range = np.array(
			[(self.p.getJointInfo(self.robotId, jointId)[8], self.p.getJointInfo(self.robotId, jointId)[9]) for jointId
			 in
			 self.ee_joints])
		self.ee_dim = len(self.ee_joints)

		# set gripper
		self._open_gripper()
		self.activated_ee = False
		self.contact_constraint = None
		self.contact_threshold = 0.50

	def reset(self):
		pass

	def get_link_PosOrn(self, link):
		return self.p.getLinkState(self.robotId, link)[:2]

	def forward_kinematics(self, return_links: list, q):
		if isinstance(q, torch.Tensor):
			q = q.detach().cpu().numpy()
		self.set_joint_position(self.body_joints, q.squeeze())
		return [(np.array(self.p.getLinkState(self.robotId, link)[4]),
				 np.array(self.p.getMatrixFromQuaternion(self.p.getLinkState(self.robotId, link)[5])).reshape(3, 3)) for
				link
				in return_links]

	def set_joint_position(self, joints: list, q):
		assert len(joints) == q.shape[0]
		for idx in range(len(joints)):
			self.p.resetJointState(self.robotId, joints[idx], targetValue=q[idx])
		self.p.performCollisionDetection()

	def get_joint_position(self, joints):
		return [self.p.getJointState(self.robotId, joint)[0] for joint in joints]

	def get_jacobian(self, joint_value: list, linkIdx: int, localPos):
		"""
		output: 3 * control_dims, 3 * control_dims
		"""
		assert len(joint_value) == self.body_dim
		return np.array(self.p.calculateJacobian(self.robotId, linkIdx, localPos,
												 objPositions=joint_value + [0 for _ in range(self.ee_dim)],
												 objVelocities=[0 for _ in range(self.body_dim + self.ee_dim)],
												 objAccelerations=[0 for _ in range(self.body_dim + self.ee_dim)])[0])[
			   :, self.body_joints], \
			np.array(self.p.calculateJacobian(self.robotId, linkIdx, localPos,
											  objPositions=joint_value + [0 for _ in range(self.ee_dim)],
											  objVelocities=[0 for _ in range(self.body_dim + self.ee_dim)],
											  objAccelerations=[0 for _ in range(self.body_dim + self.ee_dim)])[1])[:,
			self.body_joints]

	def check_self_collision_free(self):
		for all_link_idx in range(self.n_joints):
			for body_link_idx in range(all_link_idx + 2, self.body_dim):
				if self.body_dim == 7 and (all_link_idx == 6 and body_link_idx == 8):
					continue
				if bool(self.p.getClosestPoints(self.robotId, self.robotId, 0.01,
												linkIndexA=all_link_idx,
												linkIndexB=self.body_joints[body_link_idx])):
					return False
		return True

	def _link_name_to_index(self):
		self.arm_link_name = {self.p.getBodyInfo(self.robotId)[0].decode('UTF-8'): -1, }
		for _id in range(self.p.getNumJoints(self.robotId)):
			_name = self.p.getJointInfo(self.robotId, _id)[12].decode('UTF-8')
			self.arm_link_name[_name] = _id

	def _open_gripper(self):
		for idx in range(self.ee_dim):
			self.p.setJointMotorControl2(self.robotId, self.ee_joints[idx], controlMode=self.p.POSITION_CONTROL,
										 targetPosition=self.ee_range[idx][1])
		self.p.stepSimulation()

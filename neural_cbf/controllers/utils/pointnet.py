# modified from https://github.com/fxia22/pointnet.pytorch/ and https://github.com/rstrudel/nmprepr/
from __future__ import print_function

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

class PointNetfeat(nn.Module):
	"""
	Note that the feature extracted from pointcloud must be transformation sensitive.
	Input: bs * (num_sensor * ray_per_sensor) * 3
	Output: bs * (num_sensor * 16)
	"""

	def __init__(self, num_sensor, ray_per_sensor, input_channel, output_channel, use_bn=False):
		super(PointNetfeat, self).__init__()
		self.num_sensor = num_sensor
		self.ray_per_sensor = ray_per_sensor
		self.input_channel = input_channel + self.num_sensor
		self.output_channel = output_channel

		self.backbone_nn = nn.Sequential(OrderedDict([
			('fc11', nn.Linear(self.input_channel, 64)),
			# ('conv1', nn.Conv1d(self.input_channel, 64, 1)),
			('bn1', nn.BatchNorm1d(64) if use_bn else nn.Identity()),
			('relu1', nn.LeakyReLU(0.1)),
			('fc2', nn.Linear(64, 128)),
			# ('conv2', nn.Conv1d(64, 128, 1)),
			('bn2', nn.BatchNorm1d(128) if use_bn else nn.Identity()),
			('relu2', nn.LeakyReLU(0.1)),
			('fc3', nn.Linear(128, 512)),
			# ('conv3', nn.Conv1d(128, 512, 1)),
			('bn3', nn.BatchNorm1d(512) if use_bn else nn.Identity()),
			('relu3', nn.LeakyReLU(0.1)),
		]))
		self.backbone_fc = nn.Sequential(OrderedDict([
			# ('fc1', nn.Linear(512, 512)),
			# ('bn1', nn.BatchNorm1d(512) if use_bn else nn.Identity()),
			# ('relu1', nn.LeakyReLU(0.1)),
			('fc2', nn.Linear(512, 256)),
			('bn2', nn.BatchNorm1d(256) if use_bn else nn.Identity()),
			('relu2', nn.LeakyReLU(0.1)),
			('fc3', nn.Linear(256, self.output_channel)),
		]))

	def forward(self, x):
		bs = x.shape[0]
		x = x.view(bs, self.num_sensor, self.ray_per_sensor, -1)
		assert x.shape[-1] + self.num_sensor == self.input_channel
		x = torch.cat([x, F.one_hot(torch.arange(self.num_sensor), self.num_sensor).type_as(x).unsqueeze(0).unsqueeze(2).expand(bs, -1, x.shape[2], -1)], dim=-1)

		# print(self.backbone_fc[1].running_mean[:3])

		# # use conv1d
		# x = x.view(-1, self.ray_per_sensor, self.input_channel).transpose(1, 2)
		# x = self.backbone_nn(x)
		# x = torch.max(x, 2, keepdim=True)[0]

		# use fc
		x = x.view(-1, self.ray_per_sensor, self.input_channel)
		x = self.backbone_nn(x)
		x = torch.max(x, 1, keepdim=True)[0]

		x = x.view(-1, self.num_sensor, 512).view(-1, 512)
		x = self.backbone_fc(x)
		x = x.view(-1, self.num_sensor * self.output_channel)
		return x


class PointNetVanillaEncoder(nn.Module):
	"""
		Input: bs * (num_sensor * ray_per_sensor) * 3
		Output: bs * output_dim
	"""

	def __init__(self, num_sensor, ray_per_sensor, input_channel, output_dim=16):
		super(PointNetVanillaEncoder, self).__init__()
		self.num_sensor = num_sensor
		self.ray_per_sensor = ray_per_sensor

		self.encoder_nn = nn.Sequential(OrderedDict([
			('fc1', nn.Linear(input_channel * self.num_sensor, 512)),
			('relu1', nn.LeakyReLU(0.1)),
			('fc2', nn.Linear(512, 256)),
			('relu2', nn.LeakyReLU(0.1)),
			# ('fc3', nn.Linear(256, 256)),
			# ('relu3', nn.LeakyReLU(0.1)),
			('fc4', nn.Linear(256, output_dim)),
		]))

	def forward(self, x):
		return self.encoder_nn(x)

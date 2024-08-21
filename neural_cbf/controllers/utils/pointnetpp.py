# modified from https://github.com/fxia22/pointnet.pytorch/ and https://github.com/rstrudel/nmprepr/
# and https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/pointnet2_ops_lib/pointnet2_ops/pointnet2_modules.py
from __future__ import print_function
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F

channel_1 = 64


def build_shared_mlp(mlp_spec: List[int], bn: bool = True):
    layers = []
    for i in range(1, len(mlp_spec)):
        layers.append(
            nn.Conv2d(mlp_spec[i - 1], mlp_spec[i], kernel_size=1, bias=not bn)
        )
        if bn:
            layers.append(nn.BatchNorm2d(mlp_spec[i]))
        layers.append(nn.ReLU(True))

    return nn.Sequential(*layers)


class _PointnetSAModuleBase(nn.Module):
    def __init__(self):
        super(_PointnetSAModuleBase, self).__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None

    def forward(
        self, xyz: torch.Tensor, features: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features
        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        """

        new_features_list = []

        # xyz_flipped = xyz.transpose(1, 2).contiguous()
        # new_xyz = (
        #     pointnet2_utils.gather_operation(
        #         xyz_flipped, pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        #     )
        #     .transpose(1, 2)
        #     .contiguous()
        #     if self.npoint is not None
        #     else None
        # )
        new_xyz = xyz

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample)

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    r"""Pointnet set abstrction layer with multiscale grouping
    Parameters
    ----------
    npoint : int
        Number of features
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """

    def __init__(self, npoint, radii, nsamples, mlps, bn=True, use_xyz=True):
        # type: (PointnetSAModuleMSG, int, List[float], List[int], List[List[int]], bool, bool) -> None
        super(PointnetSAModuleMSG, self).__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None
                else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            self.mlps.append(build_shared_mlp(mlp_spec, bn))


class PointnetSAModule(PointnetSAModuleMSG):
    r"""Pointnet set abstrction layer
    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet before the global max_pool
    bn : bool
        Use batchnorm
    """

    def __init__(
        self, mlp, npoint=None, radius=None, nsample=None, bn=True, use_xyz=True
    ):
        # type: (PointnetSAModule, List[int], int, float, int, bool, bool) -> None
        super(PointnetSAModule, self).__init__(
            mlps=[mlp],
            npoint=npoint,
            radii=[radius],
            nsamples=[nsample],
            bn=bn,
            use_xyz=use_xyz,
        )


class PointNetfeat(nn.Module):
    """
    Note that the feature extracted from pointcloud must be transformation sensitive.
    Input: bs * (num_sensor * ray_per_sensor) * 3
    Output: bs * (num_sensor * 16)
    """

    def __init__(self, num_sensor, ray_per_sensor, input_channel, output_channel):
        super(PointNetfeat, self).__init__()
        self.num_sensor = num_sensor
        self.ray_per_sensor = ray_per_sensor
        self.input_channel = input_channel
        self.output_channel = output_channel

        self.conv1 = torch.nn.Conv1d(self.input_channel, channel_1, 1)
        self.conv2 = torch.nn.Conv1d(channel_1, channel_1, 1)
        self.conv3 = torch.nn.Conv1d(channel_1, self.output_channel, 1)
        self.bn1 = nn.BatchNorm1d(channel_1)
        self.bn2 = nn.BatchNorm1d(channel_1)
        self.bn3 = nn.BatchNorm1d(self.output_channel)

    def forward(self, x):
        assert x.shape[1]==self.num_sensor*self.ray_per_sensor*self.input_channel
        x = x.reshape(-1, self.num_sensor, self.ray_per_sensor, self.input_channel).reshape(-1, self.input_channel, self.ray_per_sensor)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = self.conv3(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.reshape(-1, self.num_sensor, self.output_channel).reshape(-1, self.num_sensor*self.output_channel)
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

        self.fc1 = nn.Linear(self.num_sensor * input_channel, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    # def __init__(self, num_sensor, ray_per_sensor, output_dim=16):
    #     super(PointNetVanillaEncoder, self).__init__()
    #     self.num_sensor = num_sensor
    #     self.ray_per_sensor = ray_per_sensor
    #
    #     self.feat = PointNetfeat(self.num_sensor, self.ray_per_sensor)
    #     self.fc1 = nn.Linear(self.num_sensor*16, 32)
    #     self.fc2 = nn.Linear(32, output_dim)
    #     self.bn1 = nn.BatchNorm1d(32)
    #
    # def forward(self, x):
    #     x = self.feat(x)
    #     x = F.relu(self.bn1(self.fc1(x)))
    #     x = self.fc2(x)
    #     return x

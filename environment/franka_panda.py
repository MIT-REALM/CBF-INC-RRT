import time
import os
import sys

import numpy as np
import torch
try:
    import pinocchio as pin
except:
    pass

from .basic_robot import BasicRobot

FILE_DIR = os.path.abspath(__file__).rsplit('/', 2)[0]
PANDA_BASE_URDF = "franka_panda/panda.urdf"

class FrankaPanda(BasicRobot):
    def __init__(self, bullet_client):
        super(FrankaPanda, self).__init__(bullet_client, urdf_file=FILE_DIR + "/utils/robot/" + PANDA_BASE_URDF)

        self.p.setCollisionFilterPair(self.robotId, self.robotId, 6, 8, 0)

        self.q0 = np.array(
            [1.00887519, 0.50546576, -1.69052917, -2.2909179, 2.95208592, 3.59793418, 2.93001438])  # body-only
        self.invalid_link = [7, 11]

        self.set_joint_position(self.body_joints, self.q0)
        # print("Successfully Initialized Franka Panda...")

    def __str__(self):
        return 'Franka_Panda'

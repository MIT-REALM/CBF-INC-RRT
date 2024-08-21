import time
import os
import sys
import warnings

import numpy as np
import torch
try:
    import pinocchio as pin
except:
    pass

from .basic_robot import BasicRobot

FILE_DIR = os.path.abspath(__file__).rsplit('/', 2)[0]
BASE_URDF = "magician/urdf/demo.urdf"

class Magician(BasicRobot):
    def __init__(self, bullet_client):
        super(Magician, self).__init__(bullet_client, urdf_file=FILE_DIR + "/utils/robot/" + BASE_URDF, global_scaling=2.)

        self.q0 = np.array([0, 0.707, 0.707, 0])  # body-only

    def __str__(self):
        return 'Magician'

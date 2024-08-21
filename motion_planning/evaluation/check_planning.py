import os
import numpy as np
import pickle

from environment import ArmEnv
from neural_cbf.systems import ArmDynamics

from motion_planning.utils.vis_path import execute_path

robot_name = 'magician'
level = 'hard'
seed = 123
gb = 85
rrt_step = 120
method = 'rrt_lidar'

# load BIT solution and method solution
BIT_file = f"../../models/motion_planning/problems/refined_pkl/{robot_name}_{level}_1000.pkl"
method_file_path = "../../models/motion_planning/" + method + '/'
all_method_file = os.listdir(method_file_path)
result_file_list = [file for file in all_method_file if file.startswith(f"{robot_name}_{level}_{seed}_{gb}_{rrt_step}")]

BIT_result = np.load(BIT_file, allow_pickle=True)
method_result = [{'result': np.load(method_file_path + file, allow_pickle=True),
                  'start_idx': int(file.rsplit('_')[-2]),
                  'end_idx': int(file.rsplit('_')[-1].rsplit('.')[0])} for file in result_file_list]

# collect all unsolved problem idx
unsolved_idx = []
for m_result in method_result:
    for i in range(m_result['end_idx'] - m_result['start_idx']):
        if not m_result['result'][i]['success']:
            unsolved_idx.append(i + m_result['start_idx'])
print(unsolved_idx)
exit()

# select certain test case
method_solution = None
problem_idx = 761
for m_result in method_result:
    if problem_idx >= m_result['start_idx'] and problem_idx < m_result['end_idx']:
        method_solution = m_result['result'][problem_idx - m_result['start_idx']]

#
nominal_params = {"m1": 5.76}
environment = ArmEnv([robot_name], GUI=True, config_file='')
robot = environment.robot_list[0]
dynamics_model = ArmDynamics(
    nominal_params,
    dt=1/30,
    dis_threshold=0.02,
    controller_dt=1/120,
    env=environment,
    robot=robot
)

environment.reset_env(enable_object=False, obs_configs=(BIT_result['obstacle_positions'][problem_idx], BIT_result['obstacle_sizes'][problem_idx]))
dynamics_model.set_goal(BIT_result['goal_config'][problem_idx])
execute_path(dynamics_model, BIT_result['path'][problem_idx], BIT_result['init_config'][problem_idx], BIT_result['goal_config'][problem_idx])


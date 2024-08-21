import os
import numpy as np
import pickle

robot = 'panda'
level = 'easy'
seeds = [123, 231, 312]
method = 'cbf'

dir_path = "../../models/motion_planning/table3_supp/"
all_file = os.listdir(dir_path)
success_rate_list = []
safe_list = []
time_list = []
for seed in seeds:
    result_file_list = [file for file in all_file if file.startswith(f"{robot}_{method}_{seed}_{level}")]

    result = {}
    assert len(result_file_list) == 1
    file = result_file_list[0]
    data = np.load(dir_path + file, allow_pickle=True)
    step = 600
    result['success'] = len(np.where(np.array(data['t_step'])<step)[0]) / 1000
    result['safe_traj'] = 1-np.sum([bool(x[1]) for x in data['safe_state']]) / 1000
    result['time_to_go'] = np.mean(np.array([i for i in data['t_step'] if i < step]))
    print("="*30)
    print(f"seed: {seed}, robot: {robot}, level: {level}")
    print(f"{method} success rate: {result['success']}")
    print(f"{method} safe traj rate: {result['safe_traj']}")
    print(f"{method} timetogo: {result['time_to_go']}")
    success_rate_list.append(result['success'])
    safe_list.append(result['safe_traj'])
    time_list.append(result['time_to_go'])
print("="*30)
print(f"success rate: {np.mean(np.array(success_rate_list))}, std: {np.std(np.array(success_rate_list))}")
print(f"safe: {np.mean(np.array(safe_list))}, std: {np.std(np.array(safe_list))}")
print(f"time: {np.mean(np.array(time_list))}, std: {np.std(np.array(time_list))}")

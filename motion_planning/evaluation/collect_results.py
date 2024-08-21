import os
import numpy as np
import pickle

robot = 'panda'
level = 'hard'
seeds = [123, 231, 312]
gb = 20  #20
rrt_step = 180 #180
method = 'rrt_docbf'

dir_path = "../../models/motion_planning/" + method + '/'
all_file = os.listdir(dir_path)
success_rate_list = []
node_list = []
time_list = []
for seed in seeds:
    result_file_list = [file for file in all_file if file.startswith(f"{robot}_{level}_{seed}_{gb}_{rrt_step}")]

    result = {'time': [], 'success': [], 'node': []}
    for file in result_file_list:
        data_i = np.load(dir_path + file, allow_pickle=True)
        for i in range(len(data_i)):
            result['time'].append(data_i[i]['time_dict']['total_time'])
            result['success'].append(data_i[i]['success'] and data_i['time_dict']['total_time'] < 0.4)
            result['node'].append(min(data_i[i]['explored_nodes'], 200))
    print("="*30)
    print(f"seed: {seed}")
    print(f"{method} success rate: {np.sum(np.array(result['success'])) / len(result['success'])}")
    print(f"{method} node: {np.mean(np.array(result['node']))}")
    print(f"{method} time: {np.sum(np.array(result['time']))}")
    success_rate_list.append(np.sum(np.array(result['success'])) / len(result['success']))
    # node_list += result['node']
    node_list.append(np.mean(np.array(result['node'])))
    time_list.append(np.sum(np.array(result['time'])))
print("="*30)
print(f"success rate: {np.mean(np.array(success_rate_list))}, std: {np.std(np.array(success_rate_list))}")
print(f"node: {np.mean(np.array(node_list))}, std: {np.std(np.array(node_list))}")
print(f"time: {np.mean(np.array(time_list))}, std: {np.std(np.array(time_list))}")

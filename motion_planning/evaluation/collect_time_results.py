import os
import numpy as np
import pickle

robot = 'panda'
level = 'easy'
seed = 23
gb = 20
rrt_step = 180
method = 'rrt_docbf'

dir_path = "../../models/motion_planning/" + method + '/'
all_file = os.listdir(dir_path)
success_rate_list = []
node_list = []
time_list = []
# result_file_list = [file for file in all_file if file.startswith(f"{robot}_{level}_{seed}_{gb}_{rrt_step}_0_100.pkl")]

result = {'total_time': [], 'nn_forward': [], 'QP': [], 'cl': [], 'observation': [], 'collision_checking': [],
          'success': [], 'node': [], 'misc': [], 'pure_steer': [], 'insert': [], 'update_dict': []}
file = dir_path + f"time_{robot}_{level}_{seed}_{gb}_{rrt_step}_0_100.pkl"
# file = dir_path + f"time_6obs_{robot}_{level}_{seed}_{gb}_{rrt_step}_0_100.pkl"
data_i = np.load(file, allow_pickle=True)
for i in range(len(data_i)):
    # print(data_i[i]['time_dict'])
    # exit()
    # print(i)
    result['total_time'].append(
        data_i[i]['time_dict']['total_time'] - data_i[i]['time_dict']['steertime_division']['misc'])
    if method in ['rrt_mindis', 'rrt_lidar', 'rrt_docbf']:
        result['nn_forward'].append(data_i[i]['time_dict']['steertime_division']['nn_forward'])
        result['QP'].append(data_i[i]['time_dict']['steertime_division']['QP'])
    else:
        result['nn_forward'].append(data_i[i]['time_dict']['steertime_division']['u'])
    if not method == 'rrt_vanilla':
        result['observation'].append(data_i[i]['time_dict']['steertime_division']['complete_sample'])
    result['pure_steer'].append(data_i[i]['time_dict']['steertime_division']['total_steer'])
    result['insert'].append(data_i[i]['time_dict']['steertime_division']['insert'])
    result['update_dict'].append(data_i[i]['time_dict']['steertime_division']['update_dict'])
    result['collision_checking'].append(data_i[i]['time_dict']['steertime_division']['collision_checking'])
    result['cl'].append(data_i[i]['time_dict']['steertime_division']['cl_dynamics'])
    result['misc'].append(data_i[i]['time_dict']['steertime_division']['misc'])
    result['success'].append(data_i[i]['success'])
    if data_i[i]['success']:
        result['node'].append(data_i[i]['explored_nodes'])
print("="*30)
print(f"seed: {seed}")
for key in result.keys():
    print(f"{method} {key}: {np.sum(np.array(result[key]))}")
print("-"*30)
print(f"success rate: {np.sum(np.array(result['success']))/len(result['success'])}")
other_time = np.sum(np.array(result['nn_forward'])) + np.sum(np.array(result['QP'])) + np.sum(np.array(result['observation']))
print(f"other time: {other_time}")
print(f"planning time: {(np.sum(np.array(result['total_time'])) - other_time)}")
# print(f"other time: {other_time/len(result['success'])}")
# print(f"planning time: {(np.sum(np.array(result['total_time'])) - other_time)/len(result['success'])}")
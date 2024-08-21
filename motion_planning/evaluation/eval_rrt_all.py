import os
import pickle
import time
from argparse import ArgumentParser
from tqdm import tqdm

import numpy as np
import torch

from environment import ArmEnv
from motion_planning.evaluation import eval_rrt_vanilla, eval_rrt_mindis, eval_rrt_lidar
from motion_planning.evaluation import eval_batchrrt_vanilla, eval_batchrrt_mindis
from motion_planning.evaluation import eval_rrt_mindis_IL, eval_rrt_mindis_RL
from motion_planning.evaluation import eval_rrt_lidar_IL, eval_rrt_lidar_RL
from motion_planning.evaluation import eval_rrt_mindis_OptRL
from motion_planning.evaluation import eval_rrt_dd
# from motion_planning.evaluation import eval_rrt_docbf

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

@torch.no_grad()
def eval_rrt_all(**kwargs):
    robot_name = kwargs['robot_name']
    # config_file = f"../../models/env_file/panda_100_8_v1_refined.npz"
    config_file = f"../../models/motion_planning/problems/refined_pkl/{robot_name}_{kwargs['level']}.pkl"
    environment = ArmEnv([robot_name], GUI=0, config_file=config_file)
    robot = environment.robot_list[0]

    positions, sizes, start_configs, end_configs = get_test_cases(config_file, **kwargs)
    seed = args.seed

    methods = {
        'rrt_vanilla': [eval_rrt_vanilla, None],
        'rrt_dd': [eval_rrt_dd, None],
        'rrt_mindis': [eval_rrt_mindis, f"../../models/neural_cbf/collection/{robot_name}_mindis/"],
        'rrt_lidar': [eval_rrt_lidar, f"../../models/neural_cbf/collection/{robot_name}_lidar/"],
        # 'batchrrt_vanilla': [eval_batchrrt_vanilla, None],
        # 'batchrrt_mindis': [eval_batchrrt_mindis, f"../../models/neural_cbf/collection/{robot_name}_mindis/"],
        'rrt_mindis_IL': [eval_rrt_mindis_IL, f"../../models/neural_cbf/IL_baseline/imitation_Mindis_{robot_name}"],
        'rrt_mindis_RL': [eval_rrt_mindis_RL, f"../../models/neural_cbf/RL_baseline/reinforcement_Mindis_{robot_name}_best"],
        'rrt_lidar_IL': [eval_rrt_lidar_IL, f"../../models/neural_cbf/IL_baseline/imitation_Lidar_{robot_name}"],
        'rrt_lidar_RL': [eval_rrt_lidar_RL, f"../../models/neural_cbf/RL_baseline/reinforcement_Lidar_{robot_name}_best"],
        'rrt_mindis_OptRL': [eval_rrt_mindis_OptRL, f"../../models/neural_cbf/RL_baseline/reinforcement_Mindis_{robot_name}_best"],
        # 'rrt_mindis_OptRL': [eval_rrt_mindis_OptRL, f"../../models/neural_cbf/Mindis/{robot_name}/reinforcement_Mindis_{robot_name}_6_kqgpsgbk"], # magician: rwbqianh; panda: kqgpsgbk
        'rrt_docbf': [eval_rrt_docbf, None],
    }

    method_name = kwargs['method_name']
    print("=" * 20, method_name, "=" * 20)
    print(f"robot: {robot_name}, level: {kwargs['level']}")
    print(f"cuda available: {torch.cuda.is_available()}")

    # for env_name, env, indexes in zip(env_names, envs, indexeses):
    method, method_model = methods[method_name]
    result = method(environment=environment, robot=robot, method_model_dir=method_model,
                    obs_positions=positions, obs_sizes=sizes, start_configs=start_configs, end_configs=end_configs, **kwargs)

    n_success = np.sum([solution['success'] for solution in result])
    avg_explored_node = np.mean([solution['explored_nodes'] for solution in result])
    success_avg_explored_node = np.mean([solution['explored_nodes'] for solution in result if solution['success']])

    print("=" * 20, method_name, "=" * 20)
    print(f'success rate: {n_success / len(result)}')
    print(f'average explored node: {avg_explored_node}')
    print(f'average explored node in success: {success_avg_explored_node}')
    print(f"total time: {np.sum([solution['time_dict']['total_time'] for solution in result])} for {len(result)} cases")

    # # with open(f"../../models/motion_planning/{method_name}/time_6obs_{robot_name}_{kwargs['level']}_{seed}_{int(kwargs['goal_biasing']*100)}_{kwargs['RRT_STEP']}_{kwargs['start_idx']}_{kwargs['end_idx']}.pkl", "wb") as handle:
    # #     pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"../../models/motion_planning/{method_name}/time_{robot_name}_{kwargs['level']}_{seed}_{int(kwargs['goal_biasing']*100)}_{kwargs['RRT_STEP']}_{kwargs['start_idx']}_{kwargs['end_idx']}.pkl", "wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_test_cases(file_name, **kwargs):
    # generate or load test cases
    if os.path.exists(file_name):
        positions = np.load(file_name, allow_pickle=True)['obstacle_positions'][kwargs['start_idx']:kwargs['end_idx']]
        sizes = np.load(file_name, allow_pickle=True)['obstacle_sizes'][kwargs['start_idx']:kwargs['end_idx']]
        # positions = [position[:6, :] for position in positions]
        # sizes = [size[:6, :] for size in sizes]

        try:
            start_configs = np.load(file_name, allow_pickle=True)['init_config'][kwargs['start_idx']:kwargs['end_idx']]
            end_configs = np.load(file_name, allow_pickle=True)['goal_config'][kwargs['start_idx']:kwargs['end_idx']]
        except:
            start_configs = np.load(file_name, allow_pickle=True)['init_configs'][kwargs['start_idx']:kwargs['end_idx']]
            end_configs = np.load(file_name, allow_pickle=True)['goal_configs'][kwargs['start_idx']:kwargs['end_idx']]

        # # two-wall-exp
        # positions = np.array([[[0.55, 0, 0.8], [0.55, 0, 0.1], [0.85, 0, 0.85], [0.85, 0, 0.15]]])
        # sizes = np.array([[[0.05, 0.5, 0.25], [0.05, 0.5, 0.1], [0.05, 0.5, 0.2], [0.05, 0.5, 0.15]]])
        # end_configs = np.array([[0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4]], dtype=np.float32)
        # start_configs = np.array([[8.83113899e-02, 5.76191382e-01, -4.92542215e-02, -1.77488283e+00,
        #                          9.94811764e-02, 3.84159051e+00, 7.40888267e-01]], dtype=np.float32)

        # # three-block-exp
        # positions = np.array([[[0.55, 0.1, 0.7], [0.65, -0.1, 0.3], [0.75, 0, 0.1]]])
        # sizes = np.array([[[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]])
        # start_configs = np.array([[0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4]], dtype=np.float32)
        # end_configs = np.array([[1.40964252e-0, 4.19911564e-01, 2.03002289e-02, -2.28147180e+00,
        #                         -1.98314446e-02, 2.70140259e+00, 9.61133575e-01]], dtype=np.float32)

    else:
        raise ValueError('No test cases found!')

    return positions, sizes, start_configs, end_configs


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--robot_name', type=str, default='panda')
    # parser.add_argument('--method_name', type=str, default='rrt_vanilla')  # OptRL
    parser.add_argument('--method_name', type=str, default='rrt_docbf')  # OptRL

    # simulation params
    parser.add_argument('--controller_period', type=float, default=1/30)
    parser.add_argument('--simulation_dt', type=float, default=1/120)
    parser.add_argument('--level', type=str, default='easy')

    # planning params
    # parser.add_argument('--accelerator', type=str, default='gpu', help='cpu or gpu')
    parser.add_argument('--seed', type=int, default=23)
    parser.add_argument('--devices', type=str, default=1, help='gpu id')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=100)

    # tuning params
    parser.add_argument('--cbf_alpha', type=float, default=0.1)
    parser.add_argument('--max_node', type=int, default=500, help="maximum planning steps")
    parser.add_argument('--goal_biasing', type=float, default=0.2)
    parser.add_argument('--RRT_STEP', type=int, default=180, help="maximum simulation steps for one node")

    # log params
    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.devices}"

    eval_rrt_all(**vars(args))

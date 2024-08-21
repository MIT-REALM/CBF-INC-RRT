import numpy as np

import torch


class SearchTree:
    def __init__(self, root):
        self.states = np.array([root])
        self.nominal_states = np.array([root])
        self.parents = [None]
        self.rewired_parents = [None]
        self.expanded_by_rrt = [None]
        self.freesp = [True]
        self.x_list = [[root]]
        self.costs = [0.]
        self.path_lengths = [-1]
        self.cumulated_collision_checks = [0]
        self.in_goal_region = [False]

        # for global exploration
        self.non_terminal_states = np.array([root])
        self.non_terminal_idxes = [0]

    def path(self):
        if not self.in_goal_region[-1]:
            return [], [], [], []
        assert self.in_goal_region[-1]
        path = []
        nominal_path = []
        path_cost = []
        x_list = []
        cost = 0
        current_index = -1
        while True:
            path.append(self.states[current_index])
            nominal_path.append(self.nominal_states[current_index])
            x_list.append(self.x_list[current_index])
            path_cost.append(cost)
            if current_index == 0:
                break
            cost -= np.linalg.norm(self.states[current_index] - self.states[self.rewired_parents[current_index]])
            current_index = self.rewired_parents[current_index]
        path.reverse()
        nominal_path.reverse()
        path_cost.reverse()
        x_list.reverse()
        return path, path_cost, nominal_path, x_list


def update_collision_checks(search_tree, collision_checks):
    search_tree.cumulated_collision_checks.append(collision_checks)


# def rewire_to(search_tree, child_idx, new_parent_idx):
#     search_tree.rewired_parents[child_idx] = new_parent_idx


def set_cost(search_tree, idx, new_cost):
    search_tree.costs[idx] = new_cost

    # Update path length if a path is found.
    if idx == -1 and search_tree.in_goal_region[-1]:
        if search_tree.path_lengths[-1] < 0 or \
                search_tree.path_lengths[-1] > new_cost:
            search_tree.path_lengths[-1] = new_cost


def insert_new_state(search_tree, state, nominal_state, x_list, parent_idx, no_collision, \
                     done, expanded_by_rrt=False, use_GP=False):
    search_tree.states = np.append(search_tree.states, [state], axis=0)
    search_tree.nominal_states = np.append(search_tree.nominal_states, [nominal_state], axis=0)
    search_tree.parents.append(parent_idx)
    search_tree.rewired_parents.append(parent_idx)
    search_tree.x_list.append(x_list)
    search_tree.expanded_by_rrt.append(expanded_by_rrt)
    search_tree.freesp.append(no_collision)
    search_tree.in_goal_region.append(done)

    # Will be updated in post-processing.
    search_tree.path_lengths.append(search_tree.path_lengths[-1])
    search_tree.costs.append(-1)

    if not done:
        search_tree.non_terminal_states = np.append(search_tree.non_terminal_states, [state], axis=0)
        search_tree.non_terminal_idxes.append(search_tree.states.shape[0] - 1)

    return search_tree.states.shape[0] - 1


# def state_kernel(env, state_A, state_B):
#     diff = env.distance(state_A, state_B) / env.RRT_EPS
#     kernel = np.exp(- (diff ** 2) * 1.)
#
#     return kernel
#
#
# def compute_w(env, search_tree, idx=None, state=None):
#     if state is None:
#         state = search_tree.states[idx]
#
#     kernel = np.maximum(state_kernel(env, search_tree.states, state), 1e-3)
#     w_ = np.sum(kernel)
#
#     return w_

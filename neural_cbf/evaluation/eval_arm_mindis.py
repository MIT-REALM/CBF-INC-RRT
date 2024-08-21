import time
import argparse
import yaml

import numpy as np
import torch
import pytorch_lightning as pl

import pybullet as p

from environment import ArmEnv

from neural_cbf.controllers import NeuralMindisCBFController
from neural_cbf.systems import ArmMindis
from neural_cbf.experiments import (
    ExperimentSuite,
    BFContourExperiment,
    RolloutStateSpaceExperiment,
)

# batch_size = 1


def init_val(path, args):
    # initialize models and parameters for loaded controllers
    nominal_params = {"m1": 5.76}
    scenarios = [
        nominal_params,
    ]

    # Define environment and agent
    environment = ArmEnv([args.robot_name], GUI=False, config_file="")
    robot = environment.robot_list[0]

    # Define the dynamics model
    dynamics_model = ArmMindis(
        nominal_params,
        dt=args.simulation_dt,
        controller_dt=args.controller_period,
        dis_threshold=args.dis_threshold,
        env=environment,
        robot=robot
    )

    # Define goal_state
    goal_state = torch.tensor(robot.q0).float()
    dynamics_model.set_goal(goal_state)

    # Initialize the rollout experiment
    ul, ll = dynamics_model.state_limits
    start_x = torch.cat([
        torch.lerp(ll, ul, 0.2 * torch.ones(ll.shape[-1]).double()).reshape(1, -1),
        # torch.lerp(ll, ul, 0.8 * torch.ones(ll.shape[-1]).double()).reshape(1, -1),
    ], dim=0).float()
    start_x = dynamics_model.complete_sample_with_observations(start_x, num_samples=start_x.shape[0])

    x_idx = 2
    y_idx = 0
    rollout_experiment = RolloutStateSpaceExperiment(
        "Rollout",
        start_x,
        x_idx,
        f"$\\theta_{x_idx}$",
        y_idx,
        f"$\\theta_{y_idx}$",
        scenarios=scenarios,
        n_sims_per_start=1,
        t_sim=5,
    )

    default_state = dynamics_model.sample_boundary(1).squeeze()

    # Define the experiment suite
    h_contour_experiment = BFContourExperiment(
        "h_Contour",
        domain=[tuple(robot.body_range[x_idx]), tuple(robot.body_range[y_idx])],
        n_grid=30,
        x_axis_index=x_idx,
        y_axis_index=y_idx,
        x_axis_label=f"$\\theta_{x_idx}$",
        y_axis_label=f"$\\theta_{y_idx}$",
        default_state=default_state,
        plot_unsafe_region=True,
    )

    experiment_suite = ExperimentSuite([rollout_experiment, h_contour_experiment])

    return NeuralMindisCBFController.load_from_checkpoint(path, dynamics_model=dynamics_model, scenarios=scenarios,
														  datamodule=None, experiment_suite=experiment_suite,
                                                          use_neural_actor=False,
														  map_location='cpu')


def vis_traj_rollout(controller):
    """
    Visualize trajectories from two-link-arm RolloutStateSpaceExperiments.
    """
    # Tweak experiment params
    controller.experiment_suite.experiments[0].t_sim = 6.0

    # Run the experiments and save the results
    controller.experiment_suite.experiments[0].run_and_plot(
        controller, display_plots=True
    )
    print('finished')


def vis_CBF_contour(controller):
    # Run the experiments and save the results
    controller.experiment_suite.experiments[1].run_and_plot(
        controller_under_test=controller, display_plots=True
    )
    print('finished CBF contour')


if __name__ == "__main__":
    pl.seed_everything(seed=20)
    robot_name = "panda"

    # Load the checkpoint file. This should include the experiment suite used during training.
    log_dir = "../../models/neural_cbf/"

    git_version = f"collection/{robot_name}_mindis/"
    log_file = ""  # specify the checkpoint file

    with open(log_dir + git_version + 'hparams.yaml', 'r') as f:
        args = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))
    args.accelerator = 'cpu'

    neural_controller = init_val(log_dir + git_version + log_file, args)
    neural_controller.h_nn.eval()

    # neural_controller.h_alpha=0.3
    # vis_traj_rollout(neural_controller)
    vis_CBF_contour(neural_controller)
"""Plot a barrier function contour"""
from typing import cast, List, Tuple, Optional, TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
import seaborn as sns
import numpy as np
import torch
import tqdm

from neural_cbf.experiments import Experiment

if TYPE_CHECKING:
    from neural_cbf.controllers import Controller, NeuralObsBFController  # noqa

class BFContourExperiment(Experiment):
    """An experiment for plotting the contours of learned BFs"""

    def __init__(
        self,
        name: str,
        domain: Optional[List[Tuple[float, float]]] = None,
        n_grid: int = 50,
        x_axis_index: int = 0,
        y_axis_index: int = 1,
        x_axis_label: str = "$x$",
        y_axis_label: str = "$y$",
        default_state: Optional[torch.Tensor] = None,
        plot_unsafe_region: bool = True,
    ):
        """Initialize an experiment for plotting the value of the BF over selected
        state dimensions.

        args:
            name: the name of this experiment
            domain: a list of two tuples specifying the plotting range,
                    one for each state dimension.
            n_grid: the number of points in each direction at which to compute h
            x_axis_index: the index of the state variable to plot on the x axis
            y_axis_index: the index of the state variable to plot on the y axis
            x_axis_label: the label for the x axis
            y_axis_label: the label for the y axis
            default_state: 1 x dynamics_model.n_dims tensor of default state
                           values. The values at x_axis_index and y_axis_index will be
                           overwritten by the grid values.
            plot_unsafe_region: True to plot the safe/unsafe region boundaries.
        """
        super(BFContourExperiment, self).__init__(name)

        # Default to plotting over [-1, 1] in all directions
        if domain is None:
            domain = [(-1.0, 1.0), (-1.0, 1.0)]
        self.domain = domain

        self.n_grid = n_grid
        self.x_axis_index = x_axis_index
        self.y_axis_index = y_axis_index
        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label
        self.default_state = default_state
        self.plot_unsafe_region = plot_unsafe_region

    @torch.no_grad()
    def run(self, controller_under_test: "Controller") -> pd.DataFrame:
        """
        Run the experiment, likely by evaluating the controller, but the experiment
        has freedom to call other functions of the controller as necessary (if these
        functions are not supported by all controllers, then experiments will be
        responsible for checking compatibility with the provided controller)

        args:
            controller_under_test: the controller with which to run the experiment
        returns:
            a pandas DataFrame containing the results of the experiment, in tidy data
            format (i.e. each row should correspond to a single observation from the
            experiment).
        """
        # Sanity check: can only be called on a NeuralObsBFController
        if not (hasattr(controller_under_test, "h")):
            raise ValueError("Controller under test must be a NeuralObs(C)BFController")

        dynamics_model = cast("ObservableSystem", controller_under_test.dynamics_model)
        if hasattr(controller_under_test.dynamics_model, 'env') and hasattr(controller_under_test.dynamics_model.env, 'reset_env'):
            env = controller_under_test.dynamics_model.env
            env.reset_env(obs_configs=env.get_env_config(-1), enable_object=False)

        # Set up a dataframe to store the results
        results_df = pd.DataFrame()

        # Set up the plotting grid
        device = "cpu"
        if hasattr(controller_under_test, "device"):
            device = controller_under_test.device  # type: ignore

        x_vals = torch.linspace(
            self.domain[0][0], self.domain[0][1], self.n_grid, device=device
        )
        y_vals = torch.linspace(
            self.domain[1][0], self.domain[1][1], self.n_grid, device=device
        )

        # Default to all zeros if no default provided
        if self.default_state is None:
            if hasattr(controller_under_test.dynamics_model, 'viz'):
                default_state = torch.ones(1, controller_under_test.dynamics_model.n_dims + controller_under_test.dynamics_model.o_dims + controller_under_test.dynamics_model.q_dims)
            elif hasattr(controller_under_test.dynamics_model, 'o_dims') and controller_under_test.dynamics_model.q_dims==7:
                default_state = torch.tensor([1.00887519, 0.50546576, -1.69052917, -2.2909179, 2.95208592, 3.59793418, 2.93001438])
            else:
                default_state = torch.ones(1, controller_under_test.dynamics_model.n_dims)
        else:
            default_state = self.default_state

        default_state = default_state.type_as(x_vals)

        # Make a copy of the default state, which we'll modify on every loop
        if hasattr(controller_under_test.dynamics_model, "o_dims"):
            x = (default_state.clone().detach().reshape(1, -1))
        else:
            x = (default_state.clone().detach().reshape(1, controller_under_test.dynamics_model.n_dims))

        # Loop through the grid
        prog_bar_range = tqdm.trange(self.n_grid, desc="Plotting BF", position=0, leave=True)
        for i in prog_bar_range:
            for j in range(self.n_grid):
                # Adjust x to be at the current grid point
                x[0, self.x_axis_index] = x_vals[i]
                x[0, self.y_axis_index] = y_vals[j]

                aa = (x_vals[i], y_vals[j])

                # Get the value of the BF from observations at this point
                if hasattr(controller_under_test.dynamics_model, "o_dims"):
                    # adjust observation
                    x = controller_under_test.dynamics_model.complete_sample_with_observations(x[:, :controller_under_test.dynamics_model.n_dims], 1)
                    h = controller_under_test.h(x)
                else:
                    obs = dynamics_model.get_observations(x)
                    h = controller_under_test.h(x, obs)

                # Get the goal, safe, or unsafe classification
                is_safe = controller_under_test.dynamics_model.safe_mask(x).all()
                is_unsafe = controller_under_test.dynamics_model.unsafe_mask(x).all()

                # Store the results
                log_packet = pd.DataFrame([{
                        self.x_axis_label: x_vals[i].cpu().numpy().item(),
                        self.y_axis_label: y_vals[j].cpu().numpy().item(),
                        "h": h.cpu().numpy().item(),
                        "Safe region": is_safe.cpu().numpy().item(),
                        "Unsafe region": is_unsafe.cpu().numpy().item(),
                    }])
                results_df = pd.concat([results_df, log_packet], ignore_index=True)

        return results_df

    def plot(
        self,
        controller_under_test: "Controller",
        results_df: pd.DataFrame,
        display_plots: bool = False,
    ) -> List[Tuple[str, figure]]:
        """
        Plot the results, and return the plot handles. Optionally
        display the plots.

        args:
            controller_under_test: the controller with which to run the experiment
            display_plots: defaults to False. If True, display the plots (blocks until
                           the user responds).
        returns: a list of tuples containing the name of each figure and the figure
                 object.
        """
        # Set the color scheme
        sns.set_theme(context="talk", style="white")

        # Plot a contour of h
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(12, 8)

        contours = ax.tricontourf(
            results_df[self.x_axis_label],
            results_df[self.y_axis_label],
            results_df["h"],
            cmap=sns.color_palette("rocket", as_cmap=True),
            levels=20,
        )
        cbar = plt.colorbar(contours, ax=ax, orientation="vertical")

        import matplotlib.ticker as ticker
        cbar.locator = ticker.MaxNLocator(nbins=6)
        cbar.update_ticks()

        linewid = 3

        # Overlay the safe/unsafe regions (if specified)
        if self.plot_unsafe_region:
            ax.plot([], [], c="green", label="gt-safe")
            ax.tricontour(
                results_df[self.x_axis_label],
                results_df[self.y_axis_label],
                results_df["Safe region"],
                colors=["green"],
                levels=[0.5],
                linewidths=linewid,
            )
            ax.plot([], [], c="magenta", label="gt-unsafe")
            ax.tricontour(
                results_df[self.x_axis_label],
                results_df[self.y_axis_label],
                results_df["Unsafe region"],
                colors=["magenta"],
                levels=[0.5],
                linewidths=linewid,
            )

            # Plot the environment if possible
            if hasattr(controller_under_test.dynamics_model, "plot_environment"):
                controller_under_test.dynamics_model.plot_environment(ax)

            if hasattr(controller_under_test, "safe_level"):
                ax.plot([], [], c="blue", label=f"predicted safe")
                ax.tricontour(
                    results_df[self.x_axis_label],
                    results_df[self.y_axis_label],
                    results_df["h"],
                    colors=["blue"],
                    levels=[-controller_under_test.safe_level],
                    linewidths=linewid,
                )
                ax.plot([], [], c="whitesmoke", label=f"predicted unsafe")
                ax.tricontour(
                    results_df[self.x_axis_label],
                    results_df[self.y_axis_label],
                    results_df["h"],
                    colors=["white"],
                    levels=[controller_under_test.unsafe_level],
                    linewidths=linewid,
                )
                # ax.plot([], [], c="brown", label=f"h={0}")
                # ax.tricontour(
                #     results_df[self.x_axis_label],
                #     results_df[self.y_axis_label],
                #     results_df["h"],
                #     colors=["brown"],
                #     levels=[0.],
                # )
            else:
                ax.plot([], [], c="blue", label="h(o(x)) = 0.0")
                ax.tricontour(
                    results_df[self.x_axis_label],
                    results_df[self.y_axis_label],
                    results_df["h"],
                    colors=["blue"],
                    levels=[0.0],
                )

        # Make the legend
        ax.legend(
            bbox_to_anchor=(0, 1.02, 1, 0.2),
            loc="lower left",
            # mode="expand",
            borderaxespad=0,
            ncol=2,
            columnspacing=0.8,
        )
        ax.set_xlabel(self.x_axis_label)
        ax.set_ylabel(self.y_axis_label)

        fig_handle = ("h Contour", fig)

        if display_plots:
            plt.show()
            return []
        else:
            return [fig_handle]

import os
os.environ['MPLBACKEND'] = 'qtagg'  # or use your preferred backend
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import master_thesis_mitchell_functions as mtmf
import copy
import coordinates_transformation_functions
import datetime
import time as time_library

dpi_setting = 300

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm


# plt.rcParams["figure.figsize"] = (12,6)
# plt.rcParams.update({'font.size': 14})
# plt.rcParams['lines.linewidth'] = 1.5
# plt.rcParams['lines.markersize'] = 3
# plt.rcParams["figure.autolayout"] = True
# plt.rcParams["axes.grid"] = True

def plot_trajectory_radialND_to_cartesianND(time_grid, states, thrust_scale=0.1, r_target=1.5, r_start=1, lim=None, N_arrows=40, thrust=True, config=None, dpi_setting=300):
    t = time_grid;      t_reshaped = t.reshape(-1, 1)
    states = np.concatenate((t_reshaped, states), axis=1)

    fig, ax = plt.subplots(1, figsize=(10, 10))
    theta = np.linspace(0, 2 * np.pi, 1000)

    # Plot the starting and target orbits using the ax object
    if r_start:
        ax.plot(r_start * np.cos(theta), r_start * np.sin(theta), label="Starting orbit", alpha=0.85, lw=1.25, ls='-', c='darkblue', zorder=1.0)
    if r_target:
        ax.plot(r_target * np.cos(theta), r_target * np.sin(theta), label="Target orbit", alpha=0.85, lw=1.25, ls='-', c='darkgoldenrod', zorder=1.0)

    y = coordinates_transformation_functions.radial_to_NDcartesian(states, config)
    # Strip time
    y = y[:, 1:]

    # Plot the trajectory line using the ax object
    ax.plot(y[:, 0], y[:, 1], label="Transfer trajectory", lw=1.25, ls='-', c='0.35')

    # Determine plot limits based on the given radius and data
    if lim is None:
        if r_start is None and r_target is None:
            lim = 1.2 * np.max(np.linalg.norm(y[:, :2], axis=1))
        else:
            lim = 1.2 * max(r_target, r_start, np.max(np.linalg.norm(y[:, :2], axis=1)))

    # Plot thrust vectors using the ax object
    if thrust:
        arrow_indices = np.linspace(0, len(y[:, 0]) - 1, N_arrows, dtype=int)
        scale = thrust_scale

        for i in arrow_indices:
            ax.arrow(y[i, 0],
                     y[i, 1],
                     y[i, 4] * 0.5 / scale,
                     y[i, 5] * 0.5 / scale, width=0.004, alpha=1.0, color='red', zorder=2.0)
        ax.arrow(-0.9 * lim, 0.87 * lim, 0.5, 0, width=0.008, color='red')
        ax.text(-0.9 * lim, 0.90 * lim, f"Thrust scale: {scale} N")

    # Set plot titles and labels using the ax object
    ax.set_title('Trajectory in Cartesian Coordinates')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('X [AU]')
    ax.set_ylabel('Y [AU]')
    ax.grid(True, alpha=0.5, zorder=0.5)

    # Add a legend
    ax.legend(bbox_to_anchor=(1, 1), loc='upper right')

    # Save the plot
    plt.savefig(f"plot_trajectory.png", dpi=dpi_setting)
    # plt.close()


def plot_trajectory_radialND_to_cartesianND_sp2_T1(time_grid, states, ga_states=None, ga_index=None, thrust_scale=0.1, r_start=1, r_ga=1.5, r_target=3, lim=None,
                                                N_arrows=40, thrust=True, configs_list=None, dpi_setting=300, config=None):
    t = time_grid;          t_reshaped = t.reshape(-1, 1)
    y = np.concatenate((t_reshaped, states[:,0:7]), axis=1)
    ga_states = np.concatenate((t_reshaped, ga_states), axis=1)

    fig, ax = plt.subplots(1, figsize=(10, 10))
    theta = np.linspace(0, 2 * np.pi, 1000)

    # Plot the starting and target orbits using the ax object
    if r_start:
        ax.plot(r_start * np.cos(theta), r_start * np.sin(theta), label="Starting orbit", alpha=0.85, lw=1.25, ls='-', c='darkblue', zorder=1.0)
    if r_ga:
        ax.plot(r_ga * np.cos(theta), r_ga * np.sin(theta), label="Mars orbit", alpha=0.85, lw=1.25, ls='-', c='darkgoldenrod', zorder=1.0)
    if r_target:
        ax.plot(r_target * np.cos(theta), r_target * np.sin(theta), label="Ceres orbit", alpha=0.85, lw=1.25, ls='-', c='darkgray', zorder=1.0)

    y = coordinates_transformation_functions.radial_to_cartesian(y, config)
    ga = coordinates_transformation_functions.radial_to_NDcartesian(ga_states, config)
    # Strip time
    y = y[:, 1:]
    ga = ga[:, 1:]

    # Plot the trajectory line using the ax object
    ax.plot(y[:, 0], y[:, 1], label="Trajectory leg 1", lw=2, ls='-', c='0.35')

    # Plot bodies
    ax.scatter(y[:, 0][0], y[:, 1][0], label="Earth", c='darkblue')
    ax.scatter(ga[:, 0][ga_index], ga[:, 1][ga_index], label="Mars", c='darkgoldenrod')

    # Determine plot limits based on the given radius and data
    if lim is None:
        if r_start is None and r_target is None:
            lim = 1.2 * np.max(np.linalg.norm(y[:, :2], axis=1))
        else:
            lim = 1.2 * max(r_target, r_start, np.max(np.linalg.norm(y[:, :2], axis=1)))

    # Plot thrust vectors using the ax object

    if thrust:
        arrow_indices = np.linspace(0, len(y[:, 0]) - 1, N_arrows, dtype=int)
        scale = thrust_scale

        for j in arrow_indices:
            ax.arrow(y[j, 0],
                     y[j, 1],
                     y[j, 4] * config['length_scale']*0.5 / scale,
                     y[j, 5] * config['length_scale']*0.5 / scale, width=0.004, alpha=1.0, color='red', zorder=2.0)
        ax.arrow(-0.9 * lim, 0.87 * lim, config['length_scale']*0.5, 0, width=0.008*config['length_scale']*0.5, color='red')
        ax.text(-0.9 * lim, 0.90 * lim, f"Thrust scale: {scale} N")

    # Set plot titles and labels using the ax object
    ax.set_title('Trajectory in Cartesian Coordinates')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.grid(True, alpha=0.5, zorder=0.5)

    # Add a legend
    ax.legend(bbox_to_anchor=(1, 1), loc='upper right')

    # Save the plot
    plt.savefig(f"plot_trajectory.png", dpi=dpi_setting)
    # plt.close()
def plot_trajectory_radialND_to_cartesianND_sp2_T2(time_grid, states, ga_states=None, ga_index=None, thrust_scale=0.1, r_start=1, r_ga=1.5, r_target=3, lim=None,
                                                N_arrows=40, thrust=True, configs_list=None, dpi_setting=300):
    t = time_grid;      t_reshaped = t.reshape(-1, 1)
    y_leg1 = np.concatenate((t_reshaped, states[:,0:7]), axis=1)
    y_leg2 = np.concatenate((t_reshaped, states[:,7:14]), axis=1)
    # ga_states = np.concatenate((t_reshaped, ga_states), axis=1)

    fig, ax = plt.subplots(1, figsize=(10, 10))
    theta = np.linspace(0, 2 * np.pi, 1000)

    # Plot the starting and target orbits using the ax object
    if r_start:
        ax.plot(r_start * np.cos(theta), r_start * np.sin(theta), label="Starting orbit", alpha=0.85, lw=1.25, ls='-', c='darkblue', zorder=1.0)
    if r_ga:
        ax.plot(r_ga * np.cos(theta), r_ga * np.sin(theta), label="Mars orbit", alpha=0.85, lw=1.25, ls='-', c='darkgoldenrod', zorder=1.0)
    if r_target:
        ax.plot(r_target * np.cos(theta), r_target * np.sin(theta), label="Ceres orbit", alpha=0.85, lw=1.25, ls='-', c='darkgray', zorder=1.0)

    y_leg1 = coordinates_transformation_functions.radial_to_cartesian(y_leg1, configs_list[0])
    y_leg2 = coordinates_transformation_functions.radial_to_cartesian(y_leg2, configs_list[1])
    # ga = coordinates_transformation_functions.radial_to_NDcartesian(ga_states, config)
    # Strip time
    y_leg1 = y_leg1[:, 1:]
    y_leg2 = y_leg2[:, 1:]
    # ga = ga[:, 1:]

    # Plot the trajectory line using the ax object
    ax.plot(y_leg1[:, 0], y_leg1[:, 1], label="Trajectory leg 1", lw=2, ls='-', c='0.35')
    ax.plot(y_leg2[:, 0], y_leg2[:, 1], label="Trajectory leg 2", lw=2, ls='-', c='0.35')

    # Plot bodies
    ax.scatter(y_leg1[:, 0][0], y_leg1[:, 1][0], label="Earth", c='darkblue')
    ax.scatter(y_leg2[:, 0][0], y_leg2[:, 1][0], label="Mars", c='darkgoldenrod')
    ax.scatter(y_leg2[:, 0][-1], y_leg2[:, 1][-1], label="Ceres", c='darkgray')

    # Determine plot limits based on the given radius and data
    if lim is None:
        if r_start is None and r_target is None:
            lim = 1.2 * np.max(np.linalg.norm(y_leg2[:, :2], axis=1))
        else:
            lim = 1.2 * max(r_target, r_start, np.max(np.linalg.norm(y_leg2[:, :2], axis=1)))

    # Plot thrust vectors using the ax object
    legs = [y_leg1, y_leg2]
    for i in range(len(legs)):
        if thrust:
            print('thrust=true')
            arrow_indices = np.linspace(0, len(legs[i][:, 0]) - 1, N_arrows, dtype=int)
            scale = thrust_scale

            for j in arrow_indices:
                ax.arrow(legs[i][j, 0],
                         legs[i][j, 1],
                         legs[i][j, 4] * configs_list[i]['length_scale']*0.5 / scale,
                         legs[i][j, 5] * configs_list[i]['length_scale']*0.5 / scale, width=0.004, alpha=1.0, color='red', zorder=2.0)
            ax.arrow(-0.9 * lim, 0.87 * lim, configs_list[i]['length_scale']*0.5, 0, width=0.008*configs_list[i]['length_scale']*0.5, color='red')
            ax.text(-0.9 * lim, 0.90 * lim, f"Thrust scale: {scale} N")

    # Set plot titles and labels using the ax object
    ax.set_title('Trajectory in Cartesian Coordinates')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.grid(True, alpha=0.5, zorder=0.5)

    # Add a legend
    ax.legend(bbox_to_anchor=(1, 1), loc='upper right')

    # Save the plot
    plt.savefig(f"plot_trajectory.png", dpi=dpi_setting)
    # plt.close()
def plot_trajectory_radialND_to_cartesianND_sp2_T3(time_grids_list, states, ga_states=None, ga_index=None, thrust_scale=0.1, r_start=1, r_ga=1.5, r_target=3, lim=None,
                                                N_arrows=40, thrust=True, configs_list=None, dpi_setting=300, save=True):
    t1 = time_grids_list[0];      t1_reshaped = t1.reshape(-1, 1)
    t2 = time_grids_list[0];      t2_reshaped = t2.reshape(-1, 1)
    y_leg1 = np.concatenate((t1_reshaped, states[:,0:7]), axis=1)
    y_leg2 = np.concatenate((t2_reshaped, states[:,7:14]), axis=1)
    # ga_states = np.concatenate((t_reshaped, ga_states), axis=1)

    fig, ax = plt.subplots(1, figsize=(10, 10))
    theta = np.linspace(0, 2 * np.pi, 1000)

    # Plot the starting and target orbits using the ax object
    if r_start:
        ax.plot(r_start * np.cos(theta), r_start * np.sin(theta), label="Starting orbit", alpha=0.85, lw=1.25, ls='-', c='darkblue', zorder=1.0)
    if r_ga:
        ax.plot(r_ga * np.cos(theta), r_ga * np.sin(theta), label="Venus orbit", alpha=0.85, lw=1.25, ls='-', c='darkgoldenrod', zorder=1.0)
    if r_target:
        ax.plot(r_target * np.cos(theta), r_target * np.sin(theta), label="Mars orbit", alpha=0.85, lw=1.25, ls='-', c='darkgray', zorder=1.0)

    y_leg1 = mtmf.coordinates_transformation_functions.radial_to_cartesian(y_leg1, configs_list[0])
    y_leg2 = mtmf.coordinates_transformation_functions.radial_to_cartesian(y_leg2, configs_list[1])
    # ga = coordinates_transformation_functions.radial_to_NDcartesian(ga_states, config)
    # Strip time
    y_leg1 = y_leg1[:, 1:]
    y_leg2 = y_leg2[:, 1:]
    # ga = ga[:, 1:]

    # Plot the trajectory line using the ax object
    ax.plot(y_leg1[:, 0], y_leg1[:, 1], label="Trajectory leg 1", lw=2, ls='-', c='0.35')
    ax.plot(y_leg2[:, 0], y_leg2[:, 1], label="Trajectory leg 2", lw=2, ls='-', c='0.35')

    # Plot bodies
    ax.scatter(y_leg1[:, 0][0], y_leg1[:, 1][0], label="Earth", c='darkblue')
    ax.scatter(y_leg2[:, 0][0], y_leg2[:, 1][0], label="Venus", c='darkgoldenrod')
    ax.scatter(y_leg2[:, 0][-1], y_leg2[:, 1][-1], label="Mars", c='darkgray')

    # Determine plot limits based on the given radius and data
    if lim is None:
        if r_start is None and r_target is None:
            lim = 1.2 * np.max(np.linalg.norm(y_leg2[:, :2], axis=1))
        else:
            lim = 1.2 * max(r_target, r_start, np.max(np.linalg.norm(y_leg2[:, :2], axis=1)))

    # Plot thrust vectors using the ax object
    legs = [y_leg1, y_leg2]
    for i in range(len(legs)):
        if thrust:
            print('thrust=true')
            arrow_indices = np.linspace(0, len(legs[i][:, 0]) - 1, N_arrows, dtype=int)
            scale = thrust_scale

            for j in arrow_indices:
                ax.arrow(legs[i][j, 0],
                         legs[i][j, 1],
                         legs[i][j, 4] * configs_list[i]['length_scale']*0.5 / scale,
                         legs[i][j, 5] * configs_list[i]['length_scale']*0.5 / scale, width=0.004, alpha=1.0, color='red', zorder=2.0)
            ax.arrow(-0.9 * lim, 0.87 * lim, configs_list[i]['length_scale']*0.5, 0, width=0.008*configs_list[i]['length_scale']*0.5, color='red')
            ax.text(-0.9 * lim, 0.90 * lim, f"Thrust scale: {scale} N")

    # Set plot titles and labels using the ax object
    ax.set_title('Trajectory in Cartesian Coordinates')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.grid(True, alpha=0.5, zorder=0.5)

    # Add a legend
    ax.legend(bbox_to_anchor=(1, 1), loc='upper right')

    # Save the plot
    if save:
        time_library.sleep(0.1)
        timestamp_file = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        plt.savefig(f"Saved_plots/{configs_list[0]['run_id_number']}/plot_trajectory_{timestamp_file}.png", dpi=dpi_setting)
    # plt.close()

def plot_compare_pcnn_tudat_states(pcnn_states,
                                   tudat_states,
                                   config,
                                   entries=6,
                                   control_entries=2,
                                   coordinates="NDcartesian",
                                   custom_labels=None,
                                   log=True,
                                   save=True):

    time = pcnn_states[coordinates][:, 0]
    states = pcnn_states[coordinates][:, 1:(1 + entries - control_entries)]
    states_bench = tudat_states[coordinates][:, 1:1 + entries - control_entries]

    fig, axes = plt.subplots(2, 2, figsize=(20, 8), gridspec_kw={'height_ratios': [5, 5]}, sharex=True)
    fig.subplots_adjust(hspace=0, wspace=0.3)
    fig.suptitle(f"Compare verification to PCNN model in [{coordinates}] coordinates", fontsize=16, y=0.98)

    for i in range(2):
        if custom_labels:
            label = custom_labels[i]
        else:
            label = f"x$_{i + 1}$"

        axes[0, 0].plot(time, states[:, i], label=f"PCNN [{label}]")
        axes[0, 0].plot(time, states_bench[:, i], linestyle='--', label=f"Verification [{label}]")

        difference = states[:, +i] - states_bench[:, i]
        difference[0] = 0.0
        axes[1, 0].plot(time, difference, label=f"[{label}]")
        # print(difference)

    for i in range(2):
        if custom_labels:
            label = custom_labels[2 + i]
        else:
            label = f"x$_{i + 3}$"

        axes[0, 1].plot(time, states[:, 2 + i], label=f"PCNN [{label}]")
        axes[0, 1].plot(time, states_bench[:, 2 + i], linestyle='--', label=f"Verification [{label}]")

        axes[1, 1].plot(time, states[:, 2 + i] - states_bench[:, 2 + i], label=f"[{label}]")

    axes[1, 1].set_xlabel("Time [-]", fontsize=16)
    axes[1, 0].set_xlabel("Time  [-]", fontsize=16)

    axes[0, 0].set_ylabel("Position [-]", fontsize=16)
    axes[0, 1].set_ylabel("Velocity [-]", fontsize=16)

    axes[1, 0].set_ylabel("Residuals [-]", fontsize=16)
    axes[1, 1].set_ylabel("Residuals [-]", fontsize=16)

    if log:
        axes[1, 0].set_yscale("log")
        axes[1, 1].set_yscale("log")

    for ax in axes.flat:
        ax.legend()
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.2)
        ax.tick_params(labelsize=16)
        ax.tick_params(axis="both", direction="in", which="both", length=4, width=1.2)
        ax.tick_params(bottom=True, top=True, left=True, right=True)
    if save:
        time_library.sleep(0.1)
        timestamp_file = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        plt.savefig(f"Saved_plots/{config['run_id_number']}/plot_compare_pcnn_tudat_states_{timestamp_file}.png", dpi=dpi_setting)

def plot_compare_pcnn_tudat_mass(pcnn_mass,
                                 tudat_mass,
                                 config,
                                 save=True):

    fig, axes = plt.subplots(2, 1, figsize = (7, 8), gridspec_kw={'height_ratios': [5, 2]}, sharex = True)
    fig.subplots_adjust(hspace=0.1)
    fig.suptitle("Spacecraft mass evolution", fontsize = 16, y = 0.96)

    axes[0].plot(pcnn_mass[:,0], pcnn_mass[:,1], label = "PCNN mass")
    axes[0].plot(tudat_mass[:,0]/config['t_scale'], tudat_mass[:,1], linestyle = '--', label = "TUDAT Mass")
    axes[0].set_ylabel("Mass [kg]", fontsize = 16)
    axes[0].legend()

    axes[1].plot(pcnn_mass[:,0], pcnn_mass[:,1] - tudat_mass[:,1])
    axes[1].set_ylabel("Mass Residual [kg]", fontsize = 16)
    axes[1].set_xlabel("Time [-]", fontsize = 16)

    for ax in axes.flat:
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.2)
        ax.tick_params(labelsize=16)
        ax.tick_params(axis="both", direction="in", which="both", length=4, width = 1.2)
        ax.tick_params(bottom=True, top=True, left=True, right=True)
    if save:
        time_library.sleep(0.1)
        timestamp_file = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        plt.savefig(f"Saved_plots/{config['run_id_number']}/plot_compare_pcnn_tudat_mass_{timestamp_file}.png", dpi=dpi_setting)
    # plt.show()

def plot_states_T1(time_grid, states, config):
    t = time_grid

    fig, axes = plt.subplots(7, 1, figsize=(12, 12), sharex=True)

    # Labels for the states and their derivatives
    labels = ['r_pred', 'theta_pred', 'v_r_pred', 'v_theta_pred', 'u_r_pred', 'u_tangential_pred', 'm_pred']
    # labels_dot = ['r_pred_dot', 'theta_pred_dot', 'v_r_pred_dot', 'v_theta_pred_dot', 'u_phi_pred_dot', 'u_T_pred_dot', 'm_pred_dot']
    for i in range(7):
        # Left column: states leg 1
        axes[i].plot(t, states[:,i], label=labels[i])
        axes[i].set_ylabel(labels[i])
        axes[i].grid(True, alpha=0.5)
        axes[i].legend(loc='upper right')

    # Set a common x-label for the last row
    axes[0].set_xlabel("Time")
    # Set a common title for the entire figure
    fig.suptitle("Predicted States ", fontsize=16)

    # Adjust layout and save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the title
    plt.savefig(f"plot_states.png", dpi=dpi_setting)

def plot_states_T3(time_grids_list, states, configs_list, save=True): # only difference w.r.t. plot_states is the time_grids_list
    t1 = time_grids_list[0]
    t2 = time_grids_list[1]

    fig, axes = plt.subplots(7, 2, figsize=(12, 12), sharex=True)

    # Labels for the states and their derivatives
    labels = ['r_pred', 'theta_pred', 'v_r_pred', 'v_theta_pred', 'u_r_pred', 'u_tangential_pred', 'm_pred']
    # labels_dot = ['r_pred_dot', 'theta_pred_dot', 'v_r_pred_dot', 'v_theta_pred_dot', 'u_phi_pred_dot', 'u_T_pred_dot', 'm_pred_dot']
    for i in range(7):
        # Left column: states leg 1
        axes[i,0].plot(t1, states[:,i], label=labels[i])
        axes[i,0].set_ylabel(labels[i])
        axes[i,0].grid(True, alpha=0.5)
        axes[i,0].legend(loc='upper right')

        # Right column: states leg 2
        axes[i,1].plot(t2, states[:, i+7], label=labels[i])
        axes[i,1].set_ylabel(labels[i])
        axes[i,1].grid(True, alpha=0.5)
        axes[i,1].legend(loc='upper right')

    # Set a common x-label for the last row
    axes[0,0].set_xlabel("Time")
    axes[0,1].set_xlabel("Time")

    # Set a common title for the entire figure
    fig.suptitle("Predicted States ", fontsize=16)

    # Adjust layout and save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the title

    if save:
        time_library.sleep(0.1)
        timestamp_file = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        plt.savefig(f"Saved_plots/{configs_list[0]['run_id_number']}/plot_states{timestamp_file}.png", dpi=dpi_setting)


def plot_loss_T1(losshistory, mass=True):
    plt.figure()
    unique_arrays_loss = losshistory.loss_train #.trainset(tuple(row) for row in losshistory.loss_train)
    unique_arrays_loss = np.array(list(unique_arrays_loss))
    unique_arrays_steps = losshistory.steps #list(set(losshistory.steps))
    unique_arrays_steps.sort()

    labels = [
        r'$\mathcal{L}_r$',  # Blue line
        r'$\mathcal{L}_{\theta}$',  # Orange line
        r'$\mathcal{L}_{v_r}$',  # Green line
        r'$\mathcal{L}_{v_{\theta}}$',  # Red line
        r'$\omega_{m}\mathcal{L}_{m}$',  # Purple line
        r'$\omega_{o}\mathcal{L}_{0}$',  # Brown line
        r'Total Loss $\mathcal{L}$',  # Black line
    ]

    total_loss = np.zeros(unique_arrays_loss[:, 0].shape)
    for i in range(len(unique_arrays_loss[0])):
        total_loss += unique_arrays_loss[:, i]
    plt.plot(unique_arrays_steps, total_loss, color='black', label=labels[6])


    for i in range(len(unique_arrays_loss[0])):
        plt.plot(unique_arrays_steps, unique_arrays_loss[:,i], label=labels[i])

    plt.yscale('log')
    plt.xlabel("Iterations", fontsize=16)
    plt.ylabel(r'Loss $\mathcal{L}_i$ [-]', fontsize=16)
    plt.title("Individual loss terms", fontsize=16)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right')
    plt.savefig(f"plot_loss.png", dpi=dpi_setting)
    plt.grid(True, alpha=0.5)
    # plt.show()
    # plt.close()

def plot_loss_T2(losshistory, mass=True):
    plt.figure()
    unique_arrays_loss = losshistory.loss_train #.trainset(tuple(row) for row in losshistory.loss_train)
    unique_arrays_loss = np.array(list(unique_arrays_loss))
    unique_arrays_steps = losshistory.steps #list(set(losshistory.steps))
    unique_arrays_steps.sort()

    labels = [
        r'$\mathcal{L}_r$',  # Blue line
        r'$\mathcal{L}_{\theta}$',  # Orange line
        r'$\mathcal{L}_{v_r}$',  # Green line
        r'$\mathcal{L}_{v_{\theta}}$',  # Red line
        r'$\omega_{m}\mathcal{L}_{m}$',  # Purple line
        r'$\mathcal{L}_r$',  # Blue line
        r'$\mathcal{L}_{\theta}$',  # Orange line
        r'$\mathcal{L}_{v_r}$',  # Green line
        r'$\mathcal{L}_{v_{\theta}}$',  # Red line
        r'$\omega_{m}\mathcal{L}_{m}$',  # Purple line
        r'$\omega_{o}\mathcal{L}_{0}$',
        r'Total Loss $\mathcal{L}$',  # Black line
    ]

    total_loss = np.zeros(unique_arrays_loss[:, 0].shape)
    for i in range(len(unique_arrays_loss[0])):
        total_loss += unique_arrays_loss[:, i]
    plt.plot(unique_arrays_steps, total_loss, color='black', label=labels[-1])


    for i in range(len(unique_arrays_loss[0])):
        plt.plot(unique_arrays_steps, unique_arrays_loss[:,i], label=labels[i])

    plt.yscale('log')
    plt.xlabel("Iterations", fontsize=16)
    plt.ylabel(r'Loss $\mathcal{L}_i$ [-]', fontsize=16)
    plt.title("Individual loss terms", fontsize=16)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right')
    plt.savefig(f"plot_loss.png", dpi=dpi_setting)
    plt.grid(True, alpha=0.5)

def plot_loss(losshistory_loaded, config, save=True):
    steps = losshistory_loaded[:, 0]
    plt.figure()
    arrays_loss = losshistory_loaded[:, 1:len(config['loss_weights'])+1] # make sure same amount of loss terms as weights
    arrays_loss = np.array(list(arrays_loss))
    arrays_steps = steps #list(set(losshistory.steps))
    arrays_steps.sort()

    labels = [
        r'$\mathcal{L}_r$',  # Blue line
        r'$\mathcal{L}_{\theta}$',  # Orange line
        r'$\mathcal{L}_{v_r}$',  # Green line
        r'$\mathcal{L}_{v_{\theta}}$',  # Red line
        r'$\omega_{m}\mathcal{L}_{m}$',  # Purple line
        r'$\omega_{ga}\mathcal{L}_{ga}$',  # Brown line
        r'$\omega_{o}\mathcal{L}_{0}$',  # Pink line
        r'Total Loss $\mathcal{L}$',  #
    ]

    total_loss = np.zeros(arrays_loss[:, 0].shape)
    for i in range(len(arrays_loss[0])):
        total_loss += arrays_loss[:, i]
    plt.plot(arrays_steps, total_loss, color='black', label=labels[-1])

    for i in range(len(arrays_loss[0])):
        if np.all(arrays_loss[:, i] == 0):
            labels[i] = labels[i] + " = 0"
        plt.plot(arrays_steps, arrays_loss[:,i], label=labels[i])

    plt.yscale('log')
    plt.xlabel("Iterations", fontsize=16)
    plt.ylabel(r'Loss $\mathcal{L}_i$ [-]', fontsize=16)
    plt.title("Individual loss terms", fontsize=16)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right')
    plt.grid(True, alpha=0.5)

    if save:
        time_library.sleep(0.1)
        timestamp_file = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        plt.savefig(f"Saved_plots/{config['run_id_number']}/plot_loss{timestamp_file}.png", dpi=dpi_setting)

def plot_loss_resampled_vs_non_resampled(losshistory_loaded_list, config):
    plt.figure()

    for idx, losshistory_loaded in enumerate(losshistory_loaded_list):
        steps = losshistory_loaded[:, 0]
        amount_of_lost_terms = len(config['loss_weights'])

        arrays_loss_train_data = losshistory_loaded[:, 1 : amount_of_lost_terms+1]
        arrays_loss_train_data = np.array(list(arrays_loss_train_data))
        array_loss_test_data = losshistory_loaded[:, amount_of_lost_terms+1: amount_of_lost_terms+1+amount_of_lost_terms ]
        arrays_loss_test_data = np.array(list(array_loss_test_data))

        total_loss_train = np.sum(arrays_loss_train_data, axis=1)  # Summing across loss terms
        total_loss_test = np.sum(arrays_loss_test_data, axis=1)  # Summing across loss terms
        plt.plot(steps, total_loss_train, label=f"Train {idx + 1}")
        plt.plot(steps, total_loss_test, label=f"Test {idx + 1}")

    plt.yscale('log')
    plt.xlabel("Iterations", fontsize=16)
    plt.ylabel(r'Total Loss $\mathcal{L}$ [-]', fontsize=16)
    plt.title("Total Loss Per Dataset", fontsize=16)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right')
    plt.savefig("plot_total_loss.png", dpi=dpi_setting)
    plt.grid(True, alpha=0.5)
    # plt.show()

def plot_metrics_best_iteration_vs_time(Dr, Dv, Dm, fuel_used, time, config, save=True):
    fig, axes = plt.subplots(3, 1, figsize=(20, 12), gridspec_kw={'height_ratios': [5, 5, 5]}, sharex=True)
    fig.subplots_adjust(hspace=0, wspace=0.3)
    fig.suptitle(f"Metrics at final iteration", fontsize=16, y=0.98)

    metrics = [Dr, Dv, Dm, fuel_used]
    labels = ['Dr', 'Dv', 'Dm', 'Fuel used']
    for i in range(3):
        if labels:
            label = labels[i]
        else:
            label = f"metric$_{i + 1}$"

        axes[i].plot(time, metrics[i], label=f"{label}")

    # Add the scalar Final_fuel_used value to the bottom-right corner of the last plot
    axes[-1].text(0.95, 0.05, f"Fuel used: {fuel_used}",
                  transform=axes[-1].transAxes,  # Use axis coordinates
                  fontsize=16, verticalalignment='bottom', horizontalalignment='right',
                  bbox=dict(facecolor='white', alpha=0.5))  # Add a background box for better visibility

    # Add labels and other plot customizations
    axes[-1].set_xlabel("Time [-]", fontsize=16)

    if save:
        time_library.sleep(0.1)
        timestamp_file = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        plt.savefig(f"Saved_plots/{config['run_id_number']}/plot_metrics_best_iteration_vs_time{timestamp_file}.png", dpi=dpi_setting)

def plot_metrics_vs_iterations(metrics_loaded, config, save=True):
    steps = metrics_loaded[:, 0]
    final_dr_list, final_dv_list, final_dm_list, fuel_used_list = metrics_loaded[:, 1], metrics_loaded[:, 2], metrics_loaded[:, 3], metrics_loaded[:, 4]
    metrics_list = final_dr_list, final_dv_list, final_dm_list, fuel_used_list
    metrics_list = tuple(map(list, metrics_list))

    # Define a list of colors to cycle through for each lowest value line
    line_colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown']

    # Determine the iterations where the minimum value occurs for each metric
    min_iterations = []
    for metric_values in metrics_list:
        min_value = min(metric_values)
        min_index = metric_values.index(min_value)
        min_iterations.append(steps[min_index])

    fig, axes = plt.subplots(len(metrics_list), 1, figsize=(8, 2 * len(metrics_list)), sharex=True)
    fig.subplots_adjust(hspace=0.1)
    fig.suptitle("Metrics vs Iterations", fontsize=16, y=0.95)

    # Create custom legend entries for each metric's lowest value line
    legend_elements = [
        Line2D([0], [0], color=line_colors[i % len(line_colors)], linestyle='--', linewidth=1,
               label=f"Lowest {config['metrics'][i]}")
        for i in range(len(metrics_list))
    ]

    for i, ax in enumerate(axes):
        metric_values = metrics_list[i]
        ax.plot(steps, metric_values, label=config['metrics'][i])

        # Find the minimum value for this metric to display in the text box
        min_value = min(metric_values)

        # Add text with final and lowest values for this plot
        ax.text(
            0.98, 0.92,
            f"Final value: {np.round(metric_values[-1], 6)} \nLowest value: {np.round(min_value, 6)}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.8')
        )

        # Draw a vertical line at each lowest value iteration across all plots
        for j, min_iter in enumerate(min_iterations):
            ax.axvline(x=min_iter, color=line_colors[j % len(line_colors)], linestyle='--', linewidth=1, alpha=0.5)

        # Set y-axis label and log scale
        ax.set_ylabel(config['metrics'][i], fontsize=16)
        ax.set_yscale("log")

    # Add a legend to the plot showing the colors associated with each metric's lowest line
    fig.legend(handles=legend_elements, loc='lower center', fontsize=10, ncol=4, bbox_to_anchor=(0.5, -0.005))

    # Set x-axis label
    axes[-1].set_xlabel("Iterations", fontsize=16)

    if save:
        time_library.sleep(0.1)
        timestamp_file = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        plt.savefig(f"Saved_plots/{config['run_id_number']}/plot_metrics_vs_iterations{timestamp_file}.png", dpi=dpi_setting)


def plot_variable_history(history, configs_list, variable_names=None, save=True):
    """
    Plot the history of each variable's values against epochs in separate subplots.

    Args:
        history (list of lists): A 2D list where each inner list contains the values
                                 of all variables at a specific epoch.
        variable_names (list of str, optional): List of names for the variables. If None,
                                                variables will be named as Var_1, Var_2, etc.
    """
    if not history:
        print("The history is empty. Nothing to plot.")
        return

    # Convert history to a 2D array for easy slicing
    history = np.array(history)
    epochs = np.arange(history.shape[0])

    # Generate default variable names if not provided
    if variable_names is None:
        variable_names = [f"Var_{i+1}" for i in range(history.shape[1])]

    # Create subplots for each variable
    num_vars = history.shape[1]
    fig, axes = plt.subplots(num_vars, 1, figsize=(10, 5 * num_vars))
    if num_vars == 1:
        axes = [axes]  # Ensure axes is always iterable

    for i, (ax, var_name) in enumerate(zip(axes, variable_names)):
        ax.plot(epochs, history[:, i], label=var_name, color='b')
        ax.set_xlabel("Iterations")
        ax.set_ylabel(f"{var_name}")
        # ax.set_title(f"Evolution of {var_name} Over Epochs")
        ax.grid(True)
        ax.legend()

        # Annotate the latest value on the graph
        latest_value = history[-1, i]
        ax.text(
            0.98, 0.92,
            f"Latest value: {np.round(latest_value, 4)}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.8')
        )

    plt.tight_layout()

    if save:
        time_library.sleep(0.1)
        timestamp_file = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        plt.savefig(f"Saved_plots/{configs_list[0]['run_id_number']}/plot_var_history{timestamp_file}.png", dpi=dpi_setting)


def plot_resampled_time_grids_evolution(time_grids):
    plt.figure(figsize=(10, 5))

    num_grids = len(time_grids)

    for i, grid in enumerate(time_grids):
        time_data = np.array(grid).flatten()  # Convert each grid to a 1D array
        offset = num_grids - i  # Reverse stacking (Grid 0 on top)
        plt.scatter(time_data, np.full_like(time_data, offset), marker='o', color='blue', alpha=0.7, label=f"Grid {i}")

    plt.xlabel("Value")
    plt.ylabel("Grid Index")
    plt.yticks(np.arange(1, num_grids + 1), labels=[f"Grid {i}" for i in range(num_grids)][::-1])  # Reverse labels
    plt.title("Evolution of resampling time grids")

    plt.grid(axis='x', linestyle='--', alpha=0.5)



























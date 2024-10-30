import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import master_thesis_mitchell_functions as mtmf
import copy
import coordinate_transformations

dpi_setting = 300

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# plt.rcParams["figure.figsize"] = (12,6)
# plt.rcParams.update({'font.size': 14})
# plt.rcParams['lines.linewidth'] = 1.5
# plt.rcParams['lines.markersize'] = 3
# plt.rcParams["figure.autolayout"] = True
# plt.rcParams["axes.grid"] = True

def plot_trajectory_radialND_to_cartesianND(states, thrust_scale=0.05, r_target=1.5, r_start=1, lim=None, N_arrows=40, thrust=True, config=None, dpi_setting=300):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    theta = np.linspace(0, 2 * np.pi, 1000)

    # Plot the starting and target orbits using the ax object
    if r_start:
        ax.plot(r_start * np.cos(theta), r_start * np.sin(theta), label="Starting orbit", alpha=0.85, lw=1.25, ls='-', c='darkblue', zorder=1.0)
    if r_target:
        ax.plot(r_target * np.cos(theta), r_target * np.sin(theta), label="Target orbit", alpha=0.85, lw=1.25, ls='-', c='darkgoldenrod', zorder=1.0)

    y = coordinate_transformations.radial_to_NDcartesian(states, config)
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
                     y[i, 5] * 0.5 / scale, width=0.008, alpha=1.0, color='red', zorder=2.0)
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


def plot_compare_pcnn_tudat_states(pcnn_states,
                                   tudat_states,
                                   entries=6,
                                   control_entries=2,
                                   coordinates="NDcartesian",
                                   custom_labels=None,
                                   log=True):

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

    plt.savefig(f"plot_compare_pcnn_tudat_states.png", dpi=dpi_setting)
    # plt.show()

def plot_compare_pcnn_tudat_mass(pcnn_mass,
                                 tudat_mass,
                                 config):

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

    plt.savefig(f"plot_compare_pcnn_tudat_mass.png", dpi=dpi_setting)
    # plt.show()

def plot_states(t, train_state):
    fig, axes = plt.subplots(7, 1, figsize=(12, 12), sharex=True)

    # Labels for the states and their derivatives
    labels = ['r_pred', 'theta_pred', 'v_r_pred', 'v_theta_pred', 'u_r_pred', 'u_tangential_pred', 'm_pred']
    # labels_dot = ['r_pred_dot', 'theta_pred_dot', 'v_r_pred_dot', 'v_theta_pred_dot', 'u_phi_pred_dot', 'u_T_pred_dot', 'm_pred_dot']
    for i in range(7):
        # Left column: states
        # axes[i, 0].plot(t, sol_pred[:, i], label=labels[i])
        axes[i].plot(t, train_state.best_y[:,i], label=labels[i])
        axes[i].set_ylabel(labels[i])
        axes[i].grid(True, alpha=0.5)
        axes[i].legend(loc='upper right')

        # # Right column: state derivatives
        # axes[i, 1].plot(t, [r_pred_dot, theta_pred_dot, v_r_pred_dot, v_theta_pred_dot, u_phi_pred_dot, u_T_pred_dot, m_pred_dot][i].numpy(), label=labels_dot[i])
        # axes[i, 1].set_ylabel(labels_dot[i])
        # axes[i, 1].grid(True)
        # axes[i, 1].legend(loc='upper right')

    # Set a common x-label for the last row
    axes[0].set_xlabel("Time")
    # axes[-1, 1].set_xlabel("Time")

    # Set a common title for the entire figure
    fig.suptitle("Predicted States ", fontsize=16)

    # Adjust layout and save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the title
    plt.savefig(f"plot_states.png", dpi=dpi_setting)
    # plt.show()
    # plt.close()

def plot_loss(losshistory):
    plt.figure()
    unique_arrays_loss = losshistory.loss_train #.trainset(tuple(row) for row in losshistory.loss_train)
    unique_arrays_loss = np.array(list(unique_arrays_loss))
    unique_arrays_steps = losshistory.steps #list(set(losshistory.steps))
    unique_arrays_steps.sort()

    labels = [
        r'Total Loss $\mathcal{L}$',  # Black line
        r'$\mathcal{L}_r$',  # Blue line
        r'$\mathcal{L}_{\theta}$',  # Orange line
        r'$\mathcal{L}_{v_r}$',  # Green line
        r'$\mathcal{L}_{v_{\theta}}$',  # Red line
        r'$\omega_{m}\mathcal{L}_{m}$',  # Purple line
        r'$\omega_{o}\mathcal{L}_{0}$'  # Brown line
    ]
    total_loss = np.zeros(unique_arrays_loss[:,0].shape)
    for i in range(len(unique_arrays_loss[0])):
        plt.plot(unique_arrays_steps, unique_arrays_loss[:,i], label=labels[i])

        total_loss += unique_arrays_loss[:,i]

    plt.plot(unique_arrays_steps, total_loss, color='black', label=labels[6])
    plt.yscale('log')
    plt.xlabel("Iterations", fontsize=16)
    plt.ylabel(r'Loss $\mathcal{L}_i$ [-]', fontsize=16)
    plt.title("Individual loss terms", fontsize=16)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right')
    plt.savefig(f"plot_loss.png", dpi=dpi_setting)
    plt.grid(True, alpha=0.5)
    # plt.show()
    # plt.close()

def plot_metrics_best_iteration_vs_time(Dr, Dv, Dm, fuel_used, time):
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
    plt.savefig(f"plot_metrics_best_iteration_vs_time.png", dpi=dpi_setting)

def plot_metrics_vs_iterations(losshistory, config):
    metrics_list  = mtmf.calculate_metrics_all_iterations(losshistory, config)

    fig, axes = plt.subplots(4, 1, figsize=(8, 2 * 4), sharex=True)
    fig.subplots_adjust(hspace=0.1)
    fig.suptitle("Metrics vs Iterations", fontsize=16, y=0.95)

    for i, ax in enumerate(axes):
        ax.plot(losshistory.steps, metrics_list[i], label=config['metrics'][i])
        ax.text(0.98, 0.92, f"Final value: {np.round(metrics_list[i][-1], 4)} \n  Lowest value: {np.round(min(metrics_list[i]), 4)}",
                     transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
                     bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.8'))

        ax.set_ylabel(config['metrics'][i], fontsize=16)
        ax.set_yscale("log")

    axes[-1].set_xlabel("Iterations", fontsize=16)

    plt.savefig(f"plot_metrics_vs_iterations.png", dpi=dpi_setting)































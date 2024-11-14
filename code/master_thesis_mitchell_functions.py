'''
This file contains several helper functions for the thesis.
'''

import deepxde as dde
import tensorflow as tf
import numpy as np
# import matplotlib
# matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
dpi_setting = 400
import tensorflow_probability as tfp
import coordinates_transformation_functions
import verification
from model_pcnn import ModelPCNN, TimeDomain_with_std
from datetime import datetime
import os
import plots
# from basic_pcnn import config, MU, m0, a, umax, isp, t0, tfinal, t_scale, length_scale, initial_state, final_state




def cart2pol_position(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart_position(rho, theta):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return(x, y)

def pol2cart_thrust_angle(theta, u_phi):
    thrust_angle_cart = theta + (0.5*np.pi - u_phi)
    return thrust_angle_cart

def generate_T_perturbed_batch(t0, tf, M):
    delta_t = (tf - t0) / M
    base_times = np.linspace(t0, tf, M + 1)
    perturbed_times = np.random.normal(loc=base_times, scale=0.2 * delta_t)
    return perturbed_times

def get_num_files(folder_path):
    # Get the number of files in the specified folder
    num_files = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
    return num_files
def read_file(file_path):
    # Read the .dat file into an array, skipping the header (the first line starting with #)
    data = np.loadtxt(file_path, delimiter=' ', skiprows=1)
    return data

def control_nodes_ref_times_3D_initial_state(states, config, mass=True): #states needs to be in NDcartesian
    t = np.linspace(0, config['tfinal'] / config['t_scale'], config['M'])
    t_reshaped = t.reshape(-1, 1)

    if mass==True:
        states = np.concatenate((t_reshaped, states[:, :-1]), axis=1)
    else:
        states = np.concatenate((t_reshaped, states), axis=1)

    control_entries = 2
    coordinates = 'radial'
    # Transform to normal cartesian, instead of ND cartesian
    transformation = getattr(coordinates_transformation_functions, f"{coordinates}_to_cartesian")
    cartesian_states = transformation(states, config)

    # Prepare control and initial state
    control = cartesian_states[:, -control_entries:]
    control_nodes = {key: value for key, value in zip(cartesian_states[:, 0], control)}
    initial_state = cartesian_states[0, 1:-control_entries].reshape(-1, 1)

    # Make 3D initial state
    if initial_state.shape[0] == 4:
        initial_state = np.concatenate((initial_state[0:2, :],
                                        np.array([[0]]),
                                        initial_state[2:4, :],
                                        np.array([[0]])),
                                        axis=0).reshape(-1, 1)

    ref_times = cartesian_states[:, 0]

    return control_nodes, ref_times, initial_state

def calculate_metrics_best_iteration(pcnn_states, tudat_states, pcnn_mass, tudat_mass, config=None, calculate_fuel='tudat'):
    dr = np.linalg.norm(pcnn_states[:,1:3] - tudat_states[:,1:3], axis=1).reshape(-1, 1)
    dv = np.linalg.norm(pcnn_states[:,3:5] - tudat_states[:,3:5], axis=1).reshape(-1, 1)
    dm = abs((pcnn_mass[:,1] - tudat_mass[:,1])).reshape(-1, 1) # not tudat_mass[:,0] because 0 index is time column

    # whether it's tudat of pcnn should not matter right..?
    if calculate_fuel == 'tudat':
        fuel_used = calculate_fuel_used(None, tudat_states, config=config)
    if calculate_fuel == 'pcnn':
        fuel_used = calculate_fuel_used(pcnn_states, None, config=config)

    time = pcnn_states[:,0]

    return dr, dv, dm, fuel_used, time

def calculate_metrics_all_iterations(losshistory, config=None, save=False, fname=None):
    y = np.array(list(losshistory.y_pred_test))

    t = np.linspace(0, config['tfinal'] / config['t_scale'], config['M'])
    t_reshaped = t.reshape(-1, 1)

    final_dr_list = []
    final_dv_list = []
    final_dm_list = []
    fuel_used_list= []

    for i in range(len(losshistory.y_pred_test)):
        y_pred_test_i = losshistory.y_pred_test[i]
        control_nodes, ref_times, initial_state = control_nodes_ref_times_3D_initial_state(y_pred_test_i, config, mass=config['mass'])
        verification_object = verification.Verification(config['m0'], config['t0'], config['tfinal'], initial_state, config['isp'], central_body="Sun", control_nodes=control_nodes,
                                                        verbose=True, ref_times=ref_times, mass_rate=True, config=config)
        verification_object.integrate()

        tudat_states_cartesian      = verification_object.states_tudat # TIME IN FIRST COLUMN
        tudat_mass                  = verification_object.mass

        # Nondimensionalize - tudat states
        tudat_states_NDcartesian    = coordinates_transformation_functions.cartesian_to_NDcartesian(tudat_states_cartesian, config)
        # convert to cartesian (ND) - y_pred_test_i
        y_pred_test_i_with_time     = np.concatenate((t_reshaped, y_pred_test_i), axis = 1)
        y_pred_test_i_NDcartesian   = coordinates_transformation_functions.radial_to_NDcartesian(y_pred_test_i_with_time, config)

        dr = np.linalg.norm(y_pred_test_i_NDcartesian[:,1:3] - tudat_states_NDcartesian[:,1:3], axis=1).reshape(-1, 1);     final_dr = dr[-1][0]
        dv = np.linalg.norm(y_pred_test_i_NDcartesian[:,3:5] - tudat_states_NDcartesian[:,3:5], axis=1).reshape(-1, 1);     final_dv = dv[-1][0]
        fuel_used = calculate_fuel_used(None, tudat_states_NDcartesian=tudat_states_NDcartesian, config=config)

        final_dr_list. append(final_dr)
        final_dv_list. append(final_dv)
        fuel_used_list.append(fuel_used)

        if config['mass']==True:
            dm = abs((y_pred_test_i[:, -1] - tudat_mass[:, 1])).reshape(-1, 1);     final_dm = dm[-1][0]  # not tudat_mass[:,0] because 0 index is time column
            final_dm_list.append(final_dm)

    if save:
        metrics_list = final_dr_list, final_dv_list, final_dm_list, fuel_used_list
        save_metrics_history(losshistory, metrics_list, fname, verbose=True)

    if config['mass']==True:
        return final_dr_list, final_dv_list, final_dm_list, fuel_used_list
    else:
        return final_dr_list, final_dv_list, fuel_used_list

def calculate_fuel_used(pcnn_states_NDcartesian=None, tudat_states_NDcartesian=None, config=None):
    if tudat_states_NDcartesian is not None:
        t = tf.reshape(tudat_states_NDcartesian[:, 0], (1, -1))[0] * config['t_scale']
        U = tudat_states_NDcartesian[:, 5:7];    U_norm = tf.norm(U, axis=1)
    if pcnn_states_NDcartesian is not None:
        t = tf.reshape(pcnn_states_NDcartesian[:, 0], (1, -1))[0] * config['t_scale']
        U = pcnn_states_NDcartesian[:, 5:7];    U_norm = tf.norm(U, axis=1)

    # Sort time and control
    idx = tf.argsort(t)
    t_sorted = tf.gather(t, idx)
    U_norm_sorted = tf.gather(U_norm, idx)

    # Propellent mass
    propellent_mass = (1 / config['isp'] / 9.81) * tfp.math.trapz(U_norm_sorted, t_sorted)
    propellent_mass_np = propellent_mass.numpy()
    return propellent_mass_np

def create_model(config, pde, constraint_layer, seed, train_distribution="uniform", std=None):
    geom = dde.geometry.TimeDomain(config['t0'] / config['t_scale'], config['tfinal'] / config['t_scale'], sampler_std=std)
    data = dde.data.PDE(geom, pde, [], config['N_train'], 2, num_test=config['N_train'], train_distribution=train_distribution)

    # Override with manually generated, sorted points, otherwise first 2 indices are the boundary points [0, 17,21, 0,086,...]
    # This should not matter, but just to be sure
    # time_points = np.linspace(config['t0'] / config['t_scale'], config['tfinal'] / config['t_scale'], config['N_train'], dtype=dde.config.real(np)).reshape(-1, 1)
    # data.train_x = time_points  # Ensure train_x is sorted and uniformly spaced

    # Overide the get test data function that includes boundary points
    test_data = np.linspace(config['t0'] / config['t_scale'], config['tfinal'] / config['t_scale'], config['N_test'], dtype=dde.config.real(np)).reshape(-1, 1)
    def new_test(self):
        return test_data, None, None
    data.test = new_test.__get__(data, dde.data.PDE)

    initializer = tf.keras.initializers.GlorotNormal(seed=seed) # FNN 20241022022825       PFNN 20241015143854
    net = dde.nn.PFNN(config["layer_architecture_PFNN"], "sin", initializer)
    net.apply_output_transform(constraint_layer)
    model = ModelPCNN(data, net, config['loss_weights'])
    return model

def restarter(config, pde, constraint_layer, lr_schedule, train_distribution="uniform", std=None, plot=True, save=False, N_attempts=40,
              save_folder="restarter_runs", run_id_number=0, max_succesful_attempts=np.inf):
    attempt = 1
    succesful_attempts = 0
    print('RESTARTER SCHEDULE\n'
          '------------------')
    while attempt <= N_attempts:
        print("Initialisation attempt:", attempt);  seed = int(datetime.now().strftime("%Y%m%d%H%M%S"));    print("time-dependent random seed:", seed)

        model = create_model(config, pde, constraint_layer, seed, train_distribution=train_distribution, std=std)

        lr_schedule = lr_schedule
        for (lr, iterations) in lr_schedule:
            print("Learning rate=", lr, "Iterations=", iterations)

            optimisation_alg = tf.keras.optimizers.Adam(learning_rate=lr)
            model.compile(optimisation_alg, lr=lr)
            losshistory, train_state = model.train(iterations=iterations, display_every=500)  # , callbacks=[checkpoint_cb])

            # RESTART
            if np.isnan(np.sum(losshistory.loss_test[-1])) or np.sum(losshistory.loss_test[-1]) > 5.0:
                print("Loss not below threshold, restarting now...")
                break

        if not np.isnan(np.sum(losshistory.loss_test[-1])):
            # # Set save=False and use lines below to save loss;  also comment out the restart if condition
            # output_dir = f'{save_folder}/{run_id_number}';  os.makedirs(output_dir, exist_ok=True)
            # fname_loss = f'{output_dir}/run_{attempt}_loss.dat'
            # save_loss_history(losshistory, fname_loss, verbose=True)

            if np.sum(losshistory.loss_test[-1]) < 5.0:
                succesful_attempts += 1
                if save:
                    output_dir = f'{save_folder}/{run_id_number}';  os.makedirs(output_dir, exist_ok=True)

                    fname_loss    = f'{output_dir}/successful_run_{succesful_attempts}_loss.dat'
                    fname_metrics = f'{output_dir}/successful_run_{succesful_attempts}_metrics.dat'

                    save_loss_history(losshistory, fname_loss, verbose=True)
                    calculate_metrics_all_iterations(losshistory, config, save=True, fname=fname_metrics)

                if plot:
                    pass

                if succesful_attempts >= max_succesful_attempts:
                    attempt = N_attempts+1

            attempt += 1


        print("amount of succesful attempts so far:", succesful_attempts)
    print("amount of succesful attempts in total:", succesful_attempts)

def single_run(config, pde, constraint_layer, lr_schedule, train_distribution="uniform", std=None, plot=True, save=False, seed=None):
    if seed == None:
        seed = int(datetime.now().strftime("%Y%m%d%H%M%S"));    print("time-dependent random seed:", seed)
    else:
        print("manually entered seed:", seed)

    model = create_model(config, pde, constraint_layer, seed=seed, train_distribution=train_distribution, std=std) # PFNN 20241015143854

    for (lr, iterations) in lr_schedule:
        print("Learning rate=", lr, "Iterations=", iterations)

        optimisation_alg = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimisation_alg, lr=lr)
        losshistory, train_state = model.train(iterations=iterations, display_every=1000)  # , callbacks=[checkpoint_cb])

        if np.isnan(np.sum(losshistory.loss_test[-1])) or np.sum(losshistory.loss_test[-1]) > 5.0:
            print("Loss not below threshold, continuing anyway...")

    if save:
        save_loss_history(losshistory, f'loss.dat', verbose=True)
        metrics_lists = calculate_metrics_all_iterations(losshistory, config)
        save_metrics_history(losshistory, metrics_lists, f'metrics.dat', verbose=True)
    if plot:
        pass

    return losshistory, train_state

def save_loss_history(loss_history, fname, verbose = True):
    # Copied this function from external.py from the DeepXDE library
    if verbose:
        print("Saving loss history to {} ...".format(fname))
    loss = np.hstack(
        (
            np.array(loss_history.steps)[:, None],
            np.array(loss_history.loss_train),
            np.array(loss_history.loss_test),
            np.array(loss_history.metrics_test),
        )
    )
    np.savetxt(fname, loss, header="iteration, loss_train, loss_test, metrics_test")

def save_metrics_history(loss_history, metrics, fname, verbose = True, mass=None):
    # Copied this function from external.py from the DeepXDE library
    if verbose:
        print("Saving metrics history to {} ...".format(fname))

    if mass==True:
        metrics = np.hstack(
            (
                np.array(loss_history.steps)[:, None],
                np.array(metrics[0])[:, None],
                np.array(metrics[1])[:, None],
                np.array(metrics[2])[:, None],
                np.array(metrics[3])[:, None],
            )
        )
        np.savetxt(fname, metrics, header="step, FinalDr, FinalDv, FinalDm, Fuel used")

    if mass==False:
        metrics = np.hstack(
            (
                np.array(loss_history.steps)[:, None],
                np.array(metrics[0])[:, None],
                np.array(metrics[1])[:, None],
                np.array(metrics[2])[:, None],
            )
        )
        np.savetxt(fname, metrics, header="step, FinalDr, FinalDv, Fuel used")


def verify_run(states, losshistory, config, showplot=True, saveplots=True):
    t = np.linspace(0, config['tfinal'] / config['t_scale'], config['M'])
    t_reshaped = t.reshape(-1, 1)

    # save time+mass seperately
    pcnn_mass = np.concatenate((t_reshaped, states[:, -1].reshape(-1, 1)), axis=1)
    # save ND states (with time and without mass)
    states_without_mass_ND = np.concatenate((t_reshaped, states[:, :-1]), axis=1)
    states_without_mass_NDcartesian = coordinates_transformation_functions.radial_to_NDcartesian(states_without_mass_ND, config)
    states_without_mass_NDcartesian_dict = {"NDcartesian": states_without_mass_NDcartesian}

    control_nodes, ref_times, initial_state = control_nodes_ref_times_3D_initial_state(states, config)


    verification_object = verification.Verification(config['m0'], config['t0'], config['tfinal'], initial_state, config['isp'], central_body="Sun", control_nodes=control_nodes,
                                                    verbose=True, ref_times=ref_times, mass_rate=True, config=config)
    verification_object.integrate()


    tudat_states_cartesian = verification_object.states_tudat
    tudat_mass = verification_object.mass
    tudat_states_NDcartesian = coordinates_transformation_functions.cartesian_to_NDcartesian(tudat_states_cartesian, config)
    tudat_states_NDcartesian_dict = {"NDcartesian": tudat_states_NDcartesian}

    if showplot:
        custom_labels = ["$r$", "$\\theta$", "$v_{r}$", "$v_{\\theta}$"]
        plots.plot_compare_pcnn_tudat_states(states_without_mass_NDcartesian_dict, tudat_states_NDcartesian_dict, custom_labels=custom_labels, log=False, save=saveplots)
        plots.plot_compare_pcnn_tudat_mass(pcnn_mass, tudat_mass, config=config, save=saveplots)

        Dr, Dv, Dm, fuel_used, time_interval = calculate_metrics_best_iteration(states_without_mass_NDcartesian, tudat_states_NDcartesian, pcnn_mass, tudat_mass, config=config)
        plots.plot_metrics_best_iteration_vs_time(Dr, Dv, Dm, fuel_used, time_interval, save=saveplots)
        plots.plot_metrics_vs_iterations(losshistory, config, save=saveplots)

        plt.show()


def verify_basic_pcnn(run_id_number):
    files_location = f'restarter_runs/{run_id_number}'
    num_files = get_num_files(files_location)

    def metrics_box_plot():
        min_dr_values, min_dv_values, min_dm_values, min_fuel_values = [], [], [], []
        for i in range(int(num_files/2)-1):
            file_path = f"{files_location}/successful_run_{i + 1}_metrics.dat"
            metrics_array = read_file(file_path)

            iterations = metrics_array[:,0]
            dr_list = metrics_array[:,1]   ;    min_dr_values.append(min(dr_list))
            dv_list = metrics_array[:,2]   ;    min_dv_values.append(min(dv_list))
            dm_list = metrics_array[:,3]   ;    min_dm_values.append(min(dm_list))
            fuel_list = metrics_array[:,4] ;    min_fuel_values.append(min(fuel_list))

        # Define boxplot color properties for blue boxplots
        # boxplot_props = {
        #     "boxprops": dict(color="blue"),
        #     "medianprops": dict(color="orange"),
        #     "whiskerprops": dict(color="blue"),
        #     "capprops": dict(color="blue"),
        #     "flierprops": dict(markeredgecolor="blue")
        # }
        # Creating subplots for each metric
        fig, axs = plt.subplots(4, 1, figsize=(1.7, 12), sharex=True, gridspec_kw={'hspace': 0.1})

        # Plotting each boxplot in a separate subplot with filled boxes
        axs[0].boxplot(min_dr_values, patch_artist=True)  # , **boxplot_props)
        axs[0].set_ylabel('dx [AU]');
        axs[0].set_yscale('log');
        axs[0].set_ylim(0.003, 100)
        axs[1].boxplot(min_dv_values, patch_artist=True)  # , **boxplot_props)
        axs[1].set_ylabel('dv [$V_⊕$]');
        axs[1].set_yscale('log');
        axs[1].set_ylim(0.003, 90)
        axs[2].boxplot(min_dm_values, patch_artist=True)  # , **boxplot_props)
        axs[2].set_ylabel('dm [Kg]');
        axs[2].set_yscale('log');
        axs[2].set_ylim(0.0002, 130)
        axs[3].boxplot(min_fuel_values, patch_artist=True)  # , **boxplot_props))
        axs[3].set_ylabel('Fuel [Kg]');
        axs[3].set_ylim(0, 100)


        # Adding x-axis label at the bottom
        plt.xlabel('[1, 10, 10, 10, 1]', rotation=270)

        # Adjust layout to prevent overlapping
        # plt.tight_layout()
        plt.subplots_adjust(left=0.4, right=0.95, top=0.95, bottom=0.2)

        plt.savefig(f'{files_location}/metrics_box_plot', dpi=dpi_setting)
        plt.show()

        print(np.round(np.median(min_dr_values),4), np.round(np.median(min_dv_values),4), np.round(np.median(min_dm_values),4), np.round(np.median(min_fuel_values),4))
    metrics_box_plot()

    def plot_total_loss(): # CAN ONLY BE USED WHEN METRICS ARE NOT SAVED AND PLOTTED -> CHANGE THIS IN RESTARTER FUNCTION
        total_loss_each_run = []
        for i in range(int(num_files)):
            file_path = f"{files_location}/run_{i + 1}_loss.dat"
            loss_array = read_file(file_path)

            iterations_array = loss_array[:, 0]
            all_loss_terms = loss_array[:, 7:]
            total_loss = np.linalg.norm(all_loss_terms, axis=1)
            total_loss_each_run.append(total_loss)

        plt.figure(figsize=(12, 9))
        for k in range(int(num_files)):
            if total_loss_each_run[k][3] < 5.0:
                color = 'green'
            elif total_loss_each_run[k][-1] < 5.0:
                color = 'orange'
            else:
                color = 'red'

            plt.plot(iterations_array, total_loss_each_run[k], c=color)

        plt.yscale('log')
        plt.xlabel("Iterations", fontsize=16)
        plt.ylabel(r'Total train loss $\mathcal{L}$ [-]', fontsize=16)
        plt.title("Loss evolution LR schedule 2 (No restart)", fontsize=16)
        plt.savefig(f'{files_location}/total_losses_plot', dpi=dpi_setting)
        plt.show()
    # plot_total_loss()























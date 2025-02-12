'''
This file contains several helper functions for the thesis.
'''

import deepxde as dde
import tensorflow as tf
import numpy as np
import math
# import matplotlib
# matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
dpi_setting = 400
import tensorflow_probability as tfp
import coordinates_transformation_functions
import verification
from model_pcnn import ModelPCNN, ModelPCNN_14, TimeDomain_with_std
from datetime import datetime
import os
import plots
from tudatpy.kernel.interface import spice
import adaptive_sampling_algs
# from basic_pcnn import config, MU, m0, a, umax, isp, t0, tfinal, t_scale, length_scale, initial_state, final_state

dde.config.set_default_float("float64")
tf.keras.backend.set_floatx('float64')

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

def control_nodes_ref_times_3D_initial_state(states, config, time_grid, mass=True): #states needs to be in NDcartesian
    t = time_grid
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
    control = cartesian_states[:, 5:7]
    control_nodes = {key: value for key, value in zip(cartesian_states[:, 0], control)}
    initial_state = cartesian_states[0, 1:5].reshape(-1, 1)

    # Make 3D initial state
    if initial_state.shape[0] == 4:
        initial_state = np.concatenate((initial_state[0:2, :],
                                        np.array([[0]]),
                                        initial_state[2:4, :],
                                        np.array([[0]])),
                                        axis=0).reshape(-1, 1)

    ref_times = cartesian_states[:, 0]

    # because indexes need to be hardcoded, I programmed a warning here since once I spent 3 hours finding this mistake
    if (control_nodes[0.0][0] > 100) or (control_nodes[0.0][1] > 100):
        raise ValueError("The indexes of the control nodes are hardcoded and are therefore incorect!!!!!!!!!!")

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

def calculate_metrics_all_iterations(losshistory, time_grid, config):
    # y = np.array(list(losshistory.y_pred_test))

    t = time_grid
    t_reshaped = t.reshape(-1, 1)

    final_dr_list = []
    final_dv_list = []
    final_dm_list = []
    fuel_used_list= []

    for i in range(len(losshistory.y_pred_test)):
        y_pred_test_i = losshistory.y_pred_test[i]
        control_nodes, ref_times, initial_state = control_nodes_ref_times_3D_initial_state(y_pred_test_i, config, time_grid, mass=config['mass'])
        m0_leg_x = losshistory.y_pred_test[-1][-1][6] #index is last epoch / last timepoint / index 6 = mass
        verification_object = verification.Verification(m0_leg_x, config['t0'], config['tfinal'], initial_state, config['isp'], central_body="Sun", control_nodes=control_nodes,
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

def create_model_and_data(config, pde, constraint_layer, time_grid_train, time_grid_test, seed, train_distribution="uniform", std=None):
    geom = dde.geometry.TimeDomain(config['t0'] / config['t_scale'], config['tfinal'] / config['t_scale'], sampler_std=std)
    data = dde.data.PDE(geom, pde, [], len(time_grid_train), 2, num_test=len(time_grid_train), train_distribution=train_distribution)
    data.train_x = time_grid_train  # Assign the custom time points
    def new_test(self):
        return time_grid_test, None, None
    data.test = new_test.__get__(data, dde.data.PDE)

    initializer = tf.keras.initializers.GlorotNormal(seed=seed) # FNN 20241022022825       PFNN 20241015143854
    net = dde.nn.FNN(config["layer_architecture_FNN"], "sin", initializer)
    net.apply_output_transform(constraint_layer)
    model = ModelPCNN(data, net, config['loss_weights'], config)
    return model, data

def create_model_sp2(configs_list, pde, constraint_layer, time_grid, seed, train_distribution="uniform", std=None):
    geom = dde.geometry.TimeDomain(configs_list[0]['t0'] / configs_list[0]['t_scale'], configs_list[0]['tfinal'] / configs_list[0]['t_scale'], sampler_std=std)
    data = dde.data.PDE(geom, pde, [], len(time_grid), 2, num_test=len(time_grid), train_distribution=train_distribution)
    data.train_x = time_grid  # Assign the custom time points

    # Override with manually generated, sorted points, otherwise first 2 indices are the boundary points [0, 17,21, 0,086,...]
    # This should not matter, but just to be sure
    # time_points = np.linspace(config['t0'] / config['t_scale'], config['tfinal'] / config['t_scale'], config['N_train'], dtype=dde.config.real(np)).reshape(-1, 1)
    # data.train_x = time_points  # Ensure train_x is sorted and uniformly spaced

    # # Overide the get test data function that includes boundary points
    # test_data = np.linspace(config['t0'] / config['t_scale'], config['tfinal'] / config['t_scale'], config['N_test'], dtype=dde.config.real(np)).reshape(-1, 1)
    def new_test(self):
        return time_grid, None, None
    data.test = new_test.__get__(data, dde.data.PDE)

    initializer = tf.keras.initializers.GlorotNormal(seed=seed) # FNN 20241022022825       PFNN 20241015143854
    net = dde.nn.FNN(configs_list[0]["layer_architecture_FNN"], "sin", initializer)
    net.apply_output_transform(constraint_layer)
    model = ModelPCNN_14(data, net, configs_list[0]['loss_weights'], configs_list)
    return model

def restarter(config, pde, constraint_layer, time_grid, lr_schedule, train_distribution="uniform", std=None, plot=True, save=False, N_attempts=40,
              save_folder="restarter_runs", run_id_number=0, max_succesful_attempts=np.inf):
    attempt = 1
    succesful_attempts = 0
    print('RESTARTER SCHEDULE\n'
          '------------------')
    while attempt <= N_attempts:
        print("Initialisation attempt:", attempt);  seed = int(datetime.now().strftime("%Y%m%d%H%M%S"));    print("time-dependent random seed:", seed)

        model, data = create_model_and_data(config, pde, constraint_layer, seed, train_distribution=train_distribution, std=std)

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
                    calculate_metrics_all_iterations(losshistory, time_grid, config, save=True, fname=fname_metrics)

                if plot:
                    pass

                if succesful_attempts >= max_succesful_attempts:
                    attempt = N_attempts+1

            attempt += 1


        print("amount of succesful attempts so far:", succesful_attempts)
    print("amount of succesful attempts in total:", succesful_attempts)

def single_run(config, pde, constraint_layer, time_grid, lr_schedule, GA_point=None, train_distribution="uniform", std=None, plot=True, save=False, seed=None):
    if seed == None:
        seed = int(datetime.now().strftime("%Y%m%d%H%M%S"));    print("time-dependent random seed:", seed)
    else:
        print("manually entered seed:", seed)

    model, data = create_model_and_data(config, pde, constraint_layer, time_grid, GA_point, seed=seed, train_distribution=train_distribution, std=std)  # PFNN 20241015143854

    for (lr, iterations) in lr_schedule:
        print("Learning rate=", lr, "Iterations=", iterations)

        # optimisation_alg = tf.keras.optimizers.Adam(learning_rate=lr)
        optimisation_alg = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
        model.compile(optimisation_alg, lr=lr)
        losshistory, train_state = model.train(iterations=iterations, display_every=1000)  # , callbacks=[checkpoint_cb])

        if np.isnan(np.sum(losshistory.loss_test[-1])) or np.sum(losshistory.loss_test[-1]) > 5.0:
            print("Loss not below threshold, continuing anyway...")

    if save:
        number = 1
        dde.utils.save_loss_history(losshistory, f'loss{number}.dat', verbose=True)
        dde.utils.save_best_state(train_state, f'train_data{number}.dat', f'test_data{number}.dat')
        metrics_lists = calculate_metrics_all_iterations(losshistory, time_grid, config)
        save_metrics_history(losshistory, metrics_lists, f'metrics{number}.dat', verbose=True)
    if plot:
        pass

    return losshistory, train_state

def single_run_with_restart(config, pde, constraint_layer, time_grid_train, time_grid_test, lr_schedule, final_state1, initial_state2, ga_trainable_variables=None, callbacks=None, configs_list=None,
                            adaptive_sampling=False, LBFGS=False, threshold=5.0, N_attempts=50, train_distribution="uniform", std=None, plot=True, save=False, seed=None):
    attempt = 1
    if seed == None:
        seed = int(datetime.now().strftime("%Y%m%d%H%M%S"));    print("time-dependent random seed:", seed)
    else:
        print("manually entered seed:", seed)

    while attempt <= N_attempts:
        print("Initialisation attempt:", attempt);  print("seed:", seed)
        resampled_time_grids_savelist = []
        # model = create_model_sp2(configs_list, pde, constraint_layer, time_grid, seed=seed, train_distribution=train_distribution, std=std)  # PFNN 20241015143854
        model, data = create_model_and_data(config, pde, constraint_layer, time_grid_train, time_grid_test, seed=seed, train_distribution=train_distribution, std=std)  # PFNN 20241015143854

        for (lr, iterations) in lr_schedule:
            print("Learning rate=", lr, "Iterations=", iterations)

            optimisation_alg = tf.keras.optimizers.Adam(learning_rate=lr)
            model.compile(optimisation_alg, lr=lr, external_trainable_variables=ga_trainable_variables)
            print('model external_trainable_variables:', model.external_trainable_variables)

            if not adaptive_sampling:
                losshistory, train_state = model.train(iterations=iterations, display_every=1000)#, callbacks=[callbacks])  # , callbacks=[checkpoint_cb])
            if adaptive_sampling:
                for i in range(int(iterations/1000)):
                    losshistory, train_state = model.train(iterations=1000, display_every=1000)
                    print('Resampling time grid')
                    time_grid_resampled = adaptive_sampling_algs.RAD(time_grid=time_grid_train, model=model, pde=constraint_layer, k=1, c=1)
                    data.replace_with_anchors(time_grid_resampled);     resampled_time_grids_savelist.append(time_grid_resampled)

            # RESTART
            if np.isnan(np.sum(losshistory.loss_test[-1])) or np.sum(losshistory.loss_test[-1]) > threshold:
                print("Loss not below threshold, restarting now...")
                attempt += 1
                seed = int(datetime.now().strftime("%Y%m%d%H%M%S"));

                break

        if LBFGS:
            model.compile("L-BFGS-B")
            losshistory, train_state = model.train()

        if not np.isnan(np.sum(losshistory.loss_test[-1])):
            if np.sum(losshistory.loss_test[-1]) < threshold:
                print("Successful run completed")
                break

    if save:
        if adaptive_sampling == True:
            number=1
        if adaptive_sampling == False:
            number=2
        dde.utils.save_loss_history(losshistory, f'loss{number}.dat', verbose=True)
        dde.utils.save_best_state(train_state, f'train_data{number}.dat', f'test_data{number}.dat')
        metrics_lists = calculate_metrics_all_iterations(losshistory, time_grid_test, config)
        save_metrics_history(losshistory, metrics_lists, f'metrics{number}.dat', verbose=True)
    if plot:
        pass

    return losshistory, train_state, resampled_time_grids_savelist

def single_run_with_restart_tp_resampling_at_ga(config, pde, constraint_layer, time_grid, lr_schedule, extra_points_amount, trainable_variables=None, GA_point=None,
                                                adding_tp_iteration=40000, threshold=5.0, N_attempts=50, train_distribution="uniform", std=None, plot=True, save=False, seed=None):
    attempt = 1
    if seed == None:
        seed = int(datetime.now().strftime("%Y%m%d%H%M%S"));    print("time-dependent random seed:", seed)
    else:
        print("manually entered seed:", seed)

    while attempt <= N_attempts:
        print("Initialisation attempt:", attempt);  print("seed:", seed)

        # Creation of model
        geom = dde.geometry.TimeDomain(config['t0'] / config['t_scale'], config['tfinal'] / config['t_scale'], sampler_std=std)
        data = dde.data.PDE(geom, pde, [], len(time_grid), 2, num_test=len(time_grid), train_distribution=train_distribution)
        data.train_x = time_grid  # Assign the custom time points
        def new_test(self):
            return time_grid, None, None
        data.test = new_test.__get__(data, dde.data.PDE)

        initializer = tf.keras.initializers.GlorotNormal(seed=seed)  # FNN 20241022022825       PFNN 20241015143854
        net = dde.nn.FNN(config["layer_architecture_FNN"], "sin", initializer)
        net.apply_output_transform(constraint_layer)
        model = ModelPCNN(data, net, config['loss_weights'], config)

        resampling_counter = 0
        total_iterations = 0
        for (lr, iterations) in lr_schedule:
            print("Learning rate=", lr, "Iterations=", iterations)

            optimisation_alg = tf.keras.optimizers.Adam(learning_rate=lr)
            model.compile(optimisation_alg, lr=lr, external_trainable_variables=trainable_variables)
            print('model external_trainable_variables:', model.external_trainable_variables)
            losshistory, train_state = model.train(iterations=iterations, display_every=1000)  # , callbacks=[checkpoint_cb])

            # Add training points if loss is below threshold
            if not np.isnan(np.sum(losshistory.loss_test[-1])):
                if (np.sum(losshistory.loss_test[-1]) < threshold) and (total_iterations > adding_tp_iteration):
                    if resampling_counter < 1:
                        resampling_counter +=1
                        t_focus = GA_point;     focus_width = (config['tfinal'] / config['t_scale']) * 0.02  # Define a small range around the focus point
                        extra_points_amount = extra_points_amount;   extra_points = np.linspace(t_focus - focus_width, t_focus + focus_width, extra_points_amount).reshape(-1, 1)
                        print('Adding', str(extra_points_amount), 'extra collocation points around the t_ga')
                        data.add_anchors(extra_points)

                        time_grid_new = np.vstack((time_grid, extra_points))
                        time_grid_new = np.sort(time_grid_new.astype(np.float32), axis=0)
                        def new_test(self):
                            return time_grid_new, None, None
                        data.test = new_test.__get__(data, dde.data.PDE)
                        data.train_x = time_grid_new

            total_iterations += iterations

            # RESTART
            if np.isnan(np.sum(losshistory.loss_test[-1])) or np.sum(losshistory.loss_test[-1]) > threshold:
                print("Loss not below threshold, restarting now...")
                attempt += 1
                seed = int(datetime.now().strftime("%Y%m%d%H%M%S"));
                break

        if not np.isnan(np.sum(losshistory.loss_test[-1])):
            if np.sum(losshistory.loss_test[-1]) < threshold:
                print("Successful run completed")
                break

    # if save:
    #     print(len(losshistory.y_pred_test[0]))
    #     save_loss_history(losshistory, f'loss.dat', verbose=True)
    #     metrics_lists = calculate_metrics_all_iterations(losshistory, time_grid, config)
    #     save_metrics_history(losshistory, metrics_lists, f'metrics.dat', verbose=True)
    if plot:
        pass

    return losshistory, train_state, time_grid_new

def parallel_pcnn_training(configs_list, loss, constraint_layers_list, time_grids_list_train, time_grids_list_test, lr_schedule, final_state1, initial_state2, ga_trainable_variables=None, callbacks=None,
                            resampling=False, resampling_epoch=44000, extra_points_amount=200, LBFGS=False, threshold=5.0, N_attempts=50, train_distribution="uniform", std=None, plot=True, save=False, seed=None):
    attempt1 = 1
    attempt2 = 1
    if seed == None:
        seed = int(datetime.now().strftime("%Y%m%d%H%M%S"));
        print("time-dependent random seed:", seed)
    else:
        print("manually entered seed:", seed)

    while attempt1 <= N_attempts:
        print("Initialisation attempt:", attempt1);
        print("seed:", seed)

        # model1 = create_model_sp2(configs_list, pde, constraint_layer, time_grid, seed=seed, train_distribution=train_distribution, std=std)  # PFNN 20241015143854
        model1, data1 = create_model_and_data(configs_list[0], loss, constraint_layers_list[0], time_grids_list_train[0], time_grids_list_test[0], seed=seed, train_distribution=train_distribution, std=std)  # FNN 20250122173225

        total_iterations = 0
        for (lr, iterations) in lr_schedule:
            total_iterations += iterations
            print("Learning rate=", lr, "Iterations=", iterations)
            optimisation_alg1 = tf.keras.optimizers.Adam(learning_rate=lr)

            # Model 1
            model1.compile(optimisation_alg1, lr=lr, external_trainable_variables=ga_trainable_variables[0:2])
            losshistory1, train_state1 = model1.train(iterations=iterations, display_every=1000, callbacks=[callbacks])  # , callbacks=[checkpoint_cb])
            # Restart
            if np.isnan(np.sum(losshistory1.loss_test[-1])) or np.sum(losshistory1.loss_test[-1]) > threshold:
                print("Loss 1 not below threshold, restarting now...")
                attempt1 += 1
                seed = int(datetime.now().strftime("%Y%m%d%H%M%S"));
                break

            while (np.sum(losshistory1.loss_test[-1]) < threshold) and attempt2 < 25:
                model2, data2 = create_model_and_data(configs_list[1], loss, constraint_layers_list[1], time_grids_list_train[1], time_grids_list_test[1], seed=seed, train_distribution=train_distribution, std=std)  # FNN 20250122173225
                configs_list[1]['m0'] = train_state1.best_y[-1,-1]
                optimisation_alg2 = tf.keras.optimizers.Adam(learning_rate=lr)
                model2.compile(optimisation_alg2, lr=lr, external_trainable_variables=ga_trainable_variables[2:4])
                losshistory2, train_state2 = model2.train(iterations=iterations, display_every=1000, callbacks=[callbacks])  # , callbacks=[checkpoint_cb])
                if np.isnan(np.sum(losshistory2.loss_test[-1])) or np.sum(losshistory2.loss_test[-1]) > threshold:
                    print("Loss 2 not below threshold, restarting model 2 now...")
                    attempt2 += 1
                    seed = int(datetime.now().strftime("%Y%m%d%H%M%S"));
                if np.sum(losshistory2.loss_test[-1]) < threshold:
                    attempt2 = 26
                    break
            if attempt2 > 26:
                configs_list[1]['m0'] = train_state1.best_y[-1, -1]
                optimisation_alg2 = tf.keras.optimizers.Adam(learning_rate=lr)
                model2.compile(optimisation_alg2, lr=lr, external_trainable_variables=ga_trainable_variables[2:4])
                losshistory2, train_state2 = model2.train(iterations=iterations, display_every=1000, callbacks=[callbacks])  # , callbacks=[checkpoint_cb])

                if np.isnan(np.sum(losshistory2.loss_test[-1])) or np.sum(losshistory2.loss_test[-1]) > threshold:
                    print("Loss 2 not below threshold in the middle of training, restarting everything now...")
                    break
            attempt2 += 1

            if resampling:
                if total_iterations > resampling_epoch:
                    print('Adding', str(extra_points_amount), 'extra collocation points')
                    # + and - 1000 because otherwise you would have 2x a t=0 and 2x a t=tf
                    new_time_grid_ND_leg1 = np.linspace((configs_list[0]['t0'] + 1000) / configs_list[0]['t_scale'], (configs_list[0]['tfinal'] - 1000) / configs_list[0]['t_scale'], extra_points_amount, dtype=np.float32).reshape(-1, 1)
                    new_time_grid_ND_leg2 = np.linspace((configs_list[1]['t0'] + 1000 )/ configs_list[1]['t_scale'], (configs_list[1]['tfinal'] - 1000) / configs_list[1]['t_scale'], extra_points_amount, dtype=np.float32).reshape(-1, 1)
                    # data1.add_anchors(new_time_grid_ND_leg1)
                    # data2.add_anchors(new_time_grid_ND_leg2)
                    time_grid_new1 = np.vstack((time_grids_list_train[0], new_time_grid_ND_leg1))
                    time_grid_new1 = np.sort(time_grid_new1, axis=0)
                    time_grid_new2 = np.vstack((time_grids_list_train[1], new_time_grid_ND_leg2))
                    time_grid_new2 = np.sort(time_grid_new2, axis=0)

                    def new_test1(self):
                        return time_grid_new1, None, None
                    def new_test2(self):
                        return time_grid_new2, None, None

                    # data1.test = new_test1.__get__(data1, dde.data.PDE)
                    data1.train_x = time_grid_new1
                    # data2.test = new_test2.__get__(data2, dde.data.PDE)
                    data2.train_x = time_grid_new2
        if LBFGS:
            model1.compile("L-BFGS-B")
            losshistory1, train_state1 = model1.train()

            configs_list[1]['m0'] = train_state1.best_y[-1, -1]
            model2.compile("L-BFGS-B")
            losshistory2, train_state2 = model2.train()

        if not np.isnan(np.sum(losshistory1.loss_test[-1])):
            if np.sum(losshistory1.loss_test[-1]) < threshold:
                print("Successful run leg 1 completed")
                if not np.isnan(np.sum(losshistory2.loss_test[-1])):
                    if np.sum(losshistory2.loss_test[-1]) < threshold:
                        print("Successful run leg 2 also completed")
                        break

    if save:
        number = 1
        dde.utils.save_loss_history(losshistory1, f'loss{number}.dat', verbose=True)
        dde.utils.save_best_state(train_state1, f'train_data{number}.dat', f'test_data{number}.dat')
        metrics_lists = calculate_metrics_all_iterations(losshistory1, time_grids_list_train[0], configs_list[0])
        save_metrics_history(losshistory1, metrics_lists, f'metrics{number}.dat', verbose=True)

        number = 2
        dde.utils.save_loss_history(losshistory2, f'loss{number}.dat', verbose=True)
        dde.utils.save_best_state(train_state2, f'train_data{number}.dat', f'test_data{number}.dat')
        metrics_lists = calculate_metrics_all_iterations(losshistory2, time_grids_list_train[0], configs_list[0])
        save_metrics_history(losshistory2, metrics_lists, f'metrics{number}.dat', verbose=True)

    if plot:
        pass

    if resampling and total_iterations > resampling_epoch:
            return time_grid_new1, losshistory1, train_state1, time_grid_new2, losshistory2, train_state2
    else:
        return time_grids_list_train[0], losshistory1, train_state1, time_grids_list_train[1], losshistory2, train_state2

def parallel_pcnn_training_sp3(configs_list, loss, constraint_layers_list, input_transform_list, time_grids_list_train, time_grids_list_test, lr_schedule, final_state1, initial_state2, ga_trainable_variables=None, callbacks=None,
                            t_ga_refinement=False, t_ga_refinement_epoch=44000, threshold=5.0, N_attempts=50, train_distribution="uniform", std=None, plot=True, save=False, seed=None):
    attempt1 = 1
    attempt2 = 1
    if seed == None:
        seed = int(datetime.now().strftime("%Y%m%d%H%M%S"));
        print("time-dependent random seed:", seed)
    else:
        print("manually entered seed:", seed)

    while attempt1 <= N_attempts:
        print("Initialisation attempt:", attempt1);
        print("seed:", seed)

        # model1 = create_model_sp2(configs_list, pde, constraint_layer, time_grid, seed=seed, train_distribution=train_distribution, std=std)  # PFNN 20241015143854
        model1, data1 = create_model_and_data(configs_list[0], loss, constraint_layers_list[0], time_grids_list_train[0], time_grids_list_test[0], seed=seed, train_distribution=train_distribution, std=std)  # FNN 20250122173225
        model2, data2 = create_model_and_data(configs_list[1], loss, constraint_layers_list[1], time_grids_list_train[1], time_grids_list_test[0], seed=seed, train_distribution=train_distribution, std=std)  # FNN 20250122173225

        total_iterations = 0
        for (lr, iterations) in lr_schedule:
            total_iterations += iterations
            print("Learning rate=", lr, "Iterations=", iterations)
            optimisation_alg1 = tf.keras.optimizers.Adam(learning_rate=lr)

            # Model 1
            model1.compile(optimisation_alg1, lr=lr, external_trainable_variables=ga_trainable_variables[0:3])
            losshistory1, train_state1 = model1.train(iterations=iterations, display_every=1000, callbacks=[callbacks])  # , callbacks=[checkpoint_cb])
            # Restart
            if np.isnan(np.sum(losshistory1.loss_test[-1])) or np.sum(losshistory1.loss_test[-1]) > threshold:
                print("Loss 1 not below threshold, restarting now...")
                attempt1 += 1
                seed = int(datetime.now().strftime("%Y%m%d%H%M%S"));
                break

            while (np.sum(losshistory1.loss_test[-1]) < threshold) and attempt2 < 25:
                model2, data2 = create_model_and_data(configs_list[1], loss, constraint_layers_list[1], time_grids_list_train[1], time_grids_list_test[0], seed=seed, train_distribution=train_distribution, std=std)  # FNN 20250122173225
                configs_list[1]['m0'] = train_state1.best_y[-1,-1]
                optimisation_alg2 = tf.keras.optimizers.Adam(learning_rate=lr)
                model2.compile(optimisation_alg2, lr=lr, external_trainable_variables=ga_trainable_variables[3:6])
                losshistory2, train_state2 = model2.train(iterations=iterations, display_every=1000, callbacks=[callbacks])  # , callbacks=[checkpoint_cb])
                if np.isnan(np.sum(losshistory2.loss_test[-1])) or np.sum(losshistory2.loss_test[-1]) > threshold:
                    print("Loss 2 not below threshold, restarting model 2 now...")
                    attempt2 += 1
                    seed = int(datetime.now().strftime("%Y%m%d%H%M%S"));
                if np.sum(losshistory2.loss_test[-1]) < threshold:
                    attempt2 = 26
                    break
            if attempt2 > 26:
                configs_list[1]['m0'] = train_state1.best_y[-1, -1]
                optimisation_alg2 = tf.keras.optimizers.Adam(learning_rate=lr)
                model2.compile(optimisation_alg2, lr=lr, external_trainable_variables=ga_trainable_variables[2:4])
                losshistory2, train_state2 = model2.train(iterations=iterations, display_every=1000, callbacks=[callbacks])  # , callbacks=[checkpoint_cb])

                if np.isnan(np.sum(losshistory2.loss_test[-1])) or np.sum(losshistory2.loss_test[-1]) > threshold:
                    print("Loss 2 not below threshold in the middle of training, restarting everything now...")
                    break
            attempt2 += 1

            if t_ga_refinement:
                if (total_iterations > t_ga_refinement_epoch) and (total_iterations < t_ga_refinement_epoch+1000):
                    print('Changing constraint layers to introduce a variable t_ga')
                    ga_trainable_variables[2]._trainable=True
                    # ga_trainable_variables[5]._trainable=True
                    # model1.net.apply_feature_transform(input_transform_list[0])
                    # model2.net.apply_feature_transform(input_transform_list[1])
                    model1.net.apply_output_transform(constraint_layers_list[2])
                    model2.net.apply_output_transform(constraint_layers_list[3])

                    model1.compile(optimisation_alg1, lr=lr, external_trainable_variables=ga_trainable_variables[0:3])
                    model2.compile(optimisation_alg2, lr=lr, external_trainable_variables=ga_trainable_variables[3:6])
                    for i in range(10):
                        losshistory1, train_state1 = model1.train(iterations=10, display_every=10, callbacks=[callbacks])  # , callbacks=[checkpoint_cb])
                        losshistory2, train_state2 = model2.train(iterations=10, display_every=10, callbacks=[callbacks])  # , callbacks=[checkpoint_cb])

                    ga_trainable_variables[2]._trainable = False
                    print('t_ga = constant again. Now refining final solution with the new ("optimized") t_ga')
                    for i in range(1):
                        losshistory1, train_state1 = model1.train(iterations=100, display_every=100, callbacks=[callbacks])  # , callbacks=[checkpoint_cb])
                        losshistory2, train_state2 = model2.train(iterations=100, display_every=100, callbacks=[callbacks])  # , callbacks=[checkpoint_cb])

                    total_iterations += 2000

        if (total_iterations > t_ga_refinement_epoch):
            print('End of run')
            break

        if not np.isnan(np.sum(losshistory1.loss_test[-1])):
            if np.sum(losshistory1.loss_test[-1]) < threshold:
                print("Successful run leg 1 completed")
                if not np.isnan(np.sum(losshistory2.loss_test[-1])):
                    if np.sum(losshistory2.loss_test[-1]) < threshold:
                        print("Successful run leg 2 also completed")
                        break

    if save:
        number = 1
        dde.utils.save_loss_history(losshistory1, f'loss{number}.dat', verbose=True)
        dde.utils.save_best_state(train_state1, f'train_data{number}.dat', f'test_data{number}.dat')
        metrics_lists = calculate_metrics_all_iterations(losshistory1, time_grids_list_train[0], configs_list[0])
        save_metrics_history(losshistory1, metrics_lists, f'metrics{number}.dat', verbose=True)
    if plot:
        pass

    return time_grids_list_train[0]*ga_trainable_variables[2].numpy(), losshistory1, train_state1, time_grids_list_train[1]*ga_trainable_variables[5].numpy(), losshistory2, train_state2


def save_metrics_history(loss_history, metrics, fname, verbose = True, mass=True):
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

def verify_run(states, config, time_grid, m0_leg_x, ga_bodies=None, showplot=True, saveplots=True):
    t = time_grid
    t_reshaped = t.reshape(-1, 1)

    # save time+mass seperately
    pcnn_mass = np.concatenate((t_reshaped, states[:, -1].reshape(-1, 1)), axis=1)
    # save ND states (with time and without mass)
    states_without_mass_ND = np.concatenate((t_reshaped, states[:, :-1]), axis=1)
    states_without_mass_NDcartesian = coordinates_transformation_functions.radial_to_NDcartesian(states_without_mass_ND, config)
    states_without_mass_NDcartesian_dict = {"NDcartesian": states_without_mass_NDcartesian}

    control_nodes, ref_times, initial_state = control_nodes_ref_times_3D_initial_state(states, config, time_grid)


    verification_object = verification.Verification(m0_leg_x, config['t0'], config['tfinal'], initial_state, config['isp'], ga_bodies=ga_bodies, central_body="Sun", control_nodes=control_nodes,
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

        # Dr, Dv, Dm, fuel_used, time_interval = calculate_metrics_best_iteration(states_without_mass_NDcartesian, tudat_states_NDcartesian, pcnn_mass, tudat_mass, config=config)
        # plots.plot_metrics_best_iteration_vs_time(Dr, Dv, Dm, fuel_used, time_interval, save=saveplots)
        # plots.plot_metrics_vs_iterations(losshistory, time_grid, config, save=saveplots)

        plt.show()

def verify_run_sp2(states, losshistory_loaded_list, metrics_loaded_list, configs_list, time_grids_list, ga_bodies=None, showplot=True, saveplots=True):
    for i in range(len(configs_list)):
        print('this is i= ', i)
        t = time_grids_list[i]
        t_reshaped = t.reshape(-1, 1)
        states_leg_x = states[:, (7*i): (7*(i+1))]

        # save time+mass seperately
        pcnn_mass = np.concatenate((t_reshaped, states_leg_x[:, -1].reshape(-1, 1)), axis=1)
        m0_leg_x = pcnn_mass[0][1]
        # save ND states (with time and without mass)
        states_without_mass_ND = np.concatenate((t_reshaped, states_leg_x[:, :-1]), axis=1)
        states_without_mass_NDcartesian = coordinates_transformation_functions.radial_to_NDcartesian(states_without_mass_ND, configs_list[i])
        states_without_mass_NDcartesian_dict = {"NDcartesian": states_without_mass_NDcartesian}

        control_nodes, ref_times, initial_state = control_nodes_ref_times_3D_initial_state(states_leg_x, configs_list[i], t)


        verification_object = verification.Verification(m0_leg_x, configs_list[i]['t0'], configs_list[i]['tfinal'], initial_state, configs_list[i]['isp'], ga_bodies=ga_bodies, central_body="Sun", control_nodes=control_nodes,
                                                        verbose=True, ref_times=ref_times, mass_rate=True, config=configs_list[i])
        verification_object.integrate()


        tudat_states_cartesian = verification_object.states_tudat
        tudat_mass = verification_object.mass
        tudat_states_NDcartesian = coordinates_transformation_functions.cartesian_to_NDcartesian(tudat_states_cartesian, configs_list[i])
        tudat_states_NDcartesian_dict = {"NDcartesian": tudat_states_NDcartesian}

        if showplot:
            custom_labels = ["$x$", "$y$", "$v_{x}$", "$v_{y}$"]
            plots.plot_compare_pcnn_tudat_states(states_without_mass_NDcartesian_dict, tudat_states_NDcartesian_dict, config=configs_list[i], custom_labels=custom_labels, log=False, save=saveplots)
            plots.plot_compare_pcnn_tudat_mass(pcnn_mass, tudat_mass, config=configs_list[i], save=saveplots)

            Dr, Dv, Dm, fuel_used, time_interval = calculate_metrics_best_iteration(states_without_mass_NDcartesian, tudat_states_NDcartesian, pcnn_mass, tudat_mass, config=configs_list[i])
            plots.plot_metrics_best_iteration_vs_time(Dr, Dv, Dm, fuel_used, time_interval, configs_list[i], save=True)
            plots.plot_metrics_vs_iterations(metrics_loaded_list[i], configs_list[i], save=True)

    return Dr

def verify_run_sp2_T2(states, configs_list, time_grid, ga_bodies=None, showplot=True, saveplots=True):
    for i in range(len(configs_list)):
        t = time_grid
        t_reshaped = t.reshape(-1, 1)
        states_leg_x = states[:, (7*i): (7*(i+1))]

        # save time+mass seperately
        pcnn_mass = np.concatenate((t_reshaped, states_leg_x[:, -1].reshape(-1, 1)), axis=1)
        m0_leg_x = pcnn_mass[0][1]
        # save ND states (with time and without mass)
        states_without_mass_ND = np.concatenate((t_reshaped, states_leg_x[:, :-1]), axis=1)
        states_without_mass_NDcartesian = coordinates_transformation_functions.radial_to_NDcartesian(states_without_mass_ND, configs_list[i])
        states_without_mass_NDcartesian_dict = {"NDcartesian": states_without_mass_NDcartesian}

        control_nodes, ref_times, initial_state = control_nodes_ref_times_3D_initial_state(states_leg_x, configs_list[i], t)


        verification_object = verification.Verification(m0_leg_x, configs_list[i]['t0'], configs_list[i]['tfinal'], initial_state, configs_list[i]['isp'], ga_bodies=ga_bodies, central_body="Sun", control_nodes=control_nodes,
                                                        verbose=True, ref_times=ref_times, mass_rate=True, config=configs_list[i])
        verification_object.integrate()


        tudat_states_cartesian = verification_object.states_tudat
        tudat_mass = verification_object.mass
        tudat_states_NDcartesian = coordinates_transformation_functions.cartesian_to_NDcartesian(tudat_states_cartesian, configs_list[i])
        tudat_states_NDcartesian_dict = {"NDcartesian": tudat_states_NDcartesian}

        if showplot:
            custom_labels = ["$r$", "$\\theta$", "$v_{r}$", "$v_{\\theta}$"]
            plots.plot_compare_pcnn_tudat_states(states_without_mass_NDcartesian_dict, tudat_states_NDcartesian_dict, config=configs_list[i], custom_labels=custom_labels, log=False, save=saveplots)
            plots.plot_compare_pcnn_tudat_mass(pcnn_mass, tudat_mass, config=configs_list[i], save=saveplots)

            # Dr, Dv, Dm, fuel_used, time_interval = calculate_metrics_best_iteration(states_without_mass_NDcartesian, tudat_states_NDcartesian, pcnn_mass, tudat_mass, config=configs_list[i])
            # plots.plot_metrics_best_iteration_vs_time(Dr, Dv, Dm, fuel_used, time_interval, save=saveplots)
            # plots.plot_metrics_vs_iterations(losshistory, time_grid, configs_list[i], save=saveplots)

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

def generate_2D_position_from_spice(observer_body, target_body, ref_frame, epoch, config, coordinates='NDradial'):
    state_tudat = spice.get_body_cartesian_state_at_epoch(target_body_name=target_body, observer_body_name=observer_body, reference_frame_name=ref_frame, aberration_corrections='NONE', ephemeris_time=epoch)
    state = np.concatenate((state_tudat[0:2], state_tudat[3:5]), axis=0)
    state = np.concatenate((np.array([0.0]), state), axis=0)

    if coordinates == 'NDradial':
        state = coordinates_transformation_functions.cartesian_to_radial(np.array([state]), config=config)
    if coordinates == 'NDcartesian':
        state = coordinates_transformation_functions.cartesian_to_NDcartesian(np.array([state]), config=config)
    if coordinates == 'cartesian':
        pass

    state = state[0][1:5]
    return state

def generate_2D_orbit_from_spice(time_grid, target_body, config, coordinates='NDradial'):
    # time_grid = np.linspace(start_epoch, end_epoch, num_points)
    states = []
    for t in time_grid:
        states.append(generate_2D_position_from_spice('Sun', target_body, 'ECLIPJ2000', t, config, coordinates=coordinates))

    states = np.array(states)
    thetas = states[:,1]
    new_thetas = coordinates_transformation_functions.get_new_thetas(thetas)
    states[:,1] = new_thetas
    return states

def ga_summary(x2_leg1, x3_leg1, x2_leg2, x3_leg2, ga_states, MU_p, configs_list, delta_ga=None):
    # Spacecraft velocity before gravity assist
    v_sc_min_vec = np.array([x2_leg1, x3_leg1])  # [v_r, v_theta]
    v_planet_vec = np.array([ga_states[-1][2], ga_states[-1][3]])  # [v_r, v_theta]
    v_inf_min_vec = v_sc_min_vec - v_planet_vec  # [v_r, v_theta]
    v_inf_mag = np.linalg.norm(v_inf_min_vec)  # v_inf_plus_mag = v_inf_min_mag
    v_inf_min_angle = math.atan2(v_inf_min_vec[0], v_inf_min_vec[1])  # Correct quadrant handling

    # # calculate v_sc_plus with given delta_ga
    # v_inf_plus_angle = v_inf_min_angle + delta_ga
    #
    # # Spacecraft velocity after gravity assist
    # v_inf_plus_vec = np.array([v_inf_mag * tf.math.cos(v_inf_plus_angle),
    #                            v_inf_mag * tf.math.sin(v_inf_plus_angle)
    #                            ])  # [v_r, v_theta]
    # v_sc_plus_vec = v_inf_plus_vec + v_planet_vec  # [v_r, v_theta]

    # calculate delta_ga with given v_sc_plus
    v_sc_plus_vec = np.array([x2_leg2, x3_leg2])  # [v_r, v_theta]
    v_inf_plus_vec = v_sc_plus_vec - v_planet_vec
    v_inf_plus_angle = math.atan2(v_inf_plus_vec[0], v_inf_plus_vec[1])

    delta_ga = v_inf_plus_angle - v_inf_min_angle

    v_inf_mag_rescaled = v_inf_mag  * (configs_list[0]['length_scale']/configs_list[0]['t_scale'])

    r_p = ((MU_p/np.sin(delta_ga/2)) - MU_p)  /  (v_inf_mag_rescaled**2)


    # Plot the vectors
    plt.figure(figsize=(8, 8))
    origin = [0, 0]  # Origin for the vectors

    # Plot v_sc_min_vec
    plt.quiver(*origin, v_sc_min_vec[0], v_sc_min_vec[1], angles='xy', scale_units='xy', scale=1, color='blue', label='v_sc_min_vec')
    # Plot v_sc_plus_vec
    plt.quiver(*origin, v_sc_plus_vec[0], v_sc_plus_vec[1], angles='xy', scale_units='xy', scale=1, color='red', label='v_sc_plus_vec')
    # Plot v_planet_vec (optional, for reference)
    plt.quiver(*origin, v_planet_vec[0], v_planet_vec[1], angles='xy', scale_units='xy', scale=1, color='green', label='v_planet_vec')
    # Plot v_inf_min_vec
    plt.quiver(*origin, v_inf_min_vec[0], v_inf_min_vec[1], angles='xy', scale_units='xy', scale=1, color='purple', label='v_inf_min_vec')
    # Plot v_inf_plus_vec
    plt.quiver(*origin, v_inf_plus_vec[0], v_inf_plus_vec[1], angles='xy', scale_units='xy', scale=1, color='orange', label='v_inf_plus_vec')

    # Set plot limits and labels
    plt.xlim(-1.5 * np.linalg.norm(v_inf_plus_vec), 1.5 * np.linalg.norm(v_inf_plus_vec))
    plt.ylim(-1.5 * np.linalg.norm(v_inf_plus_vec), 1.5 * np.linalg.norm(v_inf_plus_vec))
    plt.xlabel('v_r')
    plt.ylabel('v_theta')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid()
    plt.legend()
    plt.title('Visualization of Gravity Assist Velocity Rotation')
    plt.gca().set_aspect('equal', adjustable='box')

    print('v_inf_min_angle', v_inf_min_angle)
    print('delta_ga', delta_ga)
    print('v_inf_plus_angle', v_inf_plus_angle)
    print('v_inf_mag', v_inf_mag)
    print('r_p', r_p)

    plt.show()

    return r_p, v_sc_plus_vec

def ga_check(v_r_sc_min, v_theta_sc_min, v_r_sc_plus, v_theta_sc_plus, ga_states, MU_p, configs_list, delta_ga=None, SOI_p=None, r_planet=None):
    # Spacecraft velocity before gravity assist
    v_sc_min_vec = tf.stack([v_r_sc_min, v_theta_sc_min])  # [v_r, v_theta]
    v_planet_vec = tf.stack([tf.cast(ga_states[-1][2], tf.float64), tf.cast(ga_states[-1][3], tf.float64)])  # [v_r, v_theta]
    v_inf_min_vec = v_sc_min_vec - v_planet_vec  # [v_r, v_theta]
    v_inf_mag = tf.norm(v_inf_min_vec)  # v_inf_plus_mag = v_inf_min_mag
    v_inf_min_angle = tf.math.atan2(v_inf_min_vec[0] , v_inf_min_vec[1])

    # calculate delta_ga with given v_sc_plus
    v_sc_plus_vec = tf.stack([v_r_sc_plus, v_theta_sc_plus])  # [v_r, v_theta]
    v_inf_plus_vec = v_sc_plus_vec - v_planet_vec
    v_inf_plus_angle = tf.math.atan2(v_inf_plus_vec[0], v_inf_plus_vec[1])

    delta_ga = v_inf_plus_angle - v_inf_min_angle

    v_inf_mag_rescaled = v_inf_mag * (configs_list[0]['length_scale'] / configs_list[0]['t_scale'])

    r_p_rescaled = ((MU_p / tf.sin(delta_ga / 2)) - MU_p) / (v_inf_mag_rescaled ** 2)


    # print('v_inf_min_angle', v_inf_min_angle)
    # print('delta_ga', delta_ga)
    # print('v_inf_plus_angle', v_inf_plus_angle)
    # print('v_inf_mag', v_inf_mag)
    # print('r_p', r_p_rescaled)

    return r_p_rescaled
























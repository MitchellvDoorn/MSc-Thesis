"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, jax, paddle"""
import os
import deepxde as dde
import numpy as np
# Import tf if using backend tensorflow.compat.v1 or tensorflow
import tensorflow as tf
import master_thesis_mitchell_functions as mtmf
import matplotlib.pyplot as plt
import verification
import coordinates_transformation_functions
import plots
import time
from datetime import datetime

# dde.config.set_default_float("float64")
# tf.keras.backend.set_floatx('float64')

start_time = time.time()

run_id_number = int(datetime.now().strftime("%Y%m%d%H%M%S"))
print('Run ID number:', run_id_number)

# Constants
MU = 1.32712440042e20 # gravitational parameter of Sun
m0 = 100 # spacecraft initial mass
AU = 149597870700 # [m]
a = 10 # steepness parmater
umax = 0.1 # max allowable thrust
isp = 2500 # specific impulse
# Initial state
r0 = AU
theta0 = 0
vr0 = 0
vtheta0 = np.sqrt(MU/r0)
initial_state = np.array([r0, theta0, vr0, vtheta0])

# Final state
rfinal = 1.5*r0
theta_final = 4*np.pi
vr_final = 0
vtheta_final = np.sqrt(MU/rfinal)
final_state = np.array([rfinal, theta_final, vr_final, vtheta_final])

# Non-dimensionalization parameters
length_scale = r0
t_scale = length_scale / vtheta0

length_scale = length_scale
t_scale = t_scale

# Non-dimensionalize iniitial and final states
initial_state[0] /= length_scale
initial_state[2:] *= t_scale/length_scale
final_state[0] /= length_scale
final_state[2:] *= t_scale/length_scale

# Integration times
days = 1000
t0 = 0
tfinal = days*24*3600 # Constant time of flight
M = 200 # Amount of collocation points

# Loss weights
dyn_weight = 1
m_weigth = 1e-5 # mass term
o_weigth = 1e-7 # objective term

# create config dictionary
config = {"t0": t0,
          "tfinal": tfinal,
          "length_scale": length_scale,
          "t_scale": t_scale,
          "isp": isp,
          "m0": m0,
          "M": M,
          "metrics": ["FinalDr", "FinalDv", "FinalDm", "Fuel used"],
          "N_train": M,
          "N_test": 200,
          "layer_architecture_FNN": [1, 20, 20, 20, 20, 20, 7],
          "layer_architecture_PFNN": [1, [10,10,10,10,10,10,10], [10,10,10,10,10,10,10], [10,10,10,10,10,10,10], 7],
          "loss_weights": [dyn_weight, dyn_weight, dyn_weight, dyn_weight, m_weigth, o_weigth],
          "mass": True,
          "run_id_number": run_id_number
}

time_grid_train = np.linspace(config['t0'] / config['t_scale'], config['tfinal'] / config['t_scale'], config['N_train'], dtype=dde.config.real(np)).reshape(-1, 1)
time_grid_test  = np.linspace(config['t0'] / config['t_scale'], config['tfinal'] / config['t_scale'], config['N_test'], dtype=dde.config.real(np)).reshape(-1, 1)

def pde(t, y):
    x1 = y[:, 0:1]
    theta = y[:, 1:2]  # unused
    x2 = y[:, 2:3]
    x3 = y[:, 3:4]
    ur = y[:, 4:5]
    ut = y[:, 5:6]
    m = y[:, 6:7]
    # tf.print(m)

    # Thrust magnitude
    T = tf.reshape(tf.norm(y[:, 4:6], axis=1), (-1, 1))
    # tf.print("T:   ,", T)

    delta_t = t[1:] - t[:-1]
    # L_o = (1 / (isp * 9.81)) * 0.5 * (T[:-1] * t_scale + T[1:] * t_scale) * delta_t

    # LHS EOMs - Derivatives
    dx1_dt = dde.grad.jacobian(y, t, i=0)
    dtheta_dt = dde.grad.jacobian(y, t, i=1)
    dx2_dt = dde.grad.jacobian(y, t, i=2)
    dx3_dt = dde.grad.jacobian(y, t, i=3)
    dm_dt = dde.grad.jacobian(y, t, i=6)

    # RHS EOMs
    RHS_x1 = x2
    RHS_theta = x3 / x1
    RHS_x2 = x3 ** 2 / x1 - (MU * t_scale ** 2 / length_scale ** 3) * x1 ** (-2) + (t_scale ** 2 / length_scale) * ur / m
    RHS_x3 = - (x2 * x3) / x1 + (t_scale ** 2 / length_scale) * ut / m
    RHS_m = -T * t_scale / (isp * 9.81)

    # Return the residuals
    return [
        dx1_dt - RHS_x1,
        dtheta_dt - RHS_theta,
        dx2_dt - RHS_x2,
        dx3_dt - RHS_x3,
        dm_dt - RHS_m] # a comma here makes the loss 2D and you get an error...
        # L_o

def constraint_layer(t, y):
    c1 = tf.math.exp(-a * (t - t0))
    c2 = 1 - tf.math.exp(-a * (t - t0)) - tf.math.exp(a * (t - tfinal/t_scale))
    c3 = tf.math.exp(a * (t - tfinal/t_scale))
    c_mass = 1 - tf.math.exp(-a * (t - t0))

    # Apply sigmoid to get in [0, 1], while keeping a non-zero derivative for training
    u_norm = tf.math.sigmoid(y[:, 4:5])
    u_angle = tf.math.tanh(y[:, 5:6])
    Nm = tf.math.sigmoid(y[:, 6:7])

    # Rescale the U_R and the U_theta to their real values
    u_norm = u_norm * umax
    u_angle = u_angle * 2 * np.pi

    # Transform the control to cartesian coordinates
    ur = u_norm * tf.math.sin(u_angle)
    ut = u_norm * tf.math.cos(u_angle)

    output = tf.concat([c1 * initial_state[0] + c2 * y[:, 0:1] + c3 * final_state[0],
                        c1 * initial_state[1] + c2 * y[:, 1:2] + c3 * final_state[1],
                        c1 * initial_state[2] + c2 * y[:, 2:3] + c3 * final_state[2],
                        c1 * initial_state[3] + c2 * y[:, 3:4] + c3 * final_state[3],
                        ur,
                        ut,
                        m0 - c_mass * m0 * Nm], axis=1
                       )
    return output

# lr_schedule = [(1e-2, 3000), (1e-3, 500), (1e-4, 1000), (5e-3, 400), (1e-4, 500), (5e-3, 400), (1e-4, 500), (5e-3, 400), (1e-4, 500), (1e-5, 600)]
lr_schedule = [(1e-2, 3000), (1e-3, 5000), (1e-4, 10000), (5e-3, 4000), (1e-4, 5000), (5e-3, 4000), (1e-4, 5000), (5e-3, 4000), (1e-4, 5000), (1e-5, 6000)]


delta_t = (config['tfinal']/config['t_scale'] - config['t0']/config['t_scale']) / config['N_train'];    std = 0.2 * delta_t
# mtmf.restarter (config, pde, constraint_layer, lr_schedule, train_distribution="perturbed_uniform_tf", std=None, plot=True, save=True, N_attempts=60, run_id_number=run_id_number)
losshistory, train_state, resampled_time_grids_savelist = mtmf.single_run_with_restart(config, pde, constraint_layer, time_grid_train, time_grid_test, lr_schedule, final_state, initial_state,
                                                                                       adaptive_sampling=False, LBFGS=True, threshold=5, train_distribution="uniform", std=None, save=True, seed=20250204183139) # fill in seed=None for time dependent seed


# losshistory, train_state = mtmf.single_run_with_restart_tp_resampling_at_ga(config, pde, constraint_layer, time_grid, lr_schedule, GA_point=(tfinal/t_scale)/2, train_distribution="uniform", std=None, save=True, seed=None) # fill in seed=None for time dependent seed

states = np.loadtxt('test_data2.dat')[:, 1:] # without time column 0
losshistory_loaded = np.loadtxt('loss2.dat')
metrics_loaded = np.loadtxt('metrics2.dat')

folder_path = f"Saved_plots/{run_id_number}"
os.makedirs(folder_path, exist_ok=True)

# # Verification
# mtmf.verify_basic_pcnn(f'{run_id_number}_FNN')
# mtmf.verify_run(states, losshistory, config, time_grid, ga_bodies=None, showplot=True, saveplots=True)

# save time+mass seperately
pcnn_mass = np.concatenate((time_grid_test, states[:, -1].reshape(-1, 1)), axis=1)
# save ND states (with time and without mass)
states_without_mass_ND = np.concatenate((time_grid_test, states[:, :-1]), axis=1)
states_without_mass_NDcartesian = coordinates_transformation_functions.radial_to_NDcartesian(states_without_mass_ND, config)
states_without_mass_NDcartesian_dict = {"NDcartesian": states_without_mass_NDcartesian}

control_nodes, ref_times, initial_state = mtmf.control_nodes_ref_times_3D_initial_state(states, config, time_grid_test)


verification_object = verification.Verification(config['m0'], config['t0'], config['tfinal'], initial_state, config['isp'], ga_bodies=None, central_body="Sun", control_nodes=control_nodes,
                                                verbose=True, ref_times=ref_times, mass_rate=True, config=config)
verification_object.integrate()


tudat_states_cartesian = verification_object.states_tudat
tudat_mass = verification_object.mass
tudat_states_NDcartesian = coordinates_transformation_functions.cartesian_to_NDcartesian(tudat_states_cartesian, config)
tudat_states_NDcartesian_dict = {"NDcartesian": tudat_states_NDcartesian}

# if showplot:
custom_labels = ["$r$", "$\\theta$", "$v_{r}$", "$v_{\\theta}$"]
plots.plot_compare_pcnn_tudat_states(states_without_mass_NDcartesian_dict, tudat_states_NDcartesian_dict, config, custom_labels=custom_labels, log=False, save=True)
plots.plot_compare_pcnn_tudat_mass(pcnn_mass, tudat_mass, config=config, save=True)

Dr, Dv, Dm, fuel_used, time_interval = mtmf.calculate_metrics_best_iteration(states_without_mass_NDcartesian, tudat_states_NDcartesian, pcnn_mass, tudat_mass, config=config)
# plots.plot_metrics_best_iteration_vs_time(Dr, Dv, Dm, fuel_used, time_interval, save=True)
plots.plot_metrics_vs_iterations(metrics_loaded, config, save=True)







# plots
plots.plot_trajectory_radialND_to_cartesianND(time_grid_test, train_state.best_y, r_target = 1.5, r_start = 1, N_arrows=100,  config=config)
# plots.plot_states(time_grid, train_state.best_y, config)
plots.plot_loss(losshistory_loaded, config)
plots.plot_resampled_time_grids_evolution(resampled_time_grids_savelist)
plt.show()

end_time = time.time()
print(f"Entire run took {np.round(end_time-start_time, 1)} s")
# plt.close()

"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, jax, paddle"""
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
from tudatpy.kernel.astro import time_conversion
from tudatpy.kernel.interface import spice
spice.load_standard_kernels()

start_time = time.time()

run_id_number = int(datetime.now().strftime("%Y%m%d%H%M%S"))
print('Run ID number:', run_id_number)

# Constants
MU_SUN  = 1.32712440042e20 # [m^3 s^-2] gravitational parameter of Sun
MU_MARS = 4.2828375816e13 # [m^3 s^-2]
# MU_JUPITER = 1.26686534e17
m0 = 100 # spacecraft initial mass
AU = 149597870700 # [m]
a = 10 # steepness parmater
umax = 1.0 # max allowable thrust [N]
isp = 3000 # specific impulse [s]
day = 86400 # s

# Non-dimensionalization parameters
length_scale = AU
vtheta0 = np.sqrt(MU_SUN/AU)
t_scale = length_scale / vtheta0

# Integration domain [E-M-V]
t0 = 0
t_leg1 = 188.58 * day # seconds
t_leg2 = 289.59 * day # seconds
M = 201 # Amount of collocation points
launch_date = time_conversion.DateTime(2015,4,13,12,00,00); launch_epoch = launch_date.epoch()
GA_epoch = launch_epoch + t_leg1
arrival_epoch = GA_epoch + t_leg2

# Loss weights
dyn_weight = 1
m_weigth = 1e-5 # mass term
o_weigth = 1e-7 # objective term
# ga_dyn_weight = 1e-8 # GA dynamics weight

# create config dictionary
config_leg1 = {"t0": t0,
          "tfinal": t_leg1,
          "length_scale": length_scale,
          "t_scale": t_scale,
          "isp": isp,
          "m0": m0,
          "M": M,
          "metrics": ["FinalDr", "FinalDv", "FinalDm", "Fuel used" ],
          "N_train": M,
          "N_test": M,
          "layer_architecture_FNN": [1, 20,20,20,20,20, 7],
          "layer_architecture_PFNN": [1, [10,10,10,10,10,10,10], [10,10,10,10,10,10,10], [10,10,10,10,10,10,10], 7],
          "loss_weights": [dyn_weight, dyn_weight, dyn_weight, dyn_weight, m_weigth, o_weigth],
          "mass": True,
          # "N_delta_max": 90 * (np.pi/180),
          # "N_beta_max": 20 * (np.pi/180),
          # "r_p_mars_max" : 0.578e9, # [m]
}

config_leg2 = {"t0": t0,
          "tfinal": t_leg2,
          "length_scale": length_scale,
          "t_scale": t_scale,
          "isp": isp,
          "m0": 70,
          "M": M,
          "metrics": ["FinalDr", "FinalDv", "FinalDm", "Fuel used" ],
          "N_train": M,
          "N_test": M,
          "layer_architecture_FNN": [1, 20,20,20,20,20, 7],
          "layer_architecture_PFNN": [1, [10,10,10,10,10,10,10], [10,10,10,10,10,10,10], [10,10,10,10,10,10,10], 7],
          "loss_weights": [dyn_weight, dyn_weight, dyn_weight, dyn_weight, m_weigth, o_weigth],
          "mass": True,
}
configs_list = [config_leg1, config_leg2]
# States
initial_state1 = mtmf.generate_2D_position_from_spice('Sun', 'Earth', 'ECLIPJ2000', launch_epoch, config_leg1, coordinates='NDradial')
final_state1 = mtmf.generate_2D_position_from_spice('Sun', 'Venus', 'ECLIPJ2000', GA_epoch, config_leg1, coordinates='NDradial')
initial_state2 = mtmf.generate_2D_position_from_spice('Sun', 'Venus', 'ECLIPJ2000', GA_epoch, config_leg2, coordinates='NDradial')
final_state2 = mtmf.generate_2D_position_from_spice('Sun', 'Mars', 'ECLIPJ2000', arrival_epoch, config_leg2, coordinates='NDradial')

# initial_state1[3] += 1600*t_scale/length_scale # Add 1.6 km/s excess velocity at launch
if final_state1[1] < initial_state1[1]:
    final_state1[1] += 2*np.pi
if final_state2[1] < initial_state2[1]:
    final_state2[1] += 2*np.pi

# Time Grid
time_grid_ND_leg1 = np.linspace(config_leg2['t0'] / config_leg1['t_scale'], config_leg1['tfinal'] / config_leg1['t_scale'], config_leg1['N_train'], dtype=dde.config.real(np)).reshape(-1, 1)
time_grid_ND_leg2 = np.linspace(config_leg2['t0'] / config_leg2['t_scale'], config_leg2['tfinal'] / config_leg2['t_scale'], config_leg2['N_train'], dtype=dde.config.real(np)).reshape(-1, 1)
time_grid_RT_leg1 = time_grid_ND_leg1*config_leg1['t_scale'];   time_grid_RT_leg1 += launch_epoch
# time_grid_RT_leg2 = time_grid_ND_leg2*config_leg2['t_scale'];   time_grid_RT_leg2 += GA_epoch
time_grids_list = [time_grid_ND_leg1, time_grid_ND_leg2]

# GA states
mars_states = mtmf.generate_2D_orbit_from_spice(time_grid_RT_leg1, 'Mars', config_leg1, coordinates='NDradial')

# tf variables
# r_p_scaled = tf.Variable(initial_value=0.002*config_leg1['r_p_mars_max']/length_scale1, trainable=True, dtype=tf.float32, name="r_p_scaled")
# delta_ga = tf.Variable(initial_value=0.3*config_leg1['N_delta_max'], trainable=True, dtype=tf.float32, name="delta_ga")  # Turning angle
# beta_ga = tf.Variable(initial_value=0.1*config_leg1['N_beta_max'], trainable=True, dtype=tf.float32, name="beta_ga")

v_sc_min_r = tf.Variable(initial_value=final_state1[2], trainable=True, dtype=tf.float32, name="v_sc_min_r")
v_sc_min_theta = tf.Variable(initial_value=final_state1[3], trainable=True, dtype=tf.float32, name="v_sc_min_theta")
v_sc_plus_r = tf.Variable(initial_value=initial_state2[2], trainable=True, dtype=tf.float32, name="v_sc_plus_r")
v_sc_plus_theta = tf.Variable(initial_value=initial_state2[3], trainable=True, dtype=tf.float32, name="v_sc_plus_theta")


ga_trainable_variables  = [v_sc_min_r, v_sc_min_theta, v_sc_plus_r, v_sc_plus_theta]
trackable_variables     = [v_sc_min_r, v_sc_min_theta, v_sc_plus_r, v_sc_plus_theta]

# Callback
class VariableValueWithHistory(dde.callbacks.VariableValue):
    def __init__(self, var_list, period=1, filename='trackable_variables', precision=2):
        super().__init__(var_list, period, filename, precision)
        self.history = []  # Store values across epochs

    # def on_train_begin(self):
    #     super().on_train_begin()
    #     self.history.append(self.value)

    def on_epoch_end(self):
        super().on_epoch_end()
        self.history.append(self.value)

    def get_history(self):
        return self.history

var_callback = VariableValueWithHistory(var_list=trackable_variables, period=1, precision=4)

def loss_function(t, y):
    x1_leg1 = y[:, 0:1]
    theta_leg1 = y[:, 1:2]
    x2_leg1 = y[:, 2:3]
    x3_leg1 = y[:, 3:4]
    ur_leg1 = y[:, 4:5]
    ut_leg1 = y[:, 5:6]
    m_leg1 = y[:, 6:7]

    # Thrust magnitude
    T = tf.reshape(tf.norm(y[:, 4:6], axis=1), (-1, 1))

    # LHS EOMs - Derivatives
    dx1_leg1_dt = dde.grad.jacobian(y, t, i=0);
    dtheta_leg1_dt = dde.grad.jacobian(y, t, i=1);
    dx2_leg1_dt = dde.grad.jacobian(y, t, i=2);
    dx3_leg1_dt = dde.grad.jacobian(y, t, i=3);
    dm_leg1_dt = dde.grad.jacobian(y, t, i=6);

    # RHS EOMs leg 1
    RHS_x1_leg1 = x2_leg1
    RHS_theta_leg1 = x3_leg1 / x1_leg1
    RHS_x2_leg1 = x3_leg1 ** 2 / x1_leg1 - (MU_SUN * config_leg1['t_scale'] ** 2 / config_leg1['length_scale'] ** 3) * x1_leg1 ** (-2) + (config_leg1['t_scale'] ** 2 / config_leg1['length_scale']) * ur_leg1 / m_leg1
    RHS_x3_leg1 = - (x2_leg1 * x3_leg1) / x1_leg1 + (config_leg1['t_scale'] ** 2 / config_leg1['length_scale']) * ut_leg1 / m_leg1
    RHS_m_leg1 = -T * config_leg1['t_scale'] / (isp * 9.81)

    # Return the residuals
    return [
        dx1_leg1_dt - RHS_x1_leg1,
        dtheta_leg1_dt - RHS_theta_leg1,
        dx2_leg1_dt - RHS_x2_leg1,
        dx3_leg1_dt - RHS_x3_leg1,
        dm_leg1_dt - RHS_m_leg1,
        ]

# N_d = config_leg1['N_d_max'] * 2*tf.math.sigmoid(delta_ga) -1 # To make sigmoid range [-1, 1]
# N_beta = config_leg1['N_beta_max'] * 2*tf.math.sigmoid(beta_ga) -1 # To make sigmoid range [-1, 1]
def constraint_layer1(t, y):
    # Leg 1 scaling functions
    c1 = tf.math.exp(-a * (t - t0))
    c2 = 1 - tf.math.exp(-a * (t - t0)) - tf.math.exp(a * (t - t_leg1 / config_leg1['t_scale']))
    c3 = tf.math.exp(a * (t - t_leg1 / config_leg1['t_scale']))
    c_mass = 1 - tf.math.exp(-a * (t - t0))

    # Control inputs for leg 1
    u_norm = tf.math.sigmoid(y[:, 4:5])
    u_angle = tf.math.tanh(y[:, 5:6])
    Nm = tf.math.sigmoid(y[:, 6:7])

    # Rescale the control inputs for leg 1
    u_norm = u_norm * umax
    u_angle = u_angle * 2 * np.pi
    ur = u_norm * tf.math.sin(u_angle)
    ut = u_norm * tf.math.cos(u_angle)

    # mass
    m = config_leg1['m0'] - c_mass * config_leg1['m0'] * Nm

    # Transform the states for both legs
    output = tf.concat([
        c1 * initial_state1[0] + c2 * y[:, 0:1] + c3 * final_state1[0],
        c1 * initial_state1[1] + c2 * y[:, 1:2] + c3 * final_state1[1],
        c1 * initial_state1[2] + c2 * y[:, 2:3] + c3 * v_sc_min_r,
        c1 * initial_state1[3] + c2 * y[:, 3:4] + c3 * v_sc_min_theta,
        ur,
        ut,
        m
    ], axis=1)
    return output
def constraint_layer2(t, y):
    # Leg 2 scaling functions
    c1 = tf.math.exp(-a * (t - t0))
    c2 = 1 - tf.math.exp(-a * (t - t0)) - tf.math.exp(a * (t - t_leg2 / config_leg2['t_scale']))
    c3 = tf.math.exp(a * (t - t_leg2 / config_leg2['t_scale']))
    c_mass = 1 - tf.math.exp(-a * (t - t0))

    # Control inputs for leg 1
    u_norm = tf.math.sigmoid(y[:, 4:5])
    u_angle = tf.math.tanh(y[:, 5:6])
    Nm = tf.math.sigmoid(y[:, 6:7])

    # Rescale the control inputs for leg 1
    u_norm_ = u_norm * umax
    u_angle_ = u_angle * 2 * np.pi
    ur = u_norm_ * tf.math.sin(u_angle_)
    ut = u_norm_ * tf.math.cos(u_angle_)

    # mass
    m = config_leg2['m0'] - c_mass * config_leg2['m0'] * Nm

    # Transform the states for both legs
    output = tf.concat([
        c1 * initial_state2[0] + c2 * y[:, 0:1] + c3 * final_state2[0],
        c1 * initial_state2[1] + c2 * y[:, 1:2] + c3 * final_state2[1],
        c1 * v_sc_plus_r       + c2 * y[:, 2:3] + c3 * final_state2[2],
        c1 * v_sc_plus_theta   + c2 * y[:, 3:4] + c3 * final_state2[3],
        ur,
        ut,
        m
    ], axis=1)
    return output
constraint_layers_list = [constraint_layer1, constraint_layer2]


#TODO implement minimum r_p constraint

lr_schedule = [(1e-2, 3000), (1e-3, 5000), (1e-4, 10000), (5e-3, 4000), (1e-4, 5000), (5e-3, 4000), (1e-4, 5000), (5e-3, 4000), (1e-4, 5000), (1e-5, 6000)]


# delta_t = (config['tfinal']/config['t_scale'] - config['t0']/config['t_scale']) / config['N_train'];    std = 0.2 * delta_t
# mtmf.restarter (config, loss_function, constraint_layer, lr_schedule, train_distribution="perturbed_uniform_tf", std=None, plot=True, save=True, N_attempts=60, run_id_number=run_id_number)
# losshistory, train_state = mtmf.single_run_with_restart(config_leg1, loss_function, constraint_layer, time_grid, lr_schedule, final_state1, initial_state2, ga_trainable_variables, var_callback,
#                                                         configs_list, GA_point=None, threshold=10.0, train_distribution="uniform", std=None, save=True, seed=None) # fill in seed=None for time dependent seed
losshistory1, train_state1, losshistory2, train_state2 = mtmf.parallel_pcnn_training(configs_list, loss_function, constraint_layers_list, time_grids_list, lr_schedule, final_state1, initial_state2, ga_trainable_variables,
                                                                                     var_callback, threshold=5.0, train_distribution="uniform", std=None, save=False, seed=None) # fill in seed=None for time dependent seed

var_history = var_callback.get_history()
# print("Tracked history of variable values:", var_history)

new_train_state_best_y = np.hstack((train_state1.best_y, train_state2.best_y))

# # Verification
mtmf.verify_run_sp2(new_train_state_best_y, configs_list, time_grids_list, ga_bodies=None, showplot=True, saveplots=True)

# plots
plots.plot_trajectory_radialND_to_cartesianND_sp2_T3(time_grids_list, new_train_state_best_y, thrust_scale=1.0, r_start = initial_state1[0]*length_scale, r_ga = final_state1[0]*length_scale,
                                                  r_target = final_state2[0]*length_scale, N_arrows=100, configs_list=configs_list)
plots.plot_states_T3(time_grids_list, new_train_state_best_y, config_leg1)
plots.plot_loss(losshistory1)
plots.plot_loss(losshistory2)
variable_names = ['v_sc_min_r', 'v_sc_min_theta', 'v_sc_plus_r', 'v_sc_plus_theta']
plots.plot_variable_history(var_history, variable_names=variable_names)

plt.show()

end_time = time.time()
print(f"Entire run took {np.round(end_time-start_time, 1)} s")


# np.savetxt(
#     "predicted_states.csv",       # File name
#     train_state.best_y,           # Array to save
#     delimiter=",",                # Delimiter (comma-separated)
#     fmt="%.6f",                   # Floating-point format with 6 decimal places
#     header="r, theta, v_r, v_theta, u_r, u_theta, m, r, theta, v_r, v_theta, u_r, u_theta, m",  # Column headers
#     comments=""                   # No '#' before the header
# )

v_r_final_leg_1       = train_state1.best_y[:,2][-1]
v_theta_final_leg_1   = train_state1.best_y[:,3][-1]
v_r_initial_leg_2     = train_state2.best_y[:,2][0]
v_theta_initial_leg_2 = train_state2.best_y[:,3][0]
mtmf.ga_delta_v(v_r_final_leg_1, v_theta_final_leg_1, v_r_initial_leg_2, v_theta_initial_leg_2, mars_states, delta_ga=None)























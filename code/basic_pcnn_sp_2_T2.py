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
m0 = 1000 # spacecraft initial mass
AU = 149597870700 # [m]
a = 10 # steepness parmater
umax = 1 # max allowable thrust [N]
isp = 3000 # specific impulse [s]

# Non-dimensionalization parameters
length_scale1 = AU
vtheta0 = np.sqrt(MU_SUN/AU)
t_scale1 = length_scale1 / vtheta0

# Integration domain [E-M-C]
launch_date = time_conversion.DateTime(2003,5,6,12,00,00); launch_epoch = launch_date.epoch()
GA_date = time_conversion.DateTime(2004,2,1,12,00,00); GA_epoch = GA_date.epoch()
arrival_date = time_conversion.DateTime(2006,6,12,12,00,00); arrival_epoch = arrival_date.epoch()

TOF_EM_days = (GA_epoch - launch_epoch)/86400 # 271
TOF_MC_days = (arrival_epoch - GA_epoch)/86400 # 862
t0 = 0
t_leg1 = (TOF_EM_days)*24*3600 # seconds
t_leg2 = (TOF_MC_days)*24*3600 # seconds
M = 201 # Amount of collocation points

# Loss weights
dyn_weight = 1
m_weigth = 1e-5 # mass term
o_weigth = 1e-2 # objective term
ga_dyn_weight = 1e-8 # GA dynamics weight

# create config dictionary
config_leg1 = {"t0": t0,
          "tfinal": t_leg1,
          "length_scale": length_scale1,
          "t_scale": t_scale1,
          "isp": isp,
          "m0": m0,
          "M": M,
          "metrics": ["FinalDr", "FinalDv", "FinalDm", "Fuel used" ],
          "N_train": M,
          "N_test": M,
          "layer_architecture_FNN": [1, 35,35,35,35,35,35, 14],
          "layer_architecture_PFNN": [1, [10,10,10,10,10,10,10], [10,10,10,10,10,10,10], [10,10,10,10,10,10,10], 14],
          "loss_weights": [dyn_weight, dyn_weight, dyn_weight, dyn_weight, m_weigth, dyn_weight, dyn_weight, dyn_weight, dyn_weight, m_weigth, o_weigth],
          "mass": True,
          "N_delta_max": 90 * (np.pi/180),
          # "N_beta_max": 20 * (np.pi/180),
          "r_p_mars_max" : 0.578e9, # [m]
}

length_scale2 = length_scale1 * (TOF_MC_days/TOF_EM_days)
t_scale2 = t_scale1 * (TOF_MC_days/TOF_EM_days)
config_leg2 = {"t0": t0,
          "tfinal": t_leg2,
          "length_scale": length_scale2,
          "t_scale": t_scale2,
          "isp": isp,
          # "m0": m0, # should not be used anymore
          "M": M,
          "metrics": ["FinalDr", "FinalDv", "FinalDm", "Fuel used" ],
          "N_train": M,
          "N_test": M,
          # "layer_architecture_FNN": [1, 40, 40, 40, 40, 40, 40, 14],
          # "layer_architecture_PFNN": [1, [10,10,10,10,10,10,10], [10,10,10,10,10,10,10], [10,10,10,10,10,10,10], 14],
          "loss_weights": [dyn_weight, dyn_weight, dyn_weight, dyn_weight, m_weigth, dyn_weight, dyn_weight, dyn_weight, dyn_weight, m_weigth, o_weigth],
          "mass": True,
          # "N_delta_max": 90 * (np.pi/180), # [rad]
          # "N_beta_max": 20 * (np.pi/180), # [rad]
          # "r_p_mars_max" : 0.578e9, # [m]
}
configs_list = [config_leg1, config_leg2]

# t_ga_fraction = TOF_EM_days / (TOF_EM_days+TOF_MC_days)
# t_ga = t_ga_fraction * tfinal/t_scale1
# GA_index = t_ga_fraction * config_leg1['N_train'];    GA_index = int(round(GA_index))

# States
initial_state1 = mtmf.generate_2D_position_from_spice('Sun', 'Earth', 'ECLIPJ2000', launch_epoch, config_leg1, coordinates='NDradial')
final_state1 = mtmf.generate_2D_position_from_spice('Sun', 'Mars', 'ECLIPJ2000', GA_epoch, config_leg1, coordinates='NDradial')
initial_state2 = mtmf.generate_2D_position_from_spice('Sun', 'Mars', 'ECLIPJ2000', GA_epoch, config_leg2, coordinates='NDradial')
final_state2 = mtmf.generate_2D_position_from_spice('Sun', 'Ceres', 'ECLIPJ2000', arrival_epoch, config_leg2, coordinates='NDradial')

initial_state1[3] += 1600*t_scale1/length_scale1 # Add 1.6 km/s excess velocity at launch
if final_state1[1] < initial_state1[1]:
    final_state1[1] += 2*np.pi
if initial_state2[1] < final_state1[1]:
    initial_state2[1] += 2 * np.pi
if final_state2[1] < initial_state2[1]:
    final_state2[1] += 2*np.pi

# Time Grid
uniform_points = np.linspace(config_leg1['t0'] / config_leg1['t_scale'], config_leg1['tfinal'] / config_leg1['t_scale'], config_leg1['N_train'], dtype=dde.config.real(np)).reshape(-1, 1)
time_grid = uniform_points
time_grid_real_time = time_grid*t_scale1
time_grid_real_time += launch_epoch

# GA states
mars_states = mtmf.generate_2D_orbit_from_spice(time_grid_real_time, 'Mars', config_leg1, coordinates='NDradial')

# tf variables
r_p_scaled = tf.Variable(initial_value=0.002*config_leg1['r_p_mars_max']/length_scale1, trainable=True, dtype=tf.float32, name="r_p_scaled")
delta_ga = tf.Variable(initial_value=0.3*config_leg1['N_delta_max'], trainable=True, dtype=tf.float32, name="delta_ga")  # Turning angle
# delta_ga2 = tf.Variable(initial_value=0.1*config_leg1['N_delta_max'], trainable=True, dtype=tf.float32, name="delta_ga2")
# beta_ga = tf.Variable(initial_value=0.1*config_leg1['N_beta_max'], trainable=True, dtype=tf.float32, name="beta_ga")

v_sc_min_r = tf.Variable(initial_value=final_state1[2], trainable=True, dtype=tf.float32, name="v_sc_min_r")
v_sc_min_theta = tf.Variable(initial_value=final_state1[3], trainable=True, dtype=tf.float32, name="v_sc_min_theta")
v_sc_plus_r = tf.Variable(initial_value=initial_state2[2], trainable=True, dtype=tf.float32, name="v_sc_plus_r")
v_sc_plus_theta = tf.Variable(initial_value=initial_state2[3], trainable=True, dtype=tf.float32, name="v_sc_plus_theta")


ga_trainable_variables  = [v_sc_min_r, v_sc_min_theta, v_sc_plus_r, v_sc_plus_theta]
trackable_variables     = [delta_ga, r_p_scaled, v_sc_min_r, v_sc_min_theta, v_sc_plus_r, v_sc_plus_theta]

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

MU_MARS_scaled_1 = MU_MARS * t_scale1 ** 2 / length_scale1 ** 3

def loss_function(t, y):
    x1_leg1 = y[:, 0:1]
    theta_leg1 = y[:, 1:2]
    x2_leg1 = y[:, 2:3]
    x3_leg1 = y[:, 3:4]
    ur_leg1 = y[:, 4:5]
    ut_leg1 = y[:, 5:6]
    m_leg1 = y[:, 6:7]

    x1_leg2 = y[:, 7:8]
    theta_leg2 = y[:, 8:9]
    x2_leg2 = y[:, 9:10]
    x3_leg2 = y[:, 10:11]
    ur_leg2 = y[:, 11:12]
    ut_leg2 = y[:, 12:13]
    m_leg2 = y[:, 13:14]

    # tf.print('tf.print(x1_leg1[-1][0])', x1_leg1[-1][0])
    # tf.print('tf.print(theta_leg1[-1])', theta_leg1[-1])
    # tf.print('tf.print(x2_leg1)', x2_leg1)
    # tf.print('tf.print(x3_leg1)', x3_leg1)
    # tf.print('tf.print(ur_leg1)', ur_leg1)
    # tf.print('tf.print(ut_leg1)', ut_leg1)
    # tf.print('tf.print(m_leg1)', m_leg1)
    # tf.print('tf.print(x1_leg2)', x1_leg2)
    # tf.print('tf.print(theta_leg2)', theta_leg2)
    # tf.print('tf.print(x2_leg2)', x2_leg2)
    # tf.print('tf.print(x3_leg2)', x3_leg2)
    # tf.print('tf.print(ur_leg2)', ur_leg2)
    # tf.print('tf.print(ut_leg2)', ut_leg2)
    # tf.print('tf.print(m_leg2)', m_leg2)

    # tf.print('tf.print(x1_leg2)', x1_leg2)

    # Thrust magnitude
    T_leg1 = tf.reshape(tf.norm(y[:, 4:6], axis=1), (-1, 1))
    T_leg2 = tf.reshape(tf.norm(y[:, 11:13], axis=1), (-1, 1))

    # LHS EOMs - Derivatives
    dx1_leg1_dt = dde.grad.jacobian(y, t, i=0);     dx1_leg2_dt = dde.grad.jacobian(y, t, i=7)
    dtheta_leg1_dt = dde.grad.jacobian(y, t, i=1);  dtheta_leg2_dt = dde.grad.jacobian(y, t, i=8)
    dx2_leg1_dt = dde.grad.jacobian(y, t, i=2);     dx2_leg2_dt = dde.grad.jacobian(y, t, i=9)
    dx3_leg1_dt = dde.grad.jacobian(y, t, i=3);     dx3_leg2_dt = dde.grad.jacobian(y, t, i=10)
    dm_leg1_dt = dde.grad.jacobian(y, t, i=6);      dm_leg2_dt = dde.grad.jacobian(y, t, i=13)

    # tf.print('dm_leg1_dt', dm_leg1_dt)
    # tf.print('dm_leg2_dt', dm_leg2_dt)

    # RHS EOMs leg 1
    RHS_x1_leg1 = x2_leg1
    RHS_theta_leg1 = x3_leg1 / x1_leg1
    RHS_x2_leg1 = x3_leg1 ** 2 / x1_leg1 - (MU_SUN * t_scale1 ** 2 / length_scale1 ** 3) * x1_leg1 ** (-2) + (t_scale1 ** 2 / length_scale1) * ur_leg1 / m_leg1
    RHS_x3_leg1 = - (x2_leg1 * x3_leg1) / x1_leg1 + (t_scale1 ** 2 / length_scale1) * ut_leg1 / m_leg1
    RHS_m_leg1 = -T_leg1 * t_scale1 / (isp * 9.81)

    # RHS EOMs leg 2
    RHS_x1_leg2 = x2_leg2
    RHS_theta_leg2 = x3_leg2 / x1_leg2
    RHS_x2_leg2 = x3_leg2 ** 2 / x1_leg2 - (MU_SUN * t_scale2 ** 2 / length_scale2 ** 3) * x1_leg2 ** (-2) + (t_scale2 ** 2 / length_scale2) * ur_leg2 / m_leg2
    RHS_x3_leg2 = - (x2_leg2 * x3_leg2) / x2_leg2 + (t_scale2 ** 2 / length_scale2) * ut_leg2 / m_leg2
    RHS_m_leg2 = -T_leg2 * t_scale2 / (isp * 9.81)

    # GA extension
    v_sc_min_vec    = tf.stack([v_sc_min_r, v_sc_min_theta]) # [v_r, v_theta]
    v_planet_vec    = tf.stack([mars_states[-1][2], mars_states[-1][3]]) # [v_r, v_theta]
    v_inf_min_vec   = v_sc_min_vec - v_planet_vec # [v_r, v_theta]
    v_inf_mag       = tf.norm(v_inf_min_vec) # v_inf_plus_mag = v_inf_min_mag
    # v_inf_min_angle = tf.math.atan2(v_inf_min_vec[0] , v_inf_min_vec[1])

    # tf.print('v_sc_min_vec', v_sc_min_vec)
    # tf.print('v_planet_vec', v_planet_vec)
    # tf.print('v_inf_min_vec', v_inf_min_vec)
    # tf.print('v_inf_min_angle', v_inf_min_angle)
    # tf.print('v_inf_mag', v_inf_mag)
    # tf.print('delta_ga', delta_ga)

    # v_inf_plus_angle= v_inf_min_angle + delta_ga

    # v_inf_plus_vec  = tf.stack([v_inf_mag*tf.math.sin(v_inf_plus_angle) , v_inf_mag*tf.math.cos(v_inf_plus_angle)]) # [v_r, v_theta]
    # v_sc_plus_vec   = v_inf_plus_vec + v_planet_vec # [v_r, v_theta]
    # v_sc_plus_r.assign(v_sc_plus_vec[0])
    # v_sc_plus_theta.assign(v_sc_plus_vec[1])

    # print('tf.math.sin(v_inf_plus_angle)', tf.math.sin(v_inf_plus_angle))
    # tf.print('v_inf_plus_vec', v_inf_plus_vec)
    # tf.print('v_sc_plus_vec', v_sc_plus_vec)
    # tf.print('v_sc_plus_vec_var', v_sc_plus_vec_var)

    # RHS_ga = 2*tf.math.asin( (MU_MARS_scaled_1) / ( (r_p_scaled)*(v_inf_mag**2) + MU_MARS_scaled_1) )

    # tf.print("delta_ga:", delta_ga)
    # tf.print("RHS_ga:", RHS_ga)
    # tf.print("delta_ga shape:", tf.shape(delta_ga))
    # tf.print("RHS_ga shape:", tf.shape(RHS_ga))
    # tf.print("asin input:", (MU_MARS) / (r_p * v_inf_mag ** 2 + MU_MARS))

    # Return the residuals
    return [
        dx1_leg1_dt - RHS_x1_leg1,
        dtheta_leg1_dt - RHS_theta_leg1,
        dx2_leg1_dt - RHS_x2_leg1,
        dx3_leg1_dt - RHS_x3_leg1,
        dm_leg1_dt - RHS_m_leg1,

        dx1_leg2_dt - RHS_x1_leg2,
        dtheta_leg2_dt - RHS_theta_leg2,
        dx2_leg2_dt - RHS_x2_leg2,
        dx3_leg2_dt - RHS_x3_leg2,
        dm_leg2_dt - RHS_m_leg2,
        # dm_leg2_dt - RHS_m_leg2,

        # tf.expand_dims(delta_ga - RHS_ga, axis=0)  # Fix dimension issue
        ]

# N_d = config_leg1['N_d_max'] * 2*tf.math.sigmoid(delta_ga) -1 # To make sigmoid range [-1, 1]
# N_beta = config_leg1['N_beta_max'] * 2*tf.math.sigmoid(beta_ga) -1 # To make sigmoid range [-1, 1]
def constraint_layer(t, y):
    # Leg 1 scaling functions
    c1 = tf.math.exp(-a * (t - t0))
    c2 = 1 - tf.math.exp(-a * (t - t0)) - tf.math.exp(a * (t - t_leg1 / t_scale1))
    c3 = tf.math.exp(a * (t - t_leg1 / t_scale1))
    c_mass = 1 - tf.math.exp(-a * (t - t0))

    # # Leg 2 scaling functions
    # c1_leg2 = tf.math.exp(-a * (t - t0))
    # c2_leg2 = 1 - tf.math.exp(-a * (t - t0)) - tf.math.exp(a * (t - tfinal_leg2 / t_scale2))
    # c3_leg2 = tf.math.exp(a * (t - tfinal_leg2 / t_scale2))
    # c_mass_leg2 = 1 - tf.math.exp(-a * (t - t0))

    # Control inputs for leg 1
    u_norm_leg1 = tf.math.sigmoid(y[:, 4:5])
    u_angle_leg1 = tf.math.tanh(y[:, 5:6])
    Nm_leg1 = tf.math.sigmoid(y[:, 6:7])

    # Control inputs for leg 2
    u_norm_leg2 = tf.math.sigmoid(y[:, 11:12])
    u_angle_leg2 = tf.math.tanh(y[:, 12:13])
    Nm_leg2 = tf.math.sigmoid(y[:, 13:14])

    # Rescale the control inputs for leg 1
    u_norm_leg1 = u_norm_leg1 * umax
    u_angle_leg1 = u_angle_leg1 * 2 * np.pi
    ur_leg1 = u_norm_leg1 * tf.math.sin(u_angle_leg1)
    ut_leg1 = u_norm_leg1 * tf.math.cos(u_angle_leg1)

    # Rescale the control inputs for leg 2
    u_norm_leg2 = u_norm_leg2 * umax
    u_angle_leg2 = u_angle_leg2 * 2 * np.pi
    ur_leg2 = u_norm_leg2 * tf.math.sin(u_angle_leg2)
    ut_leg2 = u_norm_leg2 * tf.math.cos(u_angle_leg2)

    # masses per leg
    m_leg1 = m0 - c_mass * m0 * Nm_leg1
    # tf.print('m_leg1[-1]', m_leg1[-1])
    m_leg2 = m_leg1[-1] - c_mass * m_leg1[-1] * Nm_leg2

    # Transform the states for both legs
    output_leg1 = tf.concat([
        c1 * initial_state1[0] + c2 * y[:, 0:1] + c3 * final_state1[0],
        c1 * initial_state1[1] + c2 * y[:, 1:2] + c3 * final_state1[1],
        c1 * initial_state1[2] + c2 * y[:, 2:3] + c3 * v_sc_min_r,
        c1 * initial_state1[3] + c2 * y[:, 3:4] + c3 * v_sc_min_theta,
        ur_leg1,
        ut_leg1,
        m_leg1
    ], axis=1)

    output_leg2 = tf.concat([
        c1 * initial_state2[0]                                                     + c2 * y[:, 7:8]   + c3 * final_state2[0],
        c1 * initial_state2[1]                                                     + c2 * y[:, 8:9]   + c3 * final_state2[1],
        c1 * v_sc_plus_r * (length_scale1/t_scale1) * (t_scale2/length_scale2)     + c2 * y[:, 9:10]  + c3 * final_state2[2],
        c1 * v_sc_plus_theta * (length_scale1/t_scale1) * (t_scale2/length_scale2) + c2 * y[:, 10:11] + c3 * final_state2[3],
        ur_leg2,
        ut_leg2,
        m_leg2
    ], axis=1)

    # Concatenate the outputs for both legs
    output = tf.concat([output_leg1, output_leg2], axis=1)
    return output

#TODO tudat mass leg 2 is weird <-gefixt
# check de hypothesis van thomas over loss minimizen, hoe werkt het eigenlijk?
# nu heb je initial states, dynamically valid, fuel dat omstebeurt word veranderd, eerdes had je alleen dynamically valid vs fuel
# Final state Ceres is weird...
lr_schedule = [(1e-2, 3000), (1e-3, 5000)]#, (1e-4, 10000), (5e-3, 4000), (1e-4, 5000), (5e-3, 4000), (1e-4, 5000), (5e-3, 4000), (1e-4, 5000), (1e-5, 6000)]

def single_run_with_restart(config, pde, constraint_layer, time_grid, lr_schedule, final_state1, initial_state2, ga_trainable_variables, callbacks=None, configs_list=None,
                            GA_point=None, threshold=5.0, N_attempts=50, train_distribution="uniform", std=None, plot=True, save=False, seed=None):
    attempt = 1
    if seed == None:
        seed = int(datetime.now().strftime("%Y%m%d%H%M%S"));    print("time-dependent random seed:", seed)
    else:
        print("manually entered seed:", seed)

    while attempt <= N_attempts:
        print("Initialisation attempt:", attempt);  print("seed:", seed)

        model = mtmf.create_model_sp2(configs_list, pde, constraint_layer, time_grid, seed=seed, train_distribution=train_distribution, std=std)  # PFNN 20241015143854
        # model = create_model(config, pde, constraint_layer, time_grid, GA_point, seed=seed, train_distribution=train_distribution, std=std)  # PFNN 20241015143854

        for (lr, iterations) in lr_schedule:
            print("Learning rate=", lr, "Iterations=", iterations)

            optimisation_alg = tf.keras.optimizers.Adam(learning_rate=lr)
            model.compile(optimisation_alg, lr=lr, external_trainable_variables=ga_trainable_variables)
            print('model external_trainable_variables:', model.external_trainable_variables)
            losshistory, train_state = model.train(iterations=iterations, display_every=1000, callbacks=[callbacks])  # , callbacks=[checkpoint_cb])

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
    #     save_loss_history(losshistory, f'loss.dat', verbose=True)
    #     metrics_lists = calculate_metrics_all_iterations(losshistory, time_grid, config)
    #     save_metrics_history(losshistory, metrics_lists, f'metrics.dat', verbose=True)
    # if plot:
    #     pass

    return losshistory, train_state

# delta_t = (config['tfinal']/config['t_scale'] - config['t0']/config['t_scale']) / config['N_train'];    std = 0.2 * delta_t
# mtmf.restarter (config, loss_function, constraint_layer, lr_schedule, train_distribution="perturbed_uniform_tf", std=None, plot=True, save=True, N_attempts=60, run_id_number=run_id_number)
# losshistory, train_state = mtmf.single_run_with_restart(config_leg1, loss_function, constraint_layer, time_grid, lr_schedule, final_state1, initial_state2, ga_trainable_variables, var_callback,
#                                                         configs_list, GA_point=None, threshold=10.0, train_distribution="uniform", std=None, save=True, seed=None) # fill in seed=None for time dependent seed
losshistory, train_state = single_run_with_restart(config_leg1, loss_function, constraint_layer, time_grid, lr_schedule, final_state1, initial_state2, ga_trainable_variables, var_callback,
                                                        configs_list, GA_point=None, threshold=10.0, train_distribution="uniform", std=None, save=True, seed=None) # fill in seed=None for time dependent seed

var_history = var_callback.get_history()
# print("Tracked history of variable values:", var_history)

# # Verification
mtmf.verify_run_sp2_T2(train_state.best_y, configs_list, time_grid, ga_bodies=None, showplot=True, saveplots=True)

# plots
plots.plot_trajectory_radialND_to_cartesianND_sp2_T2(time_grid, train_state.best_y, thrust_scale=0.1, r_start = initial_state1[0], r_ga = final_state1[0],
                                                  r_target = final_state2[0], N_arrows=100, configs_list=configs_list)
plots.plot_states(time_grid, train_state.best_y, config_leg1)
plots.plot_loss(losshistory)
variable_names = ['delta_ga', 'r_p_scaled', 'v_sc_min_r', 'v_sc_min_theta', 'v_sc_plus_r', 'v_sc_plus_theta']
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

v_r_final_leg_1       = train_state.best_y[:,2][-1]
v_theta_final_leg_1   = train_state.best_y[:,3][-1]
v_r_initial_leg_2     = train_state.best_y[:,9][0]
v_theta_initial_leg_2 = train_state.best_y[:,10][0]
mtmf.ga_delta_v(v_r_final_leg_1, v_theta_final_leg_1, v_r_initial_leg_2, v_theta_initial_leg_2, mars_states, delta_ga=None)























































# States dataset
x1 = train_state.best_y[:,0]
theta2 = train_state.best_y[:,1]

# Mars perturbation
# Mars states
mars_states = mtmf.generate_2D_orbit_from_spice(time_grid_real_time, 'Mars', config_leg1, coordinates='NDradial')
r_mars = mars_states[:,0];      r_mars = tf.cast(r_mars, dtype=tf.float32)
theta_mars = mars_states[:,1];  theta_mars = tf.cast(theta_mars, dtype=tf.float32)

theta1 = theta_mars
r1 = r_mars;    r2 = x1
d_theta1 = theta1 - theta2
d_theta1 = np.mod(d_theta1 + np.pi, 2 * np.pi) - np.pi
r_sc_m_magnitude = tf.sqrt(r2**2 + r1**2 - 2*r1*r2*tf.cos(d_theta1))
theta3 = tf.atan2( (r2*tf.sin(d_theta1)), (r1 - r2*tf.cos(d_theta1)) )
theta4 = theta3 + theta1

a_mars = (MU_MARS * t_scale1 ** 2 / length_scale1 ** 3) * r_sc_m_magnitude ** (-2)
a_sun  = (MU_SUN  * t_scale1 ** 2 / length_scale1 ** 3) * 1 ** (-2)

# Decompose acceleration vectors to polar coordinates with Sun at the origin
d_theta_2 = theta4 - theta2
a_mars_r = a_mars * tf.cos(d_theta_2)
a_mars_t = a_mars * tf.sin(d_theta_2)

def plot_mars_acceleration(r_mars, theta_mars, r_sc_m_magnitude, theta4, a_mars):
    """
    Plots the acceleration of Mars on the spacecraft as vectors in 2D space with uniform arrow sizes.

    Parameters:
        r_mars (array): Radial position of Mars (AU).
        theta_mars (array): Angular position of Mars (radians).
        r_sc_m_magnitude (array): Magnitude of spacecraft-Mars distance vector.
        theta4 (array): Angle of spacecraft-Mars distance vector (radians).
        a_mars (array): Acceleration magnitude (in AU/t^2 units).
    """
    # Convert Mars positions to Cartesian coordinates
    mars_x = r_mars * np.cos(theta_mars)
    mars_y = r_mars * np.sin(theta_mars)

    # Convert S/C positions to Cartesian coordinates
    sc_x = x1 * np.cos(theta2)
    sc_y = x1 * np.sin(theta2)

    # Compute acceleration vectors in Cartesian coordinates
    a_mars_x = a_mars * np.cos(theta4)
    a_mars_y = a_mars * np.sin(theta4)

    # Normalize acceleration vectors to a fixed length
    arrow_length = 0.02  # Set a fixed arrow length
    magnitude = np.sqrt(a_mars_x**2 + a_mars_y**2)
    a_mars_x_normalized = a_mars_x / magnitude * arrow_length
    a_mars_y_normalized = a_mars_y / magnitude * arrow_length

    # Plot Mars positions and acceleration vectors
    plt.figure(figsize=(10, 8))
    plt.plot(mars_x, mars_y, label="Mars Orbit", color="orange")
    plt.plot(sc_x, sc_y, label="S/C Orbit", color="blue")
    # Plot the Sun at the origin
    plt.scatter(0, 0, color="yellow", s=200, label="Sun")
    plt.quiver(
        sc_x, sc_y, a_mars_x_normalized, a_mars_y_normalized,
        angles='xy', scale_units='xy', scale=0.1, color='red',
        label='Mars Acceleration', width=0.001
    )

    # Add labels, legend, and grid
    plt.title("Acceleration of Mars on Spacecraft (Uniform Arrow Size)", fontsize=14)
    plt.xlabel("X (AU)", fontsize=12)
    plt.ylabel("Y (AU)", fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.show()

# plot_mars_acceleration(r_mars, theta_mars, r_sc_m_magnitude, theta4, a_mars)
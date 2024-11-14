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
          "metrics": ["FinalDr", "FinalDv", "FinalDm", "Fuel used" ],
          "N_train": M,
          "N_test": M,
          "layer_architecture_FNN": [1, 20, 20, 20, 20, 20, 7],
          "layer_architecture_PFNN": [1, [10,10,10,10,10,10,10], [10,10,10,10,10,10,10], [10,10,10,10,10,10,10], 7],
          "loss_weights": [dyn_weight, dyn_weight, dyn_weight, dyn_weight, m_weigth, o_weigth],
          "mass": True
}
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
    L_o = (1 / (isp * 9.81)) * 0.5 * (T[:-1] * t_scale + T[1:] * t_scale) * delta_t

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
        dm_dt - RHS_m,
        # L_o
        ]
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

lr_schedule = [(1e-2, 3000), (1e-3, 5000), (1e-4, 10000), (5e-3, 4000), (1e-4, 5000), (5e-3, 4000), (1e-4, 5000), (5e-3, 4000), (1e-4, 5000), (1e-5, 6000)]

delta_t = (config['tfinal']/config['t_scale'] - config['t0']/config['t_scale']) / config['N_train'];    std = 0.2 * delta_t
# mtmf.restarter (config, pde, constraint_layer, lr_schedule, train_distribution="perturbed_uniform_tf", std=None, plot=True, save=True, N_attempts=60, run_id_number=run_id_number)
losshistory, train_state = mtmf.single_run(config, pde, constraint_layer, lr_schedule, train_distribution="uniform", std=None, save=True, seed=20241114154348) # fill in seed=None for time dependent seed

# Verification
# mtmf.verify_basic_pcnn(f'{run_id_number}_FNN')
mtmf.verify_run(train_state.best_y, losshistory, config, showplot=True, saveplots=True)

# plots
plots.plot_trajectory_radialND_to_cartesianND(train_state.best_y, r_target = 1.5, r_start = 1, N_arrows=100,  config=config)
plots.plot_states(train_state.best_y, config)
plots.plot_loss(losshistory, mass=config['mass'])
plt.show()

end_time = time.time()
print(f"Entire run took {np.round(end_time-start_time, 1)} s")
plt.close()
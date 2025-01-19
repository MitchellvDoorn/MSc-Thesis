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
MU_SUN  = 1.32712440042e20 # gravitational parameter of Sun
MU_MARS = 4.2828375816e13
# MU_MARS = 1.26686534e17
m0 = 568 # spacecraft initial mass
AU = 149597870700 # [m]
a = 30 # steepness parmater
umax = 1 # max allowable thrust [N]
isp = 3000 # specific impulse [s]

# Non-dimensionalization parameters
length_scale = AU
vtheta0 = np.sqrt(MU_SUN/AU)
t_scale = length_scale / vtheta0


# Integration domain [E-M-C]
launch_date = time_conversion.DateTime(2003,5,6,12,00,00); launch_epoch = launch_date.epoch()
GA_date = time_conversion.DateTime(2004,2,1,12,00,00); GA_epoch = GA_date.epoch()
arrival_date = time_conversion.DateTime(2006,6,12,12,00,00); arrival_epoch = arrival_date.epoch()

TOF_EM_days = (GA_epoch - launch_epoch)/86400 # 271
TOF_MC_days = (arrival_epoch - GA_epoch)/86400 # 862
t0 = 0
tfinal = (TOF_EM_days+TOF_MC_days)*24*3600 # Constant time of flight
M = 201 # Amount of collocation points
GA_point = TOF_EM_days / (TOF_EM_days+TOF_MC_days) * tfinal/t_scale

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
          "N_train": M, # boundary points are added in the creation of the model in the 'data' variable
          "N_test": M, # including 2 points for the boundary
          "layer_architecture_FNN": [1, 20, 20, 20, 20, 20, 7],
          "layer_architecture_PFNN": [1, [10,10,10,10,10,10,10], [10,10,10,10,10,10,10], [10,10,10,10,10,10,10], 7],
          "loss_weights": [dyn_weight, dyn_weight, dyn_weight, dyn_weight, m_weigth, o_weigth],
          "mass": True
}

# States
initial_state = mtmf.generate_2D_position_from_spice('Sun', 'Earth', 'ECLIPJ2000', launch_epoch, config, coordinates='NDradial')
GA_state = mtmf.generate_2D_position_from_spice('Sun', 'Mars', 'ECLIPJ2000', GA_epoch ,config, coordinates='NDradial')
final_state = mtmf.generate_2D_position_from_spice('Sun', 'Ceres', 'ECLIPJ2000', arrival_epoch ,config, coordinates='NDradial')

# GA_state[0] -= 0.01 # = 1 495 978 km above surface, but Mars has SoI of 500 000 km -> to not end at Mars cg exactly, because then r becomes 0 and acceleration becomes inf
# GA_state[3] += 0.1
initial_state[3] += 1600*t_scale/length_scale # Add 1.6 km/s excess velocity at launch
if GA_state[1] < initial_state[1]:
    GA_state[1] += 2*np.pi
if final_state[1] < GA_state[1]:
    final_state[1] += 2*np.pi

# Time Grid
uniform_points = np.linspace(config['t0'] / config['t_scale'], config['tfinal'] / config['t_scale'], config['N_train'], dtype=dde.config.real(np)).reshape(-1, 1)

time_grid = uniform_points

t_focus = GA_point  # Target focus point
focus_width = (config['tfinal'] / config['t_scale']) * 0.02  # Define a small range around the focus point
extra_points_amount = 10
extra_points = np.linspace(t_focus - focus_width, t_focus + focus_width, extra_points_amount).reshape(-1, 1)
all_points = np.vstack((uniform_points, extra_points))
time_grid_mars = np.sort(all_points.astype(np.float32), axis=0)
time_grid_real_time = all_points*t_scale
time_grid_real_time += launch_epoch

# Mars states
mars_states = mtmf.generate_2D_orbit_from_spice(time_grid_real_time, 'Mars', config, coordinates='NDradial')
r_mars = mars_states[:,0];      r_mars = tf.cast(r_mars, dtype=tf.float32)
theta_mars = mars_states[:,1];  theta_mars = tf.cast(theta_mars, dtype=tf.float32)

r_p_scaled = tf.Variable(initial_value=GA_state[0]+0.001, trainable=False, dtype=tf.float32, name="r_p_scaled")
trainable_variables  = [r_p_scaled]

def loss_function(t, y):
    x1 = tf.maximum(1e-6, y[:, 0:1])
    theta2 = tf.maximum(1e-6, y[:, 1:2])
    x2 = tf.maximum(1e-6, y[:, 2:3])
    x3 = tf.maximum(1e-6, y[:, 3:4])
    ur = tf.clip_by_value(y[:, 4:5], 1e-8, umax)
    ut = tf.clip_by_value(y[:, 4:5], 1e-8, umax)
    m = tf.maximum(1e-2, y[:, 6:7])

    # tf.print('min:')
    # tf.print("y[:, 0:1]:", tf.math.reduce_min (y[:, 0:1]), summarize=-1)
    # tf.print("y[:, 1:2]:", tf.math.reduce_min (y[:, 1:2]), summarize=-1)
    # tf.print("y[:, 2:3]:", tf.math.reduce_min (y[:, 2:3]), summarize=-1)
    # tf.print("y[:, 3:4]:", tf.math.reduce_min (y[:, 3:4]), summarize=-1)
    # tf.print("y[:, 4:5]:", tf.math.reduce_min (y[:, 4:5]), summarize=-1)
    # tf.print("y[:, 5:6]:", tf.math.reduce_min (y[:, 5:6]), summarize=-1)
    # tf.print("y[:, 6:7]:", tf.math.reduce_min (y[:, 6:7]), summarize=-1)
    # tf.print('max:')
    # tf.print("y[:, 0:1]:", tf.math.reduce_max(y[:, 0:1]), summarize=-1)
    # tf.print("y[:, 1:2]:", tf.math.reduce_max(y[:, 1:2]), summarize=-1)
    # tf.print("y[:, 2:3]:", tf.math.reduce_max(y[:, 2:3]), summarize=-1)
    # tf.print("y[:, 3:4]:", tf.math.reduce_max(y[:, 3:4]), summarize=-1)
    # tf.print("y[:, 4:5]:", tf.math.reduce_max(y[:, 4:5]), summarize=-1)
    # tf.print("y[:, 5:6]:", tf.math.reduce_max(y[:, 5:6]), summarize=-1)
    # tf.print("y[:, 6:7]:", tf.math.reduce_max(y[:, 6:7]), summarize=-1)

    # Thrust magnitude
    T = tf.reshape(tf.norm(y[:, 4:6], axis=1), (-1, 1))

    # Mars perturbation
    theta1 = tf.expand_dims(theta_mars, axis=1)
    r1 = tf.expand_dims(r_mars, axis=1)
    r2 = x1
    if len(r1) == len(r2) and len(theta1) == len(theta2):
        d_theta1 = theta1 - theta2
        r_sc_m_magnitude = tf.sqrt(r2 ** 2 + r1 ** 2 - 2 * r1 * r2 * tf.cos(d_theta1))
        r_sc_m_magnitude = tf.maximum(r_sc_m_magnitude, 1e-6)
        theta3 = tf.atan((r2 * tf.sin(d_theta1)) / (r1 - r2 * tf.cos(d_theta1)))
        theta4 = theta3 + theta1

        a_mars = (MU_MARS * t_scale ** 2 / length_scale ** 3) * r_sc_m_magnitude ** (-2)

        # Decompose acceleration vectors to polar coordinates with Sun at the origin
        d_theta_2 = theta4 - theta2
        a_mars_r = a_mars * tf.cos(d_theta_2)
        a_mars_t = a_mars * tf.sin(d_theta_2)
    else:
        a_mars_r = 0
        a_mars_t = 0

    # tf.print(len(theta1), len(theta2), len(a_mars_t), len(a_mars_r), len(a_mars), len(d_theta_2) )
    # tf.print(len(ut), len(ur), len(m), len(x2), len(x3), len(x1))
    # tf.print('a_mars:', a_mars)
    # tf.print('a_mars[-6:]:', a_mars[-6:])

    # LHS EOMs - Derivatives
    dx1_dt = dde.grad.jacobian(y, t, i=0)
    dtheta_dt = dde.grad.jacobian(y, t, i=1)
    dx2_dt = dde.grad.jacobian(y, t, i=2)
    dx3_dt = dde.grad.jacobian(y, t, i=3)
    dm_dt = dde.grad.jacobian(y, t, i=6)

    # tf.print("a_mars:", a_mars)

    # RHS EOMs
    RHS_x1 = x2
    RHS_theta = x3 / x1
    RHS_x2 = x3 ** 2 / x1 - (MU_SUN * t_scale ** 2 / length_scale ** 3) * x1 ** (-2) + (t_scale ** 2 / length_scale) * ur / m + a_mars_r
    RHS_x3 = - (x2 * x3) / x1 + (t_scale ** 2 / length_scale) * ut / m + a_mars_t
    RHS_m = -T * t_scale / (isp * 9.81)

    # Return the residuals
    return [
        dx1_dt - RHS_x1,
        dtheta_dt - RHS_theta,
        dx2_dt - RHS_x2,
        dx3_dt - RHS_x3,
        dm_dt - RHS_m,
        ]

b = 5;  theta_mid = 0.0;
GA_index = TOF_EM_days / (TOF_EM_days + TOF_MC_days) * config['N_test'];    GA_index = int(round(GA_index))
beta = theta_mid - (1 - np.exp(-a * (GA_point - t0)) - np.exp(a * (GA_point - tfinal)))
def constraint_layer(t, y):

    c1 = tf.math.exp(-a * (t - t0))
    c2 = 1 - tf.math.exp(-a * (t - t0/t_scale)) - tf.math.exp(a * (t - tfinal/t_scale))
    c2_adjustment = beta * tf.exp(-b * (t - GA_point) ** 2)
    c3 = tf.math.exp(a * (t - tfinal/t_scale))
    c_mass = 1 - tf.math.exp(-a * (t - t0))

    # Apply sigmoid to get in [0, 1], while keeping a non-zero derivative for training
    u_norm = tf.math.sigmoid(y[:, 4:5])
    u_angle = tf.math.tanh(y[:, 5:6])
    Nm = tf.math.sigmoid(y[:, 6:7])
    # r_ga = GA_state[0] - (tf.math.sigmoid(y[:, 7:8])*0.001)
    # tf.print(r_p_scaled)
    # r_ga_mean = tf.reduce_mean(r_ga)

    # Rescale the U_R and the U_theta to their real values
    u_norm = u_norm * umax
    u_angle = u_angle * 2 * np.pi

    # Transform the control to cartesian coordinates
    ur = u_norm * tf.math.sin(u_angle)
    ut = u_norm * tf.math.cos(u_angle)

    output = tf.concat([c1 * initial_state[0] + (c2+c2_adjustment) * y[:, 0:1] + c3 * final_state[0]- c2_adjustment * r_p_scaled,
                        c1 * initial_state[1] + (c2+c2_adjustment) * y[:, 1:2] + c3 * final_state[1] - c2_adjustment * GA_state[1],
                        c1 * initial_state[2] + c2 * y[:, 2:3] + c3 * final_state[2],
                        c1 * initial_state[3] + c2 * y[:, 3:4] + c3 * final_state[3],
                        ur,
                        ut,
                        m0 - c_mass * m0 * Nm], axis=1
                       )

    # output = tf.tensor_scatter_nd_update(output, [[GA_index, 0]], [GA_state[0]])
    # # tf.print(output2[GA_index, 0])

    return output


lr_schedule = [(1e-2, 3000), (1e-3, 5000)]#, (1e-4, 10000), (5e-3, 4000), (1e-4, 5000), (5e-3, 4000), (1e-4, 5000), (5e-3, 4000), (1e-4, 5000), (1e-5, 6000)]

delta_t = (config['tfinal']/config['t_scale'] - config['t0']/config['t_scale']) / config['N_train'];    std = 0.2 * delta_t
# mtmf.restarter (config, loss_function, constraint_layer, lr_schedule, train_distribution="perturbed_uniform_tf", std=None, plot=True, save=True, N_attempts=60, run_id_number=run_id_number)
losshistory, train_state, time_grid = mtmf.single_run_with_restart_tp_resampling(config, loss_function, constraint_layer, time_grid, lr_schedule, trainable_variables, GA_point, threshold=5.0,
                                                                                 train_distribution="uniform", std=None, save=True, seed=None) # fill in seed=None for time dependent seed

# # Verification
# # mtmf.verify_basic_pcnn(f'{run_id_number}_FNN')
# mtmf.verify_run(train_state.best_y, losshistory, config, time_grid, ga_bodies=None, showplot=True, saveplots=True)
# plots
plots.plot_trajectory_radialND_to_cartesianND_sp2_T1(time_grid, train_state.y_pred_test, mars_states, GA_index, thrust_scale=0.1, r_start = initial_state[0], r_ga = GA_state[0], r_target = final_state[0], N_arrows=100, config=config)
# plots.plot_states(time_grid, train_state.best_y, config)
plots.plot_loss(losshistory)
plt.show()

end_time = time.time()
print(f"Entire run took {np.round(end_time-start_time, 1)} s")




# States dataset
x1 = train_state.best_y[:,0]
theta2 = train_state.best_y[:,1]

# Mars perturbation
theta1 = theta_mars
r1 = r_mars;    r2 = x1
d_theta1 = theta1 - theta2
d_theta1 = np.mod(d_theta1 + np.pi, 2 * np.pi) - np.pi
r_sc_m_magnitude = tf.sqrt(r2**2 + r1**2 - 2*r1*r2*tf.cos(d_theta1))
theta3 = tf.atan2( (r2*tf.sin(d_theta1)), (r1 - r2*tf.cos(d_theta1)) )
theta4 = theta3 + theta1

a_mars = (MU_MARS * t_scale ** 2 / length_scale ** 3) * r_sc_m_magnitude ** (-2)
a_sun  = (MU_SUN  * t_scale ** 2 / length_scale ** 3) * 1 ** (-2)

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

plot_mars_acceleration(r_mars, theta_mars, r_sc_m_magnitude, theta4, a_mars)
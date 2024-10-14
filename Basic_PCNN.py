"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, jax, paddle"""
import deepxde as dde
import numpy as np
# Import tf if using backend tensorflow.compat.v1 or tensorflow
from deepxde.backend import tf
import master_thesis_mitchell_functions as mtmf
import matplotlib.pyplot as plt

# Constants
mu = 1.32712440042e20 # gravitational parameter of Sun
m0 = 100 # spacecraft initial mass
AU = 149597870700 # [m]
a = 10 # steepness parmater
umax = 0.1 # max allowable thrust
isp = 2500 # specific impulse

# Initial state
r0 = AU
theta0 = 0
vr0 = 0
vtheta0 = np.sqrt(mu/r0)
initial_state = np.array([r0, theta0, vr0, vtheta0])

# Final state
rfinal = 1.5*r0
theta_final = 4*np.pi
vr_final = 0
vtheta_final = np.sqrt(mu/rfinal)
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
    RHS_x2 = x3 ** 2 / x1 - (mu * t_scale ** 2 / length_scale ** 3) * x1 ** (-2) + (t_scale ** 2 / length_scale) * ur / m
    RHS_x3 = - (x2 * x3) / x1 + (t_scale ** 2 / length_scale) * ut / m
    RHS_m = -T * t_scale / (isp * 9.81)

    # Return the residuals
    return [
        dx1_dt - RHS_x1,
        dtheta_dt - RHS_theta,
        dx2_dt - RHS_x2,
        dx3_dt - RHS_x3,
        dm_dt - RHS_m,
        L_o
        ]

def my_constraint_layer(t, y):

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

geom = dde.geometry.TimeDomain(t0/t_scale, tfinal/t_scale)

data = dde.data.PDE(geom, pde, [], M, 2, num_test=M, train_distribution="uniform")


initializer = tf.keras.initializers.GlorotNormal
net = dde.nn.FNN([1, 20, 20, 20, 20, 7], "sin", initializer)
# net.regularizer = OptimalFuel.call
net.apply_output_transform(my_constraint_layer)



# Define a custom learning rate schedule as a subclass of LearningRateSchedule
class CustomLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate):
        self.initial_learning_rate = initial_learning_rate

    def __call__(self, step):
        step = tf.cast(step, tf.float32)  # Ensure step is a tensor
        lr = tf.cond(step < 3000, lambda: tf.constant(1e-2),
                     lambda: tf.cond(step < 13000, lambda: tf.constant(1e-3),
                                     lambda: tf.cond(step < 23000, lambda: tf.constant(1e-4),
                                                     lambda: tf.cond(step < 27000, lambda: tf.constant(5e-3),
                                                                     lambda: tf.cond(step < 32000, lambda: tf.constant(1e-4),
                                                                                     lambda: tf.cond(step < 36000, lambda: tf.constant(5e-3),
                                                                                                     lambda: tf.cond(step < 41000, lambda: tf.constant(1e-4),
                                                                                                                     lambda: tf.cond(step < 45000, lambda: tf.constant(5e-3),
                                                                                                                                     lambda: tf.cond(step < 50000, lambda: tf.constant(1e-4),
                                                                                                                                                     lambda: tf.constant(1e-5))))))))))
        return lr
class LearningRateLogger(dde.callbacks.Callback):
    def on_epoch_end(self):
        lr = self.model.optimizer._decayed_lr(tf.float32).numpy()
        print(f"Epoch {self.model.train_state.epoch}: Learning Rate = {lr}")


lr_schedule = CustomLRSchedule(initial_learning_rate=1e-2)
mweigth = 1e-5 # mass term
oweigth = 1e-7 # objective term

optimisation_alg = tf.keras.optimizers.Adam(learning_rate=lr_schedule) #, beta_1 = 0.9, beta_2 = 0.999)

model = dde.Model(data, net)
model.compile(optimisation_alg, lr=lr_schedule, loss_weights=[1, 1, 1, 1, mweigth, oweigth])
# model.train(iterations=15000)
# model.compile("L-BFGS")
losshistory, train_state = model.train(iterations=56000)

#
t = np.linspace(0, tfinal/t_scale, M)
# t = t.reshape(M, 1)
# sol_pred = model.predict(t)





dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# Rescale and plot trajectory
train_state.best_y[:,0] *= length_scale
train_state.best_y[:,1] *= 1
train_state.best_y[:,2] *= (length_scale/t_scale)
train_state.best_y[:,3] *= (length_scale/t_scale)

mtmf.plot_trajectory_polar_to_cart(0, train_state.best_y, tfinal, "Trajectory plot")
mtmf.plot_states(t, train_state)
import tensorflow as tf
import numpy as np
# import matplotlib
# matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
dpi_setting = 600


@tf.keras.utils.register_keras_serializable()
class MyConstraintLayer(tf.keras.layers.Layer):
    def __init__(self, time_grid, t0, tf_final, r0, rf, theta0, thetaf, v0r, vfr, v0theta, vftheta, m0, a, umax, **kwargs):
        super(MyConstraintLayer, self).__init__(**kwargs)  # Accepting and passing additional arguments
        self.time_grid = tf.cast(time_grid, tf.float32)
        self.t0 = tf.cast(t0, tf.float32)
        self.tf_final = tf.cast(tf_final, tf.float32)
        self.r0 = tf.cast(r0, tf.float32)
        self.rf = tf.cast(rf, tf.float32)
        self.theta0 = tf.cast(theta0, tf.float32)
        self.thetaf = tf.cast(thetaf, tf.float32)
        self.v0r = tf.cast(v0r, tf.float32)
        self.vfr = tf.cast(vfr, tf.float32)
        self.v0theta = tf.cast(v0theta, tf.float32)
        self.vftheta = tf.cast(vftheta, tf.float32)
        self.m0 = tf.cast(m0, tf.float32)
        self.a = tf.cast(a, tf.float32)
        self.umax = tf.cast(umax, tf.float32)

    def build(self, input_shape):
        #self.kernel = self.add_weight("kernel", shape=[int(input_shape[-1]), self.num_outputs])
        pass

    def call(self, inputs):
      Nr, Ntheta, Nvr, Nvtheta, Nu_phi, Nu_T, Nm = tf.split(inputs, num_or_size_splits=7, axis=-1)
      t = tf.cast(tf.reshape(self.time_grid, (-1, 1)), tf.float32)

      r = tf.exp(-self.a * (t - self.t0)) * self.r0 + (1 - tf.exp(-self.a * (t - self.t0)) - tf.exp(self.a * (t - self.tf_final))) * Nr + tf.exp(self.a * (t - self.tf_final)) * self.rf
      theta = tf.exp(-self.a * (t - self.t0)) * self.theta0 + (1 - tf.exp(-self.a * (t - self.t0)) - tf.exp(self.a * (t - self.tf_final))) * Ntheta + tf.exp(self.a * (t - self.tf_final)) * self.thetaf
      vr =     tf.exp(-self.a * (t - self.t0)) * self.v0r + (1 - tf.exp(-self.a * (t - self.t0)) - tf.exp(self.a * (t - self.tf_final))) * Nvr + tf.exp(self.a * (t - self.tf_final)) * self.vfr
      vtheta = tf.exp(-self.a * (t - self.t0)) * self.v0theta + (1 - tf.exp(-self.a * (t - self.t0)) - tf.exp(self.a * (t - self.tf_final))) * Nvtheta + tf.exp(self.a * (t - self.tf_final)) * self.vftheta

      u_phi = 2 * np.pi * tf.math.tanh(Nu_phi)
      u_T = self.umax * tf.math.sigmoid(Nu_T)

      ur = u_T*tf.math.sin(u_phi)
      ut = u_T*tf.math.cos(u_phi)

      m = self.m0 - (1 - tf.exp(-self.a * (t - self.t0))) * self.m0 * tf.math.sigmoid(Nm)
      return tf.concat([r, theta, vr, vtheta, ur, ut, m], axis=-1)

    def get_config(self):
      config = super(MyConstraintLayer, self).get_config()
      config.update({
          'time_grid': self.time_grid.numpy().tolist(),  # Convert tensor to numpy, then to list
          't0': self.t0.numpy().tolist(),
          'tf_final': self.tf_final.numpy().tolist(),
          'r0': self.r0.numpy().tolist(),
          'rf': self.rf.numpy().tolist(),
          'theta0': self.theta0.numpy().tolist(),
          'thetaf': self.thetaf.numpy().tolist(),
          'v0r': self.v0r.numpy().tolist(),
          'vfr': self.vfr.numpy().tolist(),
          'v0theta': self.v0theta.numpy().tolist(),
          'vftheta': self.vftheta.numpy().tolist(),
          'm0': self.m0.numpy().tolist(),
          'a': self.a.numpy().tolist(),
          'umax': self.umax.numpy().tolist()
      })
      return config

    @classmethod
    def from_config(cls, config):
      config['time_grid'] = tf.convert_to_tensor(config['time_grid'], dtype=tf.float32)
      config['t0'] = tf.convert_to_tensor(config['t0'], dtype=tf.float32)
      config['tf_final'] = tf.convert_to_tensor(config['tf_final'], dtype=tf.float32)
      config['r0'] = tf.convert_to_tensor(config['r0'], dtype=tf.float32)
      config['rf'] = tf.convert_to_tensor(config['rf'], dtype=tf.float32)
      config['theta0'] = tf.convert_to_tensor(config['theta0'], dtype=tf.float32)
      config['thetaf'] = tf.convert_to_tensor(config['thetaf'], dtype=tf.float32)
      config['v0r'] = tf.convert_to_tensor(config['v0r'], dtype=tf.float32)
      config['vfr'] = tf.convert_to_tensor(config['vfr'], dtype=tf.float32)
      config['v0theta'] = tf.convert_to_tensor(config['v0theta'], dtype=tf.float32)
      config['vftheta'] = tf.convert_to_tensor(config['vftheta'], dtype=tf.float32)
      config['m0'] = tf.convert_to_tensor(config['m0'], dtype=tf.float32)
      config['a'] = tf.convert_to_tensor(config['a'], dtype=tf.float32)
      config['umax'] = tf.convert_to_tensor(config['umax'], dtype=tf.float32)
      return cls(**config)

@tf.keras.utils.register_keras_serializable()
def custom_loss_NOT_USED(true_states, predictions, time_grid_tensor, model, M, MU_scaled, Isp, g0, weight_dynamics, weight_mass, weight_objective):
    # data loss
    # data_loss_value = tf.reduce_mean(tf.abs(predictions - true_states))

    # time derivatives
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(time_grid_tensor)
        predictions = model(time_grid_tensor)

        # r_pred has shape (M+1,),    r_pred_dot has shape (M+1,1) even without reshaping r_pred to (M+1,1),    time_grid_tensor has shape (M+1,1)
        r_pred      = tf.reshape(predictions[:, 0], [M+1,1])            ;  r_pred_dot       = tape.gradient(r_pred      , time_grid_tensor)
        theta_pred  = tf.reshape(predictions[:, 1], [M+1,1])            ;  theta_pred_dot   = tape.gradient(theta_pred  , time_grid_tensor)
        vr_pred     = tf.reshape(predictions[:, 2], [M+1,1])            ;  vr_pred_dot      = tape.gradient(vr_pred     , time_grid_tensor)
        vtheta_pred = tf.reshape(predictions[:, 3], [M+1,1])            ;  vtheta_pred_dot  = tape.gradient(vtheta_pred , time_grid_tensor)
        u_phi_pred  = tf.reshape(predictions[:, 4], [M+1,1])
        u_T_pred    = tf.reshape(predictions[:, 5], [M+1,1])
        m_pred      = tf.reshape(predictions[:, 6], [M+1,1])            ;  m_pred_dot       = tape.gradient(m_pred      , time_grid_tensor)

    # physics losses
    L_r =      tf.reduce_mean(tf.square(r_pred_dot - vr_pred))  # (M+1, 1) - (M+1) = (M+1,M+1)         (M+1, 1) - (M+1,1) = (M+1,1)
    L_theta =  tf.reduce_mean(tf.square(theta_pred_dot - vtheta_pred / r_pred))
    L_vr =     tf.reduce_mean(tf.square(vr_pred_dot - ( ((vtheta_pred ** 2) / r_pred) + (MU_scaled / (r_pred ** 2)) - (u_T_pred * tf.sin(u_phi_pred)) / m_pred) ) )
    L_vtheta = tf.reduce_mean(tf.square(vtheta_pred_dot + ((vr_pred * vtheta_pred) / r_pred)                 - ((u_T_pred * tf.cos(u_phi_pred)) / m_pred) ) )

    delta_t = time_grid_tensor[1:] - time_grid_tensor[:-1]
    L_o = (1 / (Isp * g0)) * tf.reduce_sum(0.5 * (u_T_pred[:-1] - u_T_pred[1:]) * delta_t)

    L_m = tf.reduce_mean(tf.square( m_pred_dot - (u_T_pred / (Isp * g0) ) ) )

    physics_loss_value = weight_dynamics * (L_r + L_theta + L_vr + L_vtheta) + weight_mass * (L_m) + weight_objective * (L_o)

    return physics_loss_value
@tf.keras.utils.register_keras_serializable()
def custom_loss_T_perturbed_NOT_USED(true_states, predictions, t0, t_final, model, M, MU_scaled, Isp, g0, weight_dynamics, weight_mass, weight_objective):
    # data loss
    # data_loss_value = tf.reduce_mean(tf.abs(predictions - true_states))

    # create perturbed time grid
    time_grid_perturbed = generate_training_batch(t0, t_final, M).reshape(-1, 1)
    time_grid_perturbed_tensor = tf.convert_to_tensor(time_grid_perturbed, dtype=tf.float32)

    # time derivatives
    with tf.GradientTape(persistent=True) as tape:
        time_grid_perturbed = generate_training_batch(t0, t_final, M).reshape(-1, 1)
        time_grid_perturbed_tensor = tf.convert_to_tensor(time_grid_perturbed, dtype=tf.float32)
        tape.watch(time_grid_perturbed_tensor)
        predictions = model(time_grid_perturbed_tensor)

        # r_pred has shape (M+1,),    r_pred_dot has shape (M+1,1) even without reshaping r_pred to (M+1,1),    time_grid_tensor has shape (M+1,1)
        r_pred      = tf.reshape(predictions[:, 0], [M+1,1])            ;  r_pred_dot       = tape.gradient(r_pred      , time_grid_perturbed_tensor)
        theta_pred  = tf.reshape(predictions[:, 1], [M+1,1])            ;  theta_pred_dot   = tape.gradient(theta_pred  , time_grid_perturbed_tensor)
        vr_pred     = tf.reshape(predictions[:, 2], [M+1,1])            ;  vr_pred_dot      = tape.gradient(vr_pred     , time_grid_perturbed_tensor)
        vtheta_pred = tf.reshape(predictions[:, 3], [M+1,1])            ;  vtheta_pred_dot  = tape.gradient(vtheta_pred , time_grid_perturbed_tensor)
        u_phi_pred  = tf.reshape(predictions[:, 4], [M+1,1])
        u_T_pred    = tf.reshape(predictions[:, 5], [M+1,1])
        m_pred      = tf.reshape(predictions[:, 6], [M+1,1])            ;  m_pred_dot       = tape.gradient(m_pred      , time_grid_perturbed_tensor)

    # physics losses
    L_r =      tf.reduce_mean(tf.square(r_pred_dot - vr_pred))  # (M+1, 1) - (M+1) = (M+1,M+1)         (M+1, 1) - (M+1,1) = (M+1,1)
    L_theta =  tf.reduce_mean(tf.square(theta_pred_dot - vtheta_pred / r_pred))
    L_vr =     tf.reduce_mean(tf.square(vr_pred_dot - ( ((vtheta_pred ** 2) / r_pred) + (MU_scaled / (r_pred ** 2)) - (u_T_pred * tf.sin(u_phi_pred)) / m_pred) ) )
    L_vtheta = tf.reduce_mean(tf.square(vtheta_pred_dot + ((vr_pred * vtheta_pred) / r_pred)                 - ((u_T_pred * tf.cos(u_phi_pred)) / m_pred) ) )

    delta_t = time_grid_perturbed_tensor[1:] - time_grid_perturbed_tensor[:-1]
    L_o = (1 / (Isp * g0)) * tf.reduce_sum(0.5 * (u_T_pred[:-1] - u_T_pred[1:]) * delta_t)

    L_m = tf.reduce_mean(tf.square( m_pred_dot - (u_T_pred / (Isp * g0) ) ) )

    physics_loss_value = weight_dynamics * (L_r + L_theta + L_vr + L_vtheta) + weight_mass * (L_m) + weight_objective * (L_o)

    return physics_loss_value

### Learning rate schedules
def learning_rate_schedule2000(epoch):
    if epoch < 400:
        return 1e-2
    elif epoch < 900:
        return 1e-3 # should be a float, not int
    elif epoch <1000:
        return 0.05
    elif epoch < 1300:
        return 1e-2
    elif epoch < 1500:
        return 1e-3
    elif epoch < 1700:
        return 1e-4
    elif epoch < 1850:
        return 1e-5
    # elif epoch < 27000:
    #     return 1e-4
    # elif epoch < 31000:
    #     return 5e-3  # Shaking phase 2
    # elif epoch < 36000:
    #     return 1e-4
    # elif epoch < 40000:
    #     return 5e-3  # Shaking phase 3
    # elif epoch < 46000:
    #     return 1e-4
    else:
        return 1e-6
def learning_rate_schedule51000(epoch):
    if epoch < 3000:
        return 1e-2
    elif epoch < 8000:
        return 1e-3
    elif epoch < 18000:
        return 1e-4
    elif epoch < 22000: # Shaking phase 1
        return 5e-3
    elif epoch < 27000:
        return 1e-4
    elif epoch < 31000: # Shaking phase 2
        return 5e-3
    elif epoch < 36000:
        return 1e-4
    elif epoch < 40000: # Shaking phase 3
        return 5e-3
    elif epoch < 45000:
        return 1e-4
    else:
        return 1e-6
def learning_rate_schedule35000(epoch):
    if epoch < 3000:
        return 1e-2
    elif epoch < 5000:
        return 1e-3
    elif epoch < 8000:
        return 1e-4
    elif epoch < 12000: # Shaking phase 1
        return 5e-3
    elif epoch < 16000:
        return 1e-4
    elif epoch < 20000: # Shaking phase 2
        return 5e-3
    elif epoch < 23000:
        return 1e-4
    elif epoch < 26000: # Shaking phase 3
        return 5e-3
    elif epoch < 30000:
        return 1e-4
    else:
        return 1e-6
def learning_rate_schedule10000(epoch):
    if epoch < 600:
        return 1e-2
    elif epoch < 1600:
        return 1e-3
    elif epoch < 3600:
        return 1e-4
    elif epoch < 4400:  # Shaking phase 1
        return 5e-3
    elif epoch < 5400:
        return 1e-4
    elif epoch < 6200:  # Shaking phase 2
        return 5e-3
    elif epoch < 7200:
        return 1e-4
    elif epoch < 8000:  # Shaking phase 3
        return 5e-3
    elif epoch < 9200:
        return 1e-4
    else:
        return 1e-6


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

@tf.keras.utils.register_keras_serializable()
def custom_sin(x):
    return tf.math.sin(x)

def generate_T_perturbed_batch(t0, tf, M):
    delta_t = (tf - t0) / M
    base_times = np.linspace(t0, tf, M + 1)
    perturbed_times = np.random.normal(loc=base_times, scale=0.2 * delta_t)
    return perturbed_times


# Define a custom callback to regenerate the time grid at the start of each epoch
class TimeGridCallback(tf.keras.callbacks.Callback):
    def __init__(self, t0, tf, M):
        super(TimeGridCallback, self).__init__()
        self.t0 = t0
        self.tf = tf
        self.M = M

    def on_epoch_begin(self, epoch, logs=None):
        # Regenerate perturbed time grid
        training_batch = generate_T_perturbed_batch(self.t0, self.tf, self.M)
        time_dataset = tf.data.Dataset.from_tensor_slices(training_batch)

        # Update the dataset used in training
        self.model.training_dataset = time_dataset.batch(self.M)


def plot_trajectory_polar_to_cart(attempt, predictions_rescaled, t_final, name_file):
    x, y = pol2cart_position(predictions_rescaled[:,0], predictions_rescaled[:,1])
    thrust_angle = pol2cart_thrust_angle(predictions_rescaled[:, 1], predictions_rescaled[:, 4])
    thrust_magnitude = predictions_rescaled[:, 5]  # u_T_pred

    # Calculate thrust vectors in Cartesian coordinates
    thrust_x = thrust_magnitude * np.cos(thrust_angle)
    thrust_y = thrust_magnitude * np.sin(thrust_angle)

    # Scale down the thrust vectors for visibility in the plot
    thrust_scale = 1.0 * max(np.max(x), np.max(y))  # Adjust scale factor as needed
    thrust_x *= thrust_scale
    thrust_y *= thrust_scale
    max_thrust_magnitude_scaled = max(thrust_magnitude)*thrust_scale

    fig, ax = plt.subplots(figsize=(10, 6))  # Create a figure and axis
    colors = cm.viridis(np.linspace(0, 1, len(x)))
    for i in range(1, len(x)):
        plt.plot(x[i - 1:i + 1], y[i - 1:i + 1], color=colors[i], marker='o', markersize=2, alpha=0.5)

    # Plot thrust vectors as arrows
    plt.quiver(x, y, thrust_x, thrust_y, color='red', scale_units='xy', angles='xy', scale=1, width=0.002)

    plt.title('Trajectory in Cartesian Coordinates')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.axis('equal')  # Ensures the aspect ratio is equal, so the plot is not distorted
    cbar = plt.colorbar(cm.ScalarMappable(cmap='viridis'), ax=ax, label='Time progression [s]')
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])  # Set appropriate ticks for the color bar
    cbar.set_ticklabels([0, t_final / 2, t_final / 3, t_final / 4, t_final])  # Corresponding tick labels

    # Add thrust scale to the legend
    ax.quiver([-np.max(x)*1.2], [np.max(y)*1.2], [max_thrust_magnitude_scaled], [0], color='red', scale_units='xy', angles='xy', scale=1, width=0.002)
    plt.text(-np.max(x)*1.2, np.max(y)*1.21, f'Thrust scale: {max(thrust_magnitude):.2f} N', fontsize=10, color='black', ha='center')

    plt.savefig(f"{name_file}_attempt{attempt}_trajectory.png", dpi=dpi_setting)

    # plt.show()
    plt.close()


import matplotlib.pyplot as plt


def plot_states_and_derivatives(attempt, predictions_rescaled, r_pred_dot, theta_pred_dot, v_r_pred_dot, v_theta_pred_dot, u_phi_pred_dot, u_T_pred_dot, m_pred_dot,
                                time_grid_tensor, name_file):
    # Create a figure and 7x2 subplots (7 rows, 2 columns)
    fig, axes = plt.subplots(7, 2, figsize=(12, 12), sharex=True)

    # Labels for the states and their derivatives
    labels = ['r_pred', 'theta_pred', 'v_r_pred', 'v_theta_pred', 'u_phi_pred', 'u_T_pred', 'm_pred']
    labels_dot = ['r_pred_dot', 'theta_pred_dot', 'v_r_pred_dot', 'v_theta_pred_dot', 'u_phi_pred_dot', 'u_T_pred_dot', 'm_pred_dot']

    # Plot each state and its derivative in corresponding subplots
    for i in range(7):
        # Left column: states
        axes[i, 0].plot(time_grid_tensor, predictions_rescaled[:, i], label=labels[i])
        axes[i, 0].set_ylabel(labels[i])
        axes[i, 0].grid(True)
        axes[i, 0].legend(loc='upper right')

        # Right column: state derivatives
        axes[i, 1].plot(time_grid_tensor, [r_pred_dot, theta_pred_dot, v_r_pred_dot, v_theta_pred_dot, u_phi_pred_dot, u_T_pred_dot, m_pred_dot][i].numpy(), label=labels_dot[i])
        axes[i, 1].set_ylabel(labels_dot[i])
        axes[i, 1].grid(True)
        axes[i, 1].legend(loc='upper right')

    # Set a common x-label for the last row
    axes[-1, 0].set_xlabel("Time")
    axes[-1, 1].set_xlabel("Time")

    # Set a common title for the entire figure
    fig.suptitle("Predicted States and Their Derivatives in Time", fontsize=16)

    # Adjust layout and save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the title
    plt.savefig(f"{name_file}_attempt{attempt}_states_and_derivatives.png", dpi=dpi_setting)
    # plt.show()
    plt.close()


def plot_states_pred_dot(attempt, r_pred_dot, theta_pred_dot, v_r_pred_dot, v_theta_pred_dot, u_phi_pred_dot, u_T_pred_dot, m_pred_dot, time_grid_tensor, name_file):
    plt.figure()

    plt.plot(time_grid_tensor, r_pred_dot.numpy(), label='r_pred_dot')
    plt.plot(time_grid_tensor, theta_pred_dot.numpy(), label='theta_pred_dot')
    plt.plot(time_grid_tensor, v_r_pred_dot.numpy(), label='v_r_pred_dot')
    plt.plot(time_grid_tensor, v_theta_pred_dot.numpy(), label='v_theta_pred_dot')
    plt.plot(time_grid_tensor, u_phi_pred_dot.numpy(), label='u_phi_pred_dot')
    plt.plot(time_grid_tensor, u_T_pred_dot.numpy(), label='u_T_pred_dot')
    plt.plot(time_grid_tensor, m_pred_dot.numpy(), label='m_pred_dot')

    plt.title("Predicted states derivatives in time")
    plt.xlabel("Time")
    plt.ylabel("State Derivatives")
    plt.legend(loc='upper right')  # Add legend to the plot
    plt.grid(True)
    plt.savefig(f"{name_file}_attempt{attempt}_states_dot.png", dpi=dpi_setting)
    # plt.show()
    plt.close()

def plot_loss(attempt, epochs, physics_loss_list, l_r_list, l_theta_list, l_vr_list, l_vtheta_list, l_m_list, l_o_list, name_file):
    # Plotting the individual loss terms
    plt.figure(figsize=(10, 8))
    plt.plot(epochs, physics_loss_list, label='Total Loss $\\mathcal{L}$', color='k', linewidth=0.5)
    plt.plot(epochs, l_r_list, label='$L_r$', color='blue', linestyle='-', linewidth=0.5)
    plt.plot(epochs, l_theta_list, label='$L_\\theta$', color='orange', linestyle='-', linewidth=0.5)
    plt.plot(epochs, l_vr_list, label='$L_{v_r}$', color='green', linestyle='-', linewidth=0.5)
    plt.plot(epochs, l_vtheta_list, label='$L_{v_\\theta}$', color='red', linestyle='-', linewidth=0.5)
    plt.plot(epochs, l_m_list, label='$\\omega_m L_m$', color='purple', linestyle='-', linewidth=0.5)
    plt.plot(epochs, l_o_list, label='$\\omega_o L_o$', color='brown', linestyle='-', linewidth=0.5)


    plt.yscale('log')
    plt.ylim(1e-16, 1e11)

    plt.xlabel('Epochs')
    plt.ylabel('Loss $\\mathcal{L}_i$')
    plt.title('Individual Loss Terms over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{name_file}_attempt{attempt}_loss.png", dpi=600)
    #plt.show()
    plt.close()

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
        axes[i].grid(True)
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
    plt.savefig(f"states_plot.png", dpi=dpi_setting)
    # plt.show()
    # plt.close()
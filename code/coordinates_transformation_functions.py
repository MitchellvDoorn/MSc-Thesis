import numpy as np

def get_new_thetas(thetas):

    revolutions = 0
    new_thetas = np.copy(thetas)

    for i in range(1, len(thetas)):

        if thetas[i] < thetas[i-1] and np.abs(thetas[i-1] - thetas[i]) > np.pi:
            revolutions += 1

        new_thetas[i] += revolutions*2*np.pi

    return new_thetas

def NDcartesian_to_radial(states, config):
    #  WATCH OUT: THE FIRST INDEX OF STATES SHOULD BE THE TIME COLUMN
    cartesian = np.zeros(states.shape)
    cartesian[..., 0] = states[..., 0]

    x = states[..., 1]
    y = states[..., 2]

    r = np.sqrt(x**2 + y**2)
    thetas = np.where(np.arctan2(y, x)<0, np.arctan2(y, x)+2*np.pi, np.arctan2(y, x))

    rotations = np.array([[[np.cos(theta), np.sin(theta)],
                           [-np.sin(theta),  np.cos(theta)]] for theta in thetas])

    cartesian[..., 1] = r
    cartesian[..., 2] = get_new_thetas(thetas)
    cartesian[..., 3:5] = (rotations @ states[..., 3:5].reshape(len(states), 2, 1)).reshape(len(states), 2)

    # Concert the control term to cartesian
    if states.shape[1] > 5:
        cartesian[..., 5:7] = (rotations @ states[..., 5:7].reshape(len(states), 2, 1)).reshape(len(states), 2)

    return cartesian

def radial_to_NDcartesian(states, config):
    #  WATCH OUT: THE FIRST INDEX OF STATES SHOULD BE THE TIME COLUMN
    cartesian = np.zeros(states.shape)
    cartesian[..., 0] = states[..., 0]

    radius = states[..., 1]
    thetas = states[..., 2]

    rotations = np.array([[[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta),  np.cos(theta)]] for theta in thetas])

    cartesian[..., 1]   = radius * np.cos(thetas)
    cartesian[..., 2]   = radius * np.sin(thetas)
    cartesian[..., 3:5] = (rotations @ states[..., 3:5].reshape(len(states), 2, 1)).reshape(len(states), 2)

    # Concert the control term to cartesian
    if states.shape[1] > 5:
        cartesian[..., 5:7] = (rotations @ states[..., 5:7].reshape(len(states), 2, 1)).reshape(len(states), 2)

    return cartesian


def cartesian_to_NDcartesian(states, config):

    NDcartesian = np.copy(states)
    NDcartesian[..., 0]   = states[..., 0] / config['t_scale']
    NDcartesian[..., 1:3] = states[..., 1:3] / config['length_scale']
    NDcartesian[..., 3:5] = states[..., 3:5] * (config['t_scale']/config['length_scale'])

    return NDcartesian

def NDcartesian_to_cartesian(states, config):

    cartesian = np.copy(states)
    cartesian[..., 0]   = states[..., 0] * config['t_scale']
    cartesian[..., 1:3] = states[..., 1:3] * config['length_scale']
    cartesian[..., 3:5] = states[..., 3:5] / (config['t_scale']/config['length_scale'])

    return cartesian


def radial_to_cartesian(states, config):
    return NDcartesian_to_cartesian(radial_to_NDcartesian(states, config), config)

def cartesian_to_radial(states, config):
    return NDcartesian_to_radial(cartesian_to_NDcartesian(states, config), config)
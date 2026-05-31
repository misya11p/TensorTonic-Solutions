import numpy as np

def reverse_step(x_t, t, epsilon_pred, betas, z=None):
    """
    Returns: np.ndarray x_{t-1} after one reverse diffusion step
    """
    x_t = np.array(x_t)
    epsilon_pred = np.array(epsilon_pred)
    betas = np.array(betas)

    b = betas[t - 1]
    a = 1 - b
    a_bar = np.cumprod(1 - betas)[t - 1]

    a_s = np.sqrt(a)
    a_1 = 1 - a
    a_bar_1_s = np.sqrt(1 - a_bar)

    if (z is None) or (t == 0):
        noise = 0
    else:
        noise = np.sqrt(b) * np.array(z)

    x_t_1 = (1 / a_s) * (x_t - (a_1 / a_bar_1_s) * epsilon_pred) + noise
    return x_t_1
    

    
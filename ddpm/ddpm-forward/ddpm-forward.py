import numpy as np

def get_alpha_bar(betas):
    """
    Compute cumulative product of (1 - beta).
    Returns list of floats rounded to 6 decimals.
    """
    return np.cumprod(1 - np.array(betas))

def forward_diffusion(x_0, t, betas, epsilon):
    """
    Returns: tuple of (np.ndarray x_t, np.ndarray epsilon) with same shape as x_0
    """
    x_0 = np.array(x_0)
    betas = np.array(betas)
    epsilon = np.array(epsilon)
    alpha_bar_t = get_alpha_bar(betas)[t - 1]
    x_t = np.sqrt(alpha_bar_t) * x_0 + np.sqrt(1 - alpha_bar_t) * epsilon
    return x_t
    
import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    # Write code here
    g = np.array(g)
    norm = np.sqrt((g ** 2).sum())
    if (max_norm > 0) and norm > max_norm:
        g = g * max_norm / norm
    return g
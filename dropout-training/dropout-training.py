import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    if rng is not None:
        random = rng.random(x.shape)
    else:
        random = np.random.random(x.shape)
    scale = 1 - p
    pattern = (random < scale) / scale
    x *= pattern
    return x, pattern
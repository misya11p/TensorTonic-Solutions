import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Write code here
    X = np.array(X)
    N = len(X)
    if (X.ndim != 2) or (N < 2):
        return

    X_centered = X - X.mean(axis=0)
    cov = np.dot(X_centered.T, X_centered) / (N - 1)
    return cov
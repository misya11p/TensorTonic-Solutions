import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    N, d = X.shape
    w = np.zeros(d)
    b = 0.0

    for _ in range(steps):
        pred = _sigmoid(np.dot(X, w) + b)
        grad_w = np.dot(X.T, pred - y) / N
        grad_b = pred - y
        w -= lr * grad_w
        b -= (lr * grad_b).mean()

    return w, b
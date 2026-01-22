import numpy as np

def silhouette_score(X, labels):
    """
    Compute the mean Silhouette Score for given points and cluster labels.
    X: np.ndarray of shape (n_samples, n_features)
    labels: np.ndarray of shape (n_samples,)
    Returns: float
    """
    # Write code here
    classes = np.unique(labels)
    N = len(X)
    arange = np.arange(N)

    distances = []
    for c in classes:
        idx = labels == c
        Xc = X[idx]
        dist = np.linalg.norm(X[:, np.newaxis, :] - Xc, axis=-1)
        dist = dist.sum(axis=-1) / (len(Xc) - idx)
        distances.append(dist)
    distances = np.stack(distances, axis=-1)

    is_other = np.ones_like(distances, dtype=bool)
    is_other[arange, labels] = False

    a = distances[arange, labels]
    b = distances[is_other].reshape(N, -1).min(axis=-1)
    score = ((b - a) / np.maximum(a, b)).mean()
    return score
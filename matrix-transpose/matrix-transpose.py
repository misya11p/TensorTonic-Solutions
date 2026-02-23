import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    return np.array([np.array(A)[:, i] for i in range(len(A[0]))])

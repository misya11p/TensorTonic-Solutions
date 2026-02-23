import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    max_len = max_len or len(max(seqs, key=len))
    seqs = [
        seq[:max_len] + [pad_value] * (max_len - len(seq))
        for seq in seqs
    ]
    return seqs
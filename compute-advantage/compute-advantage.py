import numpy as np

def compute_advantage(states, rewards, V, gamma):
    """
    Returns: A (NumPy array of advantages)
    """
    # Write code here
    G = []
    Gt = 0.0
    for reward in rewards[::-1]:
        Gt = gamma * Gt + reward
        G.append(Gt)
    G = np.array(G[::-1])
    V = np.array(V)
    A = G - V[states]
    return A
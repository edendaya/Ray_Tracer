import numpy as np

# Constants
EPSILON = 1e-6


def normalize(x):
    return x / np.linalg.norm(x)

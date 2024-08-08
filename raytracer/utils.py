import numpy as np

# Constants
EPSILON = 1e-6

x_axis = np.array([1, 0, 0], dtype="float")
y_axis = np.array([0, 1, 0], dtype="float")
z_axis = np.array([0, 0, 1], dtype="float")


def normalize(x):
    return x / np.linalg.norm(x)

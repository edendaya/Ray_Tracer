import numpy as np

class Ray:
    def __init__(self, origin, direction):
        """
        Initialize a ray with an origin and direction.
        """
        self.origin = np.array(origin)
        self.direction = np.array(direction)
        self.direction = self.direction / np.linalg.norm(self.direction)  # Normalize direction

    def at(self, t):
        """
        Compute the point at a given distance along the ray.
        """
        return self.origin + t * self.direction

    def __repr__(self):
        return f"Ray(origin={self.origin}, direction={self.direction})"
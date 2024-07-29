import numpy as np

class Cube:
    def __init__(self, position, scale, material_index):
        self.position = np.array(position)
        self.scale = scale
        self.material_index = material_index

    def intersect(self, ray_origin, ray_direction):
        """
        Check for intersection between a ray and the cube.
        Returns the intersection distance if intersected, otherwise returns None.
        """
        # Define cube bounds
        min_bound = self.position - self.scale / 2
        max_bound = self.position + self.scale / 2

        t_min = (min_bound - ray_origin) / ray_direction
        t_max = (max_bound - ray_origin) / ray_direction

        t1 = np.minimum(t_min, t_max)
        t2 = np.maximum(t_min, t_max)

        t_near = np.max(t1)
        t_far = np.min(t2)

        if t_near > t_far or t_far < 0:
            return None

        return t_near

    def get_normal(self, point):
        """
        Compute the normal of the cube at a given point.
        """
        min_bound = self.position - self.scale / 2
        max_bound = self.position + self.scale / 2

        # Determine which face the point is on
        if abs(point[0] - min_bound[0]) < 1e-6:
            return np.array([-1, 0, 0])
        elif abs(point[0] - max_bound[0]) < 1e-6:
            return np.array([1, 0, 0])
        elif abs(point[1] - min_bound[1]) < 1e-6:
            return np.array([0, -1, 0])
        elif abs(point[1] - max_bound[1]) < 1e-6:
            return np.array([0, 1, 0])
        elif abs(point[2] - min_bound[2]) < 1e-6:
            return np.array([0, 0, -1])
        elif abs(point[2] - max_bound[2]) < 1e-6:
            return np.array([0, 0, 1])
        else:
            raise ValueError("Point is not on the surface of the cube.")

    def __repr__(self):
        return (f"Cube(position={self.position}, scale={self.scale}, "
                f"material_index={self.material_index})")

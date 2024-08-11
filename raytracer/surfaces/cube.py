from surfaces.surface import Surface
import numpy as np
from intersection import Intersection
from surfaces.infinite_plane import InfinitePlane

from utils import EPSILON, x_axis, y_axis, z_axis


class Cube(Surface):
    def __init__(self, position, scale, material_index):
        super(Cube, self).__init__(material_index)
        self.position = np.array(position, dtype=np.float64)
        self.scale = scale


    def calc_intersection_with_ray(self, ray):
        min_bound= self.position - self.scale / 2
        max_bound = self.position + self.scale / 2
        with np.errstate(divide='ignore'):  # division by zero is ok.
            t_min = (min_bound - ray.origin) / ray.vec
            t_max = (max_bound- ray.origin) / ray.vec
            t_enter = np.max(np.minimum(t_min, t_max))
            t_exit = np.min(np.maximum(t_min, t_max))
            if t_enter > t_exit:
                return None
            return Intersection(self, ray, t_enter)

    def calculate_intersection_with_rays(self, rays):
        min_bound = self.position - self.scale / 2
        max_bound = self.position + self.scale / 2
        rays_v = np.array([ray.vec for ray in rays])
        rays_origin = np.array([ray.origin for ray in rays])
        with np.errstate(divide='ignore'):  # division by zero is ok.
            t_min = (min_bound - rays_origin) / rays_v
            t_max = (max_bound - rays_origin) / rays_v
            t_enter = np.max(np.minimum(t_min, t_max), axis=1)
            t_exit = np.min(np.maximum(t_min, t_max), axis=1)
            return [Intersection(self, rays[i], t_enter[i]) if t_exit[i] >= t_enter[i] else None for i in
                    range(len(rays))]

    def in_cube(self, point):
        for i in range(3):
            if not (self.position[i] - self.scale / 2 <= point[i] <= self.position[i] + self.scale / 2):
                return False
        return True


    def get_normal(self, point):
        """
        Compute the normal of the cube at a given point.
        """
        min_bound = self.position - self.scale / 2
        max_bound = self.position + self.scale / 2
        all_axes = [x_axis, y_axis, z_axis]
        for i in range(3):
            if abs(point[i] - min_bound[i]) < EPSILON:
                return np.array(all_axes[i] * -1)
            elif abs(point[i] - max_bound[i]) < EPSILON:
                return np.array(all_axes[i])
        # Determine which face the point is on
        raise ValueError("Point is not on the surface of the cube.")


    def __repr__(self):
        return (f"Cube(position={self.position}, scale={self.scale}, "
                f"material_index={self.material_index})")
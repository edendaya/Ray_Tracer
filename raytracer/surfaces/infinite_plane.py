from surfaces.surface import Surface
import numpy as np
from intersection import Intersection

from utils import EPSILON, normalize


class InfinitePlane(Surface):
    def __init__(self, normal, offset, material_index):
        super(InfinitePlane, self).__init__(material_index)
        self.normal = normalize(np.array(normal, dtype=np.float64))
        self.offset = offset

    def calc_intersection_with_ray(self, ray):
        dot_prod = self.normal @ ray.vec
        # ray is parallel to the plane, so there is no intersection
        if abs(dot_prod) < EPSILON:
            return None
        t = ((self.offset * self.normal - ray.origin) @ self.normal) / dot_prod
        # ray origin is in front of the plane, so there is no intersection
        if t < 0:
            return None
        return Intersection(self, ray, t)

    def calculate_intersection_with_rays(self, rays):
        if len(rays) == 1:
            return [self.get_intersection_with_ray(rays[0])]
        with np.errstate(divide='ignore'):  # allow division by zero
            rays_v_matrix = np.array([ray.vec for ray in rays])
            rays_origin = np.array([ray.origin for ray in rays])

            dot_prods = np.einsum('ij,j->i', rays_v_matrix, self.normal)
            t_values = (np.einsum('ij,j->i', (self.offset * self.normal - rays_origin), self.normal)) / dot_prods

            boolean_array = np.logical_or(abs(dot_prods) < EPSILON, t_values < 0)
            return [Intersection(self, rays[i], t_values[i]) if not boolean_array[i] else None for i in
                    range(len(rays))]

    def get_normal(self, point):
        return self.normal

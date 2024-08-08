import numpy as np

from utils import EPSILON

class IntersectionsCalculator:
    def __init__(self, scene):
        self.surfaces = scene.surfaces

    def find_all_ray_intersections_sorted(self, ray):
        intersections = []
        for surface in self.surfaces:
            intersection = surface.get_intersection_with_ray(ray)
            if intersection is not None and intersection.t > EPSILON:
                intersections.append(intersection)
        intersections.sort(key=lambda _intersection: _intersection.t)
        return intersections

    def find_closest_rays_intersections_batch(self, rays):
        min_t_values = np.array([float('inf') for _ in rays])
        closest_intersections = np.array([None for _ in rays])
        for surface in self.surfaces:
            intersections = np.array(surface.get_intersection_with_rays(rays))
            intersections_t = [intersection.t if intersection is not None else float('inf') for intersection in intersections]
            # make sure values are in range
            new_values = intersections_t < min_t_values
            # set new minimum values
            min_t_values = np.minimum(min_t_values, intersections_t)
            closest_intersections[new_values] = intersections[new_values]
        return closest_intersections

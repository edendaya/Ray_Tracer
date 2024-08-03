import numpy as np
from ray import Ray
from calcintersections import CalcIntersections

from config import EPSILON

class LightCalc:
    def __init__(self, scene):
        self.scene = scene

    def get_light_intensity_batch(self, light, intersection):
        plane_normal = intersection.hit_point - light.position
        plane_normal /= np.linalg.norm(plane_normal)
        transform_matrix = self.xy_to_general_plane(plane_normal, light.position)

        length_unit = light.radius / self.scene.scene_settings.root_number_shadow_rays
        x_values_vec, y_values_vec = np.meshgrid(range(self.scene.scene_settings.root_number_shadow_rays),
                                                 range(self.scene.scene_settings.root_number_shadow_rays))
        x_values_vec = x_values_vec.reshape(-1)
        y_values_vec = y_values_vec.reshape(-1)
        base_xy = np.array([x_values_vec * length_unit - light.radius / 2,
                            y_values_vec * length_unit - light.radius / 2,
                            np.zeros_like(x_values_vec),
                            np.zeros_like(x_values_vec)])
        offset = np.array([np.random.uniform(0, length_unit, y_values_vec.shape),
                           np.random.uniform(0, length_unit, x_values_vec.shape),
                           np.zeros_like(x_values_vec),
                           np.ones_like(x_values_vec)])
        rectangle_points = base_xy + offset

        light_points = (transform_matrix @ rectangle_points)[:3].T

        rays = [Ray(point, intersection.hit_point - point) for point in light_points]
        light_hits = CalcIntersections(self.scene).find_closest_rays_intersections_batch(rays)
        c = sum(1 for light_hit in light_hits if light_hit is not None and
                np.linalg.norm(intersection.hit_point - light_hit.hit_point) < EPSILON)
        light_intensity = (1 - light.shadow_intensity) + light.shadow_intensity * (
                c / (self.scene.scene_settings.root_number_shadow_rays ** 2))
        return light_intensity, light.color

    @staticmethod
    def xy_to_general_plane(plane_normal, plane_point):
        z_axis = np.array([0, 0, 1], dtype="float")
        if np.linalg.norm(z_axis - abs(plane_normal)) < EPSILON:  
            translation_matrix = np.eye(4)
            translation_matrix[:3, 3] = plane_point
            return translation_matrix

        rotation_axis = np.cross(z_axis, plane_normal)
        cos_theta = np.dot(z_axis, plane_normal)
        sin_theta = np.linalg.norm(rotation_axis) / np.linalg.norm(plane_normal)
        rotation_axis /= np.linalg.norm(rotation_axis)

        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = (cos_theta * np.eye(3)) + \
                                (sin_theta * np.array([
                                    [0, -rotation_axis[2], rotation_axis[1]],
                                    [rotation_axis[2], 0, -rotation_axis[0]],
                                    [-rotation_axis[1], rotation_axis[0], 0]
                                ], dtype="float") +
                                (1 - cos_theta) * np.outer(rotation_axis, rotation_axis)
                                )

        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = plane_point

        transformation_matrix = np.matmul(translation_matrix, rotation_matrix)

        return transformation_matrix

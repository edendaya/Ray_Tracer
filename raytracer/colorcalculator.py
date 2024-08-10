import numpy as np
from lightcalculator import LightCalculator
from intersectionscalculator import IntersectionsCalculator
from ray import Ray

import utils


class ColorCalculator:
    def __init__(self, scene):
        self.settings = scene.scene_settings
        self.materials = scene.materials
        self.lights = scene.lights
        self.light_calc = LightCalculator(scene)
        self.intersections_calculator = IntersectionsCalculator(scene)

    def get_diffuse_and_specular_color(self, intersection):
        if intersection is None:
            return self.settings.background_color

        # Initialize sums for diffuse and specular colors
        dif_sum = np.array([0, 0, 0], dtype='float')
        spec_sum = np.array([0, 0, 0], dtype='float')

        # Get material properties once
        material = intersection.surface.get_material(self.materials)
        diffuse_color = material.diffuse_color
        specular_color = material.specular_color
        shininess = material.shininess

        for light in self.lights:
            # Compute normal and light direction
            N = intersection.surface.get_normal(intersection.hit_point)
            L = light.position - intersection.hit_point
            L_norm = np.linalg.norm(L)
            if L_norm != 0:
                L = L / L_norm
            else:
                continue

            N_L_dot = N @ L
            if N_L_dot <= 0:
                continue

            # Compute view direction
            V = intersection.ray.origin - intersection.hit_point
            V_norm = np.linalg.norm(V)
            if V_norm != 0:
                V = V / V_norm
            else:
                continue

            # Get light intensity and color
            light_intensity, light_color = self.light_calc.calculate_light_intensity(light, intersection)

            # Compute reflected light direction
            light_ray = Ray(light.position, -L)
            R = intersection.surface.get_reflected_ray(light_ray, intersection.hit_point).v

            # Accumulate diffuse and specular components
            dif_sum += light_color * light_intensity * N_L_dot
            spec_sum += light_color * light_intensity * light.specular_intensity * \
                        (np.power(np.dot(R, V), shininess))

        # Combine diffuse and specular colors with material properties
        diffuse_color = dif_sum * diffuse_color
        specular_color = spec_sum * specular_color

        # Return combined color
        return diffuse_color + specular_color

    def get_ray_color(self, intersections, reflection_rec_level=0):
        if intersections is None or len(intersections) == 0:
            return self.settings.background_color

        # Use slicing to get the last non-transparent intersection
        for i in range(len(intersections)):
            if intersections[i].surface.get_material(self.materials).transparency == 0:
                intersections = intersections[0:i + 1]
                break

        bg_color = self.settings.background_color
        color = None
        for intersection in reversed(intersections):
            material = intersection.surface.get_material(self.materials)
            transparency = material.transparency
            d_s_color = self.get_diffuse_and_specular_color(intersection)
            reflection_color = self.get_reflection_color(intersection, reflection_rec_level) * \
                               material.reflection_color

            color = ((1 - transparency) * d_s_color + transparency * bg_color) + reflection_color
            bg_color = color

        return color

    def get_reflection_color(self, intersection, reflection_rec_level):
        if reflection_rec_level >= self.settings.max_recursions:
            return self.settings.background_color
        reflection_ray = intersection.surface.get_reflected_ray(intersection.ray, intersection.hit_point)
        intersections = self.intersections_calculator.find_all_ray_intersections_sorted(reflection_ray)
        return self.get_ray_color(intersections, reflection_rec_level + 1)

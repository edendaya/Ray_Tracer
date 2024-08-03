import numpy as np
from lightcalc import LightCalc
from calcintersections import CalcIntersections
from ray import Ray

class ColorCalc:
    def __init__(self, scene):
        self.scene = scene
        self.light_calc = LightCalc(scene)
        self.calc_intersections = CalcIntersections(scene)

    def get_diffuse_and_specular_color(self, intersection):
        if intersection is None:
            return self.scene.scene_settings.background_color

        # Initialize sums for diffuse and specular colors
        dif_sum = np.array([0, 0, 0], dtype='float')
        spec_sum = np.array([0, 0, 0], dtype='float')

        # Get material properties once
        material = intersection.surface.get_material(self.scene.materials)
        diffuse_color = material.diffuse_color
        specular_color = material.specular_color
        shininess = material.shininess

        for light in self.scene.lights:
            # Compute normal and light direction
            N = intersection.surface.get_normal(intersection.hit_point)
            L = light.position - intersection.hit_point
            L_norm = np.linalg.norm(L)
            if L_norm == 0:
                continue
            L /= L_norm
            N_L_dot = N @ L
            if N_L_dot <= 0:
                continue

            # Compute view direction
            V = intersection.ray.origin - intersection.hit_point
            V_norm = np.linalg.norm(V)
            if V_norm == 0:
                continue
            V /= V_norm

            # Get light intensity and color
            light_intensity, light_color = self.light_calc.get_light_intensity_batch(light, intersection)

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
            return self.scene.scene_settings.background_color
        
        # Use slicing to get the last non-transparent intersection
        for i in range(len(intersections)):
            if intersections[i].surface.get_material(self.scene.materials).transparency == 0:
                intersections = intersections[0:i + 1]
                break

        bg_color = self.scene.scene_settings.background_color
        color = None
        for intersection in reversed(intersections):
            material = intersection.surface.get_material(self.scene.materials)
            transparency = material.transparency
            d_s_color = self.get_diffuse_and_specular_color(intersection)
            reflection_color = self.get_reflection_color(intersection, reflection_rec_level) * \
                               material.reflection_color

            color = ((1 - transparency) * d_s_color + transparency * bg_color) + reflection_color
            bg_color = color

        return color

    def get_reflection_color(self, intersection, reflection_rec_level):
        if reflection_rec_level >= self.scene.scene_settings.max_recursions:
            return self.scene.scene_settings.background_color
        reflection_ray = intersection.surface.get_reflected_ray(intersection.ray, intersection.hit_point)
        intersections = self.calc_intersections.find_all_ray_intersections_sorted(reflection_ray)
        return self.get_ray_color(intersections, reflection_rec_level + 1)

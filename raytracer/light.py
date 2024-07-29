import numpy as np
from ray import Ray

class Light:
    def __init__(self, position, color, specular_intensity, shadow_intensity, light_radius):
        self.position = np.array(position)
        self.color = np.array(color)
        self.specular_intensity = specular_intensity
        self.shadow_intensity = shadow_intensity
        self.light_radius = light_radius


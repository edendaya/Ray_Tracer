import numpy as np

class SceneSettings:
    def __init__(self, background_color, root_number_shadow_rays, max_recursions):
        self.background_color = np.array(background_color)
        self.root_number_shadow_rays = root_number_shadow_rays
        self.max_recursions = max_recursions

    def compute_shadow_rays(self):
        """
        Compute the number of shadow rays based on the root number of shadow rays.
        """
        return self.root_number_shadow_rays ** 2
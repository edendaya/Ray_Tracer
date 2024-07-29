import numpy as np

class Material:
    def __init__(self, diffuse_color, specular_color, reflection_color, shininess, transparency):
        self.diffuse_color = np.array(diffuse_color)
        self.specular_color = np.array(specular_color)
        self.reflection_color = np.array(reflection_color)
        self.shininess = shininess
        self.transparency = transparency

    def compute_diffuse(self, light, point, normal):
        """
        Compute the diffuse component of the material's color at a given point.
        """
        light_direction = light.position - point
        light_distance = np.linalg.norm(light_direction)
        light_direction = light_direction / light_distance  # Normalize

        # Compute the diffuse intensity
        diffuse_intensity = max(np.dot(normal, light_direction), 0)
        
        # Compute the diffuse color
        return self.diffuse_color * diffuse_intensity

    def compute_specular(self, light, point, normal, view_direction):
        """
        Compute the specular component of the material's color at a given point.
        """
        light_direction = light.position - point
        light_distance = np.linalg.norm(light_direction)
        light_direction = light_direction / light_distance  # Normalize

        # Compute the reflection direction
        reflection_direction = 2 * np.dot(normal, light_direction) * normal - light_direction

        # Compute the specular intensity
        specular_intensity = max(np.dot(reflection_direction, view_direction), 0) ** self.shininess
        
        # Compute the specular color
        return self.specular_color * specular_intensity

    def compute_reflection(self, incident_direction, normal):
        """
        Compute the reflection direction given the incident direction and surface normal.
        """
        return incident_direction - 2 * np.dot(normal, incident_direction) * normal

    def compute_transparency(self, incoming_ray, normal):
        """
        Compute the refraction direction given the incoming ray and surface normal.
        This is a basic implementation; real refraction calculation involves more complex physics.
        """
        # Placeholder for refraction direction
        return incoming_ray  # Modify this according to your refraction model

import numpy as np
from intersection import Intersection

class Sphere:
    def __init__(self, center, radius, material_index):
        self.center = np.array(center)
        self.radius = radius
        self.material_index = material_index  # Store material index

    def intersect(self, ray_origin, ray_direction):
        """
        Check for intersection between a ray and the sphere.
        Returns an Intersection object if intersected, otherwise returns None.
        """
        oc = ray_origin - self.center
        a = np.dot(ray_direction, ray_direction)
        b = 2.0 * np.dot(oc, ray_direction)
        c = np.dot(oc, oc) - self.radius ** 2
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            return None
        
        sqrt_discriminant = np.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / (2.0 * a)
        t2 = (-b + sqrt_discriminant) / (2.0 * a)
        
        if t1 > t2:
            t1, t2 = t2, t1
        
        if t1 < 0:
            t1 = t2
            if t1 < 0:
                return None
        
        intersection_point = ray_origin + t1 * ray_direction
        normal = (intersection_point - self.center) / self.radius
        temp = Intersection(t1, intersection_point, normal, self.material_index)  # Pass material_index
        print(f"intersection result: {temp}")
        return temp

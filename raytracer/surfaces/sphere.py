import numpy as np
from intersection import Intersection

class Sphere:
    def __init__(self, center, radius, material_index):
        self.center = np.array(center)
        self.radius = radius
        self.material_index = material_index  # Store material index

    def intersect(self, ray):
        """
        ray_origin, ray_direction
        Check for intersection between a ray and the sphere.
        Returns an Intersection object if intersected, otherwise returns None.
        """
        ray_to_center = ray.origin - self.center
        a = 1
        b = 2 * (ray.direction @ ray_to_center)
        c = (ray_to_center @ ray_to_center) - self.radius ** 2
        disc = b ** 2 - 4 * a * c
        # print("disc is: ",disc)
        if disc <= 0:
            return None
        disc_sqrt = np.sqrt(disc)
        t1 = (-b + disc_sqrt) / 2 * a
        t2 = (-b - disc_sqrt) / 2 * a
        #print(f"t1: {t1}, t2: {t2}")
        if t1 < 0 and t2 < 0:
            return None
        t1 = t1 if t1 >= 0 else float('inf')
        t2 = t2 if t2 >= 0 else float('inf')
        t = min(t1, t2)
        #print(f"t: {t}")
        intersection_point = ray.origin + t1 * ray.direction
        normal = (intersection_point - self.center) / self.radius
        return Intersection(t, intersection_point, normal, self.material_index)  # Pass material_index
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
        """
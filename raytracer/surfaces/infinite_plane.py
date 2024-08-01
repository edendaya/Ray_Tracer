import numpy as np

class InfinitePlane:
    def __init__(self, normal, offset, material_index):
        self.normal = np.array(normal)
        self.offset = offset

    def intersect(self, ray):
        """
        Check for intersection between a ray and the plane.
        Returns the intersection distance if intersected, otherwise returns None.
        """
        denom = np.dot(self.normal, ray.direction)
        if abs(denom) < 1e-6:
            return None  # Ray is parallel to the plane

        t = (self.offset - np.dot(self.normal, ray.origin)) / denom
        if t < 0:
            return None  # Intersection is behind the ray origin
        
        return t

    def get_normal(self, point):
        """
        Compute the normal of the plane (constant for all points on the plane).
        """
        return self.normal

    def __repr__(self):
        return (f"InfinitePlane(normal={self.normal}, offset={self.offset}, "
                f"material_index={self.material_index})")

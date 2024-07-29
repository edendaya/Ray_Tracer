import numpy as np

class Intersection:
    def __init__(self, distance, point, normal, material_index):
        self.distance = distance
        self.point = point
        self.normal = normal
        self.material_index = material_index  # Add this attribute
        

    def __str__(self):
        return (f"Intersection(distance={self.distance}, point={self.point}, "
                f"normal={self.normal}, material_index={self.material_index})")

    def __repr__(self):
        return self.__str__()

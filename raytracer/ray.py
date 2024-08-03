import numpy as np

class Ray:
    def __init__(self, origin, direction):
        self.origin = np.array(origin, dtype="float")
        self.v = np.array(direction, dtype="float")
        self.v = self.v / np.linalg.norm(self.v)

    @classmethod
    def ray_from_camera(cls, camera, i, j, img):
        p = camera.screen_center + (j - img.img_width // 2) * img.ratio * camera.right - (i - img.img_height // 2) * img.ratio * camera.up_vector
        direction = p - camera.position
        direction = direction / np.linalg.norm(direction)
        return cls(camera.position, direction)

import numpy as np
from utils import normalize


class Ray:
    def __init__(self, origin, direction):
        self.origin = np.array(origin, dtype=np.float64)
        self.vec = normalize(np.array(direction, dtype=np.float64))

    @classmethod
    def ray_from_camera(cls, camera, i, j, img):
        p = (camera.screen_center + (img.img_width // 2 - j) * img.ratio * camera.right
             - (i - img.img_height // 2) * img.ratio * camera.up_vector)
        direction = normalize(p - camera.position)
        return cls(camera.position, direction)

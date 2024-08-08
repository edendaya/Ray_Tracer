from intersectionscalculator import IntersectionsCalculator
from ray import Ray
from colorcalculator import ColorCalculator

from utils import EPSILON

class ImageConstructor:
    def __init__(self, scene):
        self.scene = scene
        self.camera = scene.camera
        self.image = scene.image

    def construct_image(self, res):
        ray_tracer = IntersectionsCalculator(self.scene)
        color_calculator = ColorCalculator(self.scene)
        for i in range(self.image.img_height):
            for j in range(self.image.img_width):
                ray = Ray.ray_from_camera(self.scene.camera, i, j, self.scene.image)
                intersections = ray_tracer.find_all_ray_intersections_sorted(ray)
                color = color_calculator.get_ray_color(intersections)
                res[i, j] = color
        res[res > 1] = 1
        res[res < 0] = 0
        return res * 255

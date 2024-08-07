from calcintersections import CalcIntersections
from ray import Ray
from colorcalc import ColorCalc

from utils import EPSILON

class ImageConstructor:
    def __init__(self, scene):
        self.scene = scene

    def construct_image(self, res):
        ray_tracer = CalcIntersections(self.scene)
        color_calculator = ColorCalc(self.scene)
        for i in range(self.scene.image.img_height):
            for j in range(self.scene.image.img_width):
                ray = Ray.ray_from_camera(self.scene.camera, i, j, self.scene.image)
                intersections = ray_tracer.find_all_ray_intersections_sorted(ray)
                color = color_calculator.get_ray_color(intersections)
                res[i, j] = color
        res[res > 1] = 1
        res[res < 0] = 0
        return res * 255

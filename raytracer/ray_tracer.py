import argparse
from PIL import Image
from scene import Scene
from camera import Camera
from img import Img
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere
from imageconstructor import ImageConstructor
import numpy as np
import time

from utils import EPSILON

def parse_scene_file(file_path):
    surfaces = []
    lights = []
    materials = []
    camera = None
    scene_settings = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            obj_type = parts[0]
            params = [float(p) for p in parts[1:]]
            if obj_type == "cam":
                camera = Camera(params[:3], params[3:6], params[6:9], params[9], params[10])
            elif obj_type == "set":
                scene_settings = SceneSettings(params[:3], params[3], params[4])
            elif obj_type == "mtl":
                material = Material(params[:3], params[3:6], params[6:9], params[9], params[10])
                materials.append(material)
            elif obj_type == "sph":
                sphere = Sphere(params[:3], params[3], int(params[4]))
                surfaces.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(params[:3], params[3], int(params[4]))
                surfaces.append(plane)
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]))
                surfaces.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8])
                lights.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, surfaces, materials, lights


def save_image(image_array, img_name):
    image = Image.fromarray(np.uint8(image_array))
    image.save(f"{img_name}.png")

def main():
    start = time.time()
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, default='./scenes/scence1.txt', help='Path to the scene file')
    parser.add_argument('output_image', type=str, default='./output/output1.png', help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    parser.add_argument('--bonus', type=int, default=-1, help='Bonus (-1 for  no bonus, 1 for bonus without coloring and 2 for bonus with coloring)')

    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, surfaces, materials, lights = parse_scene_file(args.scene_file)
    image = Img(args.width, args.height, camera.screen_width)
    scene = Scene(camera, scene_settings, surfaces, materials, lights, image)
    image_constructor = ImageConstructor(scene)
    res = np.zeros((args.height, args.width, 3), dtype='float')
    image_array = image_constructor.construct_image(res)
    
    # Save the output image
    save_image(image_array, args.output_image)

    print(time.time() - start)

if __name__ == '__main__':
    main()

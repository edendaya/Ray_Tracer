import argparse
from PIL import Image
import numpy as np
from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere
from ray import Ray

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
                scene_settings = SceneSettings(params[:3], int(params[3]), params[4])  # root_number_shadow_rays as int
            elif obj_type == "mtl":
                material = Material(params[:3], params[3:6], params[6:9], params[9], params[10])
                materials.append(material)
            elif obj_type == "sph":
                sphere = Sphere(params[:3], params[3], int(params[4]) - 1)
                surfaces.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(params[:3], params[3], int(params[4]) - 1)
                surfaces.append(plane)
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]) - 1)
                surfaces.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8])
                lights.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, surfaces, materials, lights

def save_image(image_array, output_file):
    image = Image.fromarray(np.uint8(image_array * 255))  # Convert to 8-bit image
    image.save(output_file)

def intersect_scene(ray, surfaces, epsilon=0.000001):
    intersections = []
    for obj in surfaces:
        intersection = obj.intersect(ray.origin, ray.direction)
        if intersection is None or intersection.distance <= epsilon:
            continue
        intersections.append(intersection)
    intersections.sort(key=lambda x: x.distance)
    if intersections:
        return intersections[0]
    return None

def calculate_light_intensity(light, point, normal, view_direction, intersect_scene, surfaces, scene_settings):
    light_direction = light.position - point
    light_distance = np.linalg.norm(light_direction)
    light_direction /= light_distance  # Normalize

    # Calculate soft shadow contribution
    shadow_contribution = calculate_soft_shadows(light, point, normal, intersect_scene, surfaces, scene_settings)

    # Calculate light intensity
    intensity = (1 - light.shadow_intensity) + light.shadow_intensity * shadow_contribution
    return intensity

def calculate_diffuse(light, normal, light_direction):
    diffuse_intensity = max(np.dot(normal, light_direction), 0)
    diffuse_color = light.color * diffuse_intensity
    return diffuse_color

def calculate_specular(light, view_direction, light_direction, normal, specular_intensity):
    reflection_direction = 2 * np.dot(normal, light_direction) * normal - light_direction
    specular_intensity = max(np.dot(reflection_direction, view_direction), 0) ** specular_intensity
    specular_color = light.color * specular_intensity
    return specular_color

def calculate_soft_shadows(light, point, normal, intersect_scene, surfaces, scene_settings):
    num_shadow_rays = scene_settings.root_number_shadow_rays ** 2  # Total number of shadow rays (grid size squared)
    radius = light.light_radius
    shadow_hits = 0

    # Find a plane perpendicular to the light direction
    light_direction = light.position - point
    light_distance = np.linalg.norm(light_direction)
    light_direction /= light_distance  # Normalize

    # Find a perpendicular plane to the light direction
    light_plane_normal = np.cross(light_direction, np.array([0, 1, 0]))  # Arbitrary perpendicular vector
    if np.linalg.norm(light_plane_normal) < 1e-6:  # Handle the case where light_direction is close to [0,1,0]
        light_plane_normal = np.cross(light_direction, np.array([1, 0, 0]))

    light_plane_normal /= np.linalg.norm(light_plane_normal)  # Normalize

    # Define the rectangle on that plane
    light_plane_point = light.position
    grid_size = scene_settings.root_number_shadow_rays
    cell_size = radius / grid_size

    for _ in range(num_shadow_rays):
        # Sample a random point within a cell
        cell_x = np.random.uniform(-radius / 2, radius / 2)
        cell_y = np.random.uniform(-radius / 2, radius / 2)
        shadow_ray_origin = light_plane_point + cell_x * light_plane_normal + cell_y * np.cross(light_plane_normal, light_direction)
        shadow_ray_direction = point - shadow_ray_origin
        shadow_ray_direction /= np.linalg.norm(shadow_ray_direction)  # Normalize
        shadow_ray = Ray(shadow_ray_origin, shadow_ray_direction)

        if intersect_scene(shadow_ray, surfaces):
            shadow_hits += 1

    shadow_contribution = shadow_hits / num_shadow_rays
    return shadow_contribution


def compute_lighting(point, normal, ray, closest_intersection, surfaces, materials, lights, scene_settings, depth=0):
    color = np.array(scene_settings.background_color)

    if closest_intersection.material_index < 0 or closest_intersection.material_index >= len(materials):
        return color

    material = materials[closest_intersection.material_index]
    for light in lights:
        view_direction = -ray.direction
        light_intensity = calculate_light_intensity(light, point, normal, view_direction, intersect_scene, surfaces, scene_settings)
        light_direction = light.position - point
        light_direction /= np.linalg.norm(light_direction)

        diffuse_color = calculate_diffuse(light, normal, light_direction) * light_intensity
        specular_color = calculate_specular(light, view_direction, light_direction, normal, light.specular_intensity) * light_intensity

        # Apply material colors
        color += (diffuse_color + specular_color) * material.diffuse_color

    if material.transparency > 0 and depth < scene_settings.max_recursions:
        reflection_ray = Ray(point, ray.direction)
        reflected_color = compute_lighting(point, normal, reflection_ray, closest_intersection, surfaces, materials, lights, scene_settings, depth + 1)
        background_color = np.array(scene_settings.background_color)
        color = (background_color * material.transparency +
                 color * (1 - material.transparency) +
                 reflected_color)
    else:
        color = np.clip(color, 0, 1)  # Ensure color values are between 0 and 1

    return color

def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, surfaces, materials, lights = parse_scene_file(args.scene_file)

    # Image dimensions
    width = args.width
    height = args.height

    # Initialize image array
    image_array = np.zeros((height, width, 3))

    # Compute aspect ratio
    aspect_ratio = width / height
    screen_width = camera.screen_width
    screen_height = screen_width / aspect_ratio

    # Calculate pixel size
    pixel_size = screen_width / width
    half_width = screen_width / 2
    half_height = screen_height / 2

    # Ray tracing
    for y in range(height):
        for x in range(width):
            # Compute pixel position
            pixel_x = (x + 0.5) * pixel_size - half_width
            pixel_y = (y + 0.5) * pixel_size - half_height
            pixel_z = -camera.screen_distance

            # Create ray
            pixel_pos = np.array([pixel_x, pixel_y, pixel_z])
            ray_direction = pixel_pos - camera.position
            ray_direction /= np.linalg.norm(ray_direction)
            ray = Ray(camera.position, ray_direction)

            # Trace the ray
            closest_intersection = intersect_scene(ray, surfaces)
            
            if closest_intersection:
                #print(f"closest_intersection: {closest_intersection}")
                # Compute the intersection point using the distance from the ray
                point = ray.origin + closest_intersection.distance * ray.direction
                normal = closest_intersection.normal
                color = compute_lighting(point, normal, ray, closest_intersection, surfaces, materials, lights, scene_settings)
                image_array[y, x] = color
            else:
                image_array[y, x] = np.array(scene_settings.background_color)

    # Save the output image
    save_image(image_array, args.output_image)

if __name__ == '__main__':
    main()

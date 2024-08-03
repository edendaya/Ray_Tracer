class Scene:
    def __init__(self, camera, scene_settings, surfaces, materials, lights, image):
        self.camera = camera
        self.scene_settings = scene_settings
        self.surfaces = surfaces
        self.materials = materials
        self.lights = lights
        self.image = image

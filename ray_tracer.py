import argparse
from PIL import Image
import numpy as np
import math
from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere


def parse_scene_file(file_path):
    objects = []
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
                objects.append(material)
            elif obj_type == "sph":
                sphere = Sphere(params[:3], params[3], int(params[4]))
                objects.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(params[:3], params[3], int(params[4]))
                objects.append(plane)
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]))
                objects.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8])
                objects.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, objects


def seperate_objects(objects):
    materials = list(filter(lambda x: isinstance(x, Material) ,objects))
    planes = list(filter(lambda x: isinstance(x, InfinitePlane) ,objects))
    cubes = list(filter(lambda x: isinstance(x, Cube) ,objects))
    spheres = list(filter(lambda x: isinstance(x, Sphere) ,objects))
    lights = list(filter(lambda x: isinstance(x, Light) ,objects))

    return materials, planes, cubes, spheres, lights

def calc_screen_parameters(camera: Camera, screen_ratio):
    screen_vec_forward = np.array(camera.look_at)
    screen_vec_forward = screen_vec_forward / np.linalg.norm(screen_vec_forward)

    screen_vec_w = np.cross(camera.look_at, camera.up_vector)
    screen_vec_w = screen_vec_w / np.linalg.norm(screen_vec_w)
    screen_vec_w = screen_vec_w * camera.screen_width

    screen_vec_h = np.cross(screen_vec_w, camera.look_at)
    screen_vec_h = screen_vec_h / np.linalg.norm(screen_vec_h)
    screen_vec_h = screen_vec_h * camera.screen_width * screen_ratio
    
    screen_center = np.array(camera.position) + screen_vec_forward * camera.screen_distance
    screen_top_left = np.array([screen_center - camera.screen_width//2, screen_center - (camera.screen_width * screen_ratio)//2])

    return screen_top_left, screen_vec_w, screen_vec_h

def get_pixel_coordinates(row, col, top_left, screen_vec_w, screen_vec_h, width, height):
    pixel_coords = top_left + (col/width)*screen_vec_w + (row/height)*screen_vec_h
    return pixel_coords



def render_scene(camera: Camera, scene_settings, objects, width, height):
    screen_ratio = height/width
    materials, planes, cubes, spheres, lights = seperate_objects(objects)
    screen_top_left, screen_vec_w, screen_vec_h = calc_screen_parameters(camera, screen_ratio)
    for row in height:
        for col in width:
            pixel_coords = get_pixel_coordinates(row, col, screen_top_left, screen_vec_w, screen_vec_h, width, height)
            color = render_ray(camera.position, pixel_coords, scene_settings, materials, surfaces, lights)



def render_ray(start, end, scene_settings, materials, surfaces, lights):

    intersections = calc_intersections(start, end, surfaces)
    nearest_surface = find_nearest(start, intersections)

    intersection_point = calc_inter_point(start, end, nearest_surface)

    compute_lights(end, lights)
    
    t_start, t_end = get_transparency_vector()
    transparency_color =  render_ray()
    
    r_start, r_end = get_reflection_vector()
    transparency_color =  render_ray()



def calc_sphere_intersections(start, end, center, radius):
    direction = end-start
    a = np.dot(direction, direction)
    b = 2 * np.dot(direction, np.subtract(start, center))
    c = np.dot(np.subtract(start, center), np.subtract(start, center)) - radius ** 2

    discriminant = b**2 - 4*a*c

    if discriminant < 0:
        return []
    elif discriminant == 0:
        t = -b / (2 * a)
        intersection_point = start + direction*t
        return [intersection_point]
    else:
        # Two intersections
        t1 = (-b + math.sqrt(discriminant)) / (2 * a)
        t2 = (-b - math.sqrt(discriminant)) / (2 * a)
        intersection_point1 = start + direction*t1
        intersection_point2 = start + direction*t2
        return [intersection_point1, intersection_point2]



def save_image(image_array):
    image = Image.fromarray(np.uint8(image_array))

    # Save the image to a file
    image.save("scenes/Spheres.png")


def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('--scene_file', type=str, default='scenes/pool.txt', help='Path to the scene file')
    parser.add_argument('--output_image', type=str, default='output/test.png', help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)

    render_scene(camera, scene_settings, objects, args.width, args.heigth)

    # Dummy result
    image_array = np.zeros((500, 500, 3))

    # Save the output image
    save_image(image_array)


if __name__ == '__main__':
    main()

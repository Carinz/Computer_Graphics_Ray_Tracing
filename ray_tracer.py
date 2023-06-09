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
                camera = Camera(np.array(params[:3]), np.array(params[3:6]), np.array(params[6:9]), params[9], params[10])
            elif obj_type == "set":
                scene_settings = SceneSettings(np.array(params[:3]), params[3], params[4])
            elif obj_type == "mtl":
                material = Material(np.array(params[:3]), np.array(params[3:6]), np.array(params[6:9]), params[9], params[10])
                objects.append(material)
            elif obj_type == "sph":
                sphere = Sphere(np.array(params[:3]), params[3], int(params[4]))
                objects.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(np.array(params[:3]), params[3], int(params[4]))
                objects.append(plane)
            elif obj_type == "box":
                cube = Cube(np.array(params[:3]), params[3], int(params[4]))
                objects.append(cube)
            elif obj_type == "lgt":
                light = Light(np.array(params[:3]), np.array(params[3:6]), params[6], params[7], params[8])
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
    screen_vec_forward = camera.look_at - camera.position
    screen_vec_forward = normalize_vec(screen_vec_forward)

    screen_vec_w = np.cross(screen_vec_forward, camera.up_vector)
    screen_vec_w = normalize_vec(screen_vec_w)
    screen_vec_w = screen_vec_w * camera.screen_width

    screen_vec_h = np.cross(camera.look_at, screen_vec_w)
    screen_vec_h = normalize_vec(screen_vec_h)
    screen_vec_h = screen_vec_h * camera.screen_width * screen_ratio
    
    screen_center = camera.position + screen_vec_forward * camera.screen_distance
    screen_top_left = screen_center - 0.5*screen_vec_w - 0.5*screen_vec_h
    return screen_top_left, screen_vec_w, screen_vec_h

def get_pixel_coordinates(row, col, top_left, screen_vec_w, screen_vec_h, width, height):
    pixel_coords = top_left + (col/width)*screen_vec_w + (row/height)*screen_vec_h
    return pixel_coords


def render_scene(camera: Camera, scene_settings: SceneSettings, objects, width, height):
    output_image = np.zeros((height, width, 3))
    screen_ratio = height/width
    materials, planes, cubes, spheres, lights = seperate_objects(objects)
    screen_top_left, screen_vec_w, screen_vec_h = calc_screen_parameters(camera, screen_ratio)
    for row in range(height):
        for col in range(width):
            pixel_coords = get_pixel_coordinates(row, col, screen_top_left, screen_vec_w, screen_vec_h, width, height)
            if pixel_coords[0]==0 and pixel_coords[1] == 0:
                breakd = 0
            direction = calc_normalized_vec_between_2_points(camera.position, pixel_coords)
            color = render_ray(camera.position, direction, scene_settings, materials, planes, cubes, spheres, lights, scene_settings.max_recursions)
            output_image[row][col] = color * 255

    return output_image


def render_ray(start, direction, scene_settings: SceneSettings, materials, planes, cubes, spheres, lights, iter_num):

    sorted_intersect = calc_intersections(start, direction, planes, cubes, spheres)# list of tuples: (object,[ts])
    if len(sorted_intersect)==0 or iter_num==1:
        return scene_settings.background_color

    nearest_surface,nearest_ts = sorted_intersect[0] # t and surface
    in_point = start+nearest_ts[0]*direction # the point where the ray hits the object
    out_point = start+nearest_ts[-1]*direction # the point where the ray gets out of the object

    material: Material = materials[nearest_surface.material_index-1]

    transparency_factor = material.transparency
    direction_reflect = get_reflected_vector(nearest_surface, in_point, direction)

    reflection_color =  material.reflection_color * render_ray(in_point, 
                                                               direction_reflect, 
                                                               scene_settings, 
                                                               materials, planes, 
                                                               cubes, 
                                                               spheres, 
                                                               lights, 
                                                               iter_num-1)
    if transparency_factor == 0:
        transparency_color = np.zeros(3)
    else:
        transparency_color =  render_ray(out_point, direction, scene_settings, materials, planes, cubes, spheres, lights, iter_num-1)

    lights_color = get_lights_color(lights, nearest_surface, direction, in_point, material, planes, cubes, spheres)
   
    output_color = transparency_factor*transparency_color + (1-transparency_factor)*lights_color + reflection_color
    return output_color


def get_lights_color(lights, surface, ray_direction, hitting_point, material: Material, planes, cubes, spheres):
    final_color = 0
    for light in lights:
        #TODO: shadows!!!!!!!!!!!!!
        surface_2_light_ray = calc_normalized_vec_between_2_points(hitting_point, light.position)
        shadow_intensity = current_shadow_intensity(light, hitting_point, surface_2_light_ray, planes, cubes, spheres)
        diffused_color = calc_diffused_color(light, surface, ray_direction, hitting_point, surface_2_light_ray, material)
        specular_color = calc_specular_color(light, surface, ray_direction, hitting_point, surface_2_light_ray, material)
        color = shadow_intensity * (diffused_color + specular_color)

        final_color += color

    return final_color

def current_shadow_intensity(light: Light, hitting_point, light_ray, planes, cubes, spheres):
    eps_hitting_point = hitting_point + 0.00001*light_ray
    is_intersecting = is_ray_intersecting(eps_hitting_point, light_ray, planes, cubes, spheres)
    shadow_intensity = 1
    if is_intersecting:
        shadow_intensity = 1-light.shadow_intensity
    return shadow_intensity


def calc_diffused_color(light: Light, surface, ray_direction, hitting_point, surface_2_light_ray, material: Material):
    surface_normal = get_normal(surface, hitting_point, ray_direction)
    dot_product = np.dot(surface_normal, surface_2_light_ray)
    return dot_product * (light.color * material.diffuse_color)


def calc_specular_color(light: Light, surface, ray_direction, hitting_point, surface_2_light_ray, material: Material):
    reflected_light = get_reflected_vector(surface, hitting_point, surface_2_light_ray)
    dot_product = np.dot(reflected_light, -ray_direction)
    return (light.color * material.specular_color) * light.specular_intensity * dot_product**material.shininess



def is_ray_intersecting(start, direction, planes, cubes, spheres):
    for sphere in spheres:
        t_s = calc_sphere_intersections(start, direction, sphere)
        if len(t_s) > 0:
            return True
    for plane in planes:
        t_s = plane_intersect_t(plane, start, direction)
        if len(t_s) > 0:
            return True
    for cube in cubes:
        t_s = cube_intersect_ts(cube, start, direction)
        if len(t_s) > 0:
            return True
        
    return False

def calc_intersections(start, direction, planes, cubes, spheres):
    intersect_surfaces=[]

    for sphere in spheres:
        t_s = calc_sphere_intersections(start, direction, sphere)
        if len(t_s):
            intersect_surfaces.append((sphere,t_s))

    for plane in planes:
        t_s = plane_intersect_t(plane, start, direction)
        if len(t_s):
            intersect_surfaces.append((plane,t_s))

    for cube in cubes:
        t_s = cube_intersect_ts(cube, start, direction)
        if len(t_s):
            intersect_surfaces.append((cube,t_s))

    sorted_surfaces = sorted(intersect_surfaces, key=lambda x: x[1][0])
    return sorted_surfaces

def calc_sphere_intersections(start, direction, sphere : Sphere):
    center_2_start = start - sphere.position
    a = np.dot(direction, direction)
    b = 2 * np.dot(direction, center_2_start)
    c = np.dot(center_2_start, center_2_start) - sphere.radius ** 2

    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return []
    elif discriminant == 0:
        t = -b / (2 * a)
        ts = [t]
    else:
        # Two intersections
        t1 = (-b + math.sqrt(discriminant)) / (2 * a)
        t2 = (-b - math.sqrt(discriminant)) / (2 * a)
        ts = sorted([t1, t2])

    ts = list(filter(lambda x: x>0, ts))
    return ts

def plane_intersect_t(plane : InfinitePlane, start, direction_vec): #returns list of t's
    #t = -(P0 • N - d) / (V • N)
    if np.dot(direction_vec,plane.normal) == 0:
        return []
    t = - (np.dot(start,plane.normal)-plane.offset)/(np.dot(direction_vec,plane.normal))
    #intersect_point = start + t*direction_vec
    return [t] if t>0 else []

def cube_intersect_ts(cube : Cube, start, direction): #returns list of t's
    x_axis = np.array([1,0,0])
    y_axis = np.array([0,1,0])
    z_axis = np.array([0,0,1])

    offset = 0.5* cube.scale
    x_p,y_p,z_p = cube.position[0],cube.position[1],cube.position[2]

    up_point = cube.position+np.array([0,0,offset])   #z axis
    down_point = cube.position+np.array([0,0,-offset]) #z axis
    left_point = cube.position+np.array([0,0,-offset])   #y axis
    right_point = cube.position+np.array([0,0,offset])  #y axis
    near_point = cube.position+np.array([0,0,offset]) #x axis
    far_point = cube.position+np.array([0,0,-offset])   #x axis

    up_plane = InfinitePlane(z_axis, np.dot(z_axis,up_point), cube.material_index) #TODO: why?
    down_plane = InfinitePlane(z_axis, np.dot(z_axis,down_point), cube.material_index)
    left_plane = InfinitePlane(y_axis, np.dot(y_axis,left_point), cube.material_index)
    right_plane = InfinitePlane(y_axis, np.dot(y_axis,right_point), cube.material_index)
    near_plane = InfinitePlane(x_axis, np.dot(x_axis,near_point), cube.material_index)
    far_plane = InfinitePlane(x_axis, np.dot(x_axis,far_point), cube.material_index)

    cube_planes = [up_plane,down_plane,left_plane,right_plane,near_plane,far_plane]
    intersection_ts=[]
    for plane in cube_planes:
        intersection_ts+=plane_intersect_t(plane,start,direction)

    if len(intersection_ts)==0:
        return [] 

    unique_ts=[]
    for t in intersection_ts:
        if not any(np.isclose(t, t_val) for t_val in unique_ts): #TODO check that works fine
            unique_ts.append(t)
    #unique_ts=np.array(unique_ts)

    intersection_points = [start+t*direction for t in unique_ts]

    #intersection_points = np.unique(np.array(intersection_points))
    # unique_points=[]
    # for point in intersection_points:
    #     if not any(np.allclose(point, selected_vector) for selected_vector in unique_points):
    #         unique_points.append(point)

    # intersection_points = np.array(unique_points)

    intersection_points_idx = [i for i in range(len(intersection_points)) if point_in_face(intersection_points[i], x_p, y_p, z_p, offset)]

    if len(intersection_points)==0:
        return []

    intersection_ts=[unique_ts[i] for i in intersection_points_idx]
    sorted(intersection_ts)
    return intersection_ts

    #norms = np.linalg.norm(intersection_points - start, axis=1)
    #closest_point = intersection_points[np.argmin(norms)]
    #return closest_point

def point_in_face(point, x_p, y_p, z_p, offset):
    in_face = (z_p-offset < point[2] and point[2] < z_p+offset) and \
              (y_p-offset < point[1] and point[1] < y_p+offset) and \
              (x_p-offset < point[0] and point[0] < x_p+offset)
    return in_face


def get_reflected_vector(surface, point, ray_direction):
    normal = get_normal(surface, point, ray_direction)
    reflected = calculate_reflected_vector(ray_direction, normal)
    return reflected


def get_normal(surface, point, ray_direction):
    if isinstance(surface, Cube):
        normal = calculate_cube_normal(surface, point)
    elif isinstance(surface, InfinitePlane):
        normal = calculate_plane_normal(surface, ray_direction)
    elif isinstance(surface, Sphere):
        normal = calculate_sphere_normal(surface, point)

    return normal

def calculate_cube_normal(cube: Cube, point):
    diff = point - cube.position
    max_dimension = np.argmax(np.abs(diff))

    normal = np.zeros(3)
    normal[max_dimension] = np.sign(diff[max_dimension])

    return normal

def calculate_plane_normal(plane: InfinitePlane, ray_direction):
    dot_product = np.dot(ray_direction, plane.normal)
    normal = -plane.normal if dot_product > 0 else plane.normal
    return normal

def calculate_sphere_normal(sphere: Sphere, point):
    return calc_normalized_vec_between_2_points(point, sphere.position)


def calculate_reflected_vector(direction, normal):
    incident_normalized = normalize_vec(direction)

    dot_product = np.dot(incident_normalized, normal)
    reflection = incident_normalized - 2*dot_product* normal
    reflection_normalized = normalize_vec(reflection)

    return reflection_normalized


def calc_normalized_vec_between_2_points(point_start, point_end):
    vec = point_end - point_start
    return normalize_vec(vec)


def normalize_vec(vec):
    normalized = vec / np.linalg.norm(vec)
    return normalized


def save_image(image_array):
    image = Image.fromarray(np.uint8(image_array))

    # Save the image to a file
    image.save("scenes/Spheres.png")


def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('--scene_file', type=str, default='scenes/test_easy.txt', help='Path to the scene file') #TODO change to pool
    parser.add_argument('--output_image', type=str, default='output/test.png', help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)

    image_array = render_scene(camera, scene_settings, objects, args.width, args.height)

    # Save the output image
    save_image(image_array)


if __name__ == '__main__':
    main()

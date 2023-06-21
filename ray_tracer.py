import argparse
import random
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

import time
timing_dict = {}
EPSILON = 0.001
x_axis = np.array([1,0,0])
y_axis = np.array([0,1,0])
z_axis = np.array([0,0,1])

axes_normals = np.stack([x_axis, x_axis, y_axis, y_axis, z_axis, z_axis])
cube_faces = np.stack([x_axis, -x_axis, y_axis, -y_axis, z_axis, -z_axis])

def write_time(start_time, key):
    t = time.time() - start_time
    if key in timing_dict.keys():
        timing_dict[key][0] += t
        timing_dict[key][1] += 1
    else:
        timing_dict[key] = [t, 1]

def print_timings():
    for key, val in  timing_dict.items():
        print(f'{key}: total time = {round(val[0], 2)},   average time= {round(val[0]/val[1], 5) if val[1] > 0 else 0},    number of executions= {val[1]}')


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

    screen_vec_w = np.cross(camera.up_vector, screen_vec_forward)
    screen_vec_w = normalize_vec(screen_vec_w)
    screen_vec_w = screen_vec_w * camera.screen_width

    screen_vec_h = np.cross(screen_vec_w, screen_vec_forward)
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
            # pixel_time = time.time()
            pixel_coords = get_pixel_coordinates(row, col, screen_top_left, screen_vec_w, screen_vec_h, width, height)
            direction = calc_normalized_vec_between_2_points(camera.position, pixel_coords)
            # if not (col == 255 and row == 255):
            #     continue
            color = render_ray(camera.position, direction, scene_settings, materials, planes, cubes, spheres, lights, 0)
            # write_time(pixel_time,'render_pixel')
            output_image[row][col] = color*255

    return output_image


def render_ray(start, direction, scene_settings: SceneSettings, materials, planes, cubes, spheres, lights, iter_num):
    # t = time.time()
    sorted_intersect = calc_intersections(start, direction, planes, cubes, spheres)# list of tuples: (object,[ts])
    # write_time(t,'calc_intersections')

    if len(sorted_intersect) == 0 or iter_num == scene_settings.max_recursions:
        return scene_settings.background_color

    nearest_surface,nearest_ts = sorted_intersect[0] # t and surface
    in_point = start+nearest_ts[0]*direction # the point where the ray hits the object
    out_point = start+nearest_ts[-1]*direction # the point where the ray gets out of the object

    material: Material = materials[nearest_surface.material_index-1]

    transparency_factor = material.transparency

    lights_color = get_lights_color(lights, nearest_surface, direction, in_point, material, planes, cubes, spheres, scene_settings)

    direction_reflect = get_reflected_vector(nearest_surface, in_point, direction)
    reflection_color =  material.reflection_color * render_ray(in_point+EPSILON*direction_reflect, 
                                                               direction_reflect, 
                                                               scene_settings, 
                                                               materials, planes, 
                                                               cubes, 
                                                               spheres, 
                                                               lights, 
                                                               iter_num + 1)
    if transparency_factor == 0:
        transparency_color = np.zeros(3)
    else:
        transparency_color =  render_ray(out_point+EPSILON*direction, direction, scene_settings, materials, planes, cubes, spheres, lights, 0)
   
    output_color = transparency_factor*transparency_color + (1-transparency_factor)*lights_color + reflection_color
    output_color=np.clip(output_color, 0., 1.)
    return output_color


def get_lights_color(lights, surface, ray_direction, hitting_point, material: Material, planes, cubes, spheres, scene_settings):
    final_color = 0
    for light in lights:
        percentage = calc_shadow_percentage(light, hitting_point, scene_settings,planes,cubes,spheres)
        light_intensity =  (1-light.shadow_intensity)+(light.shadow_intensity*percentage)

        surface_2_light_ray = calc_normalized_vec_between_2_points(hitting_point, light.position)
        diffused_color = calc_diffused_color(light, surface, ray_direction, hitting_point, surface_2_light_ray, material, light_intensity)
        specular_color = calc_specular_color(light, surface, ray_direction, hitting_point, surface_2_light_ray, material, light_intensity)

        final_color+=(diffused_color+specular_color)

    return final_color

def calc_shadow_percentage(light:Light, hitting_point, scene_settings:SceneSettings,planes,cubes,spheres):
    N = int(scene_settings.root_number_shadow_rays)
    main_ray_direction = calc_normalized_vec_between_2_points(hitting_point, light.position)
    x = np.cross(main_ray_direction, np.array([1, 0, 0]))
    if (x == 0).all():
        x = np.cross(main_ray_direction, np.array([0, 1, 0]))
    x /= np.linalg.norm(x)
    
    y = np.cross(main_ray_direction, x)
    y /= np.linalg.norm(y)

    left_bottom = light.position-(light.radius/2)*x-(light.radius/2)*y

    cell_len = light.radius / N
    x = x*cell_len
    y = y*cell_len

    prior_surface=None
    hit_light_count= 0
    for i in range(N):
        for j in range(N):
            cell_pos = left_bottom + (i + random.random()) * x + (j + random.random()) * y
            sub_ray_direction = calc_normalized_vec_between_2_points(hitting_point,cell_pos)
            eps_hitting_point = hitting_point + EPSILON*sub_ray_direction
            current_surface,is_intersect = is_ray_intersecting(eps_hitting_point, sub_ray_direction, planes, cubes, spheres, prior_surface)
            prior_surface = current_surface if current_surface else prior_surface
            #cell_light_ray = Ray(cell_pos, ray_vector)
            #cell_surface, cell_intersect = intersect.find_intersect(scene, cell_light_ray, find_all=False)
            if not is_intersect:
                hit_light_count += 1

    percentage = float(hit_light_count) / float(N * N)
    return percentage
    #return (1 - light.shadow_intens) + (light.shadow_intens * fraction)


def calc_diffused_color(light: Light, surface, ray_direction, hitting_point, surface_2_light_ray, material: Material,light_intensity):
    surface_normal = get_normal(surface, hitting_point, surface_2_light_ray)
    dot_product = np.dot(surface_normal, surface_2_light_ray)
    if dot_product < 0:
        return np.zeros(3, dtype=float)
    return dot_product *light.color *light_intensity* material.diffuse_color


def calc_specular_color(light: Light, surface, ray_direction, hitting_point, surface_2_light_ray, material: Material, light_intensity):
    reflected_light = get_reflected_vector(surface, hitting_point, -surface_2_light_ray)
    dot_product = np.dot(reflected_light, -ray_direction)
    if dot_product < 0:
        return np.zeros(3, dtype=float)
    return light.color * light_intensity*  material.specular_color * light.specular_intensity * (dot_product**material.shininess)



def is_ray_intersecting(start, direction, planes, cubes, spheres, prior_surface):
    if prior_surface!=None:
        if isinstance(prior_surface, Cube):
            t_s = cube_intersect_ts(prior_surface, start, direction)
        elif isinstance(prior_surface, InfinitePlane):
            t_s = plane_intersect_t(prior_surface, start, direction)
        elif isinstance(prior_surface, Sphere):
            t_s =  calc_sphere_intersections(start, direction, prior_surface)
        if len(t_s) > 0:
            return prior_surface,True
        
    for sphere in spheres:
        # t = time.time()
        t_s = calc_sphere_intersections(start, direction, sphere)
        # write_time(t,'sphere_intersections')
        if len(t_s) > 0:
            return sphere,True
    for plane in planes:
        # t = time.time()
        t_s = plane_intersect_t(plane, start, direction)
        # write_time(t,'plane_intersections')
        if len(t_s) > 0:
            return plane,True
    for cube in cubes:
        # t = time.time()
        t_s = cube_intersect_ts(cube, start, direction)
        # write_time(t,'cube_intersections')
        if len(t_s) > 0:
            return cube,True
        
    return None,False


def calc_intersections(start, direction, planes, cubes, spheres):
    intersect_surfaces=[]

    for sphere in spheres:
        # t = time.time()
        t_s = calc_sphere_intersections(start, direction, sphere)
        # write_time(t,'sphere_intersections')

        if len(t_s):
            intersect_surfaces.append((sphere,t_s))

    for plane in planes:
        # t = time.time()
        t_s = plane_intersect_t(plane, start, direction)
        # write_time(t,'plane_intersections')
        if len(t_s):
            intersect_surfaces.append((plane,t_s))

    for cube in cubes:
        # t = time.time()
        t_s = cube_intersect_ts(cube, start, direction)
        # write_time(t,'cube_intersections')
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

    ts = list(filter(lambda x: x>EPSILON, ts))
    return ts

def plane_intersect_t(plane : InfinitePlane, start, direction_vec): #returns list of t's
    t_list = calc_plane_intersection(start, direction_vec, plane.normal, plane.offset)
    return t_list if (len(t_list)>0 and t_list[0]>EPSILON) else []
 
def calc_plane_intersection(start, direction_vec, plane_normal, plane_offset):
    #t = -(P0 • N - d) / (V • N)
    dot_prod = np.dot(direction_vec,plane_normal)
    if dot_prod == 0:
        return []
    return [(plane_offset - np.dot(start, plane_normal)) / (dot_prod)]

def cube_intersect_ts(cube : Cube, start, direction): #returns list of t's
    offset = 0.5* cube.scale

    faces_centers = cube.position + offset*cube_faces
    faces_offsets = np.einsum('ij,ij->i', axes_normals, faces_centers)

    dot_product = np.dot(direction, axes_normals.T)
    zero_dot = dot_product != 0

    intersection_ts_np = (faces_offsets[zero_dot] - np.dot(start, axes_normals[zero_dot].T)) / dot_product[zero_dot]

    # up_point = cube.position + offset*z_axis   #z axis
    # down_point = cube.position - offset*z_axis #z axis
    # left_point = cube.position - offset*y_axis   #y axis
    # right_point = cube.position + offset*y_axis  #y axis
    # near_point = cube.position + offset*x_axis #x axis
    # far_point = cube.position - offset*x_axis   #x axis

    # intersection_ts=[]
    # # up_plane = (z_axis, np.dot(z_axis,up_point))
    # intersection_ts += calc_plane_intersection(start, direction, z_axis, np.dot(z_axis,up_point))

    # # down_plane = (z_axis, np.dot(z_axis,down_point))
    # intersection_ts += calc_plane_intersection(start, direction, z_axis, np.dot(z_axis,down_point))

    # # left_plane = (y_axis, np.dot(y_axis,left_point))
    # intersection_ts += calc_plane_intersection(start, direction, y_axis, np.dot(y_axis,left_point))

    # # right_plane = (y_axis, np.dot(y_axis,right_point))
    # intersection_ts += calc_plane_intersection(start, direction, y_axis, np.dot(y_axis,right_point))

    # # near_plane = (x_axis, np.dot(x_axis,near_point))
    # intersection_ts += calc_plane_intersection(start, direction, x_axis, np.dot(x_axis,near_point))

    # # far_plane = (x_axis, np.dot(x_axis,far_point))
    # intersection_ts += calc_plane_intersection(start, direction, x_axis, np.dot(x_axis,far_point))

    # cube_planes = [up_plane,down_plane,left_plane,right_plane,near_plane,far_plane]
    # intersection_ts=[]
    # for plane in cube_planes:
    #     intersection_ts+=plane_intersect_t(plane,start,direction)


    # intersection_ts_np = np.array(intersection_ts)
    intersection_ts_np = intersection_ts_np[intersection_ts_np > EPSILON]
    if len(intersection_ts_np)==0:
        return [] 
    unique_ts = np.unique(np.round(intersection_ts_np, decimals=10))
    # for t in intersection_ts:
    #     if not any(np.isclose(t, t_val) for t_val in unique_ts): #TODO check that works fine
    #         unique_ts.append(t)
    #unique_ts=np.array(unique_ts)

    # intersection_points = [start+t*direction for t in unique_ts]

    #intersection_points = np.unique(np.array(intersection_points))
    # unique_points=[]
    # for point in intersection_points:
    #     if not any(np.allclose(point, selected_vector) for selected_vector in unique_points):
    #         unique_points.append(point)

    # intersection_points = np.array(unique_points)

    vec_dirs = np.dot(unique_ts.reshape(len(unique_ts),1),direction.reshape(1,3))
    intersection_points = start + vec_dirs

    in_face = np.all((cube.position - offset) < intersection_points+EPSILON, axis=1) & np.all(intersection_points-EPSILON < (cube.position + offset), axis=1)

    # intersection_points_idx = [i for i in range(len(unique_ts)) if point_in_face(start + unique_ts[i]*direction, cube.position, offset)]

    # intersection_ts=[unique_ts[i] for i in intersection_points_idx]
    intersection_ts=unique_ts[in_face]
    
    return [] if len(intersection_ts) == 0 else sorted(intersection_ts)

    #norms = np.linalg.norm(intersection_points - start, axis=1)
    #closest_point = intersection_points[np.argmin(norms)]
    #return closest_point

def point_in_face(point, cube_center, offset):
    in_face = np.all((cube_center - offset) < point+EPSILON) and np.all(point-EPSILON < (cube_center + offset))
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
    normal = plane.normal if dot_product > 0 else -plane.normal
    return normal

def calculate_sphere_normal(sphere: Sphere, point):
    return calc_normalized_vec_between_2_points(sphere.position, point)


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


def save_image(image_array, file_name):
    image = Image.fromarray(np.uint8(image_array))

    # Save the image to a file
    image.save(f"{file_name}")


def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('--scene_file', type=str, default='scenes/our_scene.txt', help='Path to the scene file') #TODO change to pool
    parser.add_argument('--output_image', type=str, default='output/ours.png', help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)

    image_array = render_scene(camera, scene_settings, objects, args.width, args.height)
    print_timings()
    # Save the output image
    save_image(image_array, args.output_image)


if __name__ == '__main__':
    main()

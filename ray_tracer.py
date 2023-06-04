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

    screen_vec_h = np.cross(camera.look_at, screen_vec_w)
    screen_vec_h = screen_vec_h / np.linalg.norm(screen_vec_h)
    screen_vec_h = screen_vec_h * camera.screen_width * screen_ratio
    
    screen_center = np.array(camera.position) + screen_vec_forward * camera.screen_distance
    screen_top_left = screen_center - 0.5*screen_vec_w - 0.5*screen_vec_h
    return screen_top_left, screen_vec_w, screen_vec_h

def get_pixel_coordinates(row, col, top_left, screen_vec_w, screen_vec_h, width, height):
    pixel_coords = top_left + (col/width)*screen_vec_w + (row/height)*screen_vec_h
    return pixel_coords


def render_scene(camera: Camera, scene_settings, objects, width, height):
    camera_position = np.array(camera.position)
    screen_ratio = height/width
    materials, planes, cubes, spheres, lights = seperate_objects(objects)
    screen_top_left, screen_vec_w, screen_vec_h = calc_screen_parameters(camera, screen_ratio)
    for row in range(height):
        for col in range(width):
            pixel_coords = get_pixel_coordinates(row, col, screen_top_left, screen_vec_w, screen_vec_h, width, height)
            direction = (pixel_coords-camera_position)/np.linalg.norm(pixel_coords-camera_position)
            color = render_ray(camera_position, direction, scene_settings, materials, planes, cubes, spheres, lights)


<<<<<<< HEAD
def render_ray(start, direction, scene_settings, materials, planes, cubes, spheres, lights, iter_num=10):
=======
def calc_color(start,direction,surfaces):
    x=0
>>>>>>> 20868802e8cea060c5c3557f22069ce07bec6798

    sorted_intersect = calc_intersections(start, direction, planes, cubes, spheres)# list of tuples: (object,[ts])
    if len(sorted_intersect)==0 or iter_num==1:
        return scene_settings.background_color

    nearest_surface,nearest_ts = sorted_intersect[0] # t and surface
    in_point = start+nearest_ts[0]*direction # the point where the ray hits the object
    out_point = start+nearest_ts[-1]*direction # the point where the ray gets out of the object

    diffuse_color = materials[nearest_surface.material_index].diffuse_color
    specular_color = materials[nearest_surface.material_index].specular_color
    transparency = materials[nearest_surface.material_index].transparency

    compute_lights(direction, lights) # specular_color is calculated

    next_start_bg = out_point 
    bg_color =  render_ray(next_start_bg, direction, scene_settings, materials, planes, cubes, spheres, lights, iter_num-1)

    next_start_reflect = in_point
    direction_reflect = 0 #TODO: calculate the reflect angle!
    reflection_color =  render_ray(next_start_reflect, direction_reflect, scene_settings, materials, planes, cubes, spheres, lights, iter_num-1)

    output_color = transparency*bg_color + (1-transparency)*(diffuse_color + specular_color) + reflection_color

def calc_intersections(start, direction, planes, cubes, spheres):
    intersect_surfaces=[]

    for sphere in spheres:
        t_s += calc_sphere_intersections(start, direction, sphere)
        intersect_surfaces+=(sphere,t_s)

    for plane in planes:
        t_s += plane_intersect_t(plane, start, direction)
        intersect_surfaces+=(plane,t_s)

    for cube in cubes:
        t_s += cube_intersect_ts(cube, start, direction)
        intersect_surfaces+=(cube,t_s)

    sorted_surfaces = sorted(intersect_surfaces, key=lambda x: x[1][0])
    return sorted_surfaces


def calc_sphere_intersections(start, direction, sphere : Sphere):
    a = np.dot(direction, direction)
    b = 2 * np.dot(direction, np.subtract(start, sphere.position))
    c = np.dot(np.subtract(start, sphere.position), np.subtract(start, sphere.position)) - sphere.radius ** 2

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
        #intersection_point1 = start + direction*t1
        #intersection_point2 = start + direction*t2
        ts = [t1, t2].sort()
        return ts

def plane_intersect_t(plane : InfinitePlane, start, direction_vec): #returns list of t's
    #t = -(P0 • N - d) / (V • N)
    if np.dot(direction_vec,plane.normal) == 0:
        return []
    t = - (np.dot(start,plane.normal)-plane.offset)/(np.dot(direction_vec,plane.normal))
    #intersect_point = start + t*direction_vec
    return [t]

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

    up_plane = InfinitePlane(z_axis,np.dot(z_axis,up_point)) 
    down_plane = InfinitePlane(z_axis,np.dot(z_axis,down_point))
    left_plane = InfinitePlane(y_axis,np.dot(y_axis,left_point))
    right_plane = InfinitePlane(y_axis,np.dot(y_axis,right_point))
    near_plane = InfinitePlane(x_axis,np.dot(x_axis,near_point))
    far_plane = InfinitePlane(x_axis,np.dot(x_axis,far_point))

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

    intersection_points[ (z_p-offset<intersection_points[:,2]) & (intersection_points[:,2]<z_p+offset) ]
    intersection_points[ (y_p-offset<intersection_points[:,1]) & (intersection_points[:,1]<y_p+offset) ]
    intersection_points[ (x_p-offset<intersection_points[:,0]) & (intersection_points[:,0]<x_p+offset) ]

    if len(intersection_points)==0:
        return []

    intersection_ts=[(point-start)[0]/direction[0] for point in intersection_points]
    intersection_ts.sort()
    return intersection_ts

    #norms = np.linalg.norm(intersection_points - start, axis=1)
    #closest_point = intersection_points[np.argmin(norms)]
    #return closest_point

def save_image(image_array):
    image = Image.fromarray(np.uint8(image_array))

    # Save the image to a file
    image.save("scenes/Spheres.png")


def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('--scene_file', type=str, default='scenes/test.txt', help='Path to the scene file') #TODO change to pool
    parser.add_argument('--output_image', type=str, default='output/test.png', help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)

    render_scene(camera, scene_settings, objects, args.width, args.height)

    # Dummy result
    image_array = np.zeros((500, 500, 3))

    # Save the output image
    save_image(image_array)


if __name__ == '__main__':
    main()

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



def render_scene(camera: Camera, scene_settings, objects, width, height):
    materials, surfaces, lights = seperate_objects(objects)

    for row in height:
        for col in width:
            pixel_coords = get_pixel_coordinates(row, col, height, width, camera)
            color = render_ray(camera.position, pixel_coords, scene_settings, materials, surfaces, lights)



def render_ray(start, direction, scene_settings, materials, surfaces, lights):

    intersections = calc_intersections(start, direction, surfaces)
    # if None return background
    nearest_surface = find_nearest(start, intersections)

    intersection_point = calc_inter_point(start, direction, nearest_surface)

    compute_lights(direction, lights)
    
    t_start, t_end = get_transparency_vector()
    transparency_color =  render_ray()
    
    r_start, r_end = get_reflection_vector()
    transparency_color =  render_ray()

def plane_intersect_point(plane : InfinitePlane, start, direction_vec):
    #t = -(P0 • N - d) / (V • N)
    if np.dot(direction_vec,plane.normal) == 0:
        return None
    t = - (np.dot(start,plane.normal)-plane.offset)/(np.dot(direction_vec,plane.normal))
    intersect_point = start + t*direction_vec
    return intersect_point

def cube_intersect(cube : Cube, start, direction_vec):
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
    intersection_points = [plane_intersect_point(p,start,direction_vec) for p in cube_planes]
    intersection_points = [i for i in intersection_points if i is not None]

    if len(intersection_points)==0:
        return None 

    intersection_points = np.unique(np.array(intersection_points))

    intersection_points[ (z_p-offset<intersection_points[:,2]) & (intersection_points[:,2]<z_p+offset) ]
    intersection_points[ (y_p-offset<intersection_points[:,1]) & (intersection_points[:,1]<y_p+offset) ]
    intersection_points[ (x_p-offset<intersection_points[:,0]) & (intersection_points[:,0]<x_p+offset) ]

    if len(intersection_points)==0:
        return None 

    norms = np.linalg.norm(intersection_points - start, axis=1)
    closest_point = intersection_points[np.argmin(norms)]
    return closest_point

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

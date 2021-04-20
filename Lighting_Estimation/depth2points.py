import numpy as np
import cv2
import struct
import os
import glob
import img_interpolation
import utils


def get_pointcloud(color_image, depth_image, remove_up_down=True):
    """ creates 3D point cloud of rgb images by taking depth information
        input : color image: numpy array[h,w,c], dtype= float32
                depth image: numpy array[h,w] values of all channels will be same
        output : camera_points, color_points - both of shape(no. of pixels, 3)
    """
    assert color_image.shape[2] == 3
    image_height = depth_image.shape[0]
    image_width = depth_image.shape[1]
    pixel_x, pixel_y = np.meshgrid(np.linspace(0, image_width - 1, image_width),
                                   np.linspace(0, image_height - 1, image_height))
    theta = pixel_x * 2.0 * np.pi / image_width - np.pi
    phi = (pixel_y+0.5) * np.pi / image_height - np.pi / 2

    camera_points_x = depth_image * np.cos(phi) * np.sin(theta)
    camera_points_y = depth_image * np.sin(phi)
    camera_points_z = depth_image * np.cos(phi) * np.cos(theta)

    camera_points = np.array([camera_points_x, camera_points_y, camera_points_z]).transpose(1, 2, 0).reshape(-1, 3)
    color_points = color_image.reshape(-1, 3)

    if remove_up_down:
        depth_image[:72, :] = 0
        depth_image[440:, :] = 0
    # Remove invalid 3D points (where depth == 0)
    valid_depth_ind = np.where(depth_image.flatten() > 0)[0]
    camera_points = camera_points[valid_depth_ind, :]
    color_points = color_points[valid_depth_ind, :]

    return camera_points, color_points


def warp_points_image(camera_points_input, color_points_input, target_p=(0, 0, 0), inter_method=None, target_size=(128, 256)):
    """
    This function generate the Illumination map given the 3D target point at space.

    :param camera_points_input: 3D position of the pixels in an image (represent as point-light-source)
    :param color_points_input: The RGB of the camera_points_input.
    :param target_p: The 3D position of the target point that we want to estimate the Illumination map
    :param inter_method: interpolation method to be used, options are: {‘linear’, ‘nearest’, ‘cubic’, None},
                         None represent for no interpolation. See scipy.interpolate.griddata for more detail
    :param target_size: the output size of the estimated Illumination map
    :return: (the estimated Illumination map,  the cooresponding depth map)
    """
    new_image = np.zeros((target_size[0], target_size[1], 3), np.float32)
    new_depth = np.zeros((target_size[0], target_size[1]), np.float32)
    image_height = target_size[0]
    image_width = target_size[1]
    points2img_dis = dict()

    r = np.sqrt(np.sum((camera_points_input - target_p) ** 2, -1))
    theta_phi = utils.world_2_angle_coordinate(camera_points_input - target_p)
    i_x_y = utils.angle_2_image_coordinate(theta_phi, image_width, image_height)

    for i in range(len(camera_points_input)):
        i_x, i_y = i_x_y[i]

        if (i_y, i_x) not in points2img_dis.keys():
            points2img_dis[(i_y, i_x)] = r[i]
            new_image[i_y, i_x, :] = color_points_input[i]
            new_depth[i_y, i_x] = r[i]
        else:
            if points2img_dis[(i_y, i_x)] > r[i]:
                points2img_dis[(i_y, i_x)] = r[i]
                new_image[i_y, i_x, :] = color_points_input[i]
                new_depth[i_y, i_x] = r[i]
            else:
                pass

    if inter_method is not None:
        new_img_inter = img_interpolation.img_inter_func(valid_index=np.where(np.sum(new_image, -1) != 0),
                                                         values=new_image, method=inter_method)
        new_depth_inter = img_interpolation.img_inter_func(valid_index=np.where(np.sum(new_image, -1) != 0),
                                                           values=new_depth, method=inter_method)
        return new_img_inter, new_depth_inter
    else:
        return new_image, new_depth


if __name__ == '__main__':
    for scene_path in glob.glob('../data/*/*'):
        scene_name = os.path.basename(scene_path)
        output_directory = scene_path
        depth_filename = os.path.join(output_directory, 'depth.npy')
        depth_data = np.load(depth_filename)

        #  read in LDR image, convert to 0~1
        color_data = cv2.imread(output_directory + '/up.png')[:,:,::-1].astype(np.float32)/255   #  read in BGR, so convert to RGB
        camera_points, color_points = get_pointcloud(color_data, depth_data)
        print("Creating the point Cloud file at : ", output_directory)
        output_filename = os.path.join(output_directory, 'camera_points.npy')
        np.save(output_filename, camera_points)
        output_filename = os.path.join(output_directory, 'ldr_color_points.npy')
        np.save(output_filename, color_points)

        #  read in HDR image (Blender synthetic dataset is captured in hdr format)
        hdr_file = os.path.join(scene_path, 'hdr.npy')
        if os.path.isfile(hdr_file):
            hdr_color_data = np.load(hdr_file)    # hdr.npy is stored in RGB, butwith shape [3 512 1024],
            hdr_color_data = np.moveaxis(hdr_color_data, 0, -1)  #  convert to shape  [512 1024 3]
        else:
            hdr_color_data = color_data ** 2.2
        camera_points, hdr_color_points = get_pointcloud(hdr_color_data, depth_data)
        output_filename = os.path.join(output_directory, 'hdr_color_points.npy')
        np.save(output_filename, hdr_color_points)

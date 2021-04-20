import cv2 as cv
import os
import numpy as np
from Lighting_Estimation import depth2points
from Lighting_Estimation import img_interpolation
import time
from multiprocessing import Pool as ThreadPool


def warpper(param):
    camera_p, color_p, pixel_l, size, pixel_num = param
    if pixel_num % 1000 == 0:
        print('Rendered pixel: ', pixel_num)

    globalhdrmap, _ = depth2points.warp_points_image(camera_p, color_p, target_p=pixel_l, target_size=size)  # global_hdr_map in RGB format
    globalhdrmap_inter = img_interpolation.img_inter_func(valid_index=np.where(np.sum(globalhdrmap, -1) != 0), values=globalhdrmap, method='nearest')

    return globalhdrmap_inter


if __name__ == '__main__':
    scene_root = './data/real/'

    for s_name in ['hall', 'room']:  #  ['barbershop', 'bedroom', 'classroom', 'livingroom', 'school']  # ['hall', 'room', 'office']
        camera_points = np.load(os.path.join(scene_root, s_name, 'camera_points.npy')).astype(np.float32)
        color_points = np.load(os.path.join(scene_root, s_name, 'hdr_color_points.npy')).astype(np.float32)  # color_points in RGB format

        # target_env_map_size = (64, 128)
        target_env_map_size = (32, 64)
        total_part = 10
        total_num_pixels = len(camera_points)

        for this_part in range(1,11):
            params = []
            start_idx = int(np.round((this_part - 1) / total_part * total_num_pixels))
            end_idx = int(np.round(this_part / total_part * total_num_pixels))
            print("start idx: %d ,   end idx: %d ,   Part: %d / %d" % (start_idx, end_idx, this_part, total_part))
            for i in range(start_idx, end_idx):
                pixel_loc = camera_points[i]*0.95
                # compute the HDR map
                params.append((camera_points, color_points, pixel_loc, target_env_map_size, i))

            pool = ThreadPool(11)
            results = pool.map(warpper, params)
            pool.close()
            pool.join()
            results_arr = np.array(results)
            print('done')
            np.save(os.path.join(scene_root, s_name, 'warped_env_map_part_%04d.npy' % this_part), results_arr)
            del results
            del params
            del results_arr



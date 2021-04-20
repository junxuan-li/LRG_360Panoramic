import numpy as np
from Lighting_Estimation import utils


def light_direction_normal_product(light_direction, normal):
    product = np.sum(light_direction * normal, axis=-1)
    occlude_idx = np.where(product < 0)
    product[occlude_idx] = 0
    return product


def shading_renderer(env_intensity, target_normal, ld_surface_map):
    """
    Shading = dot(Light_direction * surface_area, normal) * environment_intensity
    Where "Light_direction * surface_area" is pre-computed as ld_surface_map
    """
    l_contri = light_direction_normal_product(ld_surface_map, target_normal)
    if len(env_intensity.shape) == 3:
        l_contri = np.expand_dims(l_contri, axis=-1)
    result = l_contri * env_intensity
    return result.sum(axis=(0, 1))


if __name__ == '__main__':
    import cv2 as cv
    import os
    for scene_name in ['bedroom']:  #['bedroom', 'classroom', 'school', 'livingroom', 'barbershop']: #
        env_list_path = os.path.join('./data/synthetic/', scene_name)

        normal_path = os.path.join(env_list_path, 'output_normal.png')
        normal_map = (cv.imread(normal_path)[:,:,::-1].astype(np.float32) - 127) / 127
        # normal_path = os.path.join(env_list_path, 'output_normal.npy')
        # normal_map = np.load(normal_path)
        normal_map = utils.normalize_axis(normal_map, axis=-1)
        # normal_map = cv.resize(normal_map, (256, 128), interpolation=cv.INTER_NEAREST)

        # env_map = np.load('data/real/office/hdr.npy')
        # env_map = cv.resize(env_map, (128, 64), interpolation=cv.INTER_LINEAR)

        import time
        start_time = time.time()
        shading = np.zeros(shape=normal_map.shape, dtype=np.float32)
        total_num_pixel = shading.shape[0]*shading.shape[1]
        i = 0
        this_part = 1
        env_list = np.load(os.path.join(env_list_path, 'warped_env_map_part_%04d.npy' % this_part))
        height, width = env_list.shape[1], env_list.shape[2]

        light_direction_map = utils.LightDirection_of_EnvMap(height=height, width=width).astype(np.float32)

        light_direction_map_transformed = np.empty(shape=light_direction_map.shape,dtype=np.float32)
        light_direction_map_transformed[:, :, 1] = -light_direction_map[:, :, 1]
        light_direction_map_transformed[:, :, 0] = light_direction_map[:, :, 2]
        light_direction_map_transformed[:, :, 2] = -light_direction_map[:, :, 0]

        surface_coef = utils.SurfaceArea_of_EnvMap(height=height, width=width).astype(np.float32)
        li_suface_coef = light_direction_map_transformed * np.expand_dims(surface_coef, axis=-1).astype(np.float32)

        for h in range(0, 72):
            i = 0
            for w in range(shading.shape[1]):
                env_map = env_list[i]
                normal = normal_map[h,w,:]
                shading[h,w,:] = shading_renderer(env_map, normal, li_suface_coef)
                i = i+1
            print("Render pixel: [%3d , %4d]  Est: %.2f min" % (h, w, (time.time()-start_time)/(h+1) * (511-h) / 60))

        i = 0
        for h in range(72, 440):
            for w in range(shading.shape[1]):
                if i >= len(env_list):
                    this_part +=1
                    del env_list
                    env_list = np.load(os.path.join(env_list_path, 'warped_env_map_part_%04d.npy' % this_part))
                    i = 0
                env_map = env_list[i]
                normal = normal_map[h,w,:]
                shading[h,w,:] = shading_renderer(env_map, normal, li_suface_coef)
                i += 1
            print("Render pixel: [%3d , %4d]  Est: %.2f min" % (h, w, (time.time()-start_time)/(h+1) * (511-h) / 60))

        for h in range(440, 512):
            i = -1024
            for w in range(shading.shape[1]):
                env_map = env_list[i]
                normal = normal_map[h,w,:]
                shading[h,w,:] = shading_renderer(env_map, normal, li_suface_coef)
                i = i+1
            print("Render pixel: [%3d , %4d]  Est: %.2f min" % (h, w, (time.time()-start_time)/(h+1) * (511-h) / 60))

        np.save(os.path.join(env_list_path, 'shading_full.npy'), shading)
        print('done')



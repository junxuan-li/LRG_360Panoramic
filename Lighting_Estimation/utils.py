import numpy as np

"""
    In this file, the coordinate system is defined as below:
    Image coordinate system:
    
     -------------->  x
    |  ***
    |    ***
    |     ****
    y
    #####################################
    
    Sphere coordinate system:
    vertically angle (elevation) : phi,    is the angle between vector and x-z plane
    Image coordinate   y    range:   0     ~ height
    Sphere coordinate  phi  range: -0.5*pi ~ 0.5*pi
    
    horizontally angle (azimuth): theta,   is the angle between vector and z axis
    Image coordinate   x      range:   0 ~ width
    Sphere coordinate  theta  range: -pi ~ pi
    
    #####################################
    World coordinate system:
        z
       /
      /
     /
     -------------->  x
    |
    |
    |
    y

"""


def image_2_world_coordinate(img_x, img_y, width, height):
    """
    Given the image coordinate (img_x, img_y) and the size of EnvMap, compute the world coordinates of it.
    Return an normalized vector as numpy array in size (3,)    Representing (world_x, world_y, world_z)
    """
    theta = img_x * 2 * np.pi / width - np.pi
    phi = (img_y + 0.5) * np.pi / height - np.pi / 2
    return np.array([np.cos(phi) * np.sin(theta), np.sin(phi), np.cos(phi) * np.cos(theta)])


def angle_2_image_coordinate(theta_phi, width, height):
    theta = theta_phi[:, 0]
    phi = theta_phi[:, 1]

    x = (theta + np.pi) * width / 2 / np.pi
    y = (phi + np.pi / 2) * height / np.pi - 0.5
    i_x = np.round(x).astype(np.int) % width
    i_y = np.round(y).astype(np.int) % height
    return np.column_stack((i_x, i_y))


def world_2_angle_coordinate(world_p):
    """
    Input   world_coordinate : (num , 3)    like: ([x,y,z], [x,y,z], ...)
    Output  angle_coordiante : (num , 2)    like: ([theta, phi], [theta, phi], ...)
    """
    target_p_x, target_p_y, target_p_z = world_p[:, 0], world_p[:, 1], world_p[:, 2]

    r = np.sqrt(np.sum(world_p ** 2, -1))

    phi = np.arcsin(target_p_y / r)
    theta = np.arcsin(target_p_x / np.sqrt(target_p_x ** 2 + target_p_z ** 2+1e-10))

    idx = np.where((target_p_x < 0) & (target_p_z < 0))
    theta[idx] = - theta[idx] - np.pi

    idx = np.where((target_p_x > 0) & (target_p_z < 0))
    theta[idx] = - theta[idx] + np.pi

    # if target_p_x > 0 and target_p_z > 0:
    #     theta = theta
    # elif target_p_x < 0 and target_p_z < 0:
    #     theta = - theta - np.pi
    # elif target_p_x < 0 and target_p_z > 0:
    #     theta = theta
    # elif target_p_x > 0 and target_p_z < 0:
    #     theta = - theta + np.pi
    return np.column_stack((theta, phi))


def auto_render_coor_to_image_coor(xyz, width=1024, height=512):
    world_p = xyz.copy()
    world_p[:, 2] = xyz[:, 0]
    world_p[:, 1] = -xyz[:, 2]
    world_p[:, 0] = -xyz[:, 1]
    theta_phi = world_2_angle_coordinate(world_p)
    ix_iy = angle_2_image_coordinate(theta_phi, width, height)
    return ix_iy


def normalize(x):
    if len(np.shape(x)) == 1:
        return x / (np.linalg.norm(x) + 1e-12)
    else:
        return x / np.linalg.norm(x, axis=1)[:, np.newaxis]


def normalize_axis(x, axis=None):
    if len(np.shape(x)) == 1:
        return x/(np.linalg.norm(x))
    else:
        return x/np.expand_dims(np.linalg.norm(x, axis=axis), axis=axis)


def calc_foot_point_on_plane(n, v, h=0):
    """

    :param np.ndarray n: normal vector of plane
    :param np.ndarray v:
    :param float h: plane height
    :return:
    :rtype: np.ndarray
    """
    n = np.array(n).flatten() / np.linalg.norm(n)
    v = np.array(v).flatten() / np.linalg.norm(v)
    p = v - (np.dot(n, v) - h) * n
    return normalize(p)


# Rotate vector around arbitrary axis
def rotateVector(vector, axis, angle):
    cos_ang = np.reshape(np.cos(angle), (-1))
    sin_ang = np.reshape(np.sin(angle), (-1))
    vector = np.reshape(vector, (-1, 3))
    axis = np.reshape(np.array(axis), (-1, 3))
    out = vector * cos_ang[:, np.newaxis] + \
          axis * np.dot(vector, np.transpose(axis)) * (1 - cos_ang)[:, np.newaxis] + \
          np.cross(axis, vector) * sin_ang[:, np.newaxis]

    return np.reshape(out, (-1))


def vec_2_halfangle(light, view, normal):
    view = np.array(view).reshape(-1)
    normal = np.array(normal).reshape(-1)
    light = np.array(light).reshape(-1)
    H = normalize((view + light) / 2)

    theta_h = np.arccos(np.dot(normal, H) / (np.linalg.norm(normal) * np.linalg.norm(H) + 1e-12))

    foot_light = calc_foot_point_on_plane(normal, light)
    foot_H = calc_foot_point_on_plane(normal, H)
    t = foot_light
    binormal = normalize(np.cross(normal, t))

    phi_h = np.arccos(np.dot(t, foot_H) / (np.linalg.norm(t) * np.linalg.norm(foot_H) + 1e-12))

    tmp = rotateVector(light, normal, -phi_h)
    diff = rotateVector(tmp, binormal, -theta_h)
    foot_diff = calc_foot_point_on_plane(normal, diff)
    theta_d = np.arccos(np.dot(normal, diff) / (np.linalg.norm(normal) * np.linalg.norm(diff) + 1e-12))

    phi_d = np.arccos(np.dot(t, foot_diff) / (np.linalg.norm(t) * np.linalg.norm(foot_diff) + 1e-12))

    return theta_h, theta_d, phi_d


def SurfaceArea_of_EnvMap(height, width):
    """
    Given a size of the env_map, compute the cooresponding surface area of each point at a Sphere.
    Return a numpy arrary with size (height, width). And np.sum(surface_area)=1.
    Surface Area of a Sphere formula: S = 2 * pi * R * h .   where R is radius of sphere, h is the height of the 'ring'
    """
    surface_area = np.empty(shape=(height, width))
    for i in range(height):
        phi_up = np.pi * i / height
        phi_down = np.pi * (i + 1) / height
        h = np.cos(phi_up) - np.cos(phi_down)
        ds = h / width
        surface_area[i, :] = ds
    surface_area = surface_area / np.sum(surface_area)
    return surface_area


def LightDirection_of_EnvMap(height, width):
    """
    Given a size of the env_map, compute the cooresponding Light direction as a vector on each point at a Sphere.
    The vector map is computed as like an 'outgoing' vector map.
    Return a numpy arrary with size (height, width, 3). And each point as an normalized vector.
    """
    light_vector_map = np.empty(shape=(height, width, 3))
    for img_x in range(width):
        for img_y in range(height):
            light_vector_map[img_y, img_x, :] = image_2_world_coordinate(img_x=img_x, img_y=img_y, width=width, height=height)
    return light_vector_map

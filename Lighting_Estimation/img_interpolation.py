import numpy as np
from scipy.interpolate import griddata


def img_inter_func(valid_index, values, method='nearest'):
    out_img = np.empty(values.shape, np.float32)
    X, Y = np.meshgrid(np.arange(0, values.shape[0]), np.arange(0, values.shape[1]))
    if len(values.shape) == 3:
        for i in range(values.shape[2]):
            chn = values[:, :, i]
            Ti = griddata(valid_index, chn[valid_index], (X, Y), method=method)
            out_img[:, :, i] = Ti.transpose()
    else:
        Ti = griddata(valid_index, values[valid_index], (X, Y), method=method)
        out_img = Ti.transpose()
    return out_img


from nd2reader import ND2Reader
from skimage import io
import numpy as np
from tifffile import imread, imwrite

nucleus_ch = 1

file_type = 'tif'
file_type = "nd2"

img_file = "/Users/snk218/Desktop/guru/nd2/01.nd2"
#"/Users/snk218/Desktop/guru/tif/MDCK_TETCyAAmix_2W_1_max_proj.tif"

if file_type == 'nd2':
    images = ND2Reader(img_file)
    print(images.sizes)

    if 'z' in images.sizes:
        z_levels = True
    else:
        z_levels = False

    if 'v' in images.sizes:
        n_fov = images.sizes['v']
    else:
        n_fov = 1

    if not ('c' in images.sizes and 'y' in images.sizes and 'x' in images.sizes):
        raise ValueError('Image must have c, y, and x dimensions')

    images.bundle_axes = 'vczyx'
    images = images[0]  # ND2Reader adds an extra dimension to the beginning
    n_fov = len(images)
else:
    # TIF file is a z-projection of a single FOV
    images_ = io.imread(img_file)
    images = imread(img_file)

    maxch=10
    if len(images.shape) == 3 and images.shape[0] > maxch and images.shape[2] < maxch:

        levels = []
        for level in range(images.shape[2]):
            levels.append(images[:, :, level])
        images = np.stack(levels)

    # print(images.max())
    n_fov = 1
print("n_fov:", n_fov)
for i in range(n_fov):
    if file_type == 'nd2':
        fov = images[i]
    else:
        fov = images

    # z-project the dapi channel
    if file_type == 'nd2':
        dapi_img_stack = fov[nucleus_ch]
        dapi_img = np.max(dapi_img_stack, axis=0)
    else:
        dapi_img = fov[nucleus_ch]
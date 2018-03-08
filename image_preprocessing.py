import numpy as np
from scipy.misc import imread


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


def crop(image, image_size):
    if image.shape[1] > image_size:
        sz1 = int(image.shape[1] // 2)
        sz2 = int(image_size // 2)
        image = image[(sz1 - sz2):(sz1 + sz2), (sz1 - sz2):(sz1 + sz2), :]
    return image


def load_image(img, image_size, do_prewhiten=True):
    image = np.zeros((1, image_size, image_size, 3))
    _image = imread(img)
    if _image.ndim == 2:
        _image = to_rgb(_image)
    if do_prewhiten:
        _image = prewhiten(_image)
    _image = crop(_image, image_size)
    image[0, :, :, :] = _image
    return image


def reverse_channels(img):
    return img[:, :, ::-1]


import numpy as np

from PIL import Image


def make_rgb(image):
    pix = np.array(image)
    img = np.zeros((pix.shape[0], pix.shape[1], 3))
    img[:, :, 0] = pix
    img[:, :, 1] = pix
    img[:, :, 2] = pix
    image = Image.fromarray(np.uint8(img))
    return image


if __name__ == '__main__':
    make_rgb()
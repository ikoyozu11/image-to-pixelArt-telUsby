import cv2
import numpy as np
from numba import njit

@njit(fastmath=True)
def accelerate_conversion(image, width, height, color_coeff, step):
    array_of_values = []
    for x in range(0, width, step):
        for y in range(0, height, step):
            r, g, b = image[x, y] // color_coeff
            if r + g + b:
                array_of_values.append(((r, g, b), (x, y)))
    return array_of_values

def convert_to_pixel_art(image, pixel_size=5, color_lvl=10):
    height, width, _ = image.shape
    color_coeff = 255 // (color_lvl - 1)

    new_image = np.zeros((height, width, 3), dtype=np.uint8)

    for x in range(0, width, pixel_size):
        for y in range(0, height, pixel_size):
            r, g, b = image[y, x] // color_coeff
            if r + g + b:
                new_image[y:y+pixel_size, x:x+pixel_size] = (r * color_coeff, g * color_coeff, b * color_coeff)

    return new_image

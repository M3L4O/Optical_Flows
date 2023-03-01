import math

import numpy as np
import cv2
from matplotlib.colors import hsv_to_rgb


def gaussderiv(img):
    gauss_img = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
    Dx = cv2.Sobel(gauss_img, cv2.CV_64F, 1, 0, ksize=5)
    Dy = cv2.Sobel(gauss_img, cv2.CV_64F, 0, 1, ksize=5)

    return Dx, Dy


def gausssdiff(img1, img2):
    diff = cv2.subtract(img1, img2)
    print(np.mean(diff))
    return diff


def calculate_derivatives(img1, img2):
    i_t = gausssdiff(img2, img1)
    i_x, i_y = gaussderiv(img1)
    return i_x, i_y, i_t

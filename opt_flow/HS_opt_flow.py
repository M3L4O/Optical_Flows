import os
import numpy as np
import cv2 as cv
from argparse import ArgumentParser
from ..utils.tools import draw_quiver, get_derivatives



def horn_schunk(path1, path2, alpha, max_err):
    img1 = cv.imread(path1, cv.IMREAD_GRAYSCALE).astype(float)
    img2 = cv.imread(path2, cv.IMREAD_GRAYSCALE).astype(float)

    img1 = cv.resize(img1, (1920, 1080))
    img2 = cv.resize(img2, (1920, 1080))

    
    # set up initial values
    u = np.zeros(img1.shape)
    v = np.zeros(img1.shape)
    fx, fy, ft = get_derivatives(img1, img2)
    avg_kernel = np.array(
        [[1 / 12, 1 / 6, 1 / 12], [1 / 6, -1, 1 / 6], [1 / 12, 1 / 6, 1 / 12]], float
    )

    avg_kernel = cv.flip(avg_kernel, -1)
    d = alpha**2 + fx**2 + fy**2

    iter_cnt = 0
    while True:
        iter_cnt += 1

        u_avg = cv.filter2D(u, -1, avg_kernel)
        v_avg = cv.filter2D(v, -1, avg_kernel)

        p = fx * u_avg + fy * v_avg + ft

        prev = u
        u = u_avg - fx * p / d
        v = v_avg - fy * p / d

        diff = ((u - prev) ** 2).mean()
        if diff < max_err or iter_cnt > 300:
            break

    draw_quiver(u, v, img1)

    return [u, v]


if __name__ == "__main__":
    parser = ArgumentParser(description="Horn Schunck program")
    parser.add_argument("img1", type=str, help="First image name (include format)")
    parser.add_argument("img2", type=str, help="Second image name (include format)")
    args = parser.parse_args()

    img1, img2 = os.path.abspath(args.img1), os.path.abspath(args.img2)

    u, v = horn_schunk(img1, img2, alpha=35, max_err=10e-1)

import numpy as np
import cv2
import matplotlib.pyplot as plt
from os import path as ph
from utils.tools import calculate_derivatives, get_channel_permutations


def get_windows_result(window):
    u, v = 0, 0

    ix, iy, it = window[:]
    xx_sum = np.sum(ix**2)
    xy_sum = np.sum(ix * iy)
    yy_sum = np.sum(iy**2)
    xt_sum = np.sum(ix * it)
    yt_sum = np.sum(iy * it)

    A = np.vstack((ix, iy)).T
    ATA = np.array([[xx_sum, xy_sum], [xy_sum, yy_sum]])
    ATb = np.array([-xt_sum, -yt_sum])

    u, v = 0, 0
    if np.linalg.det(ATA) != 0:
        l = np.linalg.eigvals(ATA)
        min_l = np.min(l)
        max_l = np.max(l)
        if max_l / min_l < 10:
            u, v = np.linalg.inv(ATA) @ ATb

    return u, v


def lucas_kanade(img1, img2, N):
    img1 = (img1 / 255).astype(np.float32)
    img2 = (img2 / 255).astype(np.float32)
    i_x, i_y, i_t = calculate_derivatives(img1, img2)

    print(i_t.shape)
    windows = [
        (
            x,
            y,
            i_x[x : x + N, y : y + N],
            i_y[x : x + N, y : y + N],
            i_t[x : x + N, y : y + N],
        )
        for x in range(0, img1.shape[0] - N, N)
        for y in range(0, img1.shape[1] - N, N)
    ]

    U, V = [], []
    X, Y = [], []

    for window in windows:
        y, x = window[:2]
        X.append(x)
        Y.append(y)
        u, v = get_windows_result(window[2:])
        U.append(u)
        V.append(v)
    return U, V, X, Y


if __name__ == "__main__":
    from glob import glob
    from sys import argv

    images_dir = ph.join("./", argv[1])
    images = glob(ph.join(images_dir, "*"))
    img1, img2 = cv2.imread(images[0]), cv2.imread(images[1])

    images1 = get_channel_permutations(img1)
    images2 = get_channel_permutations(img2)

    # fig = plt.figure(figsize=(16, 9), dpi=80)
    # u, v, X, Y = lucas_kanade(img1, img2, 15)
    # plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    # plt.quiver(X, Y, u, v, color="r", scale=2, pivot="mid")
    # plt.show()

    for (key1, image1), (key2, image2) in zip(images1.items(), images2.items()):
        print(key1, key2)

        fig = plt.figure()
        u, v, X, Y = lucas_kanade(image1, image2, 15)
        plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        plt.quiver(X, Y, u, v, color="r", scale=2, pivot="mid")
        plt.savefig(
            f"plots/LK_{key1}_images_{ph.basename(images[0]).split('.')[0]}.png",
            dpi=100
        )

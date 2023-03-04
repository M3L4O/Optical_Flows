import numpy as np
import cv2
import matplotlib.pyplot as plt
from os import path as ph
from utils.tools import calculate_derivatives, get_channel_permutations


def get_windows_result(window, size_thresh=400, distance_thresh=10):
    u, v = 0, 0

    ix, iy, it = window[:]
    xx_sum = np.sum(ix**2)
    xy_sum = np.sum(ix * iy)
    yy_sum = np.sum(iy**2)
    xt_sum = np.sum(ix * it)
    yt_sum = np.sum(iy * it)

    ATA = np.array([[xx_sum, xy_sum], [xy_sum, yy_sum]])
    ATb = np.array([-xt_sum, -yt_sum])

    u, v = 0, 0
    if np.linalg.det(ATA) != 0:
        l = np.linalg.eigvals(ATA)
        min_l = np.min(l)
        max_l = np.max(l)
        if (min_l >= size_thresh) and ((max_l / min_l) < distance_thresh):
            u, v = np.linalg.inv(ATA) @ ATb

    return u, v


def lucas_kanade(img1, img2, N):
    img1 = (img1 / 255).astype(np.float32)
    img2 = (img2 / 255).astype(np.float32)

    i_x, i_y, i_t = calculate_derivatives(img1, img2)

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
    magnitudes = []
    angles = []

    for window in windows:
        y, x = window[:2]
        X.append(x)
        Y.append(y)
        u, v = get_windows_result(window[2:])
        U.append(u)
        V.append(v)
        magnitude, angle = 0, 0
        if 0 not in (u, v):
            magnitude = (u**2 + v**2) ** 0.5
            angle = np.arctan(u / v)
        magnitudes.append(magnitude)
        angles.append(angle)

    magnitude_optical_flow = np.zeros(
        (img1.shape[0] // window_size, img1.shape[1] // window_size)
    )
    angle_optical_flow = magnitude_optical_flow.copy()

    for x, y, mag, angle in zip(X, Y, magnitudes, angles):
        y = y // window_size
        x = x // window_size
        magnitude_optical_flow[y, x] = mag
        angle_optical_flow[y, x] = angle

    magnitude_optical_flow_thresh = np.where(
        magnitude_optical_flow
        < (
            np.max(magnitude_optical_flow)
            / (
                np.float32(
                    np.format_float_scientific(np.mean(magnitude_optical_flow)).split(
                        "e"
                    )[0]
                )
                + 2
            )
        ),
        0,
        1,
    ).astype(np.int32)

    magnitude_optical_flow_thresh = cv2.resize(
        magnitude_optical_flow_thresh,
        (img1.shape[1], img1.shape[0]),
        interpolation=cv2.INTER_LINEAR_EXACT,
    )
    angle_optical_flow = cv2.resize(
        angle_optical_flow,
        (img1.shape[1], img1.shape[0]),
        interpolation=cv2.INTER_LINEAR_EXACT,
    )

    # plt.imshow(img1)
    # plt.pcolormesh(magnitude_optical_flow_thresh, cmap='gist_yarg_r',alpha=0.6)
    # # im_ratio = magnitude_optical_flow.shape[0] / magnitude_optical_flow.shape[1]
    # # cbar = plt.colorbar(im, fraction=0.046*im_ratio, pad=0.04)
    # # cbar.set_label("Intensidade do angulo")
    # fig.savefig("./plots/LK_mag_seg.png")
    # plt.show()

    return np.array(U), np.array(V), X, Y


if __name__ == "__main__":
    from glob import glob
    from sys import argv

    images_dir = ph.join("./", argv[1])
    images = sorted(glob(ph.join(images_dir, "*")))
    img1, img2 = cv2.imread(images[30]), cv2.imread(images[31])

    window_size = 10
    images1 = get_channel_permutations(img1)
    images2 = get_channel_permutations(img2)

    background = np.full(img1.shape, 255)
    # fig = plt.figure()
    # u, v, X, Y = lucas_kanade(img1, img2, window_size)
    # plt.imshow(img1)
    # plt.quiver(X, Y, -u, -v, color="r", scale=1, pivot="mid")
    # plt.show()

    for (key1, image1), (key2, image2) in zip(images1.items(), images2.items()):
        print(key1, key2)

        fig = plt.figure()
        u, v, X, Y = lucas_kanade(image1, image2, window_size)
        plt.imshow(background)
        plt.quiver(X, Y, -u, -v, color="black", scale=1, pivot="mid")
        plt.savefig(
            f"plots/LK_{key1}_images_{ph.basename(images[0]).split('.')[0]}.png",
            dpi=100,
        )
        plt.show()

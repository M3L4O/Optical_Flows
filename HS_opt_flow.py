import cv2
import numpy as np
from pyoptflow import HornSchunck
from os import path as ph
from utils.tools import calculate_derivatives, get_channel_permutations
from matplotlib import pyplot as plt


def generate_xy_indices_list(w, h):
    X = np.tile(np.arange(w), (1, h))[0]
    Y = np.repeat(np.arange(h), w)

    return X, Y


def apply_horn_schunck(frame_1, frame_2, alpha=4, niter=100, generate_xy=True):
    U, V = HornSchunck(frame_1, frame_2, alpha=alpha, Niter=niter)

    h, w = frame_1.shape

    U = U.flatten()
    V = V.flatten()

    X, Y = generate_xy_indices_list(w, h) if generate_xy else ([], [])

    return X, Y, U, V


def apply_horn_shunck_multichannel(frame_1, frame_2, alpha=75, niter=100):
    prev_channels = cv2.split(frame_1)
    next_channels = cv2.split(frame_2)

    h, w, _ = frame_1.shape

    X, Y = generate_xy_indices_list(w, h)
    U = np.zeros_like(X, np.float32)
    V = np.zeros_like(U)
    mag = np.zeros_like(U)

    for prev, next in zip(prev_channels, next_channels):
        _, _, u, v = apply_horn_schunck(prev, next, alpha, niter, False)
        new_mag = cv2.magnitude(u, v).flatten()
        indices = np.where(new_mag > mag)[0]

        if indices.size > 0:
            mag[indices] = new_mag[indices]
            U[indices] = u[indices]
            V[indices] = v[indices]

    return X, Y, U, V


def clean_results(image, U, V, N):
    U = np.reshape(U, (image.shape[1], image.shape[0]))
    V = np.reshape(V, (image.shape[1], image.shape[0]))
    _X, _Y, _U, _V = [], [], [], []
    for x in range(0, image.shape[1] - N, N):
        for y in range(0, image.shape[0] - N, N):
            _X.append(x)
            _Y.append(y)
            _U.append(U[x, y])
            _V.append(V[x, y])
    return _X, _Y, _U, _V


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
    X, Y, u, v = apply_horn_shunck_multichannel(img1, img2, window_size)
    X, Y, u, v = clean_results(img1, u, v, window_size)
    plt.imshow(background)
    plt.quiver(X, Y, u, v, color="black", scale=1000, pivot="mid")
    plt.show()

    # for (key1, image1), (key2, image2) in zip(images1.items(), images2.items()):
    #     print(key1, key2)
    #
    #     fig = plt.figure()
    #     u, v, X, Y = apply_horn_shunck_multichannel(image1, image2)
    #     plt.imshow(background)
    #     plt.quiver(X, Y, -u, -v, color="black", scale=1, pivot="mid")
    #     plt.savefig(
    #         f"plots/LK_{key1}_images_{ph.basename(images[0]).split('.')[0]}.png",
    #         dpi=100,
    #     )
    #     plt.show()

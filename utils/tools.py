import numpy as np
import cv2
from matplotlib import pyplot as plt


def extract_channels(image: np.ndarray, ch: str) -> list[np.ndarray] | np.ndarray:
    if image.ndim < 3:
        return image
    else:
        return cv2.split(image)


def get_channel_permutations(img):
    channels = {n: c for n, c in zip(("r", "g", "b"), np.split(img, img.shape[-1], 2))}
    channels.update(
        {
            n: c
            for n, c in zip(
                ["rg", "rb", "bg"],
                map(
                    np.dstack,
                    (
                        (channels["r"], channels["g"]),
                        (channels["r"], channels["b"]),
                        (channels["b"], channels["g"]),
                    ),
                ),
            )
        }
    )
    channels["rgb"] = img
    channels["gray"] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return channels


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


def get_derivatives(img1, img2):
    # removing noise
    img1 = cv2.GaussianBlur(img1, (5, 5), 0)
    img2 = cv2.GaussianBlur(img2, (5, 5), 0)

    x_kernel = np.array([[-1, 1], [-1, 1]]) * 1 / 3
    y_kernel = np.array([[-1, -1], [1, 1]]) * 1 / 3
    t_kernel = np.ones((2, 2))

    fx = (
        1
        / 2
        * (
            cv2.filter2D(img1, -1, cv2.flip(x_kernel, -1))
            + cv2.filter2D(img2, -1, cv2.flip(x_kernel, -1))
        )
    )
    fy = (
        1
        / 2
        * (
            cv2.filter2D(img1, -1, cv2.flip(y_kernel, -1))
            + cv2.filter2D(img2, -1, cv2.flip(y_kernel, -1))
        )
    )
    ft = cv2.filter2D(img1, -1, cv2.flip(-t_kernel, -1)) + cv2.filter2D(
        img2, -1, cv2.flip(t_kernel, -1)
    )

    return [fx, fy, ft]


def get_magnitude(u, v):
    scale = 2
    sum = 0.0
    counter = 0.0

    for i in range(0, u.shape[0], 8):
        for j in range(0, u.shape[1], 8):
            counter += 1
            dy = v[i, j] * scale
            dx = u[i, j] * scale
            magnitude = (dx**2 + dy**2) ** 0.5
            sum += magnitude

    mag_avg = sum / counter

    return mag_avg


def draw_quiver(u, v, beforeImg, algorthm: str):
    scale = 4
    ax = plt.figure().gca()
    ax.imshow(beforeImg, cmap="gray")

    magnitudeAvg = get_magnitude(u, v)

    for i in range(0, u.shape[0], 8):
        for j in range(0, u.shape[1], 8):
            dy = v[i, j] * scale
            dx = u[i, j] * scale
            magnitude = (dx**2 + dy**2) ** 0.5

            if magnitude > magnitudeAvg:
                ax.arrow(j, i, dx, dy, color="red")

    plt.draw()
    plt.show()


def show_image(name, image):
    if image is None:
        return

    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    img = cv2.imread("/home/melao/Imagens/Wallpapers/flatppuccin_4k_macchiato.png")
    img = cv2.resize(img, (300, 300))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img.ndim >= 3:
        B, G, R = extract_channels(img, "rgb")

        channels = np.concatenate((R, G, B), axis=1)
        cv2.imshow("Channels", channels)
        cv2.waitKey()

    else:
        gray = extract_channels(img, "gray")

        cv2.imshow("Gray", gray)
        cv2.waitKey()

    cv2.destroyAllWindows

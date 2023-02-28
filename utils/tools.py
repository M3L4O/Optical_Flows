import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def extract_channels(image: np.ndarray, ch: str) -> list[np.ndarray] | np.ndarray:
    if image.ndim < 3:
        return image
    else:
        return cv.split(image)


def get_derivatives(img1, img2):
    # removing noise
    img1 = cv.GaussianBlur(img1, (5, 5), 0)
    img2 = cv.GaussianBlur(img2, (5, 5), 0)

    x_kernel = np.array([[-1, 1], [-1, 1]]) * 1 / 3
    y_kernel = np.array([[-1, -1], [1, 1]]) * 1 / 3
    t_kernel = np.ones((2, 2))

    fx = (
        1
        / 2
        * (
            cv.filter2D(img1, -1, cv.flip(x_kernel, -1))
            + cv.filter2D(img2, -1, cv.flip(x_kernel, -1))
        )
    )
    fy = (
        1
        / 2
        * (
            cv.filter2D(img1, -1, cv.flip(y_kernel, -1))
            + cv.filter2D(img2, -1, cv.flip(y_kernel, -1))
        )
    )
    ft = cv.filter2D(img1, -1, cv.flip(-t_kernel, -1)) + cv.filter2D(
        img2, -1, cv.flip(t_kernel, -1)
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


def draw_quiver(u, v, beforeImg):
    scale = 4
    ax = plt.figure().gca()
    ax.imshow(beforeImg, cmap="gray")

    magnitudeAvg = get_magnitude(u, v)

    for i in range(0, u.shape[0], 8):
        for j in range(0, u.shape[1], 8):
            dy = v[i, j] * scale
            dx = u[i, j] * scale
            magnitude = (dx**2 + dy**2) ** 0.5
            # draw only significant changes
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
    img = cv.imread("/home/melao/Imagens/Wallpapers/flatppuccin_4k_macchiato.png")
    img = cv.resize(img, (300, 300))
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    if img.ndim >= 3:
        B, G, R = extract_channels(img, "rgb")

        channels = np.concatenate((R, G, B), axis=1)
        cv.imshow("Channels", channels)
        cv.waitKey()

    else:
        gray = extract_channels(img, "gray")

        cv.imshow("Gray", gray)
        cv.waitKey()

    cv.destroyAllWindows

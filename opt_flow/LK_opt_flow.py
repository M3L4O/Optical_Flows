import cv2 as cv
import numpy as np
from itertools import product
from os import path as ph
from utils.config import lk_params, feature_params, extract_channels, colors


def draw_arrows(image: np.ndarray, current_points, next_points):
    for current_point, next_point in zip(current_points, next_points):
        current_point = tuple(map(int, current_point.ravel()))
        next_point = tuple(map(int, next_point.ravel()))

        image = cv.arrowedLine(
            image, current_point, next_point, colors["blue"], 1, 8, 0, 0.4
        )

    return image


def get_LK(current_frame: np.ndarray, next_frame: np.ndarray, space: str):
    current_channels = extract_channels(current_frame, space)
    next_channels = extract_channels(next_frame, space)

    n_channels = current_frame.ndim - 2

    for comb, (current_channel, next_channel) in enumerate(
        zip(current_channels, next_channels)
    ):
        current_points = cv.goodFeaturesToTrack(current_channel, **feature_params)
        next_points = cv.goodFeaturesToTrack(next_channel, **feature_params)

        next_points, st, err = cv.calcOpticalFlowPyrLK(
            current_frame,
            next_frame,
            current_points,
            None,
            **lk_params,
        )
        if next_points is None:
            next_points = next_points[st == 1]
            current_points = current_points[st == 1]

        result_image = draw_arrows(current_frame.copy(), current_points, next_points)
        diff = (
            np.abs(
                cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY)
                - cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)
            )
        )
        diff = np.expand_dims(diff, axis=-1)
        diff = np.concatenate([diff, diff, diff], axis=-1)
        print(diff.shape)
        cat_image = np.concatenate([result_image, next_frame, diff], axis=1)
        cv.imwrite(f"{comb}_LK_result.png", cat_image)


if __name__ == "__main__":
    current_frame = cv.imread("frames/original_frame_109.jpg")
    next_frame = cv.imread("frames/original_frame_111.jpg")

    get_LK(current_frame, next_frame, "bgr")

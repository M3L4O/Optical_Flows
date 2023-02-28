import cv2 as cv
import numpy as np

feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.1,
                       minDistance = 25,
                       blockSize = 7 )

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 1,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

colors = {
    "blue": (255, 0, 0),
    "green": (0, 255, 0),
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
    "magenta": (255, 0, 255),
    "cyan": (255, 255, 0),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "gray": (125, 125, 125),
    "rand": np.random.randint(0, high=256, size=(3,)).tolist(),
    "dark_gray": (50, 50, 50),
    "light_gray": (220, 220, 220),
}




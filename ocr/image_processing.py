import cv2
import math
import numpy as np

from deskew import determine_skew
from typing import *
from .custom_types import *
from .constants import *


def canny(image: CV2Image) -> CV2Image:
    return cv2.Canny(image, 100, 200)


def grayscale(image: CV2Image) -> CV2Image:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def thresholding(image: CV2Image) -> CV2Image:
    return cv2.threshold(image, 100, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def median_blur(image: CV2Image) -> CV2Image:
    return cv2.medianBlur(image, 1)


def rotate(image: CV2Image, angle: float,
           background: Union[int, Tuple[int, int, int]]) -> CV2Image:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(
        np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(
        np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image,
                          rot_mat, (int(round(height)), int(round(width))),
                          borderValue=background)


def deskew(image: CV2Image) -> CV2Image:
    angle = determine_skew(image)
    return rotate(image, angle, (0, 0, 0))


def gaussian_blur(image: CV2Image) -> CV2Image:
    return cv2.GaussianBlur(image, (3, 3), 0)


def opening(image: CV2Image) -> CV2Image:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)


def invert(image: CV2Image) -> CV2Image:
    return 255 - image


def supersample2x(image: CV2Image) -> CV2Image:
    height = int(image.shape[0] * 2.0)
    width = int(image.shape[1] * 2.0)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)


def delete_horizontal(image: CV2Image) -> CV2Image:
    kernel = np.ones((1, 5), np.uint8)
    clean_lines = cv2.erode(image, kernel, iterations=6)
    clean_lines = cv2.dilate(clean_lines, kernel, iterations=6)
    return image - clean_lines


def delete_vertical(image: CV2Image) -> CV2Image:
    kernel = np.ones((5, 1), np.uint8)
    clean_lines = cv2.erode(image, kernel, iterations=6)
    clean_lines = cv2.dilate(clean_lines, kernel, iterations=6)
    return image - clean_lines


def erode(image: CV2Image) -> CV2Image:
    kernel = np.ones((1, 1), np.uint8)
    return cv2.erode(image, kernel, iterations=NUM_ERODE_ITERATIONS)

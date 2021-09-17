import cv2
from pathlib import Path

from .custom_types import *
from .constants import *


def open_image(path: Path) -> CV2Image:
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def select_bounding_boxes(data: OCRData,
                          threshold: float = 20.0,
                          min_size: int = 2):
    n_boxes = len(data['text'])
    bounding_boxes = []
    for i in range(n_boxes):
        if float(data['conf'][i]) > threshold and len(
                data['text'][i]) >= min_size:
            bounding_boxes.append((data['top'][i], data['left'][i],
                                   data['top'][i] + data['height'][i],
                                   data['left'][i] + data['width'][i]))
    return bounding_boxes


def crop_image_with_text(image: CV2Image,
                         bounding_boxes: List[Tuple[int, int, int, int]],
                         margin: int = 10):
    h, w = image.shape
    top_with_margin = min(
        min(bounding_boxes, key=lambda x: x[0])[0] - margin, 0)
    left_with_margin = min(
        min(bounding_boxes, key=lambda x: x[1])[1] - margin, 0)
    bottom_with_margin = max(
        max(bounding_boxes, key=lambda x: x[2])[2] + margin, h)
    right_with_margin = max(
        max(bounding_boxes, key=lambda x: x[3])[3] + margin, w)
    return image[top_with_margin:bottom_with_margin,
                 left_with_margin:right_with_margin]


def apply_pipeline(image: CV2Image,
                   pipeline: List[PipelineFunction]) -> CV2Image:
    img_tmp: CV2Image = image.copy()
    for f in pipeline:
        img_tmp = f(img_tmp)
    return img_tmp


def get_fraction_of_lines(image: CV2Image,
                          pipeline: List[PipelineFunction]) -> float:
    p_img = apply_pipeline(image, pipeline)
    h, w = p_img.shape
    pixels = 1 - (np.sum(p_img, axis=1) / (255 * w))
    return float(np.sum(pixels >= 0.35)) / h

from typing import Union, Callable, List
from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import downscale_local_mean
from skimage.util import invert
from skimage.filters import threshold_yen
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb, rgb2gray
import cv2
import matplotlib.pyplot as plt

from src.tools.line_finder import remove_lines
from src.tools.prepare_page import clip_numbers


class DigitFilter:
    def __init__(
            self,
            min_area=0,
            max_area=np.inf,
            min_width=0,
            max_width=np.inf,
            min_height=0,
            max_height=np.inf,
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height

    def __call__(
            self,
            region,  # skimage.measure._RegionProperties object
    ):
        minr, minc, maxr, maxc = region.bbox
        w = maxc - minc
        h = maxr - minr
        return all([
            self.min_area <= region.area <= self.max_area,
            self.min_width <= w <= self.max_width,
            self.min_height <= h <= self.max_height,
        ])


def extract_digits(
        image: np.ndarray,
        remove_lines_threshold: int,
        threshold_method: Union[Callable, None],
        digit_filter: Callable,
        do_closing: bool = True,
) -> List[List]:

    if threshold_method is None:
        threshold_method = threshold_yen

    thresh = threshold_method(image)
    bw = image > thresh
    bw = remove_lines(bw, threshold=remove_lines_threshold)

    if do_closing:
        bw = closing(bw)

    label_image = label(bw)
    label_image = clear_border(label_image)

    digits = [r for r in regionprops(label_image, cache=True) if digit_filter(r)]

    return digits


def digits_to_table(
        digits,
        ground_truth_file,
        output_file,
        digit_shape,
):
    for d in digits:
        digit = downscale_local_mean(d, digit_shape, cval=0)
        digit = digit.ravel()
        # todo


if __name__ == '__main__':
    os.chdir('../../data')
    img_file = '5104.jpg'
    image = clip_numbers(
        img_file,
        'plot_header.jpg',
        'taxpayer_header.jpg',
        col_height=2350,
        plot_col_width=82,
        pop_col_width=1375,
    )
    h, w = image.shape
    image = invert(image)
    plt.imshow(image)
    plt.show()

    digit_filter = DigitFilter(
        min_area=100,
        max_area=500,
        min_width=20,
        min_height=30,
    )

    digits = extract_digits(image, 600, None, digit_filter, do_closing=True)
    digits = [[dig[0], downscale_local_mean(dig[1], (50, 50))] for dig in digits]

    # fig, axes = plt.subplots(10, 10, figsize=(10, 6))
    # ax = axes.ravel()
    #
    # digit_list = list(digits.values())
    # for i, dig in enumerate(digit_list[200:300]):
    #     ax[i].imshow(dig)
    # plt.show()

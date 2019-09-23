from typing import Union, Callable, Dict, Tuple
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import invert
from skimage.filters import threshold_yen
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb, rgb2gray

from src.tools.line_finder import remove_lines


def resize_digit(digit, shape):
    r, c = digit.shape
    r0, c0 = shape
    dr = r - r0
    dc = c - c0
    pad_top, pad_bottom = divmod(dr, 2)
    pad_bottom += pad_top
    pad_left, pad_right = divmod(dc, 2)
    pad_right += pad_left

    digit = np.pad(
        digit,
        pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant',
        constant_values=0,
    )
    digit = digit[:r0, :c0]

    return digit


def extract_digits(
        image: np.ndarray,
        remove_lines_threshold: int,
        threshold_method: Union[Callable, None],
        min_area: int,
        do_closing: bool = True,
) -> Dict[Tuple[int], np.ndarray]:
    if threshold_method is None:
        threshold_method = threshold_yen

    thresh = threshold_method(image)
    bw = image > thresh
    bw = remove_lines(bw, threshold=remove_lines_threshold)

    if do_closing:
        bw = closing(bw)

    label_image = label(bw)
    label_image = clear_border(label_image)

    digits = {r.bbox: r.image for r in regionprops(label_image, cache=True) if min_area <= r.area}

    return digits


def save_digits(
        digits,
        img_name,
):
    json_name = f'{img_name}_truth.json'
    outf = Path(json_name)
    characters = dict()
    for digit in digits:
        plt.imshow(digit.image)
        # todo input
        char = None
        characters[digit.centroid] = char
    data = {digit.centroid: (characters[digit.centroid], digit.image) for digit in digits}
    outf.write_text(json.dumps(data))


if __name__ == '__main__':
    image = imread('../../data/test1.jpg')
    h, w, _ = image.shape
    image = invert(rgb2gray(image))# [0:h // 2, 0:w // 2]

    digits = extract_digits(image, 700, None, 150, do_closing=True)

    fig, axes = plt.subplots(10, 10, figsize=(10, 6))
    ax = axes.ravel()

    digit_list = [d for d in digits.values()]
    for i, dig in enumerate(digit_list[100:200]):
        ax[i].imshow(dig)
    plt.show()

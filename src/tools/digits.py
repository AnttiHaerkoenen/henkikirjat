from typing import Union, Callable, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.io import imread
from skimage.util import invert
from skimage.filters import threshold_yen
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb, rgb2gray

from src.tools.line_finder import remove_lines


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

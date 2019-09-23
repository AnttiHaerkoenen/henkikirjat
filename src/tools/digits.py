from typing import Union, Callable, List
from pathlib import Path
import json

import numpy as np
from skimage.io import imread
from skimage.util import invert
from skimage.filters import threshold_yen
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb, rgb2gray
import cv2

from src.tools.line_finder import remove_lines


def resize_digit(
        digit,
        shape,
        pad_value=0,
):
    r, c = digit.shape
    r0, c0 = shape
    dr = r0 - r
    dc = c0 - c

    if 0 <= dr:
        pad_top, mod = divmod(dr, 2)
        pad_bottom = pad_top + mod
    else:
        pad_top = pad_bottom = 0

    if 0 <= dc:
        pad_left, mod = divmod(dc, 2)
        pad_right = pad_left + mod
    else:
        pad_left = pad_right = 0

    digit = np.pad(
        digit,
        pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant',
        constant_values=pad_value,
    )
    digit = digit[:r0, :c0]

    return digit


def extract_digits(
        image: np.ndarray,
        remove_lines_threshold: int,
        threshold_method: Union[Callable, None],
        min_area: int,
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

    digits = [[r.bbox, r.image] for r in regionprops(label_image, cache=True) if min_area <= r.area]

    return digits


def save_digits(
        digits,
        img_file,
        json_file=None,
):
    if json_file is None:
        json_file = '.'.join(img_file.split('.')[:-1])
        json_file = f'{json_file}_truth.json'
    outf = Path(json_file)
    img = cv2.imread(img_file)

    i = 0
    characters = [None for _ in digits]
    while True:
        if not 0 <= i < len(digits):
            break

        bbox, _ = digits[i]
        win_name = f'Digit {i} at {bbox[0]}, {bbox[1]}'
        minr, minc, maxr, maxc = bbox
        box = img[minr:maxr, minc:maxc]

        cv2.imshow(win_name, box)
        key = cv2.waitKeyEx(0)
        if key == 97:  # a
            i -= 1
        elif key == 100:  # d:
            i += 1
        elif 48 <= key <= 57:  # digits
            characters[i] = str(key - 48)
            i += 1
        elif key in (27, 98):  # exit, b
            break
        cv2.destroyAllWindows()

    data = [[digit[0], characters[i]] for i, digit in enumerate(digits)]
    outf.write_text(json.dumps(data, indent=True))


if __name__ == '__main__':
    image = imread('../../data/test1.jpg')
    h, w, _ = image.shape
    image = invert(rgb2gray(image))

    digits = extract_digits(image, 700, None, 150, do_closing=True)
    digits = [[dig[0], resize_digit(dig[1], (50, 50))] for dig in digits]

    save_digits(digits[100:], '../../data/test1.jpg')

    # fig, axes = plt.subplots(10, 10, figsize=(10, 6))
    # ax = axes.ravel()
    #
    # digit_list = list(digits.values())
    # for i, dig in enumerate(digit_list[200:300]):
    #     ax[i].imshow(dig)
    # plt.show()

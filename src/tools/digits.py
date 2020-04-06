from typing import Union, Callable, List
from pathlib import Path
import json
import os
import glob

import numpy as np
import pandas as pd
from skimage.io import imread, imshow
from skimage.transform import downscale_local_mean, resize
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
        digit_shape,
        ground_truth_file=None,
):
    images = []
    locs = []
    for d in digits:
        digit = resize(
            d.image,
            digit_shape,
            mode='constant',
            cval=0,
        )
        digit = digit.ravel()
        images.append(digit)
        locs.append(list(d.bbox))

    labels = []
    if ground_truth_file is not None:
        ground_truth_fp = Path(ground_truth_file)
        true_digits = json.loads(ground_truth_fp.read_text())
        for d in digits:
            for coords, label in true_digits:
                if coords == list(d.bbox):
                    labels.append(label)
                    break
            else:
                labels.append(None)

    images = pd.DataFrame(np.vstack(images) > 0.5)
    locs = pd.DataFrame(np.array(locs), columns='min_row min_col max_row max_col'.split())

    if ground_truth_file is not None:
        labels = pd.Series(labels, name='label')
    else:
        labels = None

    return labels, locs, images


def view_digit(data: pd.Series, shape):
    image = data.values[1:].reshape(shape).astype(bool)
    fig, ax = plt.subplots(1, 1)
    ax.imshow(image)
    ax.set_title(f"Label: {int(data[0])}")
    plt.show()


def save_training_data(
        data_dir: str,
        output_file: str,
        digit_shape: tuple,
        digit_filter: DigitFilter,
        **clip_numbers_param
):
    os.chdir(data_dir)
    outf = Path(output_file)

    data = []
    page_numbers = sorted([
        file.split('/')[-1].split('_')[0]
        for file in glob.iglob(f"./labels/*_truth.json")
    ])
    for i in page_numbers:
        img_file = f'{i}.jpg'
        image = clip_numbers(img_file, **clip_numbers_param)
        image = invert(image)

        digits = extract_digits(image, 600, None, digit_filter, do_closing=True)
        labels, locs, images = digits_to_table(
            digits,
            digit_shape,
            ground_truth_file=f'./labels/{i}_truth.json',
        )
        data.append(pd.concat([labels, images], axis=1))
        print(i)
    data = pd.concat(data, axis=0, ignore_index=True)
    data.to_csv(str(outf))


if __name__ == '__main__':
    data_dir = '../../data/train/1900'
    output_file = 'labeled_1900.csv'
    digit_filter = DigitFilter(
        min_area=50,
        max_area=500,
        min_width=20,
        min_height=20,
        max_width=100,
        max_height=100,
    )
    save_training_data(
        data_dir=data_dir,
        output_file=output_file,
        digit_shape=(40, 40),
        digit_filter=digit_filter,
        col_height=2350,
        plot_col_width=82,
        pop_col_width=1375,
        plot_header_file='../../plot_header.jpg',
        taxpayer_header_file='../../taxpayer_header.jpg',
    )
    # os.chdir('../../data')
    # data_file = 'labeled.csv'
    # data_fp = Path(data_file)
    # data = []
    # for i in {file.split('/')[-1].split('_')[0] for file in glob.iglob('./train/*.json')}:
    #     img_file = f'train/{i}.jpg'
    #     image = clip_numbers(
    #         img_file,
    #         'plot_header.jpg',
    #         'taxpayer_header.jpg',
    #         col_height=2350,
    #         plot_col_width=82,
    #         pop_col_width=1375,
    #     )
    #     h, w = image.shape
    #     image = invert(image)
    #
    #     digit_filter = DigitFilter(
    #         min_area=100,
    #         max_area=500,
    #         min_width=20,
    #         min_height=30,
    #     )
    #
    #     digits = extract_digits(image, 600, None, digit_filter, do_closing=True)
    #     labels, locs, images = digits_to_table(
    #         digits,
    #         (50, 50),
    #         ground_truth_file=f'train/{i}_truth.json',
    #     )
    #     data.append(pd.concat([labels, images], axis=1))
    #     print(i)
    # data = pd.concat(data, axis=0, ignore_index=True)
    #
    # data.to_csv(data_fp)

    # fig, axes = plt.subplots(10, 10, figsize=(10, 6))
    # ax = axes.ravel()
    #
    # digit_list = list(digits.values())
    # for i, dig in enumerate(digit_list[200:300]):
    #     ax[i].imshow(dig)
    # plt.show()

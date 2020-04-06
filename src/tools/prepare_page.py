import os
import math
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pdftabextract import imgproc
from pdftabextract.common import ROTATION, DIRECTION_HORIZONTAL
import cv2
from skimage.feature import match_template
from skimage.io import imread

from src.tools.split_page import split_page
# from src.template_matching.parameters import DetectLinesParam


def crop_page(
        data_dir: str,
        img_file: str,
        output_file: str = None,
        top: int = 0,
        bottom: int = 0,
        left: int = 0,
        right: int = 0,
):
    os.chdir(data_dir)
    if not output_file:
        output_file, extension = img_file.split('.')
        output_file = f'{output_file}_cropped.{extension}'
    img: Image = Image.open(img_file)
    x_min, y_min, x_max, y_max = img.getbbox()
    box = x_min + left, y_min + top, x_max - right, y_max - bottom
    img.crop(box).save(output_file)


def cut_names(
        data_dir: str,
        img_file: str,
        output_file: str = None,
        left: int = 0,
        right: int = 0,
):
    os.chdir(data_dir)
    if not output_file:
        output_file, extension = img_file.split('.')
        output_file = f'{output_file}_anon.{extension}'
    img = cv2.imread(img_file)
    a, _, c = np.hsplit(img, [left, right])
    img = np.hstack([a, c])
    cv2.imwrite(output_file, img)

#
# def straighten_page(
#         data_dir: str,
#         img_file: str,
#         output_file: str = None,
#         split_position: float = 0.5,
#         **kwargs
# ):
#     tmp_dir = r'tmp'
#     tmp_files = r'./tmp/left.jpg', r'./tmp/right.jpg'
#     os.chdir(data_dir)
#     if not os.path.isdir(tmp_dir):
#         os.mkdir(tmp_dir)
#     if not output_file:
#         output_file, extension = img_file.split('.')
#         output_file = f'{output_file}_straight.{extension}'
#
#     split_page(
#         img_file,
#         data_dir,
#         split_position,
#         tmp_files,
#     )
#     orig_img: Image = Image.open(img_file)
#     x_min, y_min, x_max, y_max = orig_img.getbbox()
#     split_x = int((x_max - x_min) * split_position)
#     boxes = [
#         (x_min, y_min, split_x, y_max),
#         (split_x, y_min, x_max, y_max),
#     ]
#     for img_, box in zip(tmp_files, boxes):
#         img = imgproc.ImageProc(img_)
#         parameters = DetectLinesParam(img, **kwargs)
#         img.detect_lines(**parameters.parameters)
#         rot_or_skew_type, rot_radians = img.find_rotation_or_skew(
#             only_direction=DIRECTION_HORIZONTAL,
#             rot_thresh=math.radians(1),
#             rot_same_dir_thresh=math.radians(10),
#         )
#         rot_radians = rot_radians if rot_or_skew_type == ROTATION else 0
#         region = orig_img.crop(box)
#         region = region.rotate(math.degrees(rot_radians))
#         orig_img.paste(region, box)
#     orig_img.save(output_file)


def prepare_pages(
        img_files: Sequence[str],
        data_dir: str,
        output_file: str,
):
    for file in img_files:
        crop_page(
            img_file=file,
            data_dir=data_dir,
            output_file='temp.jpg',
            top=300,
            bottom=100,
            left=450,
            right=250,
        )
        straighten_page(
            img_file='temp.jpg',
            data_dir=data_dir,
            output_file='temp.jpg',
            split_position=0.5,
        )
        cut_names(
            img_file='temp.jpg',
            data_dir=data_dir,
            output_file=output_file,
            left=100,
            right=2000,
        )


def find_columns(img_file, header_file):
    image = imread(img_file)
    header = imread(header_file)
    h, w = header.shape
    result = match_template(image, header, pad_input=True)
    r, c = np.unravel_index(result.argmax(), image.shape)
    return r, c, h, w


def clip_numbers(
        img_file,
        plot_header_file,
        taxpayer_header_file,
        *,
        col_height,
        plot_col_width,
        pop_col_width,
):
    r0, c0, h0, w0 = find_columns(img_file, plot_header_file)
    r1, c1, h1, w1 = find_columns(img_file, taxpayer_header_file)
    image = imread(img_file)
    plots_start_r = r0 + h0 // 2
    plots_start_c = c0 - w0 // 2
    pops_start_r = r1 + h1 // 2
    pops_start_c = c1 - w1 // 2
    plots = image[plots_start_r: plots_start_r + col_height,
            plots_start_c: plots_start_c + plot_col_width]
    pops = image[pops_start_r: pops_start_r + col_height,
           pops_start_c: pops_start_c + pop_col_width]
    return np.hstack([plots, pops])


if __name__ == '__main__':
    os.chdir('../../data')
    clipped = clip_numbers(
        '5104.jpg',
        'plot_header.jpg',
        'taxpayer_header.jpg',
        col_height=2350,
        plot_col_width=82,
        pop_col_width=1375,
    )
    fig = plt.imshow(clipped)
    plt.show()
    # crop_page(
    #     img_file='5104.jpg',
    #     output_file='test1.jpg',
    #     data_dir='../data',
    #     top=300,
    #     bottom=100,
    #     left=450,
    #     right=250,
    # )
    # straighten_page(
    #     img_file='test1.jpg',
    #     data_dir='../data',
    #     output_file='test1.jpg',
    #     split_position=0.5,
    # )
    # cut_names(
    #     img_file='test1.jpg',
    #     data_dir='../data',
    #     output_file='test1.jpg',
    #     left=100,
    #     right=2000,
    # )

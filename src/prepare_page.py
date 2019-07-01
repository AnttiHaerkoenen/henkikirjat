import os
import math

import numpy as np
from PIL import Image
from pdftabextract import imgproc
from pdftabextract.common import ROTATION, DIRECTION_HORIZONTAL, DIRECTION_VERTICAL

import split_page
from parameters import DetectLinesParam


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


def straighten_page(
        data_dir: str,
        img_file: str,
        output_file: str = None,
        split_position: float = 0.5,
        **kwargs
):
    tmp_dir = r'tmp'
    tmp_files = r'./tmp/left.jpg', r'./tmp/right.jpg'
    os.chdir(data_dir)
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
    if not output_file:
        output_file, extension = img_file.split('.')
        output_file = f'{output_file}_straight.{extension}'

    split_page.split_page(
        img_file,
        data_dir,
        split_position,
        tmp_files,
    )
    orig_img: Image = Image.open(img_file)
    x_min, y_min, x_max, y_max = orig_img.getbbox()
    split_x = int((x_max - x_min) * split_position)
    boxes = [
        (x_min, y_min, split_x, y_max),
        (split_x, y_min, x_max, y_max),
    ]
    for img_, box in zip(tmp_files, boxes):
        img = imgproc.ImageProc(img_)
        parameters = DetectLinesParam(img, **kwargs)
        img.detect_lines(**parameters.params)
        rot_or_skew_type, rot_radians = img.find_rotation_or_skew(
            only_direction=DIRECTION_HORIZONTAL,
            rot_thresh=math.radians(1),
            rot_same_dir_thresh=math.radians(10),
        )
        rot_radians = rot_radians if rot_or_skew_type == ROTATION else 0
        region = orig_img.crop(box)
        region = region.rotate(math.degrees(rot_radians))
        orig_img.paste(region, box)
    orig_img.save(output_file)


if __name__ == '__main__':
    crop_page(
        img_file='3355.jpg',
        output_file='test.jpg',
        data_dir='../data',
        top=500,
        bottom=240,
        left=450,
        right=250,
    )
    straighten_page(
        img_file='test.jpg',
        data_dir='../data',
        output_file='test.jpg',
        split_position=0.5,
    )

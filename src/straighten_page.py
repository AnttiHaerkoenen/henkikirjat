import os
import math

import numpy as np
from PIL import Image
from pdftabextract import imgproc
from pdftabextract.common import ROTATION, DIRECTION_HORIZONTAL, DIRECTION_VERTICAL

import split_page
from parameters import DetectLinesParam


def straighten_page(
        data_dir: str,
        img_file: str,
        output_file: str = None,
        position: float = 0.5,
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
        position,
        tmp_files,
    )
    orig_img: Image = Image.open(img_file)
    x_min, y_min, x_max, y_max = orig_img.getbbox()
    split_x = int((x_max - x_min) * position)
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
    straighten_page(
        img_file='3355.jpg',
        data_dir='../data',
        position=0.5,
    )

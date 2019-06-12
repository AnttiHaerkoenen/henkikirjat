import os
import math

import fire
from PIL import Image
from pdftabextract import imgproc
from pdftabextract.common import ROTATION, DIRECTION_HORIZONTAL, DIRECTION_VERTICAL

import split_page


def straighten_page(
        img_file: str,
        data_dir: str,
        position: float = 0.5,
        **kwargs
):
    tmp_files = r'./tmp/left', r'./tmp/right'
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
        img.detect_lines(**kwargs)
        rot_or_skew_type, rot_radians = img.find_rotation_or_skew(
            only_direction=DIRECTION_HORIZONTAL,
            rot_thresh=math.radians(1),
            rot_same_dir_thresh=math.radians(10),
        )
        rot_radians = rot_radians if rot_or_skew_type == ROTATION else 0
        region = orig_img.crop(box)
        region = region.rotate(math.degrees(rot_radians))
        orig_img.paste(region, box)


if __name__ == '__main__':
    fire.Fire(straighten_page)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import fire
import cv2
from pdftabextract import imgproc
from pdftabextract.common import DIRECTION_VERTICAL


def split_page(
        img_file: str,
        data_dir: str,
        position: float = 0.5,
):

    if not 0 <= position <= 1:
        raise ValueError("position should be between 0 and 1")

    input_filename = os.path.join(data_dir, img_file)
    img_proc_obj = imgproc.ImageProc(input_filename)
    image_1, image_2 = img_proc_obj.split_image(
        position * img_proc_obj.img_w,
        direction=DIRECTION_VERTICAL,
    )
    output_files_basename = img_file[:img_file.rindex('.')]
    output_filename_1 = os.path.join(data_dir, f'{output_files_basename}L.jpg')
    output_filename_2 = os.path.join(data_dir, f'{output_files_basename}R.jpg')
    cv2.imwrite(output_filename_1, image_1)
    cv2.imwrite(output_filename_2, image_2)
    print('split images saved')


if __name__ == '__main__':
    fire.Fire(split_page)

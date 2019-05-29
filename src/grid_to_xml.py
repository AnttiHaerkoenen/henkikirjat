#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'AnttiHaerkoenen'

import os
from collections import OrderedDict
from datetime import datetime

import xmltodict
from pdftabextract import imgproc
import numpy as np

from ocr_tools import (
    get_xml_page,
    get_page_scaling,
    get_lines,
    get_grid_pos,
    save_image_w_lines,
    repair_image
)

PAGE_TEMPLATE = r'../src/PAGE_template.xml'


def page_grid_to_xml(
        data_dir: str,
        input_file: str,
        output_path: str = None,
        p_num: int = 1,
        min_col_width: int = 20,
        min_row_height: int = 20,
        **hough_param
):
    if not output_path:
        output_path = data_dir

    xml_tree, page = get_xml_page(
        input_file,
        data_dir,
        p_num,
    )
    img_file_basename = page['image'][:page['image'].rindex('.')]
    img_file = os.path.join(data_dir, page['image'])
    img_proc_obj = imgproc.ImageProc(img_file)

    page_scaling_x, page_scaling_y = get_page_scaling(img_proc_obj, page)

    lines_hough = get_lines(img_proc_obj, **hough_param)
    img_proc_obj.lines_hough = lines_hough

    save_image_w_lines(
        img_proc_obj,
        img_file_basename,
        output_path,
    )
    repair_image(
        xml_tree,
        img_proc_obj,
        page,
        img_file,
        output_path,
    )
    page_col_pos, page_row_pos = get_grid_pos(
        img_proc_obj,
        page,
        page_scaling_x,
        page_scaling_y,
        min_col_width,
        min_row_height,
        output_path,
        img_file_basename,
    )
    print(page_row_pos)
    print(page_col_pos)

    with open(PAGE_TEMPLATE) as fin:
        doc = xmltodict.parse(fin.read())

    text_regions = OrderedDict()

    now = datetime.now()
    now = now.isoformat().split('.')[0]
    doc['PcGts']['Metadata'] = {
        'Creator': __author__,
        'Created': now,
        'LastChange': now,
    }

    doc['PcGts']['Page'] = text_regions
    print(doc)
    with open(f'{img_file_basename}.grids.xml', 'w') as fout:
        fout.writelines(xmltodict.unparse(doc, pretty=True))
    print("grid saved to XML")


if __name__ == '__main__':
    page_grid_to_xml(
        data_dir='../data',
        input_file='data.pdf',
        output_path=None,
        p_num=1,
        min_col_width=20,
        min_row_height=20,
        hough_votes_coef=0.2,
        canny_kernel_size=3,
        canny_low_thresh=50,
        canny_high_thresh=150,
        hough_rho_res=1,
        hough_theta_res=np.pi/500,
    )

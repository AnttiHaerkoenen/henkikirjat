#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'AnttiHaerkoenen'

import os
from pathlib import Path
from collections import OrderedDict
from datetime import datetime
from typing import Union

import xmltodict
from pdftabextract import imgproc, extract
import numpy as np

import ocr_tools
from rectangle import Rectangle, get_rectangle_coords
import tesseract
from parameters import DetectLinesParam


PAGE_TEMPLATE = r'../src/PAGE_template.xml'


def page_grid_to_xml(
        *,
        image: str,
        data_dir: str,
        grid_dir: str,
        output_dir: Union[str, None],
        min_col_width: int,
        min_row_height: int,
        x_offset: int,
        y_offset: int,
        vertical_cluster_method,
        horizontal_cluster_method,
        **hough_param
):
    data_path = Path(data_dir)
    if not output_dir:
        output_path = data_path
    else:
        output_path = Path(output_dir)

    img_file_basename = image.split('.')[0]
    img_file = data_path / image
    img_proc_obj = imgproc.ImageProc(str(img_file))
    hough_param = DetectLinesParam(img_proc_obj, **hough_param)

    lines_hough = img_proc_obj.detect_lines(**hough_param.params)
    img_proc_obj.lines_hough = lines_hough

    ocr_tools.save_image_w_lines(
        img_proc_obj,
        img_file_basename,
        output_path,
    )
    page_col_pos, page_row_pos = ocr_tools.get_grid_pos(
        img_proc_obj=img_proc_obj,
        min_col_width=min_col_width,
        min_row_height=min_row_height,
        output_path=output_path,
        img_file_basename=img_file_basename,
        vertical_cluster_method=vertical_cluster_method,
        horizontal_cluster_method=horizontal_cluster_method,
    )
    page_col_pos = page_col_pos.astype(int) + x_offset
    page_row_pos = page_row_pos.astype(int) + y_offset

    with open(PAGE_TEMPLATE) as fin:
        doc = xmltodict.parse(fin.read())

    now = datetime.utcnow().isoformat() + '+00:00'
    doc['PcGts']['Metadata'] = {
        'Creator': __author__,
        'Created': now,
        'LastChange': now,
    }
    reading_order = OrderedDict({
        '@caption': "Regions reading order",
        'RegionRefIndexed': [{
                '@index': 1,
                '@regionRef': 'r1',
            }],
    })
    table_region = OrderedDict({
        '@id': 'r1',
        '@lineSeparators': 'true',
        'Coords': OrderedDict(
            {'@points': get_rectangle_coords(page_col_pos, page_row_pos)}
        ),
        'TextRegion': [],
    })

    x_pairs = extract.subsequent_pairs(page_col_pos)
    y_pairs = extract.subsequent_pairs(page_row_pos)
    for i, ys in enumerate(y_pairs):
        for j, xs in enumerate(x_pairs):
            n = len(x_pairs) * i + j + 2
            rect = Rectangle(
                x_min=min(xs),
                x_max=max(xs),
                y_min=min(ys),
                y_max=max(ys),
                id=f'r{n}',
            )
            table_region['TextRegion'].append(rect.to_dict())
            reading_order['RegionRefIndexed'].append({
                '@index': n,
                '@regionRef': rect.id,
            })

    doc['PcGts']['Page']['TableRegion'] = table_region
    doc['PcGts']['Page']['ReadingOrder']['OrderedGroup'] = reading_order

    grid_path = Path(grid_dir) / f'{img_file_basename}.xml'
    output = xmltodict.unparse(
        doc,
        pretty=True,
        short_empty_elements=True,
    )
    grid_path.write_text(output)
    print("grid saved to XML")


if __name__ == '__main__':
    page_grid_to_xml(
        data_dir='../data',
        grid_dir='../data/grids',
        image='3355_straight.jpg',
        output_dir=None,
        min_col_width=200,
        min_row_height=200,
        vertical_cluster_method=np.max,
        horizontal_cluster_method=np.max,
        x_offset=0,
        y_offset=10,
        hough_votes_coef=0.25,
        canny_kernel_size=3,
        canny_low_thresh=50,
        canny_high_thresh=150,
        hough_rho_res=1,
        hough_theta_res=np.pi/500,
    )
    # tesseract.predict_page_content(
    #     r'Henkikir_3355R.jpg',
    #     r'./grids/data.pdf-2.xml',
    #     r'../data',
    # )

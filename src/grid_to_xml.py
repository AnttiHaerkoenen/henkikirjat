#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Sequence

__author__ = 'AnttiHaerkoenen'

import os
from pathlib import Path
from collections import OrderedDict
from datetime import datetime

import xmltodict
from pdftabextract import imgproc, extract
import numpy as np

import ocr_tools


PAGE_TEMPLATE = r'../src/PAGE_template.xml'


def pages_to_xml(
        *,
        data_dir: str,
        grid_dir: str = None,
        input_file: str,
        output_path: str = None,
        pages: Sequence = (1,),
        min_col_width: int = 20,
        min_row_height: int = 20,
        **hough_param
):
    data_dir = Path(data_dir)
    input_file = Path(input_file)
    if grid_dir:
        grid_dir = Path(grid_dir)
    else:
        grid_dir = data_dir / 'grids'
    if not grid_dir.is_dir():
        os.mkdir(grid_dir)
    if output_path:
        output_path = Path(output_path)
    else:
        output_path = data_dir

    xml_tree, xml_pages = ocr_tools.get_xml_pages(
        input_file,
        data_dir,
        pages,
    )
    for p_num in pages:
        page_grid_to_xml(
            xml_tree=xml_tree,
            page=xml_pages[p_num],
            data_dir=data_dir,
            grid_dir=grid_dir,
            output_path=output_path,
            min_col_width=min_col_width,
            min_row_height=min_row_height,
            **hough_param
        )


def page_grid_to_xml(
        xml_tree,
        page,
        data_dir: Path,
        grid_dir: Path,
        output_path: Path,
        min_col_width: int,
        min_row_height: int,
        **hough_param
):
    img_file_basename = page['image'][:page['image'].rindex('.')]
    img_file = data_dir / page['image']
    img_proc_obj = imgproc.ImageProc(str(img_file))

    page_scaling_x, page_scaling_y = ocr_tools.get_page_scaling(img_proc_obj, page)

    lines_hough = ocr_tools.get_lines(img_proc_obj, **hough_param)
    img_proc_obj.lines_hough = lines_hough

    ocr_tools.save_image_w_lines(
        img_proc_obj,
        img_file_basename,
        output_path,
    )
    ocr_tools.repair_image(
        xml_tree,
        img_proc_obj,
        page,
        img_file,
        output_path,
    )
    page_col_pos, page_row_pos = ocr_tools.get_grid_pos(
        img_proc_obj,
        page,
        page_scaling_x,
        page_scaling_y,
        min_col_width,
        min_row_height,
        output_path,
        img_file_basename,
    )
    page_col_pos = page_col_pos.astype(int)
    page_row_pos = page_row_pos.astype(int)

    with open(PAGE_TEMPLATE) as fin:
        doc = xmltodict.parse(fin.read())

    now = datetime.utcnow().isoformat()
    doc['PcGts']['Metadata'] = {
        'Creator': __author__,
        'Created': now,
        'LastChange': now,
    }

    table_region = OrderedDict({
        '@id': 'r1',
        '@lineSeparators': 'true',
        'Coords': OrderedDict(
            {'@points': ocr_tools.get_region_coords(page_col_pos, page_row_pos)}
        ),
        'TextRegion': [],
    })
    x_pairs = extract.subsequent_pairs(page_col_pos)
    y_pairs = extract.subsequent_pairs(page_row_pos)
    for i, ys in enumerate(y_pairs):
        for j, xs in enumerate(x_pairs):
            region_id = len(x_pairs) * i + j + 2
            coords = OrderedDict({'@points': ocr_tools.get_region_coords(xs, ys)})
            text_region = OrderedDict({
                '@id': f'r{region_id}',
                '@type': 'paragraph',
                'Coords': coords,
            })
            table_region['TextRegion'].append(text_region)

    doc['PcGts']['Page']['TableRegion'] = table_region
    print(doc)

    grid_path = grid_dir / f'{img_file_basename}.xml'
    grid_path.write_text(xmltodict.unparse(doc, pretty=True))
    print("grid saved to XML")


if __name__ == '__main__':
    pages_to_xml(
        data_dir='../data',
        grid_dir='../data/grids',
        input_file='data.pdf',
        output_path=None,
        pages=(1, 2),
        min_col_width=200,
        min_row_height=200,
        hough_votes_coef=0.25,
        canny_kernel_size=3,
        canny_low_thresh=50,
        canny_high_thresh=150,
        hough_rho_res=1,
        hough_theta_res=np.pi/500,
    )

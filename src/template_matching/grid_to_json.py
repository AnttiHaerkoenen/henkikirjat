#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'AnttiHaerkoenen'

import os
from pathlib import Path
from typing import Union, Sequence, Callable
import json

from pdftabextract import imgproc, extract
import numpy as np

from src.template_matching import ocr_tools
from src.template_matching.rectangle import Rectangle
from src.template_matching.parameters import DetectLinesParam


def make_page_grid(
        *,
        images: Sequence[str],
        grid: str,
        data_dir: str,
        output_dir: Union[str, None],
        min_col_width: int,
        min_row_height: int,
        x_offset: int,
        y_offset: int,
        vertical_cluster_method: Callable[[np.ndarray], np.ndarray],
        horizontal_cluster_method: Callable[[np.ndarray], np.ndarray],
        draw_lines: bool,
        **hough_parameters
):
    data_path = Path(data_dir)
    if not output_dir:
        output_path = data_path
    else:
        output_path = Path(output_dir)

    grid_path = Path(grid)
    doc = json.loads(grid_path.read_text())

    for image in images:
        img_file_basename = image.split('.')[0]
        img_file = data_path / image
        img_proc_obj = imgproc.ImageProc(str(img_file))
        hough_param = DetectLinesParam(img_proc_obj, **hough_parameters)

        lines_hough = img_proc_obj.detect_lines(**hough_param.parameters)
        img_proc_obj.lines_hough = lines_hough

        if draw_lines:
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
            draw_clusters=draw_lines,
        )
        page_col_pos = page_col_pos.astype(int) + x_offset
        page_row_pos = page_row_pos.astype(int) + y_offset

        rects = []

        x_pairs = extract.subsequent_pairs(page_col_pos)
        y_pairs = extract.subsequent_pairs(page_row_pos)

        for i, ys in enumerate(y_pairs):
            for j, xs in enumerate(x_pairs):
                n = len(x_pairs) * i + j
                rect = Rectangle(
                    x_min=min(xs),
                    x_max=max(xs),
                    y_min=min(ys),
                    y_max=max(ys),
                    id=f'r{n}',
                )
                rects.append(rect.to_json_dict())

        doc[img_file_basename] = rects
    grid_path.write_text(json.dumps(doc, indent=4))


if __name__ == '__main__':
    make_page_grid(
        images=['test1.jpg', 'test2.jpg'],
        data_dir='../../data',
        grid='../../data/grids/test.json',
        output_dir=None,
        min_col_width=200,
        min_row_height=200,
        vertical_cluster_method=np.median,
        horizontal_cluster_method=np.median,
        x_offset=0,
        y_offset=10,
        hough_votes_coef=0.25,
        canny_kernel_size=3,
        canny_low_thresh=50,
        canny_high_thresh=150,
        hough_rho_res=1,
        hough_theta_res=np.pi/500,
        draw_lines=False,
    )

"""
PDF tabular data extraction

Based on:
https://github.com/WZBSocialScienceCenter/pdftabextract
WZBSocialScienceCenter/pdftabextract is licensed under the Apache License 2.0

Modified by Antti Härkönen
"""

import os
from math import radians, degrees
from xml.etree import ElementTree
from collections import OrderedDict
from typing import Sequence
from pathlib import Path

import pandas as pd
import numpy as np
import cv2
import fire
from pdftabextract.common import (
    save_page_grids,
    read_xml,
    parse_pages,
    ROTATION,
    SKEW_X,
    SKEW_Y,
)
from pdftabextract.extract import (
    fit_texts_into_grid,
    datatable_to_dataframe,
    make_grid_from_positions,
)
from pdftabextract.geom import pt
from pdftabextract.textboxes import (
    rotate_textboxes,
    deskew_textboxes,
)
from pdftabextract import imgproc
from pdftabextract.clustering import (
    find_clusters_1d_break_dist,
    calc_cluster_centers_1d,
)

from parameters import DetectLinesParam

IMAGE_TYPE = 'jpg'


def save_image_w_lines(
        img_proc_obj,
        img_file,
        output_path,
):
    img_lines = img_proc_obj.draw_lines(orig_img_as_background=True)
    img_lines_file = str(output_path / f'{img_file}-lines-orig.{IMAGE_TYPE}')

    print(f"> saving image with detected lines to {img_lines_file}")
    cv2.imwrite(img_lines_file, img_lines)


def repair_image(
        xml_tree: ElementTree,
        img_proc_obj: imgproc.ImageProc,
        page: OrderedDict,
        img_file: Path,
        output_path: Path,
):
    """
    Find rotation or skew and deskew or rotate boxes
    parameters are:
    1. the minimum threshold in radians for a rotation to be counted as such
    2. the maximum threshold for the difference between horizontal and vertical line rotation (to detect skew)
    3. an optional threshold to filter out "stray" lines whose angle is too far apart from the median angle of
    all other lines that go in the same direction (no effect here)

    :param xml_tree:
    :param img_proc_obj:
    :param page:
    :param img_file:
    :param output_path:
    :return:
    """
    img_file_basename = page['image'][:page['image'].rindex('.')]
    for _ in range(10):
        rot_or_skew_type, rot_or_skew_radians = img_proc_obj.find_rotation_or_skew(
            radians(0.5),
            radians(1),
            omit_on_rot_thresh=radians(0.5),
        )

        if rot_or_skew_type == ROTATION:
            print(f"> rotating back by {-degrees(rot_or_skew_radians)}")
            rotate_textboxes(page, -rot_or_skew_radians, pt(0, 0))
        elif rot_or_skew_type in (SKEW_X, SKEW_Y):
            print(f"> deskewing in direction '{rot_or_skew_type}' by {-degrees(rot_or_skew_radians)}°")
            deskew_textboxes(page, -rot_or_skew_radians, rot_or_skew_type, pt(0, 0))
        else:
            print("> no page rotation / skew found")
            break

    save_image_w_lines(img_proc_obj, img_file_basename + '-repaired', output_path)

    output_files_basename = img_file.name.split('.')[0]
    repaired_xml_file = output_path / (output_files_basename + '.repaired.xml')

    print(f"saving repaired XML file to '{repaired_xml_file}'...")
    xml_tree.write(repaired_xml_file)


def get_grid_pos(
        *,
        img_proc_obj,
        page_scaling_x=1,
        page_scaling_y=1,
        min_col_width,
        min_row_height,
        vertical_cluster_method,
        horizontal_cluster_method,
        output_path,
        img_file_basename,
):
    vertical_clusters = img_proc_obj.find_clusters(
        imgproc.DIRECTION_VERTICAL,
        find_clusters_1d_break_dist,
        dist_thresh=min_col_width / 4,
    )
    print(f"> found {len(vertical_clusters)} clusters")

    img_w_clusters = img_proc_obj.draw_line_clusters(imgproc.DIRECTION_VERTICAL, vertical_clusters)
    save_img_file = str(output_path / f'{img_file_basename}-vertical-clusters.{IMAGE_TYPE}')
    print(f"> saving image with detected vertical clusters to '{save_img_file}'")
    cv2.imwrite(save_img_file, img_w_clusters)

    page_col_pos = np.array(
        calc_cluster_centers_1d(vertical_clusters, method=vertical_cluster_method)
    ) / page_scaling_x
    print(f'found {len(page_col_pos)} column borders')

    horizontal_clusters = img_proc_obj.find_clusters(
        imgproc.DIRECTION_HORIZONTAL,
        find_clusters_1d_break_dist,
        dist_thresh=min_row_height / 4,
    )
    print(f"> found {len(horizontal_clusters)} clusters")

    img_w_clusters = img_proc_obj.draw_line_clusters(
        imgproc.DIRECTION_HORIZONTAL,
        horizontal_clusters,
    )
    save_img_file = str(output_path / f'{img_file_basename}-horizontal-clusters.{IMAGE_TYPE}')
    print(f"> saving image with detected horizontal clusters to '{save_img_file}'")
    cv2.imwrite(save_img_file, img_w_clusters)

    page_row_pos = np.array(
        calc_cluster_centers_1d(horizontal_clusters, method=horizontal_cluster_method)
    ) / page_scaling_y
    print(f'found {len(page_row_pos)} row borders')

    return page_col_pos, page_row_pos


def extract_data_frame(
        page_col_pos,
        page_row_pos,
        p_num,
        img_file,
        output_path,
        page,
):
    grid = make_grid_from_positions(page_col_pos, page_row_pos)
    n_rows = len(grid)
    n_cols = len(grid[0]) if n_rows > 0 else 0
    print(f"> page {p_num}: grid with {n_rows} rows, {n_cols} columns")

    output_files_basename = img_file.name.split('.')[0]
    page_grids_file = output_path + (output_files_basename + '.grids.json')
    print(f"saving page grids JSON file to '{page_grids_file}'")
    save_page_grids({p_num: grid}, page_grids_file)

    table = fit_texts_into_grid(page['texts'], grid)
    return datatable_to_dataframe(table)


def get_lines(
        img_proc_obj,
        hough_param: DetectLinesParam
):
    lines_hough = img_proc_obj.detect_lines(**hough_param.param)
    print(f"> found {len(lines_hough)} lines")
    return lines_hough


def get_xml_pages(
        input_file,
        data_dir,
        pages: Sequence,
):
    if not data_dir.exists():
        raise NotADirectoryError(f"Not a valid directory name: {data_dir}")

    if not (data_dir / input_file).is_file():
        raise FileNotFoundError(f"No file named {input_file}")

    if not isinstance(pages, Sequence):
        raise ValueError(f"{pages} is not a valid page range")

    os.chdir(data_dir)
    os.system(f"pdftohtml -c -hidden -xml {input_file} {input_file}.xml -f {min(pages)} -l {max(pages)}")
    xml_tree, xml_root = read_xml(f"{input_file}.xml")
    pages = parse_pages(xml_root)
    return xml_tree, pages


# def table_extractor(
#         data_dir: str,
#         input_file: str,
#         output_path: str,
#         p_num: int,
#         min_col_width: int,
#         min_row_height: int,
#         **hough_param
# ) -> pd.DataFrame:
#     if not output_path:
#         output_path = data_dir
#
#     xml_tree, page = get_xml_pages(
#         input_file,
#         data_dir,
#         p_num,
#     )
#     img_file_basename = page['image'][:page['image'].rindex('.')]
#     img_file = os.path.join(data_dir, page['image'])
#     img_proc_obj = imgproc.ImageProc(img_file)
#
#     page_scaling_x, page_scaling_y = get_page_scaling(img_proc_obj, page)
#
#     lines_hough = get_lines(img_proc_obj, **hough_param)
#     img_proc_obj.lines_hough = lines_hough
#
#     save_image_w_lines(
#         img_proc_obj,
#         img_file_basename,
#         output_path,
#     )
#     repair_image(
#         xml_tree,
#         img_proc_obj,
#         page,
#         img_file,
#         output_path,
#     )
#     page_col_pos, page_row_pos = get_grid_pos(
#         img_proc_obj,
#         page,
#         page_scaling_x,
#         page_scaling_y,
#         min_col_width,
#         min_row_height,
#         output_path,
#         img_file_basename,
#     )
#     data_frame = extract_data_frame(
#         page_col_pos,
#         page_row_pos,
#         p_num,
#         img_file,
#         output_path,
#         page,
#     )
#     return data_frame


if __name__ == '__main__':
    pass

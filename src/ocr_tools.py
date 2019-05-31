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


def save_image_w_lines(
        img_proc_obj,
        img_file,
        output_path
):
    img_lines = img_proc_obj.draw_lines(orig_img_as_background=True)
    img_lines_file = os.path.join(output_path, f'{img_file}-lines-orig.png')

    print(f"> saving image with detected lines to {img_lines_file}")
    cv2.imwrite(img_lines_file, img_lines)


def repair_image(
        xml_tree: ElementTree,
        img_proc_obj: imgproc.ImageProc,
        page: OrderedDict,
        img_file: str,
        output_path: str,
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

    output_files_basename = img_file[:img_file.rindex('.')]
    repaired_xml_file = os.path.join(output_path, output_files_basename + '.repaired.xml')

    print(f"saving repaired XML file to '{repaired_xml_file}'...")
    xml_tree.write(repaired_xml_file)


def get_grid_pos(
        img_proc_obj,
        page,
        page_scaling_x,
        page_scaling_y,
        min_col_width,
        min_row_height,
        output_path,
        img_file_basename,
):
    # cluster the detected *vertical* lines using find_clusters_1d_break_dist as simple clustering function
    # (break on distance min_col_width / 2)
    # additionally, remove all cluster sections that are considered empty
    # a cluster is considered empty when the number of text boxes in it is below 10% of the median number of text boxes
    # per cluster section
    vertical_clusters = img_proc_obj.find_clusters(
        imgproc.DIRECTION_VERTICAL,
        find_clusters_1d_break_dist,
        remove_empty_cluster_sections_use_texts=page['texts'],
        remove_empty_cluster_sections_n_texts_ratio=0.1,
        remove_empty_cluster_sections_scaling=page_scaling_x,
        dist_thresh=min_col_width / 4,
    )
    print(f"> found {len(vertical_clusters)} clusters")

    # draw the clusters
    img_w_clusters = img_proc_obj.draw_line_clusters(imgproc.DIRECTION_VERTICAL, vertical_clusters)
    save_img_file = os.path.join(output_path, f'{img_file_basename}-vertical-clusters.png')
    print(f"> saving image with detected vertical clusters to '{save_img_file}'")
    cv2.imwrite(save_img_file, img_w_clusters)

    page_col_pos = np.array(calc_cluster_centers_1d(vertical_clusters)) / page_scaling_x
    print(f'found {len(page_col_pos)} column borders:')
    print(page_col_pos)

    # same for horizontal clusters
    horizontal_clusters = img_proc_obj.find_clusters(
        imgproc.DIRECTION_HORIZONTAL,
        find_clusters_1d_break_dist,
        remove_empty_cluster_sections_use_texts=page['texts'],
        remove_empty_cluster_sections_n_texts_ratio=0.1,
        remove_empty_cluster_sections_scaling=page_scaling_y,
        dist_thresh=min_row_height / 4,
    )
    print(f"> found {len(horizontal_clusters)} clusters")

    img_w_clusters = img_proc_obj.draw_line_clusters(
        imgproc.DIRECTION_HORIZONTAL,
        horizontal_clusters,
    )
    save_img_file = os.path.join(output_path, f'{img_file_basename}-horizontal-clusters.png')
    print(f"> saving image with detected horizontal clusters to '{save_img_file}'")
    cv2.imwrite(save_img_file, img_w_clusters)

    page_row_pos = np.array(calc_cluster_centers_1d(horizontal_clusters)) / page_scaling_y
    print(f'found {len(page_row_pos)} row borders:')
    print(page_row_pos)

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

    output_files_basename = img_file[:img_file.rindex('.')]
    page_grids_file = os.path.join(output_path, output_files_basename + '.grids.json')
    print(f"saving page grids JSON file to '{page_grids_file}'")
    save_page_grids({p_num: grid}, page_grids_file)

    table = fit_texts_into_grid(page['texts'], grid)
    return datatable_to_dataframe(table)


def get_lines(
        img_proc_obj,
        **hough_param
):
    hough_param["hough_votes_thresh"] = round(hough_param["hough_votes_coef"] * img_proc_obj.img_w)
    del hough_param["hough_votes_coef"]
    lines_hough = img_proc_obj.detect_lines(**hough_param)
    print(f"> found {len(lines_hough)} lines")
    return lines_hough


def get_xml_page(
        input_file,
        data_dir,
        p_num,
):
    if not os.path.isdir(data_dir):
        raise NotADirectoryError(f"Not a valid directory name: {data_dir}")

    if not os.path.isfile(os.path.join(data_dir, input_file)):
        raise FileNotFoundError(f"No file named {input_file}")

    if not isinstance(p_num, int):
        raise ValueError(f"{p_num} is not a valid page number")

    os.chdir(data_dir)
    os.system(f"pdftohtml -c -hidden -xml {input_file} {input_file}.xml -f {p_num} -l {p_num}")
    xml_tree, xml_root = read_xml(f"{input_file}.xml")
    page = parse_pages(xml_root)[p_num]
    return xml_tree, page


def get_page_scaling(
        img_proc_obj,
        page,
):
    page_scaling_x = img_proc_obj.img_w / page['width']
    page_scaling_y = img_proc_obj.img_h / page['height']
    return page_scaling_x, page_scaling_y


def get_region_coords(x: Sequence, y: Sequence) -> str:
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    return f"{x_min},{y_min} {x_min},{y_max} {x_max},{y_max} {x_max},{y_min}"


def table_extractor(
        data_dir: str,
        input_file: str,
        output_path: str,
        p_num: int,
        min_col_width: int,
        min_row_height: int,
        **hough_param
) -> pd.DataFrame:
    """
    Extracts table from OCR:d table
    :param data_dir:
    :param input_file:
    :param output_path:
    :param p_num:
    :param min_col_width:
    :param min_row_height:
    :return:
    """
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
    data_frame = extract_data_frame(
        page_col_pos,
        page_row_pos,
        p_num,
        img_file,
        output_path,
        page,
    )
    return data_frame


if __name__ == '__main__':
    pass

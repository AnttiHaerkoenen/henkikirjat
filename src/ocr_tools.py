"""
PDF tabular data extraction

Based on:
https://github.com/WZBSocialScienceCenter/pdftabextract
WZBSocialScienceCenter/pdftabextract is licensed under the Apache License 2.0

Modified by Antti Härkönen
"""

import os
from math import radians, degrees

import pandas as pd
import numpy as np
import cv2
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
    make_grid_from_positions
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


def save_image_w_lines(img_proc_obj, img_file, output_path):
    img_lines = img_proc_obj.draw_lines(orig_img_as_background=True)
    img_lines_file = os.path.join(output_path, f'{img_file}-lines-orig.png')

    print(f"> saving image with detected lines to {img_lines_file}")
    cv2.imwrite(img_lines_file, img_lines)


def table_extractor(
        data_dir: str,
        input_file: str,
        output_path: str,
        p_num: int,
        min_col_width: int,
        min_row_height: int,
) -> pd.DataFrame:
    if not os.path.isdir(data_dir):
        raise NotADirectoryError(f"Not a valid directory name: {data_dir}")
    if not os.path.isfile(os.path.join(data_dir, input_file)):
        raise FileNotFoundError(f"No file named {input_file}")
    if not output_path:
        output_path = data_dir

    os.chdir(data_dir)
    os.system(f"pdftohtml -c -hidden -xml {input_file} {input_file}.xml -f {p_num} -l {p_num}")

    xml_tree, xml_root = read_xml(f"{input_file}.xml")
    page = parse_pages(xml_root)[p_num]

    print(f"detecting lines in image {input_file}...")

    imgfilebasename = page['image'][:page['image'].rindex('.')]
    imgfile = os.path.join(data_dir, page['image'])
    img_proc_obj = imgproc.ImageProc(imgfile)

    # calculate the scaling of the image file in relation to the text boxes coordinate system dimensions
    page_scaling_x = img_proc_obj.img_w / page['width']  # scaling in X-direction
    page_scaling_y = img_proc_obj.img_h / page['height']  # scaling in Y-direction

    # detect the lines
    lines_hough = img_proc_obj.detect_lines(
        canny_kernel_size=3,
        canny_low_thresh=50,
        canny_high_thresh=150,
        hough_rho_res=1,
        hough_theta_res=np.pi / 500,
        hough_votes_thresh=round(0.2 * img_proc_obj.img_w),
    )
    print(f"> found {len(lines_hough)} lines")

    save_image_w_lines(img_proc_obj, imgfilebasename, output_path)

    # find rotation or skew
    # the parameters are:
    # 1. the minimum threshold in radians for a rotation to be counted as such
    # 2. the maximum threshold for the difference between horizontal and vertical line rotation (to detect skew)
    # 3. an optional threshold to filter out "stray" lines whose angle is too far apart from the median angle of
    #    all other lines that go in the same direction (no effect here)
    rot_or_skew_type, rot_or_skew_radians = img_proc_obj.find_rotation_or_skew(
        radians(0.5),
        radians(1),
        omit_on_rot_thresh=radians(0.5),
    )

    # rotate back or deskew text boxes
    needs_fix = True
    if rot_or_skew_type == ROTATION:
        print("> rotating back by %f°" % -degrees(rot_or_skew_radians))
        rotate_textboxes(page, -rot_or_skew_radians, pt(0, 0))
    elif rot_or_skew_type in (SKEW_X, SKEW_Y):
        print(f"> deskewing in direction '{rot_or_skew_type}' by {-degrees(rot_or_skew_radians)}°")
        deskew_textboxes(page, -rot_or_skew_radians, rot_or_skew_type, pt(0, 0))
    else:
        needs_fix = False
        print("> no page rotation / skew found")

    if needs_fix:
        # rotate back or deskew detected lines
        lines_hough = img_proc_obj.apply_found_rotation_or_skew(rot_or_skew_type, -rot_or_skew_radians)
        save_image_w_lines(img_proc_obj, imgfilebasename + '-repaired', output_path)

    output_files_basename = input_file[:input_file.rindex('.')]
    repaired_xmlfile = os.path.join(output_path, output_files_basename + '.repaired.xml')

    print(f"saving repaired XML file to '{repaired_xmlfile}'...")
    xml_tree.write(repaired_xmlfile)

    # cluster the detected *vertical* lines using find_clusters_1d_break_dist as simple clustering function
    # (break on distance min_col_width/2)
    # additionally, remove all cluster sections that are considered empty
    # a cluster is considered empty when the number of text boxes in it is below 10% of the median number of text boxes
    # per cluster section
    vertical_clusters = img_proc_obj.find_clusters(
        imgproc.DIRECTION_VERTICAL,
        find_clusters_1d_break_dist,
        remove_empty_cluster_sections_use_texts=page['texts'],
        remove_empty_cluster_sections_n_texts_ratio=0.1,
        remove_empty_cluster_sections_scaling=page_scaling_x,
        dist_thresh=min_col_width / 2,
    )
    print(f"> found {len(vertical_clusters)} clusters")

    # draw the clusters
    img_w_clusters = img_proc_obj.draw_line_clusters(imgproc.DIRECTION_VERTICAL, vertical_clusters)
    save_img_file = os.path.join(output_path, f'{imgfilebasename}-vertical-clusters.png')
    print(f"> saving image with detected vertical clusters to '{save_img_file}'")
    cv2.imwrite(save_img_file, img_w_clusters)

    page_colpos = np.array(calc_cluster_centers_1d(vertical_clusters)) / page_scaling_x
    print(f'found {len(page_colpos)} column borders:')
    print(page_colpos)

    # same for horizontal clusters
    horizontal_clusters = img_proc_obj.find_clusters(
        imgproc.DIRECTION_HORIZONTAL,
        find_clusters_1d_break_dist,
        remove_empty_cluster_sections_use_texts=page['texts'],
        remove_empty_cluster_sections_n_texts_ratio=0.1,
        remove_empty_cluster_sections_scaling=page_scaling_y,
        dist_thresh=min_row_height / 2,
    )
    print(f"> found {len(horizontal_clusters)} clusters")

    img_w_clusters = img_proc_obj.draw_line_clusters(
        imgproc.DIRECTION_HORIZONTAL,
        horizontal_clusters,
    )
    save_img_file = os.path.join(output_path, f'{imgfilebasename}-horizontal-clusters.png')
    print(f"> saving image with detected horizontal clusters to '{save_img_file}'")
    cv2.imwrite(save_img_file, img_w_clusters)

    page_rowpos = np.array(calc_cluster_centers_1d(horizontal_clusters)) / page_scaling_y
    print(f'found {len(page_rowpos)} row borders:')
    print(page_rowpos)

    grid = make_grid_from_positions(page_colpos, page_rowpos)
    n_rows = len(grid)
    n_cols = len(grid[0]) if n_rows > 0 else 0
    print(f"> page {p_num}: grid with {n_rows} rows, {n_cols} columns")

    page_grids_file = os.path.join(output_path, output_files_basename + '.pagegrids.json')
    print(f"saving page grids JSON file to '{page_grids_file}'")
    save_page_grids({p_num: grid}, page_grids_file)

    table = fit_texts_into_grid(page['texts'], grid)
    return datatable_to_dataframe(table)


if __name__ == '__main__':
    tbl = table_extractor(
        r"../data",
        r"esim.pdf",
        None,
        1,
        100,
        100,
    )
    print(tbl.values)
    print(tbl.iloc[1,1])

import os
import re
from math import radians, degrees

import pandas as pd
import numpy as np
import cv2
from pdftabextract.common import save_page_grids, read_xml, parse_pages, all_a_in_b,\
    ROTATION, SKEW_X, SKEW_Y, DIRECTION_VERTICAL
from pdftabextract.extract import fit_texts_into_grid, datatable_to_dataframe, make_grid_from_positions
from pdftabextract.geom import pt
from pdftabextract.textboxes import rotate_textboxes, deskew_textboxes, \
    border_positions_from_texts, split_texts_by_positions, join_texts
from pdftabextract import imgproc
from pdftabextract.clustering import find_clusters_1d_break_dist, calc_cluster_centers_1d, zip_clusters_and_values


def save_image_w_lines(img_proc_obj, img_file, output_path):
    img_lines = img_proc_obj.draw_lines(orig_img_as_background=True)
    img_lines_file = os.path.join(output_path, f'{img_file}-lines-orig.png')

    print(f"> saving image with detected lines to {img_lines_file}")
    cv2.imwrite(img_lines_file, img_lines)


def extract_table(data_dir: str, input_file: str, output_path: str=None) -> pd.DataFrame:
    if not os.path.isdir(data_dir):
        raise NotADirectoryError(f"Not a valid directory name: {data_dir}")
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"No file named {input_file}")
    if not output_path:
        output_path = data_dir

    os.chdir(data_dir)
    os.system(f"pdftohtml -c -hidden -xml {input_file} {input_file}.xml")

    xml_tree, xml_root = read_xml(os.path.join(data_dir, f"{input_file}.xml"))
    pages = parse_pages(xml_root)
    p_num = 3
    p = pages[p_num]

    print(f"detecting lines in image {input_file}...")

    imgfilebasename = p['image'][:p['image'].rindex('.')]
    imgfile = os.path.join(data_dir, p['image'])
    img_proc_obj = imgproc.ImageProc(imgfile)

    # calculate the scaling of the image file in relation to the text boxes coordinate system dimensions
    page_scaling_x = img_proc_obj.img_w / p['width']  # scaling in X-direction
    page_scaling_y = img_proc_obj.img_h / p['height']  # scaling in Y-direction

    # detect the lines
    lines_hough = img_proc_obj.detect_lines(canny_kernel_size=3,
                                            canny_low_thresh=50,
                                            canny_high_thresh=150,
                                            hough_rho_res=1,
                                            hough_theta_res=np.pi / 500,
                                            hough_votes_thresh=round(0.2 * img_proc_obj.img_w))
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
        rotate_textboxes(p, -rot_or_skew_radians, pt(0, 0))
    elif rot_or_skew_type in (SKEW_X, SKEW_Y):
        print(f"> deskewing in direction '{rot_or_skew_type}' by {-degrees(rot_or_skew_radians)}°")
        deskew_textboxes(p, -rot_or_skew_radians, rot_or_skew_type, pt(0, 0))
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
    MIN_COL_WIDTH = 60  # minimum width of a column in pixels, measured in the scanned pages

    # cluster the detected *vertical* lines using find_clusters_1d_break_dist as simple clustering function
    # (break on distance MIN_COL_WIDTH/2)
    # additionally, remove all cluster sections that are considered empty
    # a cluster is considered empty when the number of text boxes in it is below 10% of the median number of text boxes
    # per cluster section
    vertical_clusters = img_proc_obj.find_clusters(
        imgproc.DIRECTION_VERTICAL,
        find_clusters_1d_break_dist,
        remove_empty_cluster_sections_use_texts=p['texts'],
        remove_empty_cluster_sections_n_texts_ratio=0.1,
        remove_empty_cluster_sections_scaling=page_scaling_x,
        dist_thresh=MIN_COL_WIDTH / 2,
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

    words_in_footer = ('anzeige', 'annahme', 'ala')
    pttrn_table_row_beginning = re.compile(r'^[\d Oo][\d Oo]{2,} +[A-ZÄÖÜ]')

    # right border of the second column
    col2_rightborder = page_colpos[2]

    # calculate median text box height
    median_text_height = np.median([t['height'] for t in p['texts']])

    # get all texts in the first two columns with a "usual" textbox height
    # we will only use these text boxes in order to determine the line positions because they are more "stable"
    # otherwise, especially the right side of the column header can lead to problems detecting the first table row
    text_height_deviation_thresh = median_text_height / 2
    texts_cols_1_2 = [t for t in p['texts']
                      if t['right'] <= col2_rightborder
                         and abs(t['height'] - median_text_height) <= text_height_deviation_thresh]

    # get all textboxes' top and bottom border positions
    borders_y = border_positions_from_texts(texts_cols_1_2, DIRECTION_VERTICAL)

    # break into clusters using half of the median text height as break distance
    clusters_y = find_clusters_1d_break_dist(borders_y, dist_thresh=median_text_height/2)
    clusters_w_vals = zip_clusters_and_values(clusters_y, borders_y)

    # for each cluster, calculate the median as center
    pos_y = calc_cluster_centers_1d(clusters_w_vals)
    pos_y.append(p['height'])

    # 1. try to find the top row of the table
    texts_cols_1_2_per_line = split_texts_by_positions(
        texts_cols_1_2,
        pos_y,
        DIRECTION_VERTICAL,
        alignment='middle',
        enrich_with_positions=True,
    )

    # go through the texts line per line
    for line_texts, (line_top, line_bottom) in texts_cols_1_2_per_line:
        line_str = join_texts(line_texts)
        if pttrn_table_row_beginning.match(line_str):  # check if the line content matches the given pattern
            top_y = line_top
            break
    else:
        top_y = 0

    # 2. try to find the bottom row of the table
    min_footer_text_height = median_text_height * 1.5
    min_footer_y_pos = p['height'] * 0.7
    # get all texts in the lower 30% of the page that have are at least 50% bigger than the median textbox height
    bottom_texts = [t for t in p['texts']
                    if t['top'] >= min_footer_y_pos and t['height'] >= min_footer_text_height]
    bottom_texts_per_line = split_texts_by_positions(
        bottom_texts,
        pos_y + [p['height']],  # always down to the end of the page
        DIRECTION_VERTICAL,
        alignment='middle',
        enrich_with_positions=True,
    )
    # go through the texts at the bottom line per line
    page_span = page_colpos[-1] - page_colpos[0]
    min_footer_text_width = page_span * 0.8
    for line_texts, (line_top, line_bottom) in bottom_texts_per_line:
        line_str = join_texts(line_texts)
        has_wide_footer_text = any(t['width'] >= min_footer_text_width for t in line_texts)
        # check if there's at least one wide text or if all of the required words for a footer match
        if has_wide_footer_text or all_a_in_b(words_in_footer, line_str):
            bottom_y = line_top
            break
    else:
        bottom_y = p['height']

    page_rowpos = [y for y in pos_y if top_y <= y <= bottom_y]
    print(f"> page {p_num}: {len(page_rowpos)} lines between [{top_y}, {bottom_y}]")

    grid = make_grid_from_positions(page_colpos, page_rowpos)
    n_rows = len(grid)
    n_cols = len(grid[0])
    print(f"> page {p_num}: grid with {n_rows} rows, {n_cols} columns")

    page_grids_file = os.path.join(output_path, output_files_basename + '.pagegrids_p3_only.json')
    print(f"saving page grids JSON file to '{page_grids_file}'")
    save_page_grids({p_num: grid}, page_grids_file)

    table = fit_texts_into_grid(p['texts'], grid)
    return datatable_to_dataframe(table)


if __name__ == '__main__':
    tbl = extract_table()
    print(tbl)

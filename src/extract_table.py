#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PDF tabular data extraction

Based on:
https://github.com/WZBSocialScienceCenter/pdftabextract
WZBSocialScienceCenter/pdftabextract is licensed under the Apache License 2.0

Modified by AnttiHaerkoenen
"""

import fire
import numpy as np

from ocr_tools import table_extractor


def extract_table(
        input_file: str,
        data_dir: str,
        output_path: str = None,
        p_num: int = 1,
        min_col_width=100,
        min_row_height=75,
        canny_kernel_size=3,
        canny_low_thresh=50,
        canny_high_thresh=150,
        hough_rho_res=1,
        hough_theta_coef=500,
        hough_votes_coef=0.3,
):
    """
    :param input_file:
    :param data_dir:
    :param output_path:
    :param p_num:
    :param min_col_width:
    :param min_row_height:
    :param canny_kernel_size:
    :param canny_low_thresh:
    :param canny_high_thresh:
    :param hough_rho_res:
    :param hough_theta_coef: hough_theta_res = np.pi / hough_theta_coef
    :param hough_votes_coef: hough_votes_thresh = round(hough_votes_coef * img_proc_obj.img_w)
    :return:
    """
    hough_param = dict(
        canny_kernel_size=canny_kernel_size,
        canny_low_thresh=canny_low_thresh,
        canny_high_thresh=canny_high_thresh,
        hough_rho_res=hough_rho_res,
        hough_theta_res=np.pi / hough_theta_coef,
        hough_votes_coef=hough_votes_coef,
    )
    tbl = table_extractor(
        data_dir=data_dir,
        input_file=input_file,
        output_path=output_path,
        p_num=p_num,
        min_col_width=min_col_width,
        min_row_height=min_row_height,
        **hough_param
    )
    print(tbl)


if __name__ == '__main__':
    fire.Fire(extract_table)

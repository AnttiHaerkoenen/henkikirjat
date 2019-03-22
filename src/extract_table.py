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

from ocr_tools import table_extractor


def extract_table(
        input_file: str,
        data_dir: str,
        output_path: str = None,
        p_num: int = 1,
        min_col_width=50,
        min_row_height=50,
):
    tbl = table_extractor(
        data_dir=data_dir,
        input_file=input_file,
        output_path=output_path,
        p_num=p_num,
        min_col_width=min_col_width,
        min_row_height=min_row_height,
    )
    print(tbl)


if __name__ == '__main__':
    fire.Fire(extract_table)

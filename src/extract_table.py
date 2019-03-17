#!/usr/bin/env python

"""
PDF tabular data extraction

Based on:
https://github.com/WZBSocialScienceCenter/pdftabextract
WZBSocialScienceCenter/pdftabextract is licensed under the Apache License 2.0

Modified by Antti Härkönen
"""

import fire

from ocr_tools import table_extractor


def extract_table(data_dir: str, input_file: str, output_path: str = None, p_num: int = 1):
    tbl = table_extractor(
        data_dir=data_dir,
        input_file=input_file,
        output_path=output_path,
        p_num=p_num
    )
    print(tbl)


if __name__ == '__main__':
    fire.Fire(extract_table)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
from collections import OrderedDict

import fire
import xmltodict


PAGE_TEMPLATE = r'../src/PAGE_template.xml'


def page_grid_to_xml(
        grid_file: str,
        data_dir: str,
):
    """
    Saves contents of grid file to PAGE xml file
    :param grid_file:
    :param data_dir:
    :return:
    """
    os.chdir(data_dir)
    grid_file_basename = grid_file[:grid_file.rindex('.')]

    with open(grid_file) as fin:
        grids = json.load(fin)
        print(f"grids loaded from {grid_file}")

    with open(PAGE_TEMPLATE) as fin:
        doc = xmltodict.parse(fin.read())

    text_regions = OrderedDict()
    for i, v in enumerate(grids['1']):
        print(i)
        print(v)
    print(grids['1'][0])#[0][0])
    doc['PcGts']['Page'] = text_regions
    print(doc)
    with open(f'{grid_file_basename}.xml', 'w') as fout:
        fout.writelines(xmltodict.unparse(doc, pretty=True))
    print("grid saved to XML")


if __name__ == '__main__':
    fire.Fire(page_grid_to_xml)

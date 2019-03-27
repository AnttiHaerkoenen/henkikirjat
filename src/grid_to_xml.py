#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import fire
import xmltodict


def page_grid_to_xml(
        img_file: str,
        data_dir: str,
):
    """
    Saves contents of grid file to PAGE xml file
    :param img_file:
    :param data_dir:
    :return:
    """
    os.chdir(data_dir)
    img_file_basename = img_file[:img_file.rindex('.')]

    with open(img_file) as fd:
        doc = xmltodict.parse(fd.read())

    with open(f'{img_file_basename}.xml', 'w') as outf:
        outf.writelines(xmltodict.unparse(doc, pretty=True))


if __name__ == '__main__':
    fire.Fire(page_grid_to_xml)

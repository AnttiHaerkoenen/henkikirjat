from pathlib import Path
import os
from typing import Sequence
import json

import numpy as np
import pandas as pd

from src.template_matching.rectangle import Rectangle
from src.template_matching.img_tools import view_rectangle
from src.template_matching.digits import Digits
from src.template_matching.enums import TemplateMatchingMethod


def match_locations_to_rectangles(
        digits: Digits,
        rectangles: Sequence[Rectangle]
):
    # todo
    x = digits.coordinates['x'].values
    y = digits.coordinates['y'].values
    for rect in rectangles:
        inside = ((x >= rect.x_min) & (y >= rect.y_min)) & ((x <= rect.x_max) & (y <= rect.y_max))
        inside_digits = digits.coordinates[inside]
        if np.any(inside):
            # todo group by x-dim dist
            # todo df.groupby(group).max().copy()
            # todo create group-dim prob space for permutations
            #  and convert to dict[perm, prob]
            print(rect)
            print(inside_digits)


def predict_page_content(
        img_file,
        grid_file,
        data_dir,
):
    os.chdir(data_dir)
    img_path = Path(img_file)
    grid_path = Path(grid_file)


if __name__ == '__main__':
    # predict_page_content(
    #     r'test.jpg',
    #     r'./grids/test.xml',
    #     r'../../data',
    # )

    os.chdir('../../data')
    templates = {i: [Path('./digit_templates') / f"{i}.jpg"] for i in "1 2 3 4 5".split()}
    canny_parameters = {'threshold1': 400, 'threshold2': 1000}
    thresholds = {
        '1': 0.5,
        '2': 0.3,
        '3': 0.3,
        '4': 0.3,
        '5': 0.3,
    }
    digits = Digits(
        image_path=Path('test.jpg'),
        templates=templates,
        canny_parameters=canny_parameters,
        template_matching_method=TemplateMatchingMethod.CCOEF_NORM,
        threshold_values=thresholds,
    )
    grid_path = Path('./grids/test.json')
    rectangles = [Rectangle.from_json_dict(e) for e in json.loads(grid_path.read_text())['1']]
    match_locations_to_rectangles(digits, rectangles)

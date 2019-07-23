from pathlib import Path
import os
from typing import Sequence

import json

from src.template_matching.rectangle import Rectangle
from src.template_matching.img_tools import view_rectangle
from src.template_matching.digits import Digits


def match_locations_to_rectangles(
        digits: Digits,
        rectangles: Sequence[Rectangle]
):
    # todo
    pass


def predict_page_content(
        img_file,
        grid_file,
        data_dir,
):
    os.chdir(data_dir)
    img_path = Path(img_file)
    grid_path = Path(grid_file)


if __name__ == '__main__':
    predict_page_content(
        r'test.jpg',
        r'./grids/test.xml',
        r'../../data',
    )

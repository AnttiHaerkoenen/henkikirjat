from pathlib import Path
import os
from typing import Iterable, Mapping, Sequence
import json

import numpy as np
import pandas as pd

from src.template_matching.rectangle import Rectangle
from src.template_matching.img_tools import view_rectangle
from src.template_matching.digits import Digits
from src.template_matching.enums import TemplateMatchingMethod
from src.template_matching.grid_to_json import make_page_grid
from src.template_matching.parameters import CannyParam, GridParam
import cv2


def match_locations_to_rectangles(
        digits: Digits,
        rectangles: Iterable[Rectangle],
) -> Iterable[Rectangle]:
    for rect in rectangles:
        digit_list = []
        inside = digits.digit_locations.within_rectangle(rect, combined=True)
        if not inside.empty:
            probabilities = inside.sort_values(by='x')
            for _, row in probabilities.iterrows():
                candidates = row.drop(['x', 'y']).sort_values(ascending=False)
                candidates = list(candidates[candidates > 0].index)
                digit_list.append(candidates)
        rect.predicted = digit_list
    return rectangles


def predict_page_content(
        images: Sequence[str],
        grid_file,
        make_grid: bool,
        data_dir,
        templates: Mapping[str, Sequence[Path]],
        digit_threshold_values: Mapping[str, float],
        template_matching_method: TemplateMatchingMethod,
        grouping_distance: int,
        canny_parameters: CannyParam,
        grid_parameters: GridParam,
        save_edges: bool = False,
        **hough_parameters
):
    os.chdir(data_dir)
    grid_path = Path(grid_file)
    if make_grid:
        make_page_grid(
            images=images,
            data_dir=os.curdir,
            grid=grid_file,
            output_dir=None,
            **grid_parameters.parameters,
            draw_lines=False,
            **hough_parameters
        )
    grid = json.loads(grid_path.read_text())

    for image in images:
        img_basename = image.split('.')[0]
        digits = Digits(
            image_path=Path(image),
            templates=templates,
            canny_parameters=canny_parameters,
            template_matching_method=template_matching_method,
            threshold_values=digit_threshold_values,
            grouping_distance=grouping_distance,
        )
        if save_edges:
            cv2.imwrite(f"{img_basename}_edges.jpg", digits.edges)
        rectangles = [Rectangle.from_json_dict(rect) for rect in grid[img_basename]]
        rectangles = match_locations_to_rectangles(digits, rectangles)
        grid[img_basename] = [rect.to_json_dict() for rect in rectangles]

    grid_path.write_text(json.dumps(grid, indent=4))


if __name__ == '__main__':
    grid_param = GridParam(
        min_col_width=200,
        min_row_height=200,
        vertical_cluster_method=np.median,
        horizontal_cluster_method=np.median,
        x_offset=0,
        y_offset=10,
    )
    canny_par = CannyParam(400, 1000)
    digit_thresholds = {
        '1': 0.2,
        '2': 0.2,
        '3': 0.2,
        '4': 0.2,
        '5': 0.2,
    }
    templates = {
        i: [
            Path('./digit_templates/1900') / f"{i}_{j}.jpg" for j in "1 2 3 4 5".split()
        ] for i in "1 2 3 4 5".split()
    }
    predict_page_content(
        images=['test1.jpg', 'test2.jpg'],
        grid_file='./grids/test.json',
        make_grid=False,
        data_dir='../../data',
        grouping_distance=5,
        template_matching_method=TemplateMatchingMethod.CCOEF_NORM,
        templates=templates,
        grid_parameters=grid_param,
        digit_threshold_values=digit_thresholds,
        canny_parameters=canny_par,
        hough_votes_coef=0.25,
        canny_kernel_size=3,
        canny_low_thresh=50,
        canny_high_thresh=150,
        hough_rho_res=1,
        hough_theta_res=np.pi/500,
        save_edges=True,
    )

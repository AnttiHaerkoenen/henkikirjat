import os
from typing import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from .enums import TemplateMatchingMethod

__all__ = [
    'Digits',
]


class Digits:
    def __init__(
            self,
            image_path: Path,
            templates: Mapping[str, Sequence[Path]],
            template_matching_method: TemplateMatchingMethod,
            canny_parameters: Mapping,
            threshold_values: Mapping[str, float],
            grouping_distance: int = 5,
    ):
        if set(templates) != set(threshold_values):
            raise ValueError("Templates and threshold values must match!")

        self.image_path = image_path
        self.templates = templates
        self.image: np.ndarray = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        self.gray_image: np.ndarray = cv2.imread(str(self.image_path), cv2.IMREAD_GRAYSCALE)
        self.h, self.w = self.gray_image.shape
        self.grouping_distance = grouping_distance
        self.locations = None
        self.normalized = None

        self.template_matching_method = template_matching_method

        self.edges = None
        self.canny_parameters = canny_parameters
        self.threshold_values = pd.Series(threshold_values)
        self._analyse()
        self._normalise()

    def __str__(self):
        return str(self.locations)

    def _analyse(self):
        self._check_template_sizes(self.templates)
        self.edges = cv2.Canny(self.gray_image, **self.canny_parameters)
        combined_results = dict()
        for k, l in self.templates.items():
            digit_results = []
            for template_fp in l:
                template = cv2.imread(
                    str(template_fp),
                    cv2.IMREAD_GRAYSCALE,
                )
                matched = cv2.matchTemplate(
                    self.edges,
                    template,
                    self.template_matching_method.value,
                )
                digit_results.append(matched.ravel())
            combined_results[k] = np.mean(np.vstack(digit_results), axis=0)
        self.locations = pd.DataFrame.from_dict(combined_results, orient='columns')

    def _normalise(self):
        mask = self.locations.gt(self.threshold_values.T)
        normalized = self.locations.where(mask, other=0)
        normalized = normalized / normalized.max(axis=0)
        self.normalized = normalized.fillna(0)

    @property
    def coordinates(self) -> pd.DataFrame:
        norm = self.normalized
        coords = norm[norm.any(axis=1)].copy()
        y, x = divmod(coords.index, self.w)
        coords['x'], coords['y'] = x, y
        return coords

    @staticmethod
    def _check_template_sizes(templates):
        sizes = []
        for k, l in templates.items():
            for template_fp in l:
                img = cv2.imread(str(template_fp), cv2.IMREAD_GRAYSCALE)
                sizes.append(img.shape)
        if len(set(sizes)) != 1:
            raise ValueError("All templates must be same shape and size!")


if __name__ == '__main__':
    os.chdir('../data')
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
    print(digits.coordinates)
    print(digits.coordinates['1 2 3 4 5'.split()].sum(axis=1) == digits.coordinates['1 2 3 4 5'.split()].max(axis=1))
    # digits.locations.plot(kind='hist', bins=250, xlim=(0, 0.2), stacked=True)
    for i in '1 2 3 4 5'.split():
        df = digits.coordinates[[i, 'x', 'y']]
        df = df[df[i] > 0].copy()
        df.plot(x='x', y='y', kind='scatter', xlim=(0, digits.w), ylim=(0, digits.h))
    plt.show()

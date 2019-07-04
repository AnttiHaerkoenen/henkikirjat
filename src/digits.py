import os
from typing import Mapping, Sequence
from pathlib import Path
from enum import Enum

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2


class TemplateMatchingMethod(Enum):
    CCOEF = cv2.TM_CCOEFF
    CCOEF_NORM = cv2.TM_CCOEFF_NORMED
    CCORR = cv2.TM_CCORR
    CCORR_NORM = cv2.TM_CCORR_NORMED
    SQDIFF = cv2.TM_SQDIFF
    SQDIFF_NORM = cv2.TM_SQDIFF_NORMED


class Digits:
    def __init__(
            self,
            image_path: Path,
            templates: Mapping[str, Sequence[Path]],
            template_matching_method: TemplateMatchingMethod,
            canny_parameters: Mapping,
            threshold_values: Mapping[str, float],
    ):
        if set(templates) != set(threshold_values):
            raise ValueError("Templates and threshold values must match!")

        self.image_path = image_path
        self.templates = templates
        self.image: np.ndarray = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        self.gray_image: np.ndarray = cv2.imread(str(self.image_path), cv2.IMREAD_GRAYSCALE)
        self.shape = len(templates), self.image.size
        self.locations = None

        self.template_matching_method = template_matching_method.value

        self.edges = None
        self.canny_parameters = canny_parameters
        self.threshold_values = pd.Series(threshold_values)
        self._analyse()

    def __str__(self):
        return str(self.locations)

    def _analyse(self):
        self._check_template_sizes(self.templates)
        self.edges = cv2.Canny(self.gray_image, **self.canny_parameters)
        combined_results = dict()
        for k, l in self.templates.items():
            digit_results = []
            for template_fp in l:
                template = cv2.imread(str(template_fp), cv2.IMREAD_GRAYSCALE)
                matched = cv2.matchTemplate(self.edges, template, self.template_matching_method)
                digit_results.append(matched.ravel())
            combined_results[k] = np.mean(np.vstack(digit_results), axis=0)
        self.locations = pd.DataFrame(combined_results)

    @property
    def normalized(self):
        mask = self.locations.gt(self.threshold_values)
        normalized = self.locations.where(mask, other=0)
        normalized = normalized / normalized.max(axis=0)
        return normalized

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
        '2': 0.5,
        '3': 0.5,
        '4': 0.5,
        '5': 0.5,
    }
    digits = Digits(
        image_path=Path('test.jpg'),
        templates=templates,
        canny_parameters=canny_parameters,
        template_matching_method=TemplateMatchingMethod.CCOEF_NORM,
        threshold_values=thresholds,
    )
    digits.locations.plot(kind='hist', bins=250, xlim=(0, 0.2), stacked=True)
    plt.show()

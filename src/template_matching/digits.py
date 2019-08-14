import os
from typing import Mapping, Sequence, List
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import networkx as nx
from sklearn.metrics import euclidean_distances

from src.template_matching.enums import TemplateMatchingMethod
from src.template_matching.rectangle import Rectangle
from src.template_matching.parameters import CannyParam

__all__ = [
    'Digits',
    'DigitLocations',
]


class DigitLocations:
    def __init__(
            self,
            data: pd.DataFrame,
            *,
            image_path: Path,
            h: int,
            w: int,
            grouping_distance=5,
    ):
        self.image_path = image_path
        self.h = h
        self.w = w
        self.grouping_distance = grouping_distance
        coords = data[data.any(axis=1)].copy()
        y, x = divmod(coords.index, self.w)
        coords['x'] = x
        coords['y'] = y
        self._data = coords
        self._grouped = None

    def __getitem__(self, item):
        return self._data[item]

    def __getattr__(self, item):
        return getattr(self._data, item)

    def __str__(self):
        return str(self._data)

    @property
    def grouped(self):
        if self._grouped is None:
            dist = pd.DataFrame(
                euclidean_distances(self.coordinates.values),
                index=self.index,
                columns=self.index,
            )
            w = dist <= self.grouping_distance
            g = nx.from_pandas_adjacency(w)
            groups = []
            for group, c in enumerate(nx.connected_components(g)):
                groups.extend([group] * len(c))
            self._grouped = self._data
            self._grouped['group'] = groups
        return self._grouped

    @property
    def combined(self):
        values = self.grouped.drop(['x', 'y'], axis=1)
        group_values = values.groupby(['group'], sort=False).max()
        coordinates = self.grouped[['x', 'y', 'group']]
        group_coordinates = coordinates.groupby(['group']).mean()
        combined = group_values.join(group_coordinates)
        return combined

    @property
    def coordinates(self) -> pd.DataFrame:
        return self._data.loc[:, ['x', 'y']]

    def within_rectangle(
            self,
            rectangle: Rectangle,
            combined=True,
    ):
        data = self.combined if combined else self.grouped
        x = data['x']
        y = data['y']
        inside = ((x >= rectangle.x_min) & (y >= rectangle.y_min)) & ((x <= rectangle.x_max) & (y <= rectangle.y_max))
        return data[inside]


class Digits:
    def __init__(
            self,
            image_path: Path,
            templates: Mapping[str, Sequence[Path]],
            template_matching_method: TemplateMatchingMethod,
            canny_parameters: CannyParam,
            threshold_values: Mapping[str, float],
            grouping_distance: int,
    ):
        if set(templates) != set(threshold_values):
            raise ValueError("Templates and threshold values must match!")

        self.image_path = image_path
        self.templates = templates
        self.image: np.ndarray = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        self.gray_image: np.ndarray = cv2.imread(str(self.image_path), cv2.IMREAD_GRAYSCALE)
        self.h, self.w = self.gray_image.shape
        self.grouping_distance = grouping_distance
        self._locations = pd.DataFrame()
        self.normalized = pd.DataFrame()

        self.template_matching_method = template_matching_method

        self.edges = None
        self.canny_parameters = canny_parameters.parameters
        self.threshold_values = pd.Series(threshold_values)
        self._analyse()
        self._normalise()
        self.digit_locations = DigitLocations(
            self.normalized,
            image_path=self.image_path,
            w=self.w,
            h=self.h,
            grouping_distance=self.grouping_distance,
        )

    def __str__(self):
        return str(self._locations)

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
        self._locations = pd.DataFrame.from_dict(combined_results, orient='columns')

    def _normalise(self):
        mask = self._locations.gt(self.threshold_values.T)
        normalized = self._locations.where(mask, other=0)
        normalized = normalized / normalized.max(axis=0)
        self.normalized = normalized.fillna(0)

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
    templates = {
        i: [
            Path('./digit_templates/1900') / f"{i}_{j}.jpg"
            for j in '1 2'.split()
        ] for i in "1 2 3 4 5".split()
    }
    canny_parameters = CannyParam(250, 1000)
    thresholds = {
        '1': 0.5,
        '2': 0.3,
        '3': 0.3,
        '4': 0.3,
        '5': 0.3,
    }
    digits = Digits(
        image_path=Path('test1.jpg'),
        templates=templates,
        canny_parameters=canny_parameters,
        template_matching_method=TemplateMatchingMethod.CCOEF_NORM,
        threshold_values=thresholds,
        grouping_distance=3,
    )
    print(digits.digit_locations.combined)
    # print(digits.coordinates['1 2 3 4 5'.split()].sum(axis=1) == digits.coordinates['1 2 3 4 5'.split()].max(axis=1))
    # digits._locations.plot(kind='hist', bins=250, xlim=(0, 0.2), stacked=True)

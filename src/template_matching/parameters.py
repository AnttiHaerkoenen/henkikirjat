from dataclasses import dataclass
from typing import Callable

import numpy as np
from pdftabextract.imgproc import ImageProc


@dataclass
class GridParam:
    min_col_width: float = 200
    min_row_height: float = 200
    vertical_cluster_method: Callable[[np.ndarray], np.ndarray] = np.median
    horizontal_cluster_method: Callable[[np.ndarray], np.ndarray] = np.median
    x_offset: float = 0
    y_offset: float = 10

    @property
    def parameters(self):
        return {
            "min_col_width": self. min_col_width,
            "min_row_height": self.min_row_height,
            "vertical_cluster_method": self.vertical_cluster_method,
            "horizontal_cluster_method": self.horizontal_cluster_method,
            "x_offset": self.x_offset,
            "y_offset": self.y_offset,
        }


@dataclass
class CannyParam:
    lower_threshold: int
    upper_threshold: int

    @property
    def parameters(self):
        return {
            'threshold1': self.lower_threshold,
            'threshold2': self.upper_threshold,
        }


@dataclass
class DetectLinesParam:
    img_proc_obj: ImageProc
    hough_votes_coef: float = 0.25
    canny_kernel_size: float = 3
    canny_low_thresh: float = 50
    canny_high_thresh: float = 150
    hough_rho_res: float = 1
    hough_theta_res: float = np.pi / 500

    @property
    def parameters(self):
        return {
            'hough_rho_res': self.hough_rho_res,
            'hough_theta_res': self.hough_theta_res,
            'hough_votes_thresh': self.hough_votes_thresh,
            'canny_high_thresh': self.canny_high_thresh,
            'canny_low_thresh': self.canny_low_thresh,
            'canny_kernel_size': self.canny_kernel_size,
        }

    @property
    def hough_votes_thresh(self):
        return round(self.hough_votes_coef * self.img_proc_obj.img_w)

from dataclasses import dataclass

import numpy as np
from pdftabextract.imgproc import ImageProc


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
    def params(self):
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
